#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>

#include "io/camera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}";

// 两条流水线共用同一弹速，否则 pitch 无法对比。
// 注意 Planner 内部会把 <10 或 >25 夹成 22（planner.cpp:41），所以这里不能填 27
constexpr double kBulletSpeed = 24.0;

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");

  auto yaml = YAML::LoadFile(config_path);
  auto debug = yaml["debug"].as<bool>();

  tools::Plotter plotter;
  tools::Exiter exiter;

  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Planner planner(config_path);

  cv::Mat img, drawing;

  auto_aim::Target last_target;
  io::Command last_command;
  double last_t = -1;

  std::chrono::steady_clock::time_point timestamp;
  for (int frame_count = 0; !exiter.exit(); frame_count++) {
    camera.read(img, timestamp);
    if (img.empty()) break;

    Eigen::Quaterniond q{1, 0, 0, 0};
    auto w = q.w();
    auto x = q.x();
    auto y = q.y();
    auto z = q.z();

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    /// 传统流水线：Aimer
    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, kBulletSpeed, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;

    /// MPC 流水线：Planner
    // 必须显式传 optional<Target>：直接传 targets.front() 会匹配到
    // plan(Target, double) 那个重载，那条路径不做 delay_time 预测
    auto planner_start = std::chrono::steady_clock::now();
    std::optional<auto_aim::Target> target_opt;
    if (!targets.empty()) target_opt = targets.front();
    auto plan = planner.plan(target_opt, kBulletSpeed);
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms, planner: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(planner_start, aimer_start) * 1e3,
      tools::delta_time(finish, planner_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion gimbal_q = {w, x, y, z};
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    tools::draw_text(
      img,
      fmt::format(
        "plan is {},{:.2f},{:.2f},fire:{}", plan.control, plan.yaw * 57.3, plan.pitch * 57.3,
        plan.fire),
      {10, 120}, {0, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      tools::logger()->debug("armor yaw: {:.2f} deg", armor.ypr_in_world[0] * 57.3);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
      data["pnp_x"] = armor.xyz_in_camera[0];
      data["pnp_y"] = armor.xyz_in_camera[1];
      data["pnp_z"] = armor.xyz_in_camera[2];
      data["pnp_yaw"] = armor.ypr_in_camera[0];
      data["pnp_pitch"] = armor.ypr_in_camera[1];
      data["pnp_roll"] = armor.ypr_in_camera[2];
    }

    auto yaw = tools::eulers(q, 2, 1, 0)[0];
    //data["gimbal_yaw"] = yaw * 57.3;
    // 解算出的下发给下位机的 yaw/pitch（度）
    //data["cmd_yaw"] = command.yaw * 57.3;
    //data["cmd_pitch"] = command.pitch * 57.3;
    //data["cmd_control"] = command.control;
    //data["shoot"] = command.shoot;

    // MPC 流水线的对应量。plan_* / target_* 保持弧度，键名与单位都和
    // src/auto_aim_debug_mpc.cpp 一致，方便把两条流水线的曲线叠在一起看
    data["plan_yaw"] = plan.yaw;
    data["plan_pitch"] = plan.pitch;
    data["plan_yaw_vel"] = plan.yaw_vel;
    data["plan_yaw_acc"] = plan.yaw_acc;
    data["plan_pitch_vel"] = plan.pitch_vel;
    data["plan_pitch_acc"] = plan.pitch_acc;
    data["target_yaw"] = plan.target_yaw;
    data["target_pitch"] = plan.target_pitch;
    data["fire"] = plan.fire;

    // 两条流水线的直接差值（度）。只有两边都解出角度时才记，否则
    // command 无效时 yaw=0 会把差值曲线冲出一个假尖峰
    if (command.control && plan.control) {
      data["diff_yaw_deg"] = tools::limit_rad(plan.yaw - command.yaw) * 57.3;
      data["diff_pitch_deg"] = (plan.pitch - command.pitch) * 57.3;
    }

    if (!targets.empty()) {
      auto target = targets.front();

      if (last_t == -1) {
        last_target = target;
        last_t = 0;
        continue;
      }

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后
      armor_xyza_list = target.armor_xyza_list();
      for (std::size_t id = 0; id < armor_xyza_list.size(); ++id) {
        auto image_points =
          solver.reproject_armor(target.armor_pose(id), target.armor_type);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置（红）
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // planner规划瞄准位置（黄）。debug_xyza 是裸 Eigen 向量、没有 valid 标志，
      // 用 plan.control 当守卫，并挡掉未初始化时的非有限值
      if (plan.control && planner.debug_xyza.allFinite()) {
        Eigen::Vector4d plan_xyza = planner.debug_xyza;
        auto plan_image_points =
          solver.reproject_armor(plan_xyza.head(3), plan_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, plan_image_points, {0, 255, 255});
      }

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

    }

    plotter.plot(data);

    if(debug)
    {
      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
    }
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
