#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/hikrobot/hikrobot.hpp"

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace
{
constexpr double TURN_DEG = 120.0;
constexpr double REACHED_DEG = 5.0;

// 固定转向最长等待时间。
constexpr auto TURN_TIMEOUT = 2000ms;

// 转向完成后，至少给电控这段时间执行补扫。
// 这段时间内即使主相机看到目标，也先保持补扫。
constexpr auto SCAN_MIN_TIME = 300ms;

// 同侧持续可见只触发一次；无检测结果一段时间后重新允许触发。
// 注意：无检测结果也可能来自相机/推理暂停，并非严格的空帧确认。
constexpr auto REARM_TIME = 1500ms;

enum class Phase
{
  search,
  turn,
  scan
};

struct StopCameras
{
  ~StopCameras()
  {
    io::HikRobot::request_stop_all();
  }
};
}  // namespace

static int run(int argc, char **argv)
{
  tools::Exiter exiter;

  const std::string config =
    argc > 1 ? argv[1] : "configs/sentry.yaml";

  const auto yaml = tools::load(config);

  if (yaml["control_transport"].as<std::string>("sp") != "sp") {
    tools::logger()->error("Please use control_transport: sp");
    return 1;
  }

  const auto main_serial =
    yaml["camera_serial_number"].as<std::string>();
  const auto left_serial =
    yaml["omni_left_camera_serial_number"].as<std::string>();
  const auto right_serial =
    yaml["omni_right_camera_serial_number"].as<std::string>();

  if (main_serial.empty() || left_serial.empty() ||
      right_serial.empty() || main_serial == left_serial ||
      main_serial == right_serial || left_serial == right_serial) {
    tools::logger()->error("Invalid or duplicate camera serial numbers");
    return 1;
  }

  const auto vid_pid = yaml["vid_pid"].as<std::string>();

  const double exposure = yaml["omni_exposure_ms"]
    ? yaml["omni_exposure_ms"].as<double>()
    : yaml["exposure_ms"].as<double>();

  const double gain = yaml["omni_gain"]
    ? yaml["omni_gain"].as<double>()
    : yaml["gain"].as<double>();

  const float fps = yaml["omni_frame_rate"].as<float>(60.0f);
  const auto max_age = std::chrono::milliseconds(
    yaml["omni_max_detection_age_ms"].as<int>(120));

  if (max_age <= 0ms) {
    tools::logger()->error("omni_max_detection_age_ms must be positive");
    return 1;
  }

  io::Gimbal gimbal(config);
  io::Camera main_camera(config);

  io::HikRobot left_camera(
    exposure, gain, vid_pid, left_serial, fps);
  io::HikRobot right_camera(
    exposure, gain, vid_pid, right_serial, fps);

  StopCameras stop_cameras;

  auto_aim::YOLO yolo(config, false);
  auto_aim::Solver solver(config);
  auto_aim::Tracker tracker(config, solver);
  auto_aim::Aimer aimer(config);

  omniperception::Decider decider(config);
  omniperception::Perceptron perceptron(
    &left_camera, &right_camera, config);

  StopCameras stop_before_consumers;

  // 本程序始终不开火。
  auto send = [&](const io::Command &cmd) {
    io::VisionToGimbal packet{};
    packet.mode = cmd.control ? 1 : 0;
    packet.yaw = static_cast<float>(cmd.yaw);
    packet.pitch = static_cast<float>(cmd.pitch);
    packet.center_yaw = packet.yaw;
    gimbal.send(packet);
  };

  Phase phase = Phase::search;
  double turn_yaw = 0.0;
  double turn_pitch = 0.0;

  auto turn_deadline = Clock::time_point{};
  auto scan_until = Clock::time_point{};
  auto last_report = Clock::now();

  // 辅助检测一直运行，避免被永久关闭。
  std::array<bool, 2> armed{true, true};
  std::array<Clock::time_point, 2> last_seen{};

  const std::vector<omniperception::DetectionResult> no_aux_switch;

  tools::logger()->info(
    "[Omni] left +120deg / right -120deg, then MCU scan; fire disabled");

  while (!exiter.exit()) {
    cv::Mat image;
    Clock::time_point timestamp;
    main_camera.read(image, timestamp);

    if (exiter.exit()) break;

    if (image.empty()) {
      send(io::Command{false, false, 0.0, 0.0});
      continue;
    }

    const auto q = gimbal.q(timestamp - 1ms);
    if (!q.coeffs().allFinite() || q.norm() < 1e-6) {
      send(io::Command{false, false, 0.0, 0.0});
      continue;
    }

    solver.set_R_gimbal2world(q.normalized());
    const Eigen::Vector3d pose =
      tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    if (!pose.allFinite()) {
      send(io::Command{false, false, 0.0, 0.0});
      continue;
    }

    // 主相机照常检测、追踪。
    auto armors = yolo.detect(image);
    decider.armor_filter(armors);
    decider.set_priority(armors);

    auto [unused, targets] =
      tracker.track(no_aux_switch, armors, timestamp);

    io::Command main_cmd{false, false, 0.0, 0.0};
    if (!targets.empty()) {
      main_cmd = aimer.aim(
        targets, timestamp, gimbal.state().bullet_speed);
    }

    const bool main_ready =
      !armors.empty() &&
      !targets.empty() &&
      main_cmd.control &&
      std::isfinite(main_cmd.yaw) &&
      std::isfinite(main_cmd.pitch);

    const auto now = Clock::now();
    const auto detections = perceptron.get_detection_queue();

    // 更新辅助检测记录，转向期间也持续消费结果。
    std::array<bool, 2> visible{false, false};

    for (const auto &result : detections) {
      if (result.armors.empty()) continue;
      if (now < result.timestamp ||
          now - result.timestamp > max_age) continue;

      const int side = result.camera == "left" ? 0 :
                       result.camera == "right" ? 1 : -1;
      if (side < 0) continue;

      visible[side] = true;
      last_seen[side] = now;
    }

    for (int side = 0; side < 2; ++side) {
      if (!visible[side] && !armed[side] &&
          now - last_seen[side] >= REARM_TIME) {
        armed[side] = true;
      }
    }

    io::Command cmd{false, false, 0.0, 0.0};
    const char *source = "mcu_scan";

    // 1. 转向开始后，不让主瞄准提前打断。
    if (phase == Phase::turn) {
      const double error_deg =
        std::abs(tools::limit_rad(turn_yaw - pose[0]))
        * 180.0 / CV_PI;

      if (error_deg <= REACHED_DEG || now >= turn_deadline) {
        phase = Phase::scan;
        scan_until = now + SCAN_MIN_TIME;

        tools::logger()->info(
          "[Omni] MCU scan: reason={} remaining_deg={:.1f}",
          error_deg <= REACHED_DEG ? "reached" : "timeout",
          error_deg);

        // cmd 保持 control=false，即 mode=0。
      } else {
        cmd = io::Command{true, false, turn_yaw, turn_pitch};
        source = "turn";
      }
    }
    // 2. 转过去后，先让小 yaw 补扫，不立即交给主瞄准。
    else if (phase == Phase::scan && now < scan_until) {
      // mode=0：电控执行已有 Scan_Internal()。
    }
    // 3. 主瞄准有效，才接管。
    else if (main_ready) {
      cmd = main_cmd;
      cmd.shoot = false;
      source = "main";
      phase = Phase::search;
    }
    // 4. 主瞄准无目标，电控继续扫描，辅助相机继续寻找。
    else {
      phase = Phase::search;

      // 两侧同一轮都看到时，固定优先左侧。
      for (int side = 0; side < 2; ++side) {
        if (!visible[side] || !armed[side]) continue;

        armed[side] = false;

        const double direction = side == 0 ? 1.0 : -1.0;
        turn_yaw = tools::limit_rad(
          pose[0] + direction * TURN_DEG * CV_PI / 180.0);
        turn_pitch = pose[1];

        turn_deadline = now + TURN_TIMEOUT;
        phase = Phase::turn;

        cmd = io::Command{true, false, turn_yaw, turn_pitch};
        source = "turn";

        tools::logger()->info(
          "[Omni] camera={} delta_deg={:.1f} target_deg={:.1f}",
          side == 0 ? "left" : "right",
          direction * TURN_DEG,
          turn_yaw * 180.0 / CV_PI);
        break;
      }
    }

    if (now - last_report >= 500ms) {
      tools::logger()->info(
        "[control] source={} mode={} target_deg={:.1f} "
        "feedback_deg={:.1f} tracker={}",
        source,
        cmd.control ? 1 : 0,
        cmd.yaw * 180.0 / CV_PI,
        pose[0] * 180.0 / CV_PI,
        tracker.state());
      last_report = now;
    }

    send(cmd);
    std::this_thread::sleep_for(1ms);
  }

  // 按现有电控逻辑，mode=0 在自瞄开启时恢复扫描，不是急停。
  send(io::Command{false, false, 0.0, 0.0});
  return 0;
}

int main(int argc, char **argv)
{
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    tools::logger()->error("[Sentry] {}", e.what());
    return 1;
  }
}