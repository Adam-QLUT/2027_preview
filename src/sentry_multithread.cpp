#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <filesystem>
#include <memory>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <stdexcept>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/hikrobot/hikrobot.hpp"
#ifdef SP_VISION_WITH_ROS2
#include "io/ros2/ros2.hpp"
#endif
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/geometry.hpp"
#include "tasks/omniperception/handoff_policy.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tasks/omniperception/trace.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono;
namespace omni_trace = omniperception::trace;
namespace handoff = omniperception::handoff;

namespace
{
enum class OmniHandoffPhase
{
  searching,
  guiding,
  acquiring
};

struct OmniCandidate
{
  bool valid = false;
  std::string camera;
  auto_aim::ArmorName name = auto_aim::ArmorName::not_armor;
  auto_aim::ArmorPriority priority = auto_aim::ArmorPriority::fifth;
  int confirm_count = 0;
  double world_yaw = 0;
  double pitch = 0;
  std::chrono::steady_clock::time_point timestamp{};
};

struct SuppressedOmniCandidate
{
  bool valid = false;
  std::string camera;
  auto_aim::ArmorName name = auto_aim::ArmorName::not_armor;
  double world_yaw = 0;
  std::chrono::steady_clock::time_point until{};
};

const char * phase_name(OmniHandoffPhase phase)
{
  switch (phase) {
    case OmniHandoffPhase::searching:
      return "searching";
    case OmniHandoffPhase::guiding:
      return "guiding";
    case OmniHandoffPhase::acquiring:
      return "acquiring";
  }
  return "unknown";
}

int candidate_slot(const std::string & camera)
{
  if (camera == "left") return 0;
  if (camera == "right") return 1;
  return -1;
}
// FOV fallback assumes a pinhole camera with its principal point at image center,
// and the existing yaw-only mount. Local axes match R_camera2gimbal: x forward,
// y left, z up. A measured calibration is still needed for tilt and parallax.
Eigen::Vector3d fallback_ray(
  double x, double y, double horizontal_fov, double vertical_fov, double mount_yaw)
{
  const Eigen::Vector3d camera_ray(
    1.0, (1.0 - 2.0 * x) * std::tan(horizontal_fov / 2.0),
    (1.0 - 2.0 * y) * std::tan(vertical_fov / 2.0));
  return Eigen::AngleAxisd(mount_yaw, Eigen::Vector3d::UnitZ()) * camera_ray;
}
}  // namespace

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{startup-check  | false               | 仅检查控制链路启动，不创建相机或发送控制指令}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

static int run_sentry(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  const auto yaml = tools::load(config_path);
  if (omni_trace::level()) {
    omni_trace::event(1,
      "stage=start executable={} config={} cwd={} yolo={} variant={} device={} "
      "min_confidence={} use_roi={} use_traditional={}",
      argv[0], std::filesystem::absolute(config_path).string(), std::filesystem::current_path().string(),
      yaml["yolo_name"].as<std::string>(),
      yaml["yolov5_variant"] ? yaml["yolov5_variant"].as<std::string>() : "original",
      yaml["device"].as<std::string>(), yaml["min_confidence"].as<double>(),
      yaml["use_roi"].as<std::string>("unset"), yaml["use_traditional"].as<std::string>("unset"));
  }
  if (!yaml["camera_name"] || yaml["camera_name"].as<std::string>() != "hikrobot") {
    tools::logger()->error("[Sentry] Three-camera mode requires camera_name: hikrobot");
    return 1;
  }
  const auto main_camera_serial = yaml["camera_serial_number"]
                                    ? yaml["camera_serial_number"].as<std::string>()
                                    : "";
  const auto left_camera_serial = yaml["omni_left_camera_serial_number"]
                                    ? yaml["omni_left_camera_serial_number"].as<std::string>()
                                    : "";
  const auto right_camera_serial = yaml["omni_right_camera_serial_number"]
                                     ? yaml["omni_right_camera_serial_number"].as<std::string>()
                                     : "";
  if (
    main_camera_serial.empty() || left_camera_serial.empty() ||
    right_camera_serial.empty()) {
    tools::logger()->error(
      "[Sentry] Set camera_serial_number, omni_left_camera_serial_number and "
      "omni_right_camera_serial_number in {}",
      config_path);
    return 1;
  }
  if (
    main_camera_serial == left_camera_serial || main_camera_serial == right_camera_serial ||
    left_camera_serial == right_camera_serial) {
    tools::logger()->error("[Sentry] Main/left/right camera serial numbers must be different");
    return 1;
  }

  const auto exposure_ms = tools::read<double>(yaml, "exposure_ms");
  const auto gain = tools::read<double>(yaml, "gain");
  const auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
  const auto omni_exposure_ms =
    yaml["omni_exposure_ms"] ? yaml["omni_exposure_ms"].as<double>() : exposure_ms;
  const auto omni_gain = yaml["omni_gain"] ? yaml["omni_gain"].as<double>() : gain;
  const auto omni_frame_rate =
    yaml["omni_frame_rate"] ? static_cast<float>(yaml["omni_frame_rate"].as<double>()) : 60.0f;
  const auto handoff_timeout = std::chrono::milliseconds(
    yaml["omni_handoff_timeout_ms"] ? yaml["omni_handoff_timeout_ms"].as<int>() : 1500);
  const auto takeover_timeout = std::chrono::milliseconds(
    yaml["omni_takeover_timeout_ms"] ? yaml["omni_takeover_timeout_ms"].as<int>() : 800);
  const auto omni_scan_amplitude =
    (yaml["omni_scan_amplitude_deg"] ? yaml["omni_scan_amplitude_deg"].as<double>() : 4.0) *
    CV_PI / 180.0;
  const auto omni_scan_period = std::chrono::milliseconds(
    yaml["omni_scan_period_ms"] ? yaml["omni_scan_period_ms"].as<int>() : 300);
  const auto failed_candidate_cooldown = std::chrono::milliseconds(
    yaml["omni_failed_candidate_cooldown_ms"]
      ? yaml["omni_failed_candidate_cooldown_ms"].as<int>()
      : 500);
  const auto omni_detection_max_age = std::chrono::milliseconds(
    yaml["omni_max_detection_age_ms"] ? yaml["omni_max_detection_age_ms"].as<int>() : 120);
  const auto candidate_max_gap = std::chrono::milliseconds(
    yaml["omni_candidate_max_gap_ms"] ? yaml["omni_candidate_max_gap_ms"].as<int>()
                                        : static_cast<int>(omni_detection_max_age.count()));
  const auto scan_settle_time = std::chrono::milliseconds(
    yaml["omni_scan_settle_ms"] ? yaml["omni_scan_settle_ms"].as<int>() : 100);
  const double scan_start_tolerance =
    (yaml["omni_scan_start_tolerance_deg"] ? yaml["omni_scan_start_tolerance_deg"].as<double>() : 6.0) *
    CV_PI / 180.0;
  const auto candidate_max_yaw_jump =
    (yaml["omni_candidate_max_yaw_jump_deg"]
       ? yaml["omni_candidate_max_yaw_jump_deg"].as<double>()
       : 10.0) *
    CV_PI / 180.0;
  const auto main_match_max_yaw_error =
    (yaml["omni_main_match_max_yaw_error_deg"]
       ? yaml["omni_main_match_max_yaw_error_deg"].as<double>()
       : 20.0) *
    CV_PI / 180.0;
  if (
    handoff_timeout <= 0ms || takeover_timeout <= 0ms || failed_candidate_cooldown < 0ms ||
    omni_detection_max_age <= 0ms || candidate_max_gap <= 0ms || omni_scan_period <= 0ms ||
    !std::isfinite(omni_scan_amplitude) || omni_scan_amplitude < 0 || scan_settle_time < 0ms ||
    !std::isfinite(scan_start_tolerance) || scan_start_tolerance <= 0 ||
    !std::isfinite(candidate_max_yaw_jump) || candidate_max_yaw_jump <= 0 ||
    !std::isfinite(main_match_max_yaw_error) || main_match_max_yaw_error <= 0) {
    tools::logger()->error("[Sentry] Invalid omnidirectional handoff timing/FOV configuration");
    return 1;
  }
  const auto control_transport =
    yaml["control_transport"] ? yaml["control_transport"].as<std::string>() : "cboard";
  if (control_transport != "sp" && control_transport != "cboard") {
    tools::logger()->error(
      "[Sentry] Invalid control_transport '{}'; expected 'sp' or 'cboard'", control_transport);
    return 1;
  }
  const bool use_sp_transport = control_transport == "sp";
  const bool real_vehicle_test_no_fire = yaml["real_vehicle_test_no_fire"]
                                           ? yaml["real_vehicle_test_no_fire"].as<bool>()
                                           : true;
  if (omni_trace::level()) {
    omni_trace::event(1,
      "stage=handoff_config transport={} max_age_ms={} confirm_count=2 yaw_jump_deg={} "
      "main_match_deg={} handoff_ms={} takeover_ms={} cooldown_ms={} imu_freshness_ms={} "
      "min_detect_count={} no_fire={}",
      control_transport, omni_detection_max_age.count(), candidate_max_yaw_jump * 180.0 / CV_PI,
      main_match_max_yaw_error * 180.0 / CV_PI, handoff_timeout.count(), takeover_timeout.count(),
      failed_candidate_cooldown.count(),
      yaml["imu_freshness_timeout_ms"] ? yaml["imu_freshness_timeout_ms"].as<int>() : 50,
      yaml["min_detect_count"].as<int>(), real_vehicle_test_no_fire);
  }

#ifdef SP_VISION_WITH_ROS2
  io::ROS2 ros2;
#else
  tools::logger()->info("[Sentry] ROS2 integration disabled");
#endif
  // Report the actual build and wire ABI before entering any blocking constructor.
  // These are entry-point diagnostics, outside every camera capture path.
  tools::logger()->info(
    "[Sentry][startup] build={} {} source={} config={} transport={} RX_wire={} TX={}",
    __DATE__, __TIME__, __FILE__, std::filesystem::absolute(config_path).string(),
    control_transport,
    yaml["gimbal_rx_frame_bytes"] ? yaml["gimbal_rx_frame_bytes"].as<int>() : 59,
    sizeof(io::VisionToGimbal));
  if (use_sp_transport) {
    tools::logger()->info("[Sentry][startup] stage=control_begin port={}",
                          tools::read<std::string>(yaml, "com_port"));
  }
  std::unique_ptr<io::CBoard> cboard;
  std::unique_ptr<io::Gimbal> gimbal;
  if (use_sp_transport) {
    gimbal = std::make_unique<io::Gimbal>(config_path);
  } else {
    cboard = std::make_unique<io::CBoard>(config_path);
  }
  tools::logger()->info("[Sentry][startup] stage=control_ready");
  if (cli.get<bool>("startup-check")) {
    tools::logger()->info("[Sentry][startup] check complete; exiting before camera/control loops");
    return 0;
  }
  tools::logger()->info(
    "[Sentry] transport={}, real_vehicle_test_no_fire={}", control_transport,
    real_vehicle_test_no_fire);
  tools::logger()->info(
    "[Sentry] HikRobot serial mapping: main={}, left={}, right={}", main_camera_serial,
    left_camera_serial, right_camera_serial);
  io::Camera camera(config_path);
  io::HikRobot left_camera(omni_exposure_ms, omni_gain, vid_pid, left_camera_serial, omni_frame_rate);
  io::HikRobot right_camera(omni_exposure_ms, omni_gain, vid_pid, right_camera_serial, omni_frame_rate);
  struct StopAllCameras
  {
    ~StopAllCameras() { io::HikRobot::request_stop_all(); }
  } stop_all_cameras;

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  omniperception::Decider decider(config_path);
  const omniperception::Geometry omni_geometry(config_path);
  tools::logger()->info(
    "[Sentry] Auxiliary geometry: left={}, right={}; SP_OMNI_TRACE=2 enables detection boundaries",
    omni_geometry.calibrated("left") ? "calibrated_pnp" : "angle_fallback",
    omni_geometry.calibrated("right") ? "calibrated_pnp" : "angle_fallback");
  const double fallback_fov_h = (yaml["omni_horizontal_fov_deg"]
    ? yaml["omni_horizontal_fov_deg"].as<double>() : yaml["new_fov_h"].as<double>()) * CV_PI / 180.0;
  const double fallback_fov_v = (yaml["omni_vertical_fov_deg"]
    ? yaml["omni_vertical_fov_deg"].as<double>() : yaml["new_fov_v"].as<double>()) * CV_PI / 180.0;
  const std::array<double, 2> fallback_mount_yaw{
    yaml["omni_left_yaw_deg"].as<double>(120.0) * CV_PI / 180.0,
    yaml["omni_right_yaw_deg"].as<double>(-120.0) * CV_PI / 180.0};
  if (!std::isfinite(fallback_fov_h) || !std::isfinite(fallback_fov_v) ||
      fallback_fov_h <= 0 || fallback_fov_h >= CV_PI ||
      fallback_fov_v <= 0 || fallback_fov_v >= CV_PI ||
      !std::isfinite(fallback_mount_yaw[0]) || !std::isfinite(fallback_mount_yaw[1])) {
    throw std::runtime_error("Invalid auxiliary FOV or mounting yaw");
  }
  tools::logger()->info(
    "[Sentry] Auxiliary fallback: model=pinhole_rotation left_yaw={}deg right_yaw={}deg "
    "horizontal_fov={}deg vertical_fov={}deg",
    fallback_mount_yaw[0] * 180.0 / CV_PI, fallback_mount_yaw[1] * 180.0 / CV_PI,
    fallback_fov_h * 180.0 / CV_PI, fallback_fov_v * 180.0 / CV_PI);
  // The main camera owns fire authority. Auxiliary HikRobot cameras only
  // search their configured left/right 120-degree sectors.
  omniperception::Perceptron perceptron(&left_camera, &right_camera, config_path);
  StopAllCameras stop_before_consumers;

  omniperception::DetectionResult switch_target;
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  OmniHandoffPhase handoff_phase = OmniHandoffPhase::searching;
  std::array<OmniCandidate, 2> pending_candidates;
  SuppressedOmniCandidate suppressed_candidate;
  io::Command last_omni_command{false, false, 0, 0};
  auto handoff_started_at = std::chrono::steady_clock::time_point{};
  auto takeover_started_at = std::chrono::steady_clock::time_point{};
  auto handoff_scan_started_at = std::chrono::steady_clock::time_point{};
  std::string locked_omni_camera;
  auto_aim::ArmorName locked_omni_name = auto_aim::ArmorName::not_armor;
  double locked_world_yaw = 0;
  double scan_center_yaw = 0;
  bool main_reacquiring = false;
  auto scan_settled_since = steady_clock::time_point{};
  auto last_p_query = steady_clock::time_point{};
  Eigen::Quaterniond last_p_value = Eigen::Quaterniond::Identity();
  bool last_p_valid = false;
  auto last_gate_report = steady_clock::now();
  auto last_invalid_center_report = steady_clock::time_point{};
  auto last_invalid_attitude_report = steady_clock::time_point{};
  auto last_direction_report = steady_clock::time_point{};
  auto locked_auxiliary_frame = steady_clock::time_point{};
  double locked_auxiliary_pitch = 0;
  std::array<steady_clock::time_point, 2> last_suppressed_report{};

  // Existing p() consumes history. Never query backwards across cameras/loops.
  auto query_auxiliary_attitude = [&](steady_clock::time_point frame_time)
    -> std::optional<Eigen::Quaterniond> {
      const auto query_time = frame_time - 1ms;
      if (query_time < last_p_query) {
        omni_trace::event(1, "stage=candidate_drop reason=attitude_time_reversed frame_us={}",
                         omni_trace::time_us(frame_time));
        return std::nullopt;
      }
      if (query_time == last_p_query) {
        return last_p_valid ? std::optional<Eigen::Quaterniond>(last_p_value) : std::nullopt;
      }
      const auto p = use_sp_transport ? gimbal->p(query_time) : cboard->chassis_imu_at(query_time);
      last_p_query = query_time;
      last_p_valid = p.coeffs().allFinite() && p.norm() > 1e-6;
      if (!last_p_valid) {
        const auto report_time = steady_clock::now();
        if (report_time - last_invalid_attitude_report >= 1s) {
          tools::logger()->warn(
            "[Sentry] Auxiliary attitude rejected: no fresh pose for frame, age_ms={:.1f}",
            omni_trace::age_ms(frame_time));
          last_invalid_attitude_report = report_time;
        }
        return std::nullopt;
      }
      last_p_value = p.normalized();
      return last_p_value;
    };

  auto reset_pending_candidates = [&] {
    pending_candidates = {};
  };

  auto clear_omni_handoff = [&] {
    handoff_phase = OmniHandoffPhase::searching;
    reset_pending_candidates();
    locked_omni_camera.clear();
    locked_omni_name = auto_aim::ArmorName::not_armor;
    locked_world_yaw = 0;
    locked_auxiliary_frame = steady_clock::time_point{};
    locked_auxiliary_pitch = 0;
    scan_center_yaw = 0;
    last_omni_command = io::Command{false, false, 0, 0};
    handoff_started_at = std::chrono::steady_clock::time_point{};
    takeover_started_at = std::chrono::steady_clock::time_point{};
    handoff_scan_started_at = std::chrono::steady_clock::time_point{};
    scan_settled_since = steady_clock::time_point{};
    main_reacquiring = false;
  };

  auto release_failed_handoff =
    [&](std::chrono::steady_clock::time_point now, const std::string & reason) {
      tools::logger()->info(
        "[Sentry][handoff_timeout] camera={}, target={}, target_yaw={:.1f}deg, phase={}, "
        "reason={}",
        locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
        locked_world_yaw * 180.0 / CV_PI, phase_name(handoff_phase), reason);
      suppressed_candidate = SuppressedOmniCandidate{
        true, locked_omni_camera, locked_omni_name, locked_world_yaw,
        now + failed_candidate_cooldown};
      clear_omni_handoff();
  };

  // Sweep around the auxiliary estimate while the main camera is reacquiring.
  // The controller still receives a world-frame yaw setpoint; only that
  // setpoint moves within a small bounded window around the locked direction.
  auto make_handoff_scan_command =
    [&](std::chrono::steady_clock::time_point now, double main_yaw) {
      auto scan_command = last_omni_command;
      if (omni_scan_amplitude == 0) return scan_command;
      if (handoff_scan_started_at == std::chrono::steady_clock::time_point{}) {
        // First hold a fixed bearing until the main gimbal reaches the neighborhood.
        if (handoff::angle_error(main_yaw, scan_center_yaw) > scan_start_tolerance) {
          scan_settled_since = steady_clock::time_point{};
          return scan_command;
        }
        if (scan_settled_since == steady_clock::time_point{}) scan_settled_since = now;
        if (now - scan_settled_since < scan_settle_time) return scan_command;
        handoff_scan_started_at = now;
      }

      const auto elapsed = std::chrono::duration<double>(now - handoff_scan_started_at).count();
      const auto period = std::chrono::duration<double>(omni_scan_period).count();
      if (period <= 0.0) return scan_command;

      const auto scan_offset =
        omni_scan_amplitude * std::sin(2.0 * CV_PI * elapsed / period);
      scan_command.yaw = tools::limit_rad(scan_center_yaw + scan_offset);
      scan_command.shoot = false;
      return scan_command;
    };

  size_t omni_loop = 0;
  auto last_omni_heartbeat = steady_clock::now();
  while (!exiter.exit()) {
    ++omni_loop;
    const auto loop_started = steady_clock::now();
    if (omni_trace::level() && loop_started - last_omni_heartbeat >= 1s) {
      omni_trace::event(1, "stage=main loop={} tracker={} phase={} enabled={}",
                       omni_loop, tracker.state(), phase_name(handoff_phase), perceptron.enabled());
      last_omni_heartbeat = loop_started;
    }
    const bool omni_searching = handoff_phase == OmniHandoffPhase::searching;
    perceptron.set_enabled(tracker.state() == "lost" && omni_searching);
    camera.read(img, timestamp);
    if (img.empty()) continue;
    omni_trace::event(2, "stage=main_attitude_begin loop={} frame_us={}",
                     omni_loop, omni_trace::time_us(timestamp));
    Eigen::Quaterniond q = use_sp_transport ? gimbal->q(timestamp - 1ms)
                                             : cboard->imu_at(timestamp - 1ms);
    recorder.record(img, q, timestamp);
    /// 自瞄核心逻辑
    solver.set_R_gimbal2world(q);

    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    omni_trace::event(2, "stage=main_infer_begin loop={}", omni_loop);
    auto armors = yolo.detect(img);
    omni_trace::event(2, "stage=main_infer_end loop={} n={} frame_age_ms={:.2f}",
                     omni_loop, armors.size(), omni_trace::age_ms(timestamp));

#ifdef SP_VISION_WITH_ROS2
    decider.get_invincible_armor(ros2.subscribe_enemy_status());
#endif

    decider.armor_filter(armors);

    decider.set_priority(armors);

    bool main_match_visible = false;
    double main_match_world_yaw = 0;
    size_t main_same_id = 0;
    double nearest_main_error = std::numeric_limits<double>::infinity();
    double nearest_main_yaw = 0;
    double nearest_main_pitch = 0;
    cv::Point2f nearest_main_center{};
    if (!omni_searching) {
      std::optional<auto_aim::Armor> best_match;
      for (const auto & armor : armors) {
        if (armor.name != locked_omni_name) continue;
        ++main_same_id;
        const auto ray = omni_geometry.main_ray(armor.center);
        if (!ray) continue;
        const Eigen::Vector3d world_ray = solver.R_gimbal2world() *
          Eigen::Vector3d((*ray)[0], (*ray)[1], (*ray)[2]);
        if (!world_ray.allFinite() || std::hypot(world_ray.x(), world_ray.y()) < 1e-9) continue;
        const auto observed_yaw = std::atan2(world_ray.y(), world_ray.x());
        const auto error = handoff::angle_error(observed_yaw, locked_world_yaw);
        if (error < nearest_main_error) {
          nearest_main_error = error;
          nearest_main_yaw = observed_yaw;
          nearest_main_pitch = -std::atan2(world_ray.z(), std::hypot(world_ray.x(), world_ray.y()));
          nearest_main_center = armor.center;
          if (error <= main_match_max_yaw_error) {
            best_match = armor;
            main_match_world_yaw = observed_yaw;
          }
        }
      }
      armors.clear();
      main_match_visible = best_match.has_value();
      if (best_match) armors.push_back(std::move(*best_match));
    }

    // Compare measured sight rays before the matching gate discards detections.
    // Keep the original auxiliary pitch separate from Aimer's ballistic command.
    const auto direction_report_time = steady_clock::now();
    if (!omni_searching && std::isfinite(nearest_main_error) &&
        direction_report_time - last_direction_report >= 1s) {
      tools::logger()->info(
        "[Sentry][direction_check] camera={} target={} phase={} accepted={} "
        "aux_yaw_deg={:.1f} main_yaw_deg={:.1f} yaw_difference_deg={:.1f} "
        "aux_pitch_deg={:.1f} main_pitch_deg={:.1f} pitch_difference_deg={:.1f} "
        "main_center_px=({:.1f},{:.1f}) main_pose_ypr_deg=({:.1f},{:.1f},{:.1f}) "
        "frame_gap_ms={:.1f} main_age_ms={:.1f}",
        locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
        phase_name(handoff_phase), main_match_visible,
        locked_world_yaw * 180.0 / CV_PI, nearest_main_yaw * 180.0 / CV_PI,
        tools::limit_rad(nearest_main_yaw - locked_world_yaw) * 180.0 / CV_PI,
        locked_auxiliary_pitch * 180.0 / CV_PI, nearest_main_pitch * 180.0 / CV_PI,
        (nearest_main_pitch - locked_auxiliary_pitch) * 180.0 / CV_PI,
        nearest_main_center.x, nearest_main_center.y,
        gimbal_pos[0] * 180.0 / CV_PI, gimbal_pos[1] * 180.0 / CV_PI,
        gimbal_pos[2] * 180.0 / CV_PI,
        duration<double, std::milli>(timestamp - locked_auxiliary_frame).count(),
        omni_trace::age_ms(timestamp));
      last_direction_report = direction_report_time;
    }

    auto detection_queue = perceptron.get_detection_queue();
    const auto received_batches = detection_queue.size();

    decider.sort(detection_queue);

    // Latest-path postprocessing: sorting/filtering can change the front armor.
    // Keep its identity and fallback bearing bound together without changing Decider's old callers.
    const auto filtered_at = steady_clock::now();
    for (auto & result : detection_queue) {
      result.armors.remove_if([&](const auto & armor) {
        const auto x = armor.center_norm.x;
        const auto y = armor.center_norm.y;
        const bool valid = std::isfinite(x) && std::isfinite(y) &&
                           x >= 0 && x <= 1 && y >= 0 && y <= 1;
        if (!valid && filtered_at - last_invalid_center_report >= 1s) {
          tools::logger()->warn(
            "[Sentry] Auxiliary detection rejected: camera={} center_norm=({:.3f},{:.3f}) outside image",
            result.camera, x, y);
          last_invalid_center_report = filtered_at;
        }
        return !valid;
      });
    }
    detection_queue.erase(std::remove_if(detection_queue.begin(), detection_queue.end(),
      [&](const auto & result) {
        return candidate_slot(result.camera) < 0 || result.armors.empty() ||
               !handoff::fresh(result.timestamp, filtered_at, omni_detection_max_age);
      }), detection_queue.end());
    for (auto & result : detection_queue) {
      const auto delta = decider.delta_angle(result.armors, result.camera);
      result.delta_yaw = delta[0] * CV_PI / 180.0;
      result.delta_pitch = delta[1] * CV_PI / 180.0;
    }

    const auto previous_tracker_state = tracker.state();
    // The latest lost-only handoff owns auxiliary arbitration. Do not activate the
    // legacy Tracker::switching path with an unconfirmed auxiliary result.
    const std::vector<omniperception::DetectionResult> no_legacy_switch;
    auto [switch_target, targets] = tracker.track(no_legacy_switch, armors, timestamp);

    io::Command command{false, false, 0, 0};
    const bool p_ready = use_sp_transport ? gimbal->p_ready() : cboard->chassis_imu_ready();
    const auto now = std::chrono::steady_clock::now();
    const auto bullet_speed = use_sp_transport ? gimbal->state().bullet_speed : cboard->bullet_speed;
    omni_trace::event(2,
      "stage=handoff_gate loop={} main_frame_us={} n_received={} n_valid={} main_n={} "
      "tracker_before={} tracker={} phase={} enabled={} p_ready={} switch_n={} main_match={}",
      omni_loop, omni_trace::time_us(timestamp), received_batches, detection_queue.size(), armors.size(),
      previous_tracker_state, tracker.state(), phase_name(handoff_phase), perceptron.enabled(),
      p_ready, switch_target.armors.size(), main_match_visible);

    /// 全向感知逻辑
    if (handoff_phase != OmniHandoffPhase::searching && !p_ready) {
      release_failed_handoff(now, "chassis attitude became unavailable");
    }
    else if (handoff_phase == OmniHandoffPhase::searching && tracker.state() == "lost") {
      OmniCandidate * selected = nullptr;

      if (!detection_queue.empty() && !p_ready) {
        tools::logger()->warn(
          "[Sentry] Auxiliary candidates ignored: waiting for valid p/chassis IMU");
      }

      if (p_ready) {
        // At most two results; order attitude queries by capture time, not target priority.
        std::array<const omniperception::DetectionResult *, 2> chronological{};
        size_t result_count = 0;
        for (const auto & result : detection_queue) {
          if (result_count < chronological.size()) chronological[result_count++] = &result;
        }
        std::sort(chronological.begin(), chronological.begin() + result_count,
          [](const auto * a, const auto * b) { return a->timestamp < b->timestamp; });
        for (size_t index = 0; index < result_count; ++index) {
          const auto & result = *chronological[index];
          if (result.armors.empty()) continue;
          const int slot = candidate_slot(result.camera);
          if (slot < 0) continue;
          if (!handoff::fresh(result.timestamp, steady_clock::now(), omni_detection_max_age)) continue;

          const auto p = query_auxiliary_attitude(result.timestamp);
          if (!p) { pending_candidates[slot] = {}; continue; }
          const auto big_ypr = tools::eulers(*p, 2, 1, 0);
          const auto big_yaw = big_ypr[0];
          auto local_bearing = result.delta_yaw;
          double world_yaw = 0;
          double world_pitch = 0;
          if (omni_geometry.calibrated(result.camera)) {
            const auto vector = omni_geometry.auxiliary_vector(result.camera, result.armors.front());
            if (!vector) {
              pending_candidates[slot] = {};
              omni_trace::event(1, "stage=candidate_drop camera={} reason=calibrated_geometry_rejected",
                               result.camera);
              continue;
            }
            const Eigen::Vector3d world = *p * Eigen::Vector3d((*vector)[0], (*vector)[1], (*vector)[2]);
            if (!world.allFinite() || std::hypot(world.x(), world.y()) < 1e-6) continue;
            world_yaw = std::atan2(world.y(), world.x());
            world_pitch = -std::atan2(world.z(), std::hypot(world.x(), world.y()));
          } else {
            const auto center = result.armors.front().center_norm;
            const Eigen::Vector3d local = fallback_ray(
              center.x, center.y, fallback_fov_h, fallback_fov_v, fallback_mount_yaw[slot]);
            const Eigen::Vector3d world = *p * local;
            if (!world.allFinite() || std::hypot(world.x(), world.y()) < 1e-6) {
              pending_candidates[slot] = {};
              continue;
            }
            local_bearing = std::atan2(local.y(), local.x());
            world_yaw = std::atan2(world.y(), world.x());
            world_pitch = -std::atan2(world.z(), std::hypot(world.x(), world.y()));
          }
          const auto name = result.armors.front().name;

          if (suppressed_candidate.valid && now >= suppressed_candidate.until) {
            suppressed_candidate = {};
          }
          const bool suppressed =
            suppressed_candidate.valid && result.camera == suppressed_candidate.camera &&
            name == suppressed_candidate.name &&
            std::abs(tools::limit_rad(world_yaw - suppressed_candidate.world_yaw)) <=
              main_match_max_yaw_error;
          if (suppressed) {
            pending_candidates[slot] = {};
            if (now - last_suppressed_report[slot] >= 1s) {
              tools::logger()->debug(
                "[Sentry][candidate_suppressed] camera={}, target={}, target_yaw={:.1f}deg",
                result.camera, auto_aim::ARMOR_NAMES[name], world_yaw * 180.0 / CV_PI);
              last_suppressed_report[slot] = now;
            }
            continue;
          }

          auto & pending = pending_candidates[slot];
          const bool continuous =
            pending.valid && pending.name == name && result.timestamp > pending.timestamp &&
            result.timestamp - pending.timestamp <= candidate_max_gap &&
            std::abs(tools::limit_rad(world_yaw - pending.world_yaw)) <=
              candidate_max_yaw_jump;
          omni_trace::event(2,
            "stage=candidate camera={} frame_us={} age_ms={:.2f} id={} previous_valid={} "
            "previous_id={} previous_frame_us={} gap_ms={:.2f} yaw_jump_deg={:.2f} "
            "continuous={} next_count={} p_ready={}",
            result.camera, omni_trace::time_us(result.timestamp), omni_trace::age_ms(result.timestamp),
            static_cast<int>(name), pending.valid, static_cast<int>(pending.name),
            omni_trace::time_us(pending.timestamp),
            pending.valid ? duration<double, std::milli>(result.timestamp - pending.timestamp).count() : -1.0,
            pending.valid ? std::abs(tools::limit_rad(world_yaw - pending.world_yaw)) * 180.0 / CV_PI : -1.0,
            continuous, continuous ? pending.confirm_count + 1 : 1, p_ready);

          pending.valid = true;
          pending.camera = result.camera;
          pending.name = name;
          pending.priority = result.armors.front().priority;
          pending.confirm_count = continuous ? pending.confirm_count + 1 : 1;
          pending.world_yaw = world_yaw;
          pending.pitch = world_pitch;
          pending.timestamp = result.timestamp;

          tools::logger()->debug(
            "[Sentry][candidate_confirming] camera={}, target={}, count={}, big_yaw={:.1f}deg, "
            "bearing={:.1f}deg, target_yaw={:.1f}deg, center_norm=({:.3f},{:.3f}), "
            "big_pitch={:.1f}deg, big_roll={:.1f}deg, target_pitch={:.1f}deg, age_ms={:.1f}",
            pending.camera, auto_aim::ARMOR_NAMES[pending.name], pending.confirm_count,
            big_yaw * 180.0 / CV_PI, local_bearing * 180.0 / CV_PI,
            pending.world_yaw * 180.0 / CV_PI,
            result.armors.front().center_norm.x, result.armors.front().center_norm.y,
            big_ypr[1] * 180.0 / CV_PI, big_ypr[2] * 180.0 / CV_PI,
            world_pitch * 180.0 / CV_PI, omni_trace::age_ms(result.timestamp));

        }
      }

      const auto select_now = steady_clock::now();
      selected = handoff::select_candidate(
        pending_candidates, select_now, omni_detection_max_age, gimbal_pos[0], 2, candidate_max_gap);
      // A cached confirmed slot must not bypass loss of the chassis-attitude gate.
      if (!(use_sp_transport ? gimbal->p_ready() : cboard->chassis_imu_ready())) selected = nullptr;

      if (selected != nullptr) {
        locked_omni_camera = selected->camera;
        locked_omni_name = selected->name;
        locked_world_yaw = selected->world_yaw;
        locked_auxiliary_frame = selected->timestamp;
        locked_auxiliary_pitch = selected->pitch;
        scan_center_yaw = locked_world_yaw;
        last_omni_command = io::Command{true, false, locked_world_yaw, selected->pitch};
        handoff_started_at = select_now;
        handoff_scan_started_at = steady_clock::time_point{};
        scan_settled_since = steady_clock::time_point{};
        handoff_phase = OmniHandoffPhase::guiding;
        command = last_omni_command;
        perceptron.set_enabled(false);

        const auto yaw_error = tools::limit_rad(locked_world_yaw - gimbal_pos[0]);
        tools::logger()->info(
          "[Sentry][handoff_locked] camera={}, target={}, target_yaw={:.1f}deg, "
          "small_yaw={:.1f}deg, shortest_error={:.1f}deg, phase={}",
          locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
          locked_world_yaw * 180.0 / CV_PI, gimbal_pos[0] * 180.0 / CV_PI,
          yaw_error * 180.0 / CV_PI, phase_name(handoff_phase));
        reset_pending_candidates();
      }
      command.shoot = false;
    }

    else if (handoff_phase == OmniHandoffPhase::guiding) {
      if (
        main_match_visible &&
        (tracker.state() == "detecting" || tracker.state() == "tracking")) {
        handoff_phase = OmniHandoffPhase::acquiring;
        takeover_started_at = now;
        main_reacquiring = false;
        tools::logger()->info(
          "[Sentry][main_matched] camera={}, target={}, locked_yaw={:.1f}deg, "
          "observed_yaw={:.1f}deg, error={:.1f}deg, tracker_state={}",
          locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
          locked_world_yaw * 180.0 / CV_PI, main_match_world_yaw * 180.0 / CV_PI,
          tools::limit_rad(main_match_world_yaw - locked_world_yaw) * 180.0 / CV_PI,
          tracker.state());
        command = aimer.aim(targets, timestamp, bullet_speed);
        if (command.control) {
          last_omni_command = command;
          scan_center_yaw = command.yaw;
        }
        handoff_scan_started_at = steady_clock::time_point{};
        scan_settled_since = steady_clock::time_point{};
      } else if (now - handoff_started_at > handoff_timeout) {
        release_failed_handoff(now, "main camera did not find matching target");
      } else {
        command = make_handoff_scan_command(now, gimbal_pos[0]);
      }
      command.shoot = false;
    }

    else if (handoff_phase == OmniHandoffPhase::acquiring) {
      const auto action = handoff::acquisition_action(
        now, takeover_started_at, takeover_timeout,
        tracker.state() == "tracking", main_match_visible, !targets.empty());
      if (action == handoff::AcquireAction::timeout) {
        release_failed_handoff(now, "main camera failed to reach stable tracking");
      } else if (action == handoff::AcquireAction::complete) {
        command = aimer.aim(targets, timestamp, bullet_speed);
        tools::logger()->info(
          "[Sentry][handoff_completed] camera={}, target={}, target_yaw={:.1f}deg, "
          "tracker_state=tracking",
          locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
          locked_world_yaw * 180.0 / CV_PI);
        clear_omni_handoff();
      } else if (action == handoff::AcquireAction::follow) {
        if (main_reacquiring) {
          tools::logger()->info(
            "[Sentry][main_reacquired] camera={}, target={}, error={:.1f}deg",
            locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
            tools::limit_rad(main_match_world_yaw - locked_world_yaw) * 180.0 / CV_PI);
          main_reacquiring = false;
        }
        command = aimer.aim(targets, timestamp, bullet_speed);
        if (command.control) {
          last_omni_command = command;
          scan_center_yaw = command.yaw;
        }
        handoff_scan_started_at = steady_clock::time_point{};
        scan_settled_since = steady_clock::time_point{};
      } else {
        command = make_handoff_scan_command(now, gimbal_pos[0]);
        if (!main_reacquiring) {
          main_reacquiring = true;
          tools::logger()->info(
            "[Sentry][main_reacquiring] camera={}, target={}, target_yaw={:.1f}deg, "
            "tracker_state={}",
            locked_omni_camera, auto_aim::ARMOR_NAMES[locked_omni_name],
            locked_world_yaw * 180.0 / CV_PI, tracker.state());
        }
      }
      command.shoot = false;
    }

    else {
      perceptron.set_enabled(false);
      command = aimer.aim(targets, timestamp, bullet_speed);
    }

    /// Fire authority belongs exclusively to a stable main-camera track. Omni
    /// perception can move the gimbal for reacquisition, but can never fire.
    command.shoot =
      !real_vehicle_test_no_fire && handoff_phase == OmniHandoffPhase::searching &&
          tracker.state() == "tracking"
        ? shooter.shoot(command, aimer, targets, gimbal_pos)
        : false;
    // command.shoot = false;

    // Low-rate upper-layer report also works with SP_OMNI_TRACE unset.
    if (now - last_gate_report >= 1s) {
      tools::logger()->debug(
        "[Sentry][gate_summary] tracker={} phase={} enabled={} received={} valid={} p_ready={} "
        "main_same_id={} main_match={} nearest_match_deg={:.1f} small_yaw_deg={:.1f} "
        "control={} command_yaw_deg={:.1f} remaining_yaw_deg={:.1f} "
        "small_pitch_deg={:.1f} command_pitch_deg={:.1f}",
        tracker.state(), phase_name(handoff_phase), perceptron.enabled(), received_batches,
        detection_queue.size(), p_ready, main_same_id, main_match_visible,
        std::isfinite(nearest_main_error) ? nearest_main_error * 180.0 / CV_PI : -1.0,
        gimbal_pos[0] * 180.0 / CV_PI, command.control, command.yaw * 180.0 / CV_PI,
        command.control ? handoff::angle_error(command.yaw, gimbal_pos[0]) * 180.0 / CV_PI : -1.0,
        gimbal_pos[1] * 180.0 / CV_PI, command.pitch * 180.0 / CV_PI);
      last_gate_report = now;
    }

    omni_trace::event(2,
      "stage=command loop={} tracker={} phase={} control={} shoot={} yaw_rad={:.4f} "
      "pitch_rad={:.4f} p_ready={} loop_to_command_ms={:.2f}",
      omni_loop, tracker.state(), phase_name(handoff_phase), command.control, command.shoot,
      command.yaw, command.pitch, p_ready, omni_trace::age_ms(loop_started));
    if (use_sp_transport) {
      io::VisionToGimbal packet{};
      packet.mode = command.control ? (command.shoot ? 2 : 1) : 0;
      packet.center_yaw = command.yaw;
      if (!targets.empty()) {
        const auto target_state = targets.front().ekf_x();
        if (target_state.size() > 2) {
          const auto center_yaw = std::atan2(target_state[2], target_state[0]);
          if (std::isfinite(center_yaw)) packet.center_yaw = center_yaw;
        }
      }
      packet.yaw = command.yaw;
      packet.yaw_vel = 0;
      packet.yaw_acc = 0;
      packet.pitch = command.pitch;
      packet.pitch_vel = 0;
      packet.pitch_acc = 0;
      omni_trace::event(2,
        "stage=packet loop={} mode={} yaw_rad={:.4f} center_yaw_rad={:.4f} pitch_rad={:.4f} bytes={}",
        omni_loop, static_cast<int>(packet.mode), static_cast<double>(packet.yaw),
        static_cast<double>(packet.center_yaw), static_cast<double>(packet.pitch), sizeof(packet));
      gimbal->send(packet);
    } else {
      cboard->send(command);
    }
    omni_trace::event(2, "stage=send_return loop={} transport={}", omni_loop, control_transport);

#ifdef SP_VISION_WITH_ROS2
    /// ROS2通信
    Eigen::Vector4d target_info = decider.get_target_info(armors, targets);

    ros2.publish(target_info);
#endif
  }

  io::HikRobot::request_stop_all();
  return 0;
}

int main(int argc, char * argv[])
{
  try {
    return run_sentry(argc, argv);
  } catch (const std::exception & error) {
    // Unwind run_sentry first so already-created cameras and workers shut down
    // even when construction of a later camera fails. Avoid the logger here:
    // reporting the original exception must not require a working log sink.
    std::fprintf(stderr, "[Sentry] Startup/runtime failure: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "[Sentry] Startup/runtime failure: unknown exception\n");
    return 1;
  }
}