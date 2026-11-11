#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

long long timestamp_ns(Clock::time_point timestamp)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
}

// One worker owns each replay source. The source repeats until stopped, and
// stamps frames at delivery time minus an explicitly injected age. Video PTS
// is intentionally not used as a steady_clock timestamp.
class ReplayCamera final : public io::CameraBase
{
public:
  ReplayCamera(const std::string & path, int interval_ms, int age_ms)
  : interval_(interval_ms), age_(age_ms)
  {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("Replay input must be a local image or video file: " + path);
    }
    still_ = cv::imread(path, cv::IMREAD_COLOR);
    if (still_.empty() && (!video_.open(path) || !video_.read(first_))) {
      throw std::runtime_error("Cannot decode replay input: " + path);
    }
    const auto & sample = still_.empty() ? first_ : still_;
    if (sample.empty() || sample.type() != CV_8UC3) {
      throw std::runtime_error("Replay expects an 8-bit BGR image (CV_8UC3)");
    }
    fmt::print("REPLAY_INPUT path={} width={} height={} type={} channels={} order=BGR\n",
               path, sample.cols, sample.rows, sample.type(), sample.channels());
  }

  void start() { enabled_.store(true, std::memory_order_release); }
  void stop() { enabled_.store(false, std::memory_order_release); }
  unsigned long long frames() const { return frames_.load(std::memory_order_relaxed); }

  bool try_read(cv::Mat & image, Clock::time_point & timestamp) override
  {
    if (!enabled_.load(std::memory_order_acquire) || Clock::now() < next_frame_) return false;
    if (!still_.empty()) {
      image = still_.clone();
    } else if (!first_.empty()) {
      image = std::move(first_);
    } else if (!video_.read(image)) {
      if (!video_.set(cv::CAP_PROP_POS_FRAMES, 0) || !video_.read(image)) {
        throw std::runtime_error("Cannot rewind replay video");
      }
    }
    if (image.empty() || image.type() != CV_8UC3) {
      throw std::runtime_error("Replay frame must be nonempty CV_8UC3 BGR");
    }
    const auto delivered_at = Clock::now();
    timestamp = delivered_at - age_;
    next_frame_ = delivered_at + interval_;
    frames_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void read(cv::Mat &, Clock::time_point &) override
  {
    // Perceptron must use its nonblocking CameraBase interface.
    throw std::logic_error("ReplayCamera::read called; expected try_read");
  }

private:
  cv::Mat still_, first_;
  cv::VideoCapture video_;
  Milliseconds interval_, age_;
  Clock::time_point next_frame_{};
  std::atomic<bool> enabled_{false};
  std::atomic<unsigned long long> frames_{0};
};

struct Observations
{
  unsigned long long published = 0;
  unsigned long long published_armors = 0;
  unsigned long long sorted = 0;
  unsigned long long valid = 0;
  unsigned long long confirmable_pairs = 0;
  Clock::time_point last_published{}, last_valid{};
  auto_aim::ArmorName last_name = auto_aim::ArmorName::not_armor;
  double last_yaw = 0;
};

int role_slot(const std::string & role)
{
  if (role == "left") return 0;
  if (role == "right") return 1;
  return -1;
}
}  // namespace

// This verifies actual inference -> publication -> consumption -> Decider::sort.
// Pair checks assume a stationary chassis (big_yaw = 0). They do not instantiate
// a tracker, ROS node, IMU, gimbal or the sentry handoff state machine.
int main(int argc, char ** argv)
{
  const cv::String keys =
    "{help h usage ? | | Show usage}"
    "{@input | | Local armor image/video; images and videos repeat}"
    "{config c | configs/sentry.yaml | Existing model and omni configuration}"
    "{role | both | left, right or both; both uses independent replay sources}"
    "{right-input | | Optional different right image/video for role=both}"
    "{frame-interval-ms | 33 | Minimum delivery interval per source}"
    "{timeout-ms | 5000 | Observation time after model construction}"
    "{timestamp-age-ms | 0 | Inject age before inference}"
    "{consumer-delay-ms | 0 | Hold each nonempty dequeued batch before sort}"
    "{min-results | 2 | Minimum received publications for every requested role}"
    "{expect | fresh | fresh, published or stale; see output for criteria}";
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    std::cout << "fresh: require publications, valid outputs and at least one confirmable pair per role.\n"
                 "published: require publications only; report freshness and pairs separately.\n"
                 "stale: require publications but zero outputs after Decider::sort.\n"
                 "Exit: 0 expected behavior; 1 input/runtime error; 2 assertion failed; 130 interrupted.\n"
                 "Use Linux timeout for an outer bound including model loading and worker teardown.\n";
    return 0;
  }

  try {
    const auto input = cli.get<std::string>(0);
    const auto config = cli.get<std::string>("config");
    const auto role = cli.get<std::string>("role");
    const auto right_input = cli.get<std::string>("right-input");
    const int interval_ms = cli.get<int>("frame-interval-ms");
    const int timeout_ms = cli.get<int>("timeout-ms");
    const int age_ms = cli.get<int>("timestamp-age-ms");
    const int delay_ms = cli.get<int>("consumer-delay-ms");
    const int min_results = cli.get<int>("min-results");
    const auto expect = cli.get<std::string>("expect");
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }
    if (input.empty() || (role != "left" && role != "right" && role != "both") ||
        (expect != "fresh" && expect != "published" && expect != "stale") ||
        interval_ms < 1 || timeout_ms < 1 || age_ms < 0 || delay_ms < 0 || min_results < 1 ||
        (!right_input.empty() && role != "both")) {
      throw std::runtime_error("Invalid replay options; use --help");
    }
    const auto yaml = YAML::LoadFile(config);
    const int max_age_ms = yaml["omni_max_detection_age_ms"]
                             ? yaml["omni_max_detection_age_ms"].as<int>() : 120;
    const double yaw_jump_deg = yaml["omni_candidate_max_yaw_jump_deg"]
                                  ? yaml["omni_candidate_max_yaw_jump_deg"].as<double>() : 10.0;
    if (max_age_ms < 1 || !std::isfinite(yaw_jump_deg) || yaw_jump_deg <= 0) {
      throw std::runtime_error("Invalid omni age or candidate yaw-jump configuration");
    }
    const auto max_age = Milliseconds(max_age_ms);
    const double yaw_jump_rad = yaw_jump_deg * CV_PI / 180.0;
    std::array<std::unique_ptr<ReplayCamera>, 2> cameras;
    if (role != "right") cameras[0] = std::make_unique<ReplayCamera>(input, interval_ms, age_ms);
    if (role != "left") {
      cameras[1] = std::make_unique<ReplayCamera>(
        right_input.empty() ? input : right_input, interval_ms, age_ms);
    }

    tools::Exiter exiter;
    omniperception::Decider decider(config);
    omniperception::Perceptron perceptron(cameras[0].get(), cameras[1].get(), config);
    // Model construction is outside the observation interval; no frame is
    // available until both Perceptron workers have been constructed.
    const auto deadline = Clock::now() + Milliseconds(timeout_ms);
    for (auto & camera : cameras) if (camera) camera->start();
    fmt::print("REPLAY_START config={} role={} expect={} max_age_ms={} yaw_jump_deg={} "
               "interval_ms={} injected_age_ms={} consumer_delay_ms={} observation_ms={}\n",
               config, role, expect, max_age_ms, yaw_jump_deg, interval_ms, age_ms, delay_ms, timeout_ms);

    std::array<Observations, 2> observations;
    bool failed = false;
    while (!exiter.exit() && Clock::now() < deadline) {
      auto batch = perceptron.get_detection_queue();
      if (batch.empty()) {
        std::this_thread::sleep_for(Milliseconds(2));
        continue;
      }
      const size_t published_batch_size = batch.size();
      for (const auto & result : batch) {
        const int slot = role_slot(result.camera);
        if (slot < 0 || !cameras[slot]) {
          fmt::print(stderr, "REPLAY_FAIL unexpected_role={}\n", result.camera);
          failed = true;
          continue;
        }
        auto & observed = observations[slot];
        const bool valid_stamp = result.timestamp != Clock::time_point{} &&
                                 result.timestamp <= Clock::now() &&
                                 result.timestamp > observed.last_published;
        bool valid_payload = !result.armors.empty() && std::isfinite(result.delta_yaw) &&
                             std::isfinite(result.delta_pitch);
        for (const auto & armor : result.armors) {
          valid_payload = valid_payload && std::isfinite(armor.confidence) &&
                          std::isfinite(armor.center_norm.x) && std::isfinite(armor.center_norm.y) &&
                          static_cast<size_t>(armor.name) < auto_aim::ARMOR_NAMES.size();
        }
        failed = failed || !valid_stamp || !valid_payload;
        ++observed.published;
        observed.published_armors += result.armors.size();
        observed.last_published = result.timestamp;
        fmt::print("REPLAY_RECEIVED now_ns={} frame_ns={} role={} batch_size={} armors={} "
                   "age_ms={:.3f} yaw_rad={:.6f} pitch_rad={:.6f} monotonic={} payload_ok={}\n",
                   timestamp_ns(Clock::now()), timestamp_ns(result.timestamp), result.camera,
                   published_batch_size, result.armors.size(),
                   std::chrono::duration<double, std::milli>(Clock::now() - result.timestamp).count(),
                   result.delta_yaw, result.delta_pitch, valid_stamp, valid_payload);
      }
      if (failed) break;  // Do not feed malformed enum/payload values into Decider.
      if (delay_ms) {
        const auto resume_at = Clock::now() + Milliseconds(delay_ms);
        while (!exiter.exit() && Clock::now() < resume_at && Clock::now() < deadline) {
          std::this_thread::sleep_for(Milliseconds(2));
        }
        if (exiter.exit() || Clock::now() >= deadline) break;
      }
      for (const auto & result : batch) ++observations[role_slot(result.camera)].sorted;
      decider.sort(batch);
      fmt::print("REPLAY_SORT now_ns={} received_results={} valid_results={} filtered_results={}\n",
                 timestamp_ns(Clock::now()), published_batch_size, batch.size(),
                 published_batch_size - batch.size());
      for (const auto & result : batch) {
        auto & observed = observations[role_slot(result.camera)];
        const auto name = result.armors.front().name;
        // Only the data prerequisites for two-frame confirmation are checked.
        // A fresh result is not proof that the sentry is ready to hand off.
        const bool pair = observed.valid > 0 && name == observed.last_name &&
                          result.timestamp > observed.last_valid &&
                          result.timestamp - observed.last_valid <= max_age &&
                          std::abs(tools::limit_rad(result.delta_yaw - observed.last_yaw)) <= yaw_jump_rad;
        ++observed.valid;
        if (pair) ++observed.confirmable_pairs;
        observed.last_valid = result.timestamp;
        observed.last_name = name;
        observed.last_yaw = result.delta_yaw;
        fmt::print("REPLAY_VALID frame_ns={} role={} id={} armors={} confirmable_pair={}\n",
                   timestamp_ns(result.timestamp), result.camera, auto_aim::ARMOR_NAMES[name],
                   result.armors.size(), pair);
      }
    }
    for (auto & camera : cameras) if (camera) camera->stop();
    for (size_t i = 0; i < cameras.size(); ++i) {
      if (!cameras[i]) continue;
      const auto & observed = observations[i];
      bool role_ok = observed.published >= static_cast<unsigned long long>(min_results);
      if (expect == "fresh") role_ok = role_ok && observed.valid >= 2 && observed.confirmable_pairs > 0;
      if (expect == "stale") {
        role_ok = role_ok && observed.sorted >= static_cast<unsigned long long>(min_results) &&
                  observed.valid == 0;
      }
      failed = failed || !role_ok;
      fmt::print("REPLAY_SUMMARY role={} frames={} published_results={} published_armors={} "
                 "sorted_results={} valid_results={} confirmable_pairs={} expected={} pass={}\n",
                 i == 0 ? "left" : "right", cameras[i]->frames(), observed.published,
                 observed.published_armors, observed.sorted, observed.valid, observed.confirmable_pairs,
                 expect, role_ok);
    }
    if (exiter.exit()) return 130;
    fmt::print("REPLAY_{} scope=candidate_output_only stationary_chassis_assumption=true\n",
               failed ? "FAIL" : "PASS");
    return failed ? 2 : 0;
  } catch (const std::exception & error) {
    fmt::print(stderr, "REPLAY_ERROR {}\n", error.what());
    return 1;
  }
}