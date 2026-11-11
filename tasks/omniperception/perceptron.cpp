#include "perceptron.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "trace.hpp"

namespace omniperception
{
namespace
{
struct AuxiliaryCamera
{
  io::CameraBase * camera;
  std::string role;
};
using CameraList = std::vector<AuxiliaryCamera>;
}

Perceptron::Perceptron(
  io::CameraBase * left_camera, io::CameraBase * right_camera,
  const std::string & config_path)
: detection_queue_(10), decider_(config_path), stop_flag_(false), enabled_(true)
{
  const CameraList cameras{{left_camera, "left"}, {right_camera, "right"}};
  detectors_.reserve(cameras.size());
  for (size_t i = 0; i < cameras.size(); ++i) {
    detectors_.push_back(std::make_shared<auto_aim::YOLO>(config_path, false));
  }

  for (size_t i = 0; i < cameras.size(); ++i) {
    if (cameras[i].camera == nullptr) continue;
    threads_.emplace_back([this, camera = cameras[i], detector = detectors_[i]] {
      parallel_infer(camera.camera, camera.role, detector);
    });
  }

  tools::logger()->info("Omni perception initialized with {} auxiliary cameras", threads_.size());
}

void Perceptron::set_enabled(bool enabled)
{
  std::lock_guard<std::mutex> publication_lock(publication_mutex_);
  const bool previous = enabled_.exchange(enabled, std::memory_order_acq_rel);
  if (previous != enabled) generation_.fetch_add(1, std::memory_order_acq_rel);
  if (trace::level() && previous != enabled) {
    trace::event(1, "stage=enable previous={} enabled={} q_snapshot={}",
                 previous, enabled, detection_queue_.size());
  }
  if (!enabled) {
    const auto before = trace::level() >= 2 ? detection_queue_.size() : 0;
    detection_queue_.clear();
    if (trace::level() >= 2 && (before != 0 || previous)) {
      trace::event(2, "stage=clear reason=disabled q_before_snapshot={} q_after_snapshot={}",
                   before, detection_queue_.size());
    }
  }
}

bool Perceptron::enabled() const { return enabled_.load(std::memory_order_acquire); }

Perceptron::~Perceptron()
{
  stop_flag_.store(true, std::memory_order_release);
  for (auto & thread : threads_) {
    if (thread.joinable()) thread.join();
  }
  tools::logger()->info("Omni perception destructed");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::lock_guard<std::mutex> publication_lock(publication_mutex_);
  std::vector<DetectionResult> results;
  if (!enabled()) return results;
  const auto active_generation = generation_.load(std::memory_order_acquire);
  DetectionResult result;
  size_t drained = 0;
  // Bound work even if producers continue publishing while the consumer drains.
  // Ten is this queue's construction capacity; leave new arrivals for the next loop.
  for (size_t budget = 0; budget < 10; ++budget) {
    if (!detection_queue_.try_pop(result)) break;
    ++drained;
    if (result.generation != active_generation) continue;
    if (trace::level() >= 2) {
      trace::event(2, "stage=pop camera={} frame_us={} age_ms={:.2f} n={} q_snapshot={}",
                   result.camera, trace::time_us(result.timestamp), trace::age_ms(result.timestamp),
                   result.armors.size(), detection_queue_.size());
    }

    auto same_camera = std::find_if(
      results.begin(), results.end(),
      [&result](const DetectionResult & current) { return current.camera == result.camera; });
    if (same_camera == results.end()) {
      results.push_back(std::move(result));
    } else if (result.timestamp > same_camera->timestamp) {
      *same_camera = std::move(result);
    }
  }
  if (trace::level() >= 2) {
    trace::event(2, "stage=drain drained={} latest_per_camera={} q_snapshot={}",
                 drained, results.size(), detection_queue_.size());
  }
  return results;
}

void Perceptron::parallel_infer(
  io::CameraBase * camera, const std::string & camera_role,
  const std::shared_ptr<auto_aim::YOLO> & detector)
{
  if (camera == nullptr || detector == nullptr) return;

  size_t frames = 0, no_frame = 0, disabled = 0, inferred = 0;
  size_t raw_count = 0, kept_count = 0, filtered_count = 0, published = 0;
  bool first_detection_logged = false;
  double infer_total_ms = 0, infer_max_ms = 0;
  auto last_heartbeat = std::chrono::steady_clock::now();

  try {
    while (true) {
      if (stop_flag_.load(std::memory_order_acquire)) break;
      const auto now = std::chrono::steady_clock::now();
      if (trace::level() && now - last_heartbeat >= std::chrono::seconds(1)) {
        const auto seconds = std::chrono::duration<double>(now - last_heartbeat).count();
        trace::event(1,
          "stage=worker camera={} enabled={} window_s={:.2f} frames={} read_fps={:.1f} "
          "no_frame_polls={} disabled={} inferred={} infer_fps={:.1f} infer_mean_ms={:.2f} "
          "infer_max_ms={:.2f} raw={} kept={} filtered={} published={} q_snapshot={} counters=interval",
          camera_role, enabled(), seconds, frames, frames / seconds, no_frame, disabled, inferred,
          inferred / seconds, inferred ? infer_total_ms / inferred : 0.0, infer_max_ms,
          raw_count, kept_count, filtered_count, published, detection_queue_.size());
        frames = no_frame = disabled = inferred = 0;
        raw_count = kept_count = filtered_count = published = 0;
        infer_total_ms = infer_max_ms = 0;
        last_heartbeat = now;
      }

      cv::Mat image;
      std::chrono::steady_clock::time_point timestamp;
      if (!camera->try_read(image, timestamp) || image.empty()) {
        ++no_frame;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        continue;
      }

      ++frames;
      if (trace::level() >= 2) {
        trace::event(2,
          "stage=frame camera={} frame_us={} age_ms={:.2f} enabled={} width={} height={} "
          "type={} channels={} step={} continuous={} q_snapshot={}",
          camera_role, trace::time_us(timestamp), trace::age_ms(timestamp), enabled(),
          image.cols, image.rows, image.type(), image.channels(), image.step[0],
          image.isContinuous(), detection_queue_.size());
      }
      if (!enabled()) { ++disabled; continue; }
      const auto generation = generation_.load(std::memory_order_acquire);

      trace::event(2, "stage=infer_begin camera={} frame_us={}", camera_role, trace::time_us(timestamp));
      const auto infer_start = std::chrono::steady_clock::now();
      auto armors = detector->detect(image);
      const auto infer_end = std::chrono::steady_clock::now();
      const auto infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
      infer_total_ms += infer_ms;
      infer_max_ms = std::max(infer_max_ms, infer_ms);
      ++inferred;
      raw_count += armors.size();
      trace::event(2,
        "stage=infer_end camera={} frame_us={} infer_ms={:.2f} age_ms={:.2f} n={} enabled={}",
        camera_role, trace::time_us(timestamp),
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count(),
        std::chrono::duration<double, std::milli>(infer_end - timestamp).count(), armors.size(), enabled());
      if (trace::level() >= 2) {
        for (const auto & armor : armors) {
          trace::event(2,
            "stage=raw_armor camera={} frame_us={} id={} color={} conf={:.4f} cx={:.4f} cy={:.4f}",
            camera_role, trace::time_us(timestamp), static_cast<int>(armor.name),
            static_cast<int>(armor.color), armor.confidence, armor.center_norm.x, armor.center_norm.y);
        }
      }
      if (!enabled()) { ++disabled; continue; }
      if (armors.empty()) continue;

      const auto before_filter = armors.size();
      decider_.armor_filter(armors);
      filtered_count += before_filter - armors.size();
      kept_count += armors.size();
      trace::event(2, "stage=worker_filter camera={} frame_us={} n={}",
                   camera_role, trace::time_us(timestamp), armors.size());
      if (armors.empty()) continue;

      decider_.set_priority(armors);
      armors.sort([](const auto_aim::Armor & lhs, const auto_aim::Armor & rhs) {
        const cv::Point2f image_center{0.5F, 0.5F};
        return cv::norm(lhs.center_norm - image_center) <
               cv::norm(rhs.center_norm - image_center);
      });
      armors.sort([](const auto_aim::Armor & lhs, const auto_aim::Armor & rhs) {
        return lhs.priority < rhs.priority;
      });

      auto delta = decider_.delta_angle(armors, camera_role);
      DetectionResult result;
      result.armors = std::move(armors);
      result.timestamp = timestamp;
      result.delta_yaw = delta[0] * CV_PI / 180.0;
      result.delta_pitch = delta[1] * CV_PI / 180.0;
      result.camera = camera_role;
      if (!first_detection_logged) {
        first_detection_logged = true;
        tools::logger()->info(
          "[OMNI][first_detection] camera={} target={} confidence={:.3f} age_ms={:.1f}",
          camera_role, auto_aim::ARMOR_NAMES[result.armors.front().name],
          result.armors.front().confidence, trace::age_ms(timestamp));
      }
      if (trace::level() >= 2) {
        trace::event(2,
          "stage=publish OMNI_DETECTED camera={} frame_us={} age_ms={:.2f} n={} id={} "
          "conf={:.4f} yaw_rad={:.4f} pitch_rad={:.4f} q_snapshot={} capacity=10",
          camera_role, trace::time_us(timestamp), trace::age_ms(timestamp), result.armors.size(),
          static_cast<int>(result.armors.front().name), result.armors.front().confidence,
          result.delta_yaw, result.delta_pitch, detection_queue_.size());
      }
      // Keep the newest frame for each camera. A stale auxiliary result must
      // never delay or override the main camera tracker.
      {
        std::lock_guard<std::mutex> publication_lock(publication_mutex_);
        if (!enabled() || generation != generation_.load(std::memory_order_acquire)) continue;
        result.generation = generation;
        detection_queue_.push(std::move(result));
      }
      ++published;
      if (trace::level() >= 2) {
        trace::event(2, "stage=publish_done camera={} frame_us={} q_snapshot={}",
                     camera_role, trace::time_us(timestamp), detection_queue_.size());
      }
    }
  } catch (const std::exception & exception) {
    tools::logger()->error(
      "[OMNI] stage=worker_exit mono_us={} camera={} frames={} inferred={} published={} error={}",
      trace::time_us(std::chrono::steady_clock::now()), camera_role, frames, inferred, published,
      exception.what());
  }
}

}  // namespace omniperception