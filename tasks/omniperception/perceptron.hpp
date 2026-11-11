#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "decider.hpp"
#include "detection.hpp"
#include "io/camera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/thread_safe_queue.hpp"

namespace omniperception
{

class Perceptron
{
public:
  Perceptron(
    io::CameraBase * left_camera, io::CameraBase * right_camera,
    const std::string & config_path);

  ~Perceptron();

  std::vector<DetectionResult> get_detection_queue();

  // Disable inference while the main camera is acquiring a target. Workers
  // still drain frames, so re-enabling never reuses an old image.
  void set_enabled(bool enabled);
  bool enabled() const;

  void parallel_infer(
    io::CameraBase * cam, const std::string & camera_role,
    const std::shared_ptr<auto_aim::YOLO> & yolo_parallel);

private:
  std::vector<std::thread> threads_;
  tools::ThreadSafeQueue<DetectionResult, true> detection_queue_;

  std::vector<std::shared_ptr<auto_aim::YOLO>> detectors_;

  Decider decider_;
  std::atomic<bool> stop_flag_;
  std::atomic<bool> enabled_;
  std::atomic<std::uint64_t> generation_{0};
  std::mutex publication_mutex_;
};

}  // namespace omniperception
#endif