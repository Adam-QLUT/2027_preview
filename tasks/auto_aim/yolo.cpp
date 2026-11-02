#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolo11.hpp"
#include "yolos/yolov5.hpp"
#include "yolos/yolov8.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  if (yolo_name == "yolov8") {
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
  }

  else if (yolo_name == "yolo11") {
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
  }

  else if (yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::optional<YOLO::AsyncResult> YOLO::detect_async(
  const cv::Mat & img, std::chrono::steady_clock::time_point timestamp, int frame_count)
{
  auto * yolov5 = dynamic_cast<YOLOV5 *>(yolo_.get());
  if (yolov5 == nullptr) {
    throw std::logic_error("Asynchronous inference is only implemented for YOLOv5");
  }

  auto result = yolov5->detect_async(img, timestamp, frame_count);
  if (!result) return std::nullopt;
  return AsyncResult{
    std::move(result->armors), result->timestamp, std::move(result->image), result->frame_count};
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
