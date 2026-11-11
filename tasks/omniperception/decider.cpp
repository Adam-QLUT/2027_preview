#include "decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "trace.hpp"

namespace omniperception
{
Decider::Decider(const std::string & config_path) : count_(0)
{
  auto yaml = YAML::LoadFile(config_path);
  img_width_ = yaml["image_width"].as<double>();
  img_height_ = yaml["image_height"].as<double>();
  fov_h_ = yaml["fov_h"].as<double>();
  fov_v_ = yaml["fov_v"].as<double>();
  new_fov_h_ = yaml["omni_horizontal_fov_deg"]
                 ? yaml["omni_horizontal_fov_deg"].as<double>()
                 : yaml["new_fov_h"].as<double>();
  new_fov_v_ = yaml["omni_vertical_fov_deg"]
                 ? yaml["omni_vertical_fov_deg"].as<double>()
                 : yaml["new_fov_v"].as<double>();
  left_yaw_offset_deg_ = yaml["omni_left_yaw_deg"] ? yaml["omni_left_yaw_deg"].as<double>() : 120.0;
  right_yaw_offset_deg_ = yaml["omni_right_yaw_deg"] ? yaml["omni_right_yaw_deg"].as<double>() : -120.0;
  max_detection_age_ms_ = yaml["omni_max_detection_age_ms"]
                           ? yaml["omni_max_detection_age_ms"].as<int>()
                           : 120;
  enemy_color_ =
    (yaml["enemy_color"].as<std::string>() == "red") ? auto_aim::Color::red : auto_aim::Color::blue;
  mode_ = yaml["mode"].as<double>();
  trace::event(1,
    "stage=decider_config path={} enemy_color={} mode={} max_age_ms={} left_yaw_deg={} "
    "right_yaw_deg={} horizontal_fov_deg={} vertical_fov_deg={}",
    config_path, static_cast<int>(enemy_color_), mode_, max_detection_age_ms_,
    left_yaw_offset_deg_, right_yaw_offset_deg_, new_fov_h_, new_fov_v_);
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
  io::USBCamera & usbcam2, io::Camera & back_camera)
{
  Eigen::Vector2d delta_angle;
  io::USBCamera * cams[] = {&usbcam1, &usbcam2};

  cv::Mat usb_img;
  std::chrono::steady_clock::time_point timestamp;
  if (count_ < 0 || count_ > 2) {
    throw std::runtime_error("count_ out of valid range [0,2]");
  }
  if (count_ == 2) {
    back_camera.read(usb_img, timestamp);
  } else {
    cams[count_]->read(usb_img, timestamp);
  }
  auto armors = yolo.detect(usb_img);
  auto empty = armor_filter(armors);

  if (!empty) {
    if (count_ == 2) {
      delta_angle = this->delta_angle(armors, "back");
    } else {
      delta_angle = this->delta_angle(armors, cams[count_]->device_name);
    }

    tools::logger()->debug(
      "[{} camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      (count_ == 2 ? "back" : cams[count_]->device_name), delta_angle[0], delta_angle[1],
      armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    count_ = (count_ + 1) % 3;

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  count_ = (count_ + 1) % 3;
  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera)
{
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  back_cammera.read(img, timestamp);
  auto armors = yolo.detect(img);
  auto empty = armor_filter(armors);

  if (!empty) {
    auto delta_angle = this->delta_angle(armors, "back");
    tools::logger()->debug(
      "[back camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      delta_angle[0], delta_angle[1], armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) {
    return io::Command{false, false, 0, 0};
  }

  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  tools::logger()->info(
    "omniperceptron find {},delta yaw is {:.4f}", auto_aim::ARMOR_NAMES[dr.armors.front().name],
    dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
};

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  Eigen::Vector2d delta_angle;
  if (camera == "left") {
    delta_angle[0] = left_yaw_offset_deg_ + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
    return delta_angle;
  }

  else if (camera == "right") {
    delta_angle[0] = right_yaw_offset_deg_ + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
    return delta_angle;
  }

  else {
    delta_angle[0] = 170 + (54.2 / 2) - armors.front().center_norm.x * 54.2;
    delta_angle[1] = armors.front().center_norm.y * 44.5 - 44.5 / 2;
    return delta_angle;
  }
}

bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return true;
  const auto before = armors.size();
  // 过滤非敌方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });
  const auto after_color = armors.size();

  // 25赛季没有5号装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::five; });
  const auto after_five = armors.size();
  // 不打工程
  // armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::two; });
  // 不打前哨站
  armors.remove_if(
    [&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::outpost; });
  const auto after_outpost = armors.size();

  // 过滤掉刚复活无敌的装甲板
  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
           invincible_armor_.end();
  });

  trace::event(2,
    "stage=armor_filter n_in={} n_out={} reject_color={} reject_five={} reject_outpost={} "
    "reject_invincible={}", before, armors.size(), before - after_color,
    after_color - after_five, after_five - after_outpost, after_outpost - armors.size());
  return armors.empty();
}

void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return;

  const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;

  if (!armors.empty()) {
    for (auto & armor : armors) {
      armor.priority = priority_map.at(armor.name);
    }
  }
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return;

  const auto now = std::chrono::steady_clock::now();

  // 对每个 DetectionResult 调用 armor_filter 和 set_priority
  for (auto & dr : detection_queue) {
    const auto before = dr.armors.size();
    const int previous_id = dr.armors.empty() ? -1 : static_cast<int>(dr.armors.front().name);
    const auto decision_age_ms = std::chrono::duration<double, std::milli>(now - dr.timestamp).count();
    trace::event(2, "stage=sort_in camera={} frame_us={} checked_age_ms={:.2f} n={} id={}",
                 dr.camera, trace::time_us(dr.timestamp), decision_age_ms, before, previous_id);
    if (
      dr.timestamp == std::chrono::steady_clock::time_point{} ||
      dr.timestamp > now ||
      now - dr.timestamp > std::chrono::milliseconds(max_detection_age_ms_)) {
      trace::event(2,
        "stage=sort_drop camera={} frame_us={} checked_age_ms={:.2f} n={} reason=stale max_age_ms={}",
        dr.camera, trace::time_us(dr.timestamp), decision_age_ms, before,
        max_detection_age_ms_);
      dr.armors.clear();
      continue;
    }
    armor_filter(dr.armors);
    set_priority(dr.armors);

    // 对每个 DetectionResult 中的 armors 进行排序
    dr.armors.sort(
      [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
    if (!dr.armors.empty()) {
      const auto angle = delta_angle(dr.armors, dr.camera);
      dr.delta_yaw = angle[0] * CV_PI / 180.0;
      dr.delta_pitch = angle[1] * CV_PI / 180.0;
    }
    trace::event(2,
      "stage=sort_out camera={} frame_us={} n_in={} n_out={} previous_id={} id={} "
      "yaw_rad={:.4f} pitch_rad={:.4f}",
      dr.camera, trace::time_us(dr.timestamp), before, dr.armors.size(), previous_id,
      dr.armors.empty() ? -1 : static_cast<int>(dr.armors.front().name), dr.delta_yaw, dr.delta_pitch);
  }

  detection_queue.erase(
    std::remove_if(
      detection_queue.begin(), detection_queue.end(),
      [](const DetectionResult & result) { return result.armors.empty(); }),
    detection_queue.end());
  if (detection_queue.empty()) return;

  // 根据优先级对 DetectionResult 进行排序
  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      if (a.armors.front().priority != b.armors.front().priority) {
        return a.armors.front().priority < b.armors.front().priority;
      }
      return a.timestamp > b.timestamp;
    });
}

Eigen::Vector4d Decider::get_target_info(
  const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets)
{
  if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

  auto target = targets.front();

  for (const auto & armor : armors) {
    if (armor.name == target.name) {
      return Eigen::Vector4d{
        armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
        static_cast<double>(armor.name) + 1};  //避免歧义+1(详见通信协议)
    }
  }

  return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
  invincible_armor_.clear();

  if (invincible_enemy_ids.empty()) return;

  for (const auto & id : invincible_enemy_ids) {
    tools::logger()->info("invincible armor id: {}", id);
    invincible_armor_.push_back(auto_aim::ArmorName(id - 1));
  }
}

void Decider::get_auto_aim_target(
  std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
  if (auto_aim_target.empty()) return;

  std::vector<auto_aim::ArmorName> auto_aim_targets;

  for (const auto & target : auto_aim_target) {
    if (target <= 0 || static_cast<size_t>(target) > auto_aim::ARMOR_NAMES.size()) {
      tools::logger()->warn("Received invalid auto_aim target value: {}", int(target));
      continue;
    }
    auto_aim_targets.push_back(static_cast<auto_aim::ArmorName>(target - 1));
    tools::logger()->info("nav send auto_aim target is {}", auto_aim::ARMOR_NAMES[target - 1]);
  }

  if (auto_aim_targets.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
           auto_aim_targets.end();
  });
}

}  // namespace omniperception