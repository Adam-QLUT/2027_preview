#include "geometry.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace omniperception
{
namespace
{
int slot(const std::string & role)
{
  return role == "left" ? 0 : (role == "right" ? 1 : -1);
}

std::vector<double> numbers(const YAML::Node & node, const char * key, size_t size)
{
  if (!node[key]) throw std::runtime_error(std::string("Missing calibration field: ") + key);
  auto values = node[key].as<std::vector<double>>();
  if (values.size() != size ||
      !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); })) {
    throw std::runtime_error(std::string("Invalid calibration field: ") + key);
  }
  return values;
}

cv::Matx33d matrix(const YAML::Node & node, const char * key)
{
  const auto values = numbers(node, key, 9);
  return cv::Matx33d(values.data());
}

void check_rotation(const cv::Matx33d & R)
{
  if (cv::norm(cv::Mat(R.t() * R - cv::Matx33d::eye())) > 1e-3 ||
      std::abs(cv::determinant(cv::Mat(R)) - 1.0) > 1e-3) {
    throw std::runtime_error("Calibration rotation must be orthonormal with determinant +1");
  }
}

void load_intrinsics(const YAML::Node & node, cv::Matx33d & K, cv::Mat & D)
{
  K = matrix(node, "camera_matrix");
  if (K(0, 0) <= 0 || K(1, 1) <= 0 || std::abs(K(2, 2) - 1) > 1e-9 ||
      std::abs(K(0, 1)) > 1e-9 || std::abs(K(1, 0)) > 1e-9 ||
      std::abs(K(2, 0)) > 1e-9 || std::abs(K(2, 1)) > 1e-9) {
    throw std::runtime_error("Invalid pinhole camera_matrix");
  }
  if (!node["distort_coeffs"]) throw std::runtime_error("Missing distort_coeffs");
  const auto count = node["distort_coeffs"].size();
  if (count != 4 && count != 5 && count != 8 && count != 12 && count != 14) {
    throw std::runtime_error("Expected OpenCV pinhole distortion coefficients, not fisheye");
  }
  const auto coefficients = numbers(node, "distort_coeffs", count);
  D = cv::Mat(coefficients, true).reshape(1, 1).clone();
}

bool finite(const cv::Vec3d & v)
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
}  // namespace

Geometry::Geometry(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);
  // The same existing main-camera calibration as Solver; no second main calibration.
  load_intrinsics(yaml, main_.K, main_.D);
  main_.R = matrix(yaml, "R_camera2gimbal");
  check_rotation(main_.R);
  main_.enabled = true;

  if (!yaml["omni_calibration_file"] || yaml["omni_calibration_file"].as<std::string>().empty()) return;
  std::filesystem::path path(yaml["omni_calibration_file"].as<std::string>());
  if (path.is_relative()) path = std::filesystem::path(config_path).parent_path() / path;
  const auto calibration = YAML::LoadFile(path.string());
  if (!calibration["schema_version"] || calibration["schema_version"].as<int>() != 1) {
    throw std::runtime_error("Unsupported omni calibration schema_version");
  }
  bool any_enabled = false;
  for (const auto * role : {"left", "right"}) {
    const auto node = calibration[role];
    if (!node || !node["enabled"] || !node["enabled"].as<bool>()) continue;
    any_enabled = true;
    auto & camera = auxiliary_[slot(role)];
    const auto serial_key = std::string("omni_") + role + "_camera_serial_number";
    if (!node["serial_number"] || !yaml[serial_key] ||
        node["serial_number"].as<std::string>() != yaml[serial_key].as<std::string>()) {
      throw std::runtime_error(std::string("Calibration serial mismatch: ") + role);
    }
    camera.image_size = {node["image_width"].as<int>(), node["image_height"].as<int>()};
    if (camera.image_size.width <= 0 || camera.image_size.height <= 0) {
      throw std::runtime_error("Calibration needs the actual capture image dimensions");
    }
    load_intrinsics(node, camera.K, camera.D);
    camera.R = matrix(node, "R_camera2chassis");
    check_rotation(camera.R);
    const auto t = numbers(node, "t_camera2chassis", 3);
    camera.t = cv::Vec3d(t.data());
    if (node["max_reprojection_error_px"])
      camera.max_reprojection_error_px = node["max_reprojection_error_px"].as<double>();
    if (node["min_distance_m"]) camera.min_distance_m = node["min_distance_m"].as<double>();
    if (node["max_distance_m"]) camera.max_distance_m = node["max_distance_m"].as<double>();
    if (!std::isfinite(camera.max_reprojection_error_px) || camera.max_reprojection_error_px <= 0 ||
        !std::isfinite(camera.min_distance_m) || camera.min_distance_m <= 0 ||
        !std::isfinite(camera.max_distance_m) || camera.max_distance_m <= camera.min_distance_m) {
      throw std::runtime_error("Invalid omni PnP validation limits");
    }
    camera.enabled = true;
  }
  if (!any_enabled) return;
  const auto origin = numbers(calibration, "handoff_origin_in_chassis", 3);
  handoff_origin_ = cv::Vec3d(origin.data());
  const auto dimensions = numbers(calibration, "armor_dimensions_m", 3);
  if (std::any_of(dimensions.begin(), dimensions.end(), [](double v) { return v <= 0; })) {
    throw std::runtime_error("Armor dimensions must be positive metres");
  }
  small_width_m_ = dimensions[0];
  big_width_m_ = dimensions[1];
  lightbar_length_m_ = dimensions[2];
}

bool Geometry::calibrated(const std::string & role) const
{
  const auto index = slot(role);
  return index >= 0 && auxiliary_[index].enabled;
}

bool Geometry::frame_size_matches(const std::string & role, cv::Size size) const
{
  const auto index = slot(role);
  return index >= 0 && (!auxiliary_[index].enabled || auxiliary_[index].image_size == size);
}

std::optional<cv::Vec3d> Geometry::main_ray(const cv::Point2f & pixel) const
{
  if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) return std::nullopt;
  std::vector<cv::Point2f> corrected;
  cv::undistortPoints(std::vector<cv::Point2f>{pixel}, corrected, main_.K, main_.D);
  const auto ray = main_.R * cv::Vec3d(corrected[0].x, corrected[0].y, 1.0);
  return finite(ray) ? std::optional<cv::Vec3d>(ray) : std::nullopt;
}

std::optional<cv::Vec3d> Geometry::auxiliary_vector(
  const std::string & role, const auto_aim::Armor & armor) const
{
  if (!calibrated(role) || armor.points.size() != 4 ||
      (armor.type != auto_aim::ArmorType::small && armor.type != auto_aim::ArmorType::big)) {
    return std::nullopt;
  }
  const auto & camera = auxiliary_[slot(role)];
  // DetectionResult must stay unchanged. YOLO already supplies center in full-frame
  // pixels AND normalized by the actual frame size; cross-check that correspondence.
  if (!std::isfinite(armor.center.x) || !std::isfinite(armor.center.y) ||
      !std::isfinite(armor.center_norm.x) || !std::isfinite(armor.center_norm.y) ||
      armor.center_norm.x <= 0 || armor.center_norm.x >= 1 ||
      armor.center_norm.y <= 0 || armor.center_norm.y >= 1 ||
      std::abs(armor.center.x / armor.center_norm.x - camera.image_size.width) > 0.1 ||
      std::abs(armor.center.y / armor.center_norm.y - camera.image_size.height) > 0.1) {
    return std::nullopt;
  }
  for (const auto & point : armor.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        point.x < 0 || point.x >= camera.image_size.width ||
        point.y < 0 || point.y >= camera.image_size.height) return std::nullopt;
  }
  if (!cv::isContourConvex(armor.points) || std::abs(cv::contourArea(armor.points)) < 1.0) {
    return std::nullopt;
  }
  const auto width = armor.type == auto_aim::ArmorType::big ? big_width_m_ : small_width_m_;
  // Same corner order as Armor/Solver: left-top, right-top, right-bottom, left-bottom.
  const std::vector<cv::Point3d> object{
    {0, width / 2, lightbar_length_m_ / 2}, {0, -width / 2, lightbar_length_m_ / 2},
    {0, -width / 2, -lightbar_length_m_ / 2}, {0, width / 2, -lightbar_length_m_ / 2}};
  cv::Vec3d rvec, tvec;
  try {
    if (!cv::solvePnP(object, armor.points, camera.K, camera.D, rvec, tvec, false,
                      cv::SOLVEPNP_IPPE) || !finite(rvec) || !finite(tvec)) return std::nullopt;
    cv::Matx33d rotation;
    cv::Rodrigues(rvec, rotation);
    for (const auto & point : object) {
      if ((rotation * cv::Vec3d(point.x, point.y, point.z) + tvec)[2] <= 0) return std::nullopt;
    }
    const auto distance = cv::norm(tvec);
    if (distance < camera.min_distance_m || distance > camera.max_distance_m) return std::nullopt;
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object, rvec, tvec, camera.K, camera.D, projected);
    double sum_squared_error = 0;
    for (size_t i = 0; i < projected.size(); ++i) {
      const auto error = projected[i] - cv::Point2d(armor.points[i]);
      sum_squared_error += error.dot(error);
    }
    if (!std::isfinite(sum_squared_error) ||
        std::sqrt(sum_squared_error / 4.0) > camera.max_reprojection_error_px) return std::nullopt;
    const auto vector = camera.R * tvec + camera.t - handoff_origin_;
    if (!finite(vector) || cv::norm(vector) < 1e-6) return std::nullopt;
    return vector;
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
}
}  // namespace omniperception