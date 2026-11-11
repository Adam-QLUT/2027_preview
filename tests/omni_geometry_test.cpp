#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "tasks/omniperception/geometry.hpp"

namespace
{
void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}
std::vector<double> entries(const cv::Matx33d & m) { return {m.val, m.val + 9}; }
void write_yaml(const std::filesystem::path & path, const YAML::Node & node)
{
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write temporary test fixture");
  out << node;
}
template <typename Function>
void rejects(Function f, const char * message)
{
  bool threw = false;
  try { f(); } catch (const std::exception &) { threw = true; }
  require(threw, message);
}
}

int main(int argc, char ** argv)
{
  try {
    if (argc != 2) throw std::runtime_error("Pass a disposable fixture directory");
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    const auto config_path = directory / "sentry.synthetic.yaml";
    const auto calibration_path = directory / "aux.synthetic.yaml";
    const cv::Matx33d K(800, 0, 320, 0, 820, 240, 0, 0, 1);
    const cv::Matx33d optical_to_chassis(0, 0, 1, -1, 0, 0, 0, -1, 0);
    YAML::Node config;
    config["camera_matrix"] = entries(K);
    config["distort_coeffs"] = std::vector<double>{-0.12, 0.02, 0.001, -0.002, 0};
    config["R_camera2gimbal"] = entries(optical_to_chassis);
    config["omni_left_camera_serial_number"] = "SYNTHETIC-LEFT";
    config["omni_right_camera_serial_number"] = "SYNTHETIC-RIGHT";
    write_yaml(config_path, config);
    const omniperception::Geometry fallback(config_path.string());
    require(!fallback.calibrated("left") && !fallback.calibrated("right"),
            "absent calibration must preserve fallback");
    const auto center_ray = fallback.main_ray({320, 240});
    require(center_ray && cv::norm(*center_ray - cv::Vec3d(1, 0, 0)) < 1e-6,
            "main optical center must map to forward axis");

    YAML::Node calibration;
    calibration["schema_version"] = 1;
    calibration["handoff_origin_in_chassis"] = std::vector<double>{0.05, 0, 0.1};
    calibration["armor_dimensions_m"] = std::vector<double>{0.135, 0.230, 0.056};
    std::array<cv::Matx33d, 2> rotations;
    for (int side = 0; side < 2; ++side) {
      const auto role = side == 0 ? "left" : "right";
      const double yaw = (side == 0 ? 120.0 : -120.0) * CV_PI / 180.0;
      const cv::Matx33d mount(std::cos(yaw), -std::sin(yaw), 0,
                             std::sin(yaw), std::cos(yaw), 0, 0, 0, 1);
      rotations[side] = mount * optical_to_chassis;
      auto node = calibration[role];
      node["enabled"] = true;
      node["serial_number"] = side == 0 ? "SYNTHETIC-LEFT" : "SYNTHETIC-RIGHT";
      node["image_width"] = 640;
      node["image_height"] = 480;
      node["camera_matrix"] = config["camera_matrix"];
      node["distort_coeffs"] = config["distort_coeffs"];
      node["R_camera2chassis"] = entries(rotations[side]);
      node["t_camera2chassis"] = std::vector<double>{0.12, 0.04, 0.02};
      node["max_reprojection_error_px"] = 0.5;
    }
    write_yaml(calibration_path, calibration);
    config["omni_calibration_file"] = calibration_path.filename().string();
    write_yaml(config_path, config);
    const omniperception::Geometry geometry(config_path.string());

    const cv::Vec3d position(0.12, -0.08, 3.0);
    cv::Matx33d tilt;
    cv::Rodrigues(cv::Vec3d(0.08, 0.12, 0.03), tilt);
    cv::Vec3d rvec;
    cv::Rodrigues(tilt * optical_to_chassis.t(), rvec);
    const cv::Mat D(config["distort_coeffs"].as<std::vector<double>>(), true);
    // Exercise both armor widths, distortion and nonzero translations for BOTH cameras.
    for (const auto type : {auto_aim::ArmorType::small, auto_aim::ArmorType::big}) {
      const double width = type == auto_aim::ArmorType::small ? 0.135 : 0.230;
      const std::vector<cv::Point3d> object{{0, width/2, 0.028}, {0, -width/2, 0.028},
                                          {0, -width/2, -0.028}, {0, width/2, -0.028}};
      std::vector<cv::Point2d> pixels;
      cv::projectPoints(object, rvec, position, K, D, pixels);
      std::vector<cv::Point2f> corners(pixels.begin(), pixels.end());
      auto_aim::Armor armor(7, 0.99F, {}, corners);
      armor.type = type;
      armor.center_norm = {armor.center.x / 640.0F, armor.center.y / 480.0F};
      for (int side = 0; side < 2; ++side) {
        const auto role = side == 0 ? "left" : "right";
        const auto vector = geometry.auxiliary_vector(role, armor);
        const auto expected = rotations[side] * position + cv::Vec3d(0.12, 0.04, 0.02) -
                              cv::Vec3d(0.05, 0, 0.1);
        require(vector && cv::norm(*vector - expected) < 0.005,
                "PnP must recover metric target vector including each camera baseline");
      }
      auto bad = armor;
      bad.center_norm.x = bad.center.x / 1280.0F;
      require(!geometry.auxiliary_vector("left", bad), "wrong capture resolution rejected");
      bad = armor;
      bad.points[1] = bad.points[0];
      require(!geometry.auxiliary_vector("left", bad), "degenerate corners rejected");
      bad = armor;
      bad.points[0].x -= 25;
      require(!geometry.auxiliary_vector("left", bad), "large reprojection residual rejected");
    }
    auto invalid = YAML::Clone(calibration);
    invalid["left"]["serial_number"] = "WRONG-CAMERA";
    write_yaml(calibration_path, invalid);
    rejects([&] { omniperception::Geometry g(config_path.string()); }, "serial mismatch rejected");
    invalid = YAML::Clone(calibration);
    invalid["left"]["R_camera2chassis"] = std::vector<double>{1,0,0,0,1,0,0,0,-1};
    write_yaml(calibration_path, invalid);
    rejects([&] { omniperception::Geometry g(config_path.string()); }, "reflection rejected");
    invalid = YAML::Clone(calibration);
    invalid["left"]["camera_matrix"] = std::vector<double>{0,0,320,0,820,240,0,0,1};
    write_yaml(calibration_path, invalid);
    rejects([&] { omniperception::Geometry g(config_path.string()); }, "zero focal length rejected");
    std::cout << "PASS: production calibration/PnP regressions (synthetic data only)\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}