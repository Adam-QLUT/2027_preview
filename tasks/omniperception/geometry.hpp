#ifndef OMNIPERCEPTION__GEOMETRY_HPP
#define OMNIPERCEPTION__GEOMETRY_HPP

#include <array>
#include <optional>
#include <string>
#include <opencv2/core.hpp>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
// Upper-layer geometry only. No camera, serial, packet or DetectionResult changes.
class Geometry
{
public:
  explicit Geometry(const std::string & config_path);
  bool calibrated(const std::string & role) const;
  bool frame_size_matches(const std::string & role, cv::Size size) const;

  // Metric vector from the small-yaw handoff origin to the armor, in p's local frame.
  // Null on missing calibration, invalid corners or an unreliable PnP result.
  std::optional<cv::Vec3d> auxiliary_vector(
    const std::string & role, const auto_aim::Armor & armor) const;

  // Undistorted main-camera viewing ray, expressed in the existing gimbal frame.
  std::optional<cv::Vec3d> main_ray(const cv::Point2f & pixel) const;

private:
  class CameraModel
  {
  public:
    bool enabled = false;
    cv::Size image_size;
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Mat D;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0, 0, 0};
    double max_reprojection_error_px = 3.0;
    double min_distance_m = 0.1;
    double max_distance_m = 30.0;
  };
  std::array<CameraModel, 2> auxiliary_;
  CameraModel main_;
  cv::Vec3d handoff_origin_{0, 0, 0};
  double small_width_m_ = 0.135;
  double big_width_m_ = 0.230;
  double lightbar_length_m_ = 0.056;
};
}  // namespace omniperception
#endif
