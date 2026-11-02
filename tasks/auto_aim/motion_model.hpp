#ifndef AUTO_AIM__MOTION_MODEL_HPP
#define AUTO_AIM__MOTION_MODEL_HPP

#include <Eigen/Geometry>
#include <array>
#include <cmath>

#include "armor.hpp"

namespace auto_aim::motion_model
{
// Geometry used only by the PnP/Ceres pose solver; no tracking observation model.
constexpr double ARMOR_PITCH = 15.0 * CV_PI / 180.0;

inline double armor_pitch(ArmorName name)
{
  return name == ArmorName::outpost ? -ARMOR_PITCH : ARMOR_PITCH;
}

template <typename T>
Eigen::Matrix<T, 3, 3> rotation(const T & yaw, const T & pitch, const T & roll = T(0))
{
  using std::cos;
  using std::sin;
  const T cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
  const T cr = cos(roll), sr = sin(roll);
  Eigen::Matrix<T, 3, 3> result;
  result << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr;
  return result;
}

inline std::array<Eigen::Vector3d, 4> object_points(ArmorType type)
{
  const double half_width = (type == ArmorType::big ? 0.230 : 0.135) / 2.0;
  constexpr double half_height = 0.056 / 2.0;
  return {{{0, half_width, half_height},
           {0, -half_width, half_height},
           {0, -half_width, -half_height},
           {0, half_width, -half_height}}};
}

struct Camera
{
  Eigen::Matrix3d matrix;
  Eigen::Matrix<double, 5, 1> distortion;
  Eigen::Matrix3d R_world2camera;
  Eigen::Vector3d t_world2camera;

  template <typename T>
  bool project(const Eigen::Transform<T, 3, Eigen::Isometry> & pose, ArmorType type,
               std::array<Eigen::Matrix<T, 2, 1>, 4> & pixels) const
  {
    const auto points = object_points(type);
    for (int i = 0; i < 4; ++i) {
      const Eigen::Matrix<T, 3, 1> p =
        R_world2camera.cast<T>() * (pose * points[i].cast<T>()) + t_world2camera.cast<T>();
      if (p.z() <= T(1e-6)) return false;
      const T x = p.x() / p.z(), y = p.y() / p.z();
      const T r2 = x * x + y * y;
      const T radial =
        T(1) + r2 * (T(distortion[0]) + r2 * (T(distortion[1]) + r2 * T(distortion[4])));
      const T xd =
        x * radial + T(2 * distortion[2]) * x * y + T(distortion[3]) * (r2 + T(2) * x * x);
      const T yd =
        y * radial + T(distortion[2]) * (r2 + T(2) * y * y) + T(2 * distortion[3]) * x * y;
      pixels[i] << T(matrix(0, 0)) * xd + T(matrix(0, 2)), T(matrix(1, 1)) * yd + T(matrix(1, 2));
    }
    return true;
  }
};

}  // namespace auto_aim::motion_model

#endif
