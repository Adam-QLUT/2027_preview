#ifndef AUTO_AIM__MOTION_MODEL_HPP
#define AUTO_AIM__MOTION_MODEL_HPP

#include <Eigen/Geometry>
#include <array>
#include <cmath>

#include "armor.hpp"

namespace auto_aim::motion_model
{
constexpr int STATE_DIM = 12;
constexpr int CAR_PITCH = 11;
constexpr int UV_DIM = 8;
constexpr double ARMOR_PITCH = 15.0 * CV_PI / 180.0;

inline double armor_pitch(ArmorName name)
{
  return name == ArmorName::outpost ? -ARMOR_PITCH : ARMOR_PITCH;
}

template <typename T>
Eigen::Matrix<T, 3, 3> rotation(const T & yaw, const T & pitch)
{
  using std::cos;
  using std::sin;
  const T cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
  Eigen::Matrix<T, 3, 3> result;
  result << cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, T(0), cp;
  return result;
}

template <typename T>
Eigen::Transform<T, 3, Eigen::Isometry> armor_pose(const T * x, int id, int armor_num,
                                                   ArmorName name)
{
  using std::cos;
  using std::sin;
  const bool alternate = armor_num == 4 && (id & 1);
  const T radius = alternate ? x[8] + x[9] : x[8];
  const T phase = T(id * 2.0 * CV_PI / armor_num);
  T height = alternate ? x[10] : T(0);
  if (name == ArmorName::outpost && armor_num == 3) {
    if (id == 1) height = x[9];
    if (id == 2) height = x[10];
  }
  // Roll is absent. Pitch tilts both the armor orbit and its orientation.
  const T pitch = name == ArmorName::outpost || name == ArmorName::base ? T(0) : x[CAR_PITCH];
  const Eigen::Matrix<T, 3, 3> car_rotation = rotation(x[6], pitch);
  auto pose = Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
  pose.translation() =
    Eigen::Matrix<T, 3, 1>(x[0], x[2], x[4]) +
    car_rotation * Eigen::Matrix<T, 3, 1>(-radius * cos(phase), -radius * sin(phase), height);
  pose.linear() = car_rotation * rotation(phase, T(armor_pitch(name)));
  return pose;
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

template <typename T>
Eigen::Matrix<T, UV_DIM, 1> light_observation(const std::array<Eigen::Matrix<T, 2, 1>, 4> & pixels)
{
  using std::atan2;
  using std::sqrt;
  Eigen::Matrix<T, UV_DIM, 1> observation;
  for (int light = 0; light < 2; ++light) {
    const auto & top = pixels[light];
    const auto & bottom = pixels[3 - light];
    const Eigen::Matrix<T, 2, 1> delta = top - bottom;
    const Eigen::Matrix<T, 2, 1> center = (top + bottom) / T(2);
    observation.template segment<4>(light * 4) << atan2(delta.x(), delta.y()), center.x(),
      center.y(), sqrt(delta.squaredNorm());
  }
  return observation;
}
}  // namespace auto_aim::motion_model

#endif
