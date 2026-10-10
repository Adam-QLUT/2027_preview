#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <list>
#include <mutex>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0.0f;
  float target_pitch = 0.0f;
  float yaw = 0.0f;
  float yaw_vel = 0.0f;
  float yaw_acc = 0.0f;
  float pitch = 0.0f;
  float pitch_vel = 0.0f;
  float pitch_acc = 0.0f;
};

class Planner
{
public:
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);
  std::optional<Eigen::Isometry3d> debug_armor_pose() const;

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double outpost_delay_time_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  mutable std::mutex debug_pose_mutex_;
  std::optional<Eigen::Isometry3d> debug_armor_pose_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
  void set_debug_armor_pose(std::optional<Eigen::Isometry3d> pose);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
