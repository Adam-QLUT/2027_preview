#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
class Solver;

class Target
{
public:
  ArmorName name = ArmorName::not_armor;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, double v1 = 100, double v2 = 400, double q_pitch = 1e-4);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor, const Solver & solver);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  Eigen::Isometry3d armor_pose(int id) const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  int outpost_rotation_direction_ = 0;  // 0: unknown, 1: CCW, -1: CW
  std::array<bool, 3> outpost_height_initialized_{{true, false, false}};

  double v1_;  // 加速度方差
  double v2_;  // 角加速度方差
  double q_pitch_ = 1e-4;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_uv(const Armor & armor, int id, const Solver & solver);
  void lock_outpost_rotation_direction();
  void enforce_outpost_radius();
  void initialize_outpost_height(const Armor & armor, int id);
  void clamp_outpost_height_offsets();
  double association_error(const Armor & armor, int id, const Solver & solver) const;

};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
