#include "target.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "motion_model.hpp"
#include "solver.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_RADIUS = 0.2765;                // m
constexpr double OUTPOST_ROTATION_SPEED = 0.8 * CV_PI;  // 0.4 r/s
constexpr double OUTPOST_DIRECTION_THRESHOLD = 2.0;     // rad/s
constexpr double OUTPOST_HEIGHT_MIN = 0;              // m
constexpr double OUTPOST_HEIGHT_MAX = 0.21;              // m
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, double v1, double v2, double q_pitch)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  v1_(v1),
  v2_(v2),
  q_pitch_(q_pitch),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  outpost_rotation_direction_(0),
  switch_count_(0)
{
  const bool is_outpost = name == ArmorName::outpost && armor_num_ == 3;
  auto r = is_outpost ? OUTPOST_RADIUS : radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const bool fixed_pitch = name == ArmorName::outpost || name == ArmorName::base;
  const double car_pitch = fixed_pitch ? 0.0 : ypr[1] - motion_model::armor_pitch(name);
  const Eigen::Vector3d center = xyz +
    motion_model::rotation(ypr[0], car_pitch) * Eigen::Vector3d(r, 0, 0);
  const double center_z = center.z();

  // x vx y vy z vz a w r l h car_pitch; car roll is fixed at zero.
  // a: angle
  // w: angular velocity
  // l: r2 - r1; for the outpost, z1 - z0
  // h: z2 - z1; for the outpost, z2 - z0
  Eigen::VectorXd x0{{center.x(), 0, center.y(), 0, center_z, 0, ypr[0], 0, r, 0, 0, car_pitch}};
  if (P0_dig.size() == motion_model::STATE_DIM - 1) {
    P0_dig.conservativeResize(motion_model::STATE_DIM);
    P0_dig[motion_model::CAR_PITCH] = 0.1;
  }
  if (P0_dig.size() != motion_model::STATE_DIM) {
    throw std::invalid_argument("Target covariance must have 12 entries");
  }
  if (fixed_pitch) P0_dig[motion_model::CAR_PITCH] = 0.0;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  if (name == ArmorName::outpost && armor_num_ == 3) {
    tools::logger()->debug(
      "[Target][Outpost] init: armor 0 anchored at z={:.3f} m; armor 1/2 heights unknown",
      center_z);
  }
  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  enforce_outpost_radius();
}

Target::Target(double x, double vyaw, double radius, double h)
: name(ArmorName::three), armor_type(ArmorType::small), priority(ArmorPriority::first),
  jumped(false), last_id(0), armor_num_(4), switch_count_(0), update_count_(0),
  is_switch_(false), is_converged_(false), outpost_rotation_direction_(0), v1_(100), v2_(400)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h, 0}};
  Eigen::VectorXd P0_dig = Eigen::VectorXd::Zero(motion_model::STATE_DIM);
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(motion_model::STATE_DIM, motion_model::STATE_DIM);
  for (int position : {0, 2, 4, 6}) F(position, position + 1) = dt;

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1 = v1_, v2 = v2_;
  if (name == ArmorName::outpost) {
    v1 = 10;  // 前哨站加速度方差
    v2 = (outpost_rotation_direction_ == 0) ? 200 : 0.1;
  }

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(motion_model::STATE_DIM, motion_model::STATE_DIM);
  for (int position : {0, 2, 4, 6}) {
    const double variance = position == 6 ? v2 : v1;
    Q(position, position) = a * variance;
    Q(position, position + 1) = Q(position + 1, position) = b * variance;
    Q(position + 1, position + 1) = c * variance;
  }
  if (name != ArmorName::outpost && name != ArmorName::base) {
    Q(motion_model::CAR_PITCH, motion_model::CAR_PITCH) = std::abs(dt) * q_pitch_;
  }

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (name == ArmorName::outpost && outpost_rotation_direction_ != 0) {
    ekf_.x[7] = outpost_rotation_direction_ * OUTPOST_ROTATION_SPEED;
  }

  ekf_.predict(F, Q, f);
  enforce_outpost_radius();
}

void Target::update(const Armor & armor, const Solver & solver)
{
  // 装甲板匹配
  int id = 0;
  auto min_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 普通车辆只考虑距离相机最近的三块板；前哨站恰好有三块板。
  const int candidate_count = std::min(3, armor_num_);
  for (int i = 0; i < candidate_count; i++) {
    const int candidate_id = xyza_i_list[i].second;
    const auto error = association_error(armor, candidate_id, solver);

    if (error < min_error) {
      id = candidate_id;
      min_error = error;
    }
  }
  if (!std::isfinite(min_error) || min_error == 1e10) return;

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  initialize_outpost_height(armor, id);
  update_uv(armor, id, solver);
  enforce_outpost_radius();
  clamp_outpost_height_offsets();
  lock_outpost_rotation_direction();
}

double Target::association_error(const Armor & armor, int id, const Solver & solver) const
{
  Eigen::VectorXd prediction;
  if (!solver.armor_uv_prediction(ekf_.x, id, armor_num_, armor_type, name, prediction)) {
    return std::numeric_limits<double>::infinity();
  }
  Eigen::VectorXd residual = solver.armor_uv_observation(armor) - prediction;
  residual[0] = tools::limit_rad(residual[0]);
  residual[4] = tools::limit_rad(residual[4]);
  if (name == ArmorName::outpost && !outpost_height_initialized_[id]) {
    // Unknown armor heights must not prevent the first association.
    residual[2] = 0;
    residual[6] = 0;
  }
  return residual.dot(solver.armor_uv_noise(armor).ldlt().solve(residual));
}

void Target::initialize_outpost_height(const Armor & armor, int id)
{
  if (
    name != ArmorName::outpost || armor_num_ != 3 || id <= 0 ||
    id >= outpost_height_initialized_.size() || outpost_height_initialized_[id])
    return;

  const int state_index = (id == 1) ? 9 : 10;
  ekf_.x[state_index] = armor.xyz_in_world[2] - ekf_.x[4];
  outpost_height_initialized_[id] = true;
  clamp_outpost_height_offsets();
  tools::logger()->debug(
    "[Target][Outpost] armor {} height offset initialized to {:.3f} m", id,
    ekf_.x[state_index]);
}

void Target::clamp_outpost_height_offsets()
{
  if (name != ArmorName::outpost || armor_num_ != 3) return;

  for (int id = 1; id < 3; ++id) {
    if (!outpost_height_initialized_[id]) continue;

    const int state_index = (id == 1) ? 9 : 10;
    const double sign = ekf_.x[state_index] < 0.0 ? -1.0 : 1.0;
    const double magnitude = std::clamp(
      std::abs(ekf_.x[state_index]), OUTPOST_HEIGHT_MIN, OUTPOST_HEIGHT_MAX);
    ekf_.x[state_index] = sign * magnitude;
  }
}

void Target::enforce_outpost_radius()
{
  if (name != ArmorName::outpost || armor_num_ != 3) return;

  ekf_.x[8] = OUTPOST_RADIUS;
  ekf_.P.row(8).setZero();
  ekf_.P.col(8).setZero();
}

void Target::update_uv(const Armor & armor, int id, const Solver & solver)
{
  Eigen::VectorXd prediction;
  Eigen::MatrixXd H;
  if (!solver.armor_uv_prediction(ekf_.x, id, armor_num_, armor_type, name, prediction, &H)) return;
  const Eigen::VectorXd z = solver.armor_uv_observation(armor);
  const Eigen::MatrixXd R = solver.armor_uv_noise(armor);
  const auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd uv;
    if (!solver.armor_uv_prediction(x, id, armor_num_, armor_type, name, uv)) return prediction;
    return uv;
  };
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[4] = tools::limit_rad(c[4]);
    return c;
  };
  // Chi-square 95% threshold for eight independent light observations.
  ekf_.update(z, H, R, h, z_subtract, 15.5073130559);
  const Eigen::VectorXd residual = z_subtract(z, h(ekf_.x));
  for (int light = 0; light < 2; ++light) {
    const std::string prefix = light == 0 ? "residual_left_" : "residual_right_";
    for (int component = 0; component < 4; ++component) {
      const std::array<std::string, 4> names{{"angle", "u", "v", "length"}};
      ekf_.data[prefix + names[component]] = residual[light * 4 + component];
    }
  }
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    const auto pose = armor_pose(i);
    auto angle = tools::eulers(pose.linear(), 2, 1, 0)[0];
    const Eigen::Vector3d xyz = pose.translation();
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

Eigen::Isometry3d Target::armor_pose(int id) const
{
  return motion_model::armor_pose(ekf_.x.data(), id, armor_num_, name);
}

bool Target::diverged() const
{
  if (!ekf_.x.allFinite() || !ekf_.P.allFinite()) return true;
  if (std::abs(ekf_.x[motion_model::CAR_PITCH]) > CV_PI / 4.0) return true;
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  if (name == ArmorName::outpost) return !r_ok;

  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 20 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

void Target::lock_outpost_rotation_direction()
{
  if (
    name != ArmorName::outpost || outpost_rotation_direction_ != 0 || update_count_ <= 10 ||
    std::abs(ekf_.x[7]) <= OUTPOST_DIRECTION_THRESHOLD)
    return;

  outpost_rotation_direction_ = ekf_.x[7] > 0 ? 1 : -1;
  ekf_.x[7] = outpost_rotation_direction_ * OUTPOST_ROTATION_SPEED;

  // Once the random direction is identified, the known outpost speed is exact.
  ekf_.P.row(7).setZero();
  ekf_.P.col(7).setZero();
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
