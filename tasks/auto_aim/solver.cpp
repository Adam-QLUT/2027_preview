#include "solver.hpp"

#include <ceres/ceres.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

namespace
{
struct PixelResidual
{
  motion_model::Camera camera;
  ArmorType type;
  std::vector<cv::Point2f> pixels;

  template <typename T>
  bool operator()(const T * parameters, T * residuals) const
  {
    auto pose = Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
    pose.translation() << parameters[0], parameters[1], parameters[2];
    pose.linear() = motion_model::rotation(parameters[3], parameters[4]);
    std::array<Eigen::Matrix<T, 2, 1>, 4> projected;
    if (!camera.project(pose, type, projected)) return false;
    for (int i = 0; i < 4; ++i) {
      residuals[2 * i] = projected[i].x() - T(pixels[i].x);
      residuals[2 * i + 1] = projected[i].y() - T(pixels[i].y);
    }
    return true;
  }
};
}  // namespace

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
  if (yaml["uv_sigma_center_ratio"]) {
    uv_sigma_center_ratio_ = yaml["uv_sigma_center_ratio"].as<double>();
  }
  if (yaml["uv_sigma_length_ratio"]) {
    uv_sigma_length_ratio_ = yaml["uv_sigma_length_ratio"].as<double>();
  }
  if (yaml["uv_sigma_angle"]) uv_sigma_angle_ = yaml["uv_sigma_angle"].as<double>();
  if (!std::isfinite(uv_sigma_center_ratio_) || !std::isfinite(uv_sigma_length_ratio_) ||
      !std::isfinite(uv_sigma_angle_) || uv_sigma_center_ratio_ <= 0 ||
      uv_sigma_length_ratio_ <= 0 || uv_sigma_angle_ <= 0) {
    throw std::invalid_argument("UV measurement noise must be positive");
  }
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

//solvePnP（获得姿态）
void Solver::solve(Armor & armor) const
{
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
  cv::SOLVEPNP_IPPE);  // cv::SOLVEPNP_ITERATIVE感觉效果也不错  cv::SOLVEPNP_IPPE 

  cv::cv2eigen(tvec, armor.xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * armor.xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  armor.ypr_in_camera = tools::eulers(R_armor2camera, 2, 1, 0);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  armor.yaw_raw = armor.ypr_in_world[0];
  optimize_pose(armor);
}

motion_model::Camera Solver::camera() const
{
  motion_model::Camera result;
  cv::cv2eigen(camera_matrix_, result.matrix);
  for (int i = 0; i < 5; ++i) result.distortion[i] = distort_coeffs_.at<double>(0, i);
  result.R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  result.t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;
  return result;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Isometry3d & pose, ArmorType type) const
{
  std::array<Eigen::Vector2d, 4> projected;
  if (!camera().project(pose, type, projected)) return {};
  std::vector<cv::Point2f> pixels;
  for (const auto & point : projected) pixels.emplace_back(point.x(), point.y());
  return pixels;
}

Eigen::VectorXd Solver::armor_uv_observation(const Armor & armor) const
{
  std::array<Eigen::Vector2d, 4> pixels;
  for (int i = 0; i < 4; ++i) pixels[i] << armor.points[i].x, armor.points[i].y;
  return motion_model::light_observation(pixels);
}

Eigen::MatrixXd Solver::armor_uv_noise(const Armor & armor) const
{
  Eigen::VectorXd diagonal(motion_model::UV_DIM);
  for (int light = 0; light < 2; ++light) {
    const double length = cv::norm(armor.points[light] - armor.points[3 - light]);
    const double sigma_center = std::max(0.5, uv_sigma_center_ratio_ * length);
    const double sigma_length = std::max(1.0, uv_sigma_length_ratio_ * length);
    diagonal.segment<4>(light * 4) << tools::square(uv_sigma_angle_) / 2.0,
      tools::square(sigma_center) / 2.0, tools::square(sigma_center) / 2.0,
      tools::square(sigma_length) / 2.0;
  }
  return diagonal.asDiagonal();
}

bool Solver::armor_uv_prediction(
  const Eigen::VectorXd & x, int id, int armor_num, ArmorType type, ArmorName name,
  Eigen::VectorXd & observation, Eigen::MatrixXd * jacobian) const
{
  if (!jacobian) {
    std::array<Eigen::Vector2d, 4> projected;
    if (!camera().project(motion_model::armor_pose(x.data(), id, armor_num, name), type, projected)) {
      return false;
    }
    observation = motion_model::light_observation(projected);
    return observation.allFinite();
  }
  using Jet = ceres::Jet<double, motion_model::STATE_DIM>;
  std::array<Jet, motion_model::STATE_DIM> state;
  for (int i = 0; i < motion_model::STATE_DIM; ++i) {
    state[i] = Jet(x[i]);
    state[i].v[i] = 1.0;
  }
  std::array<Eigen::Matrix<Jet, 2, 1>, 4> projected;
  if (!camera().project(motion_model::armor_pose(state.data(), id, armor_num, name), type, projected)) {
    return false;
  }
  const auto uv = motion_model::light_observation(projected);
  observation.resize(motion_model::UV_DIM);
  jacobian->resize(motion_model::UV_DIM, motion_model::STATE_DIM);
  for (int i = 0; i < motion_model::UV_DIM; ++i) {
    observation[i] = uv[i].a;
    jacobian->row(i) = uv[i].v.transpose();
  }
  return observation.allFinite() && jacobian->allFinite();
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // solve
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

void Solver::optimize_pose(Armor & armor) const
{
  const bool fixed_pitch = armor.name == ArmorName::outpost || armor.name == ArmorName::base;
  const double pitch = fixed_pitch ? motion_model::armor_pitch(armor.name) : armor.ypr_in_world[1];
  double parameters[5] = {armor.xyz_in_world.x(), armor.xyz_in_world.y(), armor.xyz_in_world.z(),
                          armor.ypr_in_world[0], pitch};
  ceres::Problem problem;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<PixelResidual, 8, 5>(
      new PixelResidual{camera(), armor.type, armor.points}),
    nullptr, parameters);
  if (fixed_pitch) {
    problem.SetManifold(parameters, new ceres::SubsetManifold(5, {4}));
  } else {
    problem.SetParameterLowerBound(parameters, 4, motion_model::ARMOR_PITCH - CV_PI / 4.0);
    problem.SetParameterUpperBound(parameters, 4, motion_model::ARMOR_PITCH + CV_PI / 4.0);
    parameters[4] = std::clamp(parameters[4], motion_model::ARMOR_PITCH - CV_PI / 4.0,
                               motion_model::ARMOR_PITCH + CV_PI / 4.0);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 20;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable()) return;

  armor.xyz_in_world << parameters[0], parameters[1], parameters[2];
  armor.xyz_in_gimbal = R_gimbal2world_.transpose() * armor.xyz_in_world;
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  const Eigen::Matrix3d R_armor2world = motion_model::rotation(parameters[3], parameters[4]);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);
  armor.ypr_in_gimbal = tools::eulers(R_gimbal2world_.transpose() * R_armor2world, 2, 1, 0);
}

// 世界坐标到像素坐标的转换
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim
