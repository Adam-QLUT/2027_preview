#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "armor.hpp"
#include "motion_model.hpp"

namespace auto_aim
{
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // 使用 IPPE 的全部候选解，返回是否找到朝向相机的合法解。
  bool solve(Armor & armor) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Isometry3d & pose, ArmorType type) const;

  Eigen::VectorXd armor_uv_observation(const Armor & armor) const;
  Eigen::MatrixXd armor_uv_noise(const Armor & armor) const;
  bool armor_uv_prediction(
    const Eigen::VectorXd & x, int id, int armor_num, ArmorType type, ArmorName name,
    Eigen::VectorXd & observation, Eigen::MatrixXd * jacobian = nullptr) const;

  double oupost_reprojection_error(Armor armor, const double & picth);

  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  double uv_sigma_center_ratio_ = 0.2;
  double uv_sigma_length_ratio_ = 0.3;
  double uv_sigma_angle_ = 0.1;

  motion_model::Camera camera() const;
  void optimize_pose(Armor & armor) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
