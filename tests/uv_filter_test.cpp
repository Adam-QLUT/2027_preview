#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "tasks/auto_aim/motion_model.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/math_tools.hpp"

namespace
{
using namespace auto_aim;
using namespace auto_aim::motion_model;

void require(bool condition, const std::string & message)
{
  if (!condition) throw std::runtime_error(message);
}

Eigen::VectorXd state(double yaw = 0.3, double pitch = 0.12)
{
  Eigen::VectorXd x = Eigen::VectorXd::Zero(STATE_DIM);
  x[0] = 4.0;
  x[2] = 0.1;
  x[4] = 0.2;
  x[6] = yaw;
  x[8] = 0.23;
  x[9] = 0.04;
  x[10] = 0.06;
  x[CAR_PITCH] = pitch;
  return x;
}

Armor observation(Solver & solver, const Eigen::Isometry3d & pose, ArmorType type, ArmorName name)
{
  auto pixels = solver.reproject_armor(pose, type);
  require(pixels.size() == 4, "Armor must be in front of camera");
  // This constructor is used by the network detector; lightbar members need not be populated.
  Armor armor(0, 0.99f, cv::Rect(), pixels);
  armor.type = type;
  armor.name = name;
  armor.priority = ArmorPriority::first;
  armor.color = Color::blue;
  solver.solve(armor);
  return armor;
}

void geometry_test()
{
  const auto x = state();
  const Eigen::Matrix3d car = (Eigen::AngleAxisd(x[6], Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(x[CAR_PITCH], Eigen::Vector3d::UnitY()))
                                .toRotationMatrix();
  require(std::abs(tools::eulers(car, 2, 1, 0)[2]) < 1e-12, "Car roll must be zero");
  for (int id = 0; id < 4; ++id) {
    const double phase = id * CV_PI / 2.0;
    const double radius = (id & 1) ? x[8] + x[9] : x[8];
    const Eigen::Vector3d offset(-radius * std::cos(phase), -radius * std::sin(phase),
                                 (id & 1) ? x[10] : 0);
    const auto pose = armor_pose(x.data(), id, 4, ArmorName::three);
    const Eigen::Vector3d expected = Eigen::Vector3d(x[0], x[2], x[4]) + car * offset;
    require((pose.translation() - expected).norm() < 1e-12, "Pitch must tilt armor orbit");
    const Eigen::Matrix3d relative = car.transpose() * pose.linear();
    const Eigen::Vector3d ypr = tools::eulers(relative, 2, 1, 0);
    require(std::abs(ypr[1] - ARMOR_PITCH) < 1e-12, "Relative armor pitch must be 15 degrees");
    require(std::abs(ypr[2]) < 1e-12, "Relative armor roll must be zero");
  }
}

void projection_test(Solver & solver)
{
  for (ArmorName name : {ArmorName::three, ArmorName::outpost, ArmorName::base}) {
    const int count = name == ArmorName::three ? 4 : 3;
    for (ArmorType type : {ArmorType::small, ArmorType::big}) {
      for (int id = 0; id < count; ++id) {
        const auto x = state();
        Eigen::VectorXd predicted;
        Eigen::MatrixXd H;
        require(solver.armor_uv_prediction(x, id, count, type, name, predicted, &H),
                "UV prediction must be finite");
        for (int col = 0; col < STATE_DIM; ++col) {
          auto plus = x, minus = x;
          plus[col] += 1e-6;
          minus[col] -= 1e-6;
          Eigen::VectorXd zp, zm;
          solver.armor_uv_prediction(plus, id, count, type, name, zp);
          solver.armor_uv_prediction(minus, id, count, type, name, zm);
          Eigen::VectorXd delta = zp - zm;
          delta[0] = tools::limit_rad(delta[0]);
          delta[4] = tools::limit_rad(delta[4]);
          require((delta / 2e-6 - H.col(col)).cwiseAbs().maxCoeff() < 1e-4,
                  "Automatic UV Jacobian differs from finite differences");
        }
        if (name != ArmorName::three) {
          require(H.col(CAR_PITCH).isZero(), "Outpost/base pitch must remain fixed");
        }
      }
    }
  }
  // Compare the new projection against the existing OpenCV projection with distortion.
  for (ArmorType type : {ArmorType::small, ArmorType::big}) {
    const Eigen::Vector3d xyz(3.0, 0.3, 0.4);
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation() = xyz;
    pose.linear() = rotation(0.4, ARMOR_PITCH);
    const auto expected = solver.reproject_armor(xyz, 0.4, type, ArmorName::three);
    const auto actual = solver.reproject_armor(pose, type);
    for (int i = 0; i < 4; ++i) {
      require(cv::norm(expected[i] - actual[i]) < 1e-3, "Projection differs from OpenCV");
    }
  }
  auto behind = state();
  behind[0] = -1;
  Eigen::VectorXd unused;
  require(!solver.armor_uv_prediction(behind, 0, 4, ArmorType::small, ArmorName::three, unused),
          "Projection must reject points behind camera");
}

void least_squares_test(Solver & solver)
{
  for (ArmorType type : {ArmorType::small, ArmorType::big}) {
    for (double distance : {3.0, 8.0}) {
      for (double yaw : {-0.7, 0.0, 0.7}) {
        for (double pitch : {-0.15, 0.15}) {
          auto pose = Eigen::Isometry3d::Identity();
          pose.translation() << distance, 0.2, 0.4;
          pose.linear() = rotation(yaw, ARMOR_PITCH + pitch);
          const auto armor = observation(solver, pose, type, ArmorName::three);
          require((armor.xyz_in_world - pose.translation()).norm() < 0.002,
                  "Least-squares position differs from exact pose");
          require(
            (rotation(armor.ypr_in_world[0], armor.ypr_in_world[1]) - pose.linear()).norm() < 0.002,
            "Least-squares orientation differs from exact pose");
          require(std::abs(armor.ypr_in_world[2]) < 1e-10, "Initial car roll must be zero");
        }
      }
    }
  }
  auto pose = Eigen::Isometry3d::Identity();
  pose.translation() << 4, 0.2, 0.3;
  pose.linear() = rotation(0.5, ARMOR_PITCH + 0.1);
  auto armor = observation(solver, pose, ArmorType::small, ArmorName::three);
  armor.points[0].x += 0.8f;
  armor.points[1].y -= 0.5f;
  armor.points[2].x -= 0.4f;
  solver.solve(armor);
  auto refined_pose = Eigen::Isometry3d::Identity();
  refined_pose.translation() = armor.xyz_in_world;
  refined_pose.linear() = rotation(armor.ypr_in_world[0], armor.ypr_in_world[1]);
  const auto refined = solver.reproject_armor(refined_pose, armor.type);
  const auto baseline = solver.reproject_armor(pose, armor.type);
  double refined_cost = 0, baseline_cost = 0;
  for (int i = 0; i < 4; ++i) {
    refined_cost += tools::square(cv::norm(refined[i] - armor.points[i]));
    baseline_cost += tools::square(cv::norm(baseline[i] - armor.points[i]));
  }
  require(refined_cost < baseline_cost, "Pixel least squares must reduce reprojection SSE");
  std::cout << "Noisy pixel SSE: " << baseline_cost << " -> " << refined_cost << '\n';
}

void filter_test(Solver & solver)
{
  auto truth = state(0.0, 0.12);
  auto armor = observation(solver, armor_pose(truth.data(), 0, 4, ArmorName::three),
                           ArmorType::small, ArmorName::three);
  armor.xyz_in_world += Eigen::Vector3d(0.03, -0.02, 0.02);
  Eigen::VectorXd covariance = Eigen::VectorXd::Constant(STATE_DIM, 0.1);
  const auto start = std::chrono::steady_clock::now();
  Target target(armor, start, 0.2, 4, covariance, 5, 350);
  const auto before = (target.armor_pose(0).translation() -
                       armor_pose(truth.data(), 0, 4, ArmorName::three).translation())
                        .norm();
  std::mt19937 generator(26);
  std::normal_distribution<double> noise(0, 0.05);
  for (int frame = 0; frame < 400; ++frame) {
    truth[6] = tools::limit_rad(frame * 0.01);
    target.predict(0.01);
    const int id = static_cast<int>(std::round(-truth[6] / (CV_PI / 2.0)) + 4) % 4;
    auto measured = observation(solver, armor_pose(truth.data(), id, 4, ArmorName::three),
                                ArmorType::small, ArmorName::three);
    for (auto & point : measured.points) {
      point.x += noise(generator);
      point.y += noise(generator);
    }
    // UV filtering must use the detected pixels, independent of the PnP position/yaw.
    measured.xyz_in_world << 99, 99, 99;
    measured.ypr_in_world << 2, 2, 2;
    measured.ypd_in_world << 2, 2, 99;
    target.update(measured, solver);
    require(target.last_id == id, "UV association selected wrong armor");
    require(!target.diverged(), "Rotating tilted vehicle filter diverged");
    require(std::isfinite(target.ekf().data.at("nees")), "NEES must be finite");
    if (frame == 30) {
      const double after = (target.armor_pose(id).translation() -
                            armor_pose(truth.data(), id, 4, ArmorName::three).translation())
                             .norm();
      require(after < before, "UV updates must improve armor position");
    }
  }
  require(std::abs(target.ekf_x()[CAR_PITCH] - truth[CAR_PITCH]) < 0.015,
          "Vehicle pitch must remain observable during rotation");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenvalues(target.ekf().P);
  require(eigenvalues.eigenvalues().minCoeff() > -1e-10, "Covariance must remain PSD");
  std::cout << "Rotating vehicle pitch error: " << target.ekf_x()[CAR_PITCH] - truth[CAR_PITCH]
            << " rad; NIS=" << target.ekf().data.at("nis") << '\n';
}

void outpost_test(Solver & solver)
{
  auto truth = state(0.2, 0.4);
  truth[8] = 0.2765;
  truth[9] = 0.1;
  truth[10] = -0.1;
  const auto start = std::chrono::steady_clock::now();
  auto first = observation(solver, armor_pose(truth.data(), 0, 3, ArmorName::outpost),
                           ArmorType::small, ArmorName::outpost);
  Eigen::VectorXd covariance = Eigen::VectorXd::Constant(STATE_DIM, 0.1);
  Target target(first, start, truth[8], 3, covariance, 5, 350);
  for (int frame = 0; frame < 400; ++frame) {
    truth[6] = tools::limit_rad(0.2 + frame * 0.01 * 0.8 * CV_PI);
    target.predict(0.01);
    const int id = static_cast<int>(std::round(-truth[6] / (2 * CV_PI / 3)) + 3) % 3;
    auto measured = observation(solver, armor_pose(truth.data(), id, 3, ArmorName::outpost),
                                ArmorType::small, ArmorName::outpost);
    target.update(measured, solver);
    require(target.last_id == id, "Outpost selected wrong armor/height");
    require(target.ekf_x()[8] == 0.2765, "Outpost radius must stay fixed");
    require(target.ekf_x()[CAR_PITCH] == 0, "Outpost car pitch must stay fixed");
    require(std::isfinite(target.ekf().data.at("nees")), "Fixed-state NEES must remain finite");
    require(!target.diverged(), "Outpost filter diverged");
  }
  require(std::abs(target.ekf_x()[9] - truth[9]) < 0.01 &&
            std::abs(target.ekf_x()[10] - truth[10]) < 0.01,
          "Outpost heights must converge");
}

void tracker_test(const std::string & config)
{
  Solver solver(config);
  Tracker tracker(config, solver);
  auto truth = state(0.2, 0.1);
  auto t = std::chrono::steady_clock::now();
  for (int frame = 0; frame < 20; ++frame) {
    std::list<Armor> armors{observation(solver, armor_pose(truth.data(), 0, 4, ArmorName::three),
                                        ArmorType::small, ArmorName::three)};
    t += std::chrono::milliseconds(10);
    const auto targets = tracker.track(armors, t);
    require(!targets.empty(), "Tracker must retain a target with valid UV observations");
    require(targets.front().ekf_x().size() == STATE_DIM, "Tracker must construct 12D state");
  }
  require(tracker.state() == "tracking", "Tracker must reach tracking state");
  Target simulated(3, 5, 0.2, 0.1);
  simulated.predict(0.01);
  require(simulated.ekf_x().allFinite(),
          "Offline planner target must predict without uninitialized data");
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    require(argc == 2, "Pass the UV test camera configuration");
    geometry_test();
    Solver solver(argv[1]);
    projection_test(solver);
    least_squares_test(solver);
    filter_test(solver);
    outpost_test(solver);
    const Eigen::Quaterniond q(Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitX()));
    solver.set_R_gimbal2world(q);
    projection_test(solver);
    least_squares_test(solver);
    tracker_test(argv[1]);
    std::cout
      << "UV model, pixel least squares, rotating vehicle, outpost and tracker checks passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
