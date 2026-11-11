#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: idle, 1: auto aim, 2: small buff, 3: big buff
  float q[4];    // Small-yaw/main-camera attitude, wxyz.
  float p[4];    // Big-yaw/chassis attitude, wxyz.
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) == 59);
static_assert(offsetof(GimbalToVision, q) == 3);
static_assert(offsetof(GimbalToVision, p) == 19);
static_assert(offsetof(GimbalToVision, crc16) == 57);

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: no control, 1: aim, 2: aim and fire
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  float center_yaw; // World-frame handoff target; the controller closes the small-yaw loop.
  uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) == 33);
static_assert(offsetof(VisionToGimbal, center_yaw) == 27);
static_assert(offsetof(VisionToGimbal, yaw) == 3);
static_assert(offsetof(VisionToGimbal, crc16) == 31);

enum class GimbalMode
{
  IDLE,
  AUTO_AIM,
  SMALL_BUFF,
  BIG_BUFF
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);
  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point timestamp);
  // Returns a zero quaternion when no sufficiently recent sample supports timestamp.
  Eigen::Quaterniond p(std::chrono::steady_clock::time_point timestamp);
  bool p_ready() const;
  size_t queue_size() const;

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  void send(io::VisionToGimbal vision_to_gimbal);

private:
  using AttitudeSample =
    std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>;

  serial::Serial serial_;
  std::mutex serial_mutex_;
  std::thread thread_;
  std::atomic<bool> quit_ = false;
  std::atomic<size_t> rx_bytes_{0};
  std::atomic<size_t> rx_crc_errors_{0};
  std::atomic<size_t> rx_invalid_q_{0};
  std::atomic<bool> chassis_imu_received_{false};
  std::atomic<int64_t> last_chassis_imu_ns_{0};
  std::chrono::milliseconds imu_freshness_timeout_{50};
  mutable std::mutex mutex_;

  GimbalToVision rx_data_{};
  VisionToGimbal tx_data_{};

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_{};
  // Keep recent IMU history even when auxiliary detection pauses consumption.
  tools::ThreadSafeQueue<AttitudeSample, true> gimbal_queue_{1000};
  tools::ThreadSafeQueue<AttitudeSample, true> chassis_queue_{1000};
  AttitudeSample gimbal_ahead_{
    Eigen::Quaterniond::Identity(), std::chrono::steady_clock::time_point{}};
  AttitudeSample gimbal_behind_{
    Eigen::Quaterniond::Identity(), std::chrono::steady_clock::time_point{}};
  AttitudeSample chassis_ahead_{
    Eigen::Quaterniond::Identity(), std::chrono::steady_clock::time_point{}};
  AttitudeSample chassis_behind_{
    Eigen::Quaterniond::Identity(), std::chrono::steady_clock::time_point{}};

  Eigen::Quaterniond attitude_at(
    tools::ThreadSafeQueue<AttitudeSample, true> & queue, AttitudeSample & ahead,
    AttitudeSample & behind,
    std::chrono::steady_clock::time_point timestamp, bool require_fresh = false);
  size_t read(uint8_t * buffer, size_t capacity);
  void read_thread(size_t wire_frame_size);
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP