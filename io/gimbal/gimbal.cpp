#include "gimbal.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include "tools/exiter.hpp"
#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
// Wire length includes the CRC. It need not equal sizeof(GimbalToVision):
// the configured 64-byte format has five extension bytes before its CRC.
// Keep the stream decoder here so this translation unit is self-contained.
class SpFrameDecoder
{
public:
  explicit SpFrameDecoder(size_t frame_size) : frame_size_(frame_size)
  {
    if (frame_size != 59 && frame_size != 64)
      throw std::invalid_argument("SP RX frame size must be 59 or 64");
  }

  void reset() { used_ = 0; }
  size_t crc_errors() const { return crc_errors_; }

  template <typename OnFrame>
  void feed(const uint8_t * data, size_t count, OnFrame on_frame)
  {
    for (size_t i = 0; i < count; ++i) {
      bytes_[used_++] = data[i];
      // Search byte by byte, preserving a trailing 'S' across short reads.
      // Never discard a whole candidate on CRC failure: it may contain the
      // start of the next valid frame after a dropped or inserted byte.
      while (used_ >= 2 && (bytes_[0] != 'S' || bytes_[1] != 'P')) discard_one();
      if (used_ != frame_size_) continue;

      if (tools::check_crc16(bytes_.data(), static_cast<uint32_t>(frame_size_))) {
        on_frame(bytes_.data(), frame_size_);
        used_ = 0;
      } else {
        ++crc_errors_;
        discard_one();
      }
    }
  }

private:
  void discard_one()
  {
    --used_;
    std::memmove(bytes_.data(), bytes_.data() + 1, used_);
  }

  const size_t frame_size_;
  std::array<uint8_t, 64> bytes_{};
  size_t used_ = 0;
  size_t crc_errors_ = 0;
};

bool valid_quaternion(const Eigen::Quaterniond & q)
{
  const auto norm2 = q.squaredNorm();
  return norm2 > 0.5 && norm2 < 1.5 && !std::isnan(norm2);
}

int64_t steady_ns(std::chrono::steady_clock::time_point timestamp)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
}
}  // namespace

Gimbal::Gimbal(const std::string & config_path)
{
  const auto yaml = tools::load(config_path);
  const auto com_port = tools::read<std::string>(yaml, "com_port");
  imu_freshness_timeout_ = std::chrono::milliseconds(
    yaml["imu_freshness_timeout_ms"] ? yaml["imu_freshness_timeout_ms"].as<int>() : 50);

  // The established 57-byte payload prefix is unchanged. Extended firmware
  // appends five opaque bytes before its CRC; never interpret them as fields.
  const int wire_frame_size = yaml["gimbal_rx_frame_bytes"]
                                ? yaml["gimbal_rx_frame_bytes"].as<int>() : 59;
  if (wire_frame_size != 59 && wire_frame_size != 64) {
    throw std::invalid_argument("gimbal_rx_frame_bytes must be 59 or 64");
  }
  const int startup_timeout_ms = yaml["gimbal_startup_timeout_ms"]
                                   ? yaml["gimbal_startup_timeout_ms"].as<int>() : 10000;
  if (startup_timeout_ms <= 0)
    throw std::invalid_argument("gimbal_startup_timeout_ms must be positive");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);
    auto timeout = serial::Timeout::simpleTimeout(100);
    serial_.setTimeout(timeout);
    serial_.open();
    tools::logger()->info("[Gimbal] Serial opened on {}", com_port);
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    throw;
  }

  tools::logger()->info("[Gimbal] RX wire frame={} bytes, known payload={} bytes, TX={} bytes",
                        wire_frame_size, offsetof(GimbalToVision, crc16), sizeof(VisionToGimbal));
  thread_ = std::thread(&Gimbal::read_thread, this, static_cast<size_t>(wire_frame_size));

  // A constructor that throws must join its worker explicitly: its destructor
  // will not run, and destroying a joinable std::thread would terminate.
  Eigen::Quaterniond q0;
  std::chrono::steady_clock::time_point t0;
  try {
    const auto started = std::chrono::steady_clock::now();
    auto next_report = started + std::chrono::seconds(1);
    AttitudeSample first;
    while (true) {
      if (tools::exit_requested())
        throw std::runtime_error("Gimbal startup interrupted");
      if (gimbal_queue_.try_pop(first)) {
        std::tie(q0, t0) = first;
        break;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_report) {
        tools::logger()->warn(
          "[Gimbal] Waiting for q: RX_bytes={} CRC_errors={} invalid_q={} wire_bytes={}",
          rx_bytes_.load(), rx_crc_errors_.load(), rx_invalid_q_.load(), wire_frame_size);
        next_report = now + std::chrono::seconds(1);
      }
      if (now - started >= std::chrono::milliseconds(startup_timeout_ms))
        throw std::runtime_error("Gimbal startup timed out waiting for valid q; inspect RX diagnostics");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    gimbal_ahead_ = {q0, t0};
    gimbal_behind_ = gimbal_ahead_;
  } catch (...) {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
    try { serial_.close(); } catch (...) {}
    throw;
  }

  const Eigen::Vector3d euler0 = tools::eulers(q0, 2, 1, 0);
  tools::logger()->info(
    "[Gimbal] First valid q received: yaw={:.3f}deg pitch={:.3f}deg roll={:.3f}deg",
    euler0[0] * 57.3, euler0[1] * 57.3, euler0[2] * 57.3);
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

size_t Gimbal::queue_size() const { return gimbal_queue_.size(); }

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point timestamp)
{
  return attitude_at(gimbal_queue_, gimbal_ahead_, gimbal_behind_, timestamp);
}

Eigen::Quaterniond Gimbal::p(std::chrono::steady_clock::time_point timestamp)
{
  return attitude_at(chassis_queue_, chassis_ahead_, chassis_behind_, timestamp, true);
}

bool Gimbal::p_ready() const
{
  if (!chassis_imu_received_.load()) return false;

  const auto last_ns = last_chassis_imu_ns_.load();
  const auto now_ns = steady_ns(std::chrono::steady_clock::now());
  const auto timeout_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(imu_freshness_timeout_).count();
  return last_ns > 0 && now_ns - last_ns <= timeout_ns;
}

Eigen::Quaterniond Gimbal::attitude_at(
  tools::ThreadSafeQueue<AttitudeSample, true> & queue, AttitudeSample & ahead,
  AttitudeSample & behind, std::chrono::steady_clock::time_point timestamp, bool require_fresh)
{
  const auto invalid = Eigen::Quaterniond(0, 0, 0, 0);
  if (std::get<1>(behind) == std::chrono::steady_clock::time_point{}) {
    if (!queue.try_pop(behind)) return invalid;
    ahead = behind;
  }

  // Preserve the cached bracket across nearby queries. Do not pop another
  // future sample when ahead <= timestamp <= behind already holds.
  while (std::get<1>(behind) < timestamp) {
    AttitudeSample next;
    if (!queue.try_pop(next)) break;
    ahead = behind;
    behind = std::move(next);
  }

  const auto q_a = std::get<0>(ahead).normalized();
  const auto q_b = std::get<0>(behind).normalized();
  const auto t_a = std::get<1>(ahead);
  const auto t_b = std::get<1>(behind);
  const auto recent = [&](std::chrono::steady_clock::time_point sample_time) {
    const auto gap = sample_time > timestamp ? sample_time - timestamp : timestamp - sample_time;
    return gap <= imu_freshness_timeout_;
  };
  // RX freshness alone does not establish that the pose used for this image
  // is fresh. Reject stale endpoints and gaps instead of clamping an old pose.
  if (timestamp <= t_a)
    return require_fresh && !recent(t_a) ? invalid : q_a;
  if (timestamp >= t_b)
    return require_fresh && !recent(t_b) ? invalid : q_b;
  if (require_fresh && (!recent(t_a) || !recent(t_b))) return invalid;

  const auto interval = std::chrono::duration<double>(t_b - t_a).count();
  const auto elapsed = std::chrono::duration<double>(timestamp - t_a).count();
  return q_a.slerp(elapsed / interval, q_b).normalized();
}

void Gimbal::send(io::VisionToGimbal vision_to_gimbal)
{
  tx_data_ = vision_to_gimbal;
  tx_data_.head[0] = 'S';
  tx_data_.head[1] = 'P';
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    if (serial_.isOpen()) {
      serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.center_yaw = yaw;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    if (serial_.isOpen()) {
      serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

size_t Gimbal::read(uint8_t * buffer, size_t capacity)
{
  try {
    // Read only currently available bytes; wait for one byte when idle.
    // Return the actual count so a short read never discards partial frames.
    const auto count = std::min(capacity, std::max(size_t{1}, serial_.available()));
    return serial_.read(buffer, count);
  } catch (const std::exception &) {
    return 0;
  }
}

void Gimbal::read_thread(size_t wire_frame_size)
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  SpFrameDecoder decoder(wire_frame_size);
  std::array<uint8_t, 256> bytes{};

  while (!quit_) {
    if (error_count > 100) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      decoder.reset();
      continue;
    }

    const auto count = read(bytes.data(), bytes.size());
    if (count == 0) {
      ++error_count;
      continue;
    }
    rx_bytes_ += count;
    error_count = 0;  // Only transport failures trigger a port reconnect.
    const auto timestamp = std::chrono::steady_clock::now();
    decoder.feed(bytes.data(), count, [&](const uint8_t * wire, size_t frame_size) {
      constexpr auto payload_size = offsetof(GimbalToVision, crc16);
      std::memcpy(&rx_data_, wire, payload_size);
      rx_data_.crc16 = static_cast<uint16_t>(wire[frame_size - 2]) |
                        (static_cast<uint16_t>(wire[frame_size - 1]) << 8);

      const Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
      const Eigen::Quaterniond p(rx_data_.p[0], rx_data_.p[1], rx_data_.p[2], rx_data_.p[3]);
      if (!valid_quaternion(q)) {
        ++rx_invalid_q_;
        return;
      }
      gimbal_queue_.push({q, timestamp});
      if (valid_quaternion(p)) {
        chassis_queue_.push({p, timestamp});
        last_chassis_imu_ns_ = steady_ns(timestamp);
        chassis_imu_received_ = true;
      }

      std::lock_guard<std::mutex> lock(mutex_);
      state_.yaw = rx_data_.yaw;
      state_.yaw_vel = rx_data_.yaw_vel;
      state_.pitch = rx_data_.pitch;
      state_.pitch_vel = rx_data_.pitch_vel;
      state_.bullet_speed = rx_data_.bullet_speed;
      state_.bullet_count = rx_data_.bullet_count;

      switch (rx_data_.mode) {
        case 0:
          mode_ = GimbalMode::IDLE;
          break;
        case 1:
          mode_ = GimbalMode::AUTO_AIM;
          break;
        case 2:
          mode_ = GimbalMode::SMALL_BUFF;
          break;
        case 3:
          mode_ = GimbalMode::BIG_BUFF;
          break;
        default:
          mode_ = GimbalMode::IDLE;
          break;
      }
    });
    rx_crc_errors_ = decoder.crc_errors();
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  chassis_imu_received_ = false;
  last_chassis_imu_ns_ = 0;
  constexpr int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    {
      std::lock_guard<std::mutex> lock(serial_mutex_);
      try {
        if (serial_.isOpen()) serial_.close();
      } catch (...) {
      }
    }

    for (int wait = 0; wait < 100 && !quit_; ++wait)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (quit_) break;
    try {
      std::lock_guard<std::mutex> lock(serial_mutex_);
      serial_.open();
      if (serial_.isOpen()) {
        gimbal_queue_.clear();
        chassis_queue_.clear();
        tools::logger()->info("[Gimbal] Reconnected serial successfully.");
        break;
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
    }
  }
}

}  // namespace io