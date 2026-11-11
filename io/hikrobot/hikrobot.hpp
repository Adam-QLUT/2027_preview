#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "io/camera.hpp"

namespace io
{
class HikRobot : public CameraBase
{
public:
  enum class State { Starting, Capturing, Recovering, Stopped, Quarantined };
  struct Status
  {
    State state;
    int last_worker_pid;
    int last_reset_pid;
    unsigned int last_timeout_operation;
  };
  HikRobot(
    double exposure_ms, double gain, const std::string & vid_pid,
    const std::string & serial_number = "", float frame_rate = 150.0f);
  ~HikRobot() override;

  // Returns an empty image after 100 ms without data; callers must check empty().
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  bool try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  void update_params(double exposure_ms, double gain);
  void request_stop() noexcept;
  // Request all cameras before joining consumers: they share one shutdown interval.
  static void request_stop_all();
  Status status() const noexcept;

private:
  struct Process;
  std::unique_ptr<Process> process_;
  std::thread daemon_thread_;
  std::atomic<bool> daemon_quit_{false};
  std::atomic<bool> retired_{false};

  void supervise() noexcept;
};
}  // namespace io

#endif  // IO__HIKROBOT_HPP
