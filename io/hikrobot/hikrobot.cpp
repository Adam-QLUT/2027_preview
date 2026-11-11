#include "hikrobot.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <new>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "worker_protocol.hpp"

extern char ** environ;
using namespace std::chrono_literals;

namespace io
{
namespace
{
using Clock = std::chrono::steady_clock;
using namespace hikrobot_worker;
std::mutex cameras_mutex;
std::vector<HikRobot *> cameras;

// Supervisor deadlines cannot cancel an uninterruptible Linux ioctl.
constexpr auto call_timeout = 500ms;
constexpr auto shutdown_timeout = 1500ms;
constexpr auto reset_timeout = 500ms;
constexpr auto reap_timeout = 100ms;

bool reap(pid_t & pid, int & status)
{
  if (pid <= 0) return true;
  const auto result = ::waitpid(pid, &status, WNOHANG);
  if (result == pid) {
    pid = -1;
    return true;
  }
  if (result < 0 && errno == ECHILD) {
    // Lost ownership, e.g. a process-wide SIGCHLD reaper: never signal this PID.
    pid = -1;
    status = 0xff00;
    return true;
  }
  return false;
}

bool wait_until(pid_t & pid, int & status, Clock::time_point deadline)
{
  do {
    if (reap(pid, status)) return true;
    std::this_thread::sleep_for(5ms);
  } while (Clock::now() < deadline);
  return reap(pid, status);
}

void terminate(pid_t pid)
{
  if (pid > 0) ::kill(pid, SIGKILL);
}

std::filesystem::path worker_path()
{
  if (const char * path = std::getenv("SP_VISION_HIKROBOT_WORKER")) {
    return std::filesystem::absolute(path);
  }
  return std::filesystem::read_symlink("/proc/self/exe").parent_path() / "hikrobot_worker";
}

int above_reserved_fds(int fd)
{
  if (fd < 0 || fd > 4) return fd;
  const int duplicate = ::fcntl(fd, F_DUPFD_CLOEXEC, 5);
  const int duplicate_error = errno;
  ::close(fd);
  if (duplicate < 0) errno = duplicate_error;
  return duplicate;
}

std::string serial_key(const std::string & serial)
{
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char value : serial) {
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
}
}  // namespace

struct HikRobot::Process
{
  SharedMemory * shared = nullptr;
  int shared_fd = -1;
  int lease_fd = -1;
  pid_t worker = -1;
  pid_t reset_worker = -1;
  int vid = -1;
  int pid = -1;
  std::string serial;
  std::string executable;
  std::string reset_executable;
  std::string launcher;
  std::string failure_reason;
  std::atomic<State> state{State::Starting};
  std::atomic<int> last_worker_pid{-1};
  std::atomic<int> last_reset_pid{-1};
  std::atomic<unsigned int> last_timeout_operation{0};
  std::atomic<std::uint64_t> generation{0};

  ~Process()
  {
    // A stuck child may keep its mapping and lease. During kernel exit the lease
    // can close before a USB fd blocks; it is not a durable quarantine marker.
    // Never wait here or destroy shared atomics still used by an unreaped child.
    if (shared != nullptr) ::munmap(shared, sizeof(SharedMemory));
    if (shared_fd >= 0) ::close(shared_fd);
    if (lease_fd >= 0) ::close(lease_fd);
  }

  bool spawn(bool reset)
  {
    std::vector<std::string> arguments = reset
      ? std::vector<std::string>{launcher, reset_executable, "--reset", std::to_string(vid),
                                std::to_string(pid), serial, failure_reason}
      : std::vector<std::string>{launcher, executable, "--capture", "3", serial};
    std::vector<char *> argv;
    for (auto & argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int ret = ::posix_spawn_file_actions_init(&actions);
    if (ret != 0) return false;
    if (!reset) ret = ::posix_spawn_file_actions_adddup2(&actions, shared_fd, 3);
    if (ret == 0) ret = ::posix_spawn_file_actions_adddup2(&actions, lease_fd, 4);
    // The small launcher closes inherited FDs before exec'ing MVS. Doing this
    // inside the MVS-linked worker could close FDs created by vendor constructors.
    posix_spawnattr_t attributes;
    if (ret != 0 || ::posix_spawnattr_init(&attributes) != 0) {
      ::posix_spawn_file_actions_destroy(&actions);
      return false;
    }
    sigset_t empty, defaults;
    ::sigemptyset(&empty);
    ::sigemptyset(&defaults);
    ::sigaddset(&defaults, SIGINT);
    ::sigaddset(&defaults, SIGTERM);
    ret = ::posix_spawnattr_setsigmask(&attributes, &empty);
    if (ret == 0) ret = ::posix_spawnattr_setsigdefault(&attributes, &defaults);
    if (ret == 0) ret = ::posix_spawnattr_setpgroup(&attributes, 0);
    if (ret == 0) ret = ::posix_spawnattr_setflags(
      &attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP);
    auto & child = reset ? reset_worker : worker;
    if (ret == 0) ret = ::posix_spawn(&child, arguments.front().c_str(), &actions, &attributes,
                                    argv.data(), environ);
    if (ret == 0) {
      (reset ? last_reset_pid : last_worker_pid).store(child, std::memory_order_release);
    } else {
      child = -1;
    }
    ::posix_spawnattr_destroy(&attributes);
    ::posix_spawn_file_actions_destroy(&actions);
    return ret == 0;
  }

  bool reset_usb()
  {
    if (!spawn(true)) return false;
    int status = 0;
    if (!wait_until(reset_worker, status, Clock::now() + reset_timeout)) {
      terminate(reset_worker);
      wait_until(reset_worker, status, Clock::now() + reap_timeout);
      return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }

  bool retire_and_reset()
  {
    int status = 0;
    terminate(worker);
    wait_until(worker, status, Clock::now() + reap_timeout);
    // Reset runs in another exec process, including enumeration/SN reads/close.
    // No further call can use the old SDK handle after retirement begins.
    const bool reset_ok = reset_usb();
    const bool reaped = wait_until(worker, status, Clock::now() + reap_timeout);
    return reset_ok && reaped && reset_worker <= 0;
  }
};

HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid,
  const std::string & serial_number, float frame_rate)
: process_(std::make_unique<Process>())
{
  if (serial_number.empty() || serial_number.size() > 100 ||
      !std::isfinite(exposure_ms) || exposure_ms <= 0 || !std::isfinite(gain) || gain < 0 ||
      !std::isfinite(frame_rate) || frame_rate <= 0) {
    throw std::invalid_argument("HikRobot requires a nonempty SN and valid exposure/gain/FPS");
  }
  auto & process = *process_;
  const auto separator = vid_pid.find(':');
  std::size_t used_vid = 0, used_pid = 0;
  if (separator == std::string::npos) throw std::invalid_argument("Invalid HikRobot VID:PID");
  process.vid = std::stoi(vid_pid.substr(0, separator), &used_vid, 16);
  process.pid = std::stoi(vid_pid.substr(separator + 1), &used_pid, 16);
  if (used_vid != separator || used_pid != vid_pid.size() - separator - 1 ||
      process.vid <= 0 || process.vid > 0xffff || process.pid < 0 || process.pid > 0xffff) {
    throw std::invalid_argument("Invalid HikRobot VID:PID");
  }
  process.serial = serial_number;
  process.executable = worker_path().string();
  process.reset_executable =
    (std::filesystem::path(process.executable).parent_path() / "hikrobot_usb_reset").string();
  process.launcher =
    (std::filesystem::path(process.executable).parent_path() / "hikrobot_launcher").string();
  if (const char * path = std::getenv("SP_VISION_HIKROBOT_RESET")) {
    process.reset_executable = std::filesystem::absolute(path).string();
  }
  if (const char * path = std::getenv("SP_VISION_HIKROBOT_LAUNCHER")) {
    process.launcher = std::filesystem::absolute(path).string();
  }
  if (::access(process.executable.c_str(), X_OK) != 0) {
    throw std::runtime_error("Missing executable: " + process.executable);
  }
  if (::access(process.reset_executable.c_str(), X_OK) != 0) {
    throw std::runtime_error("Missing executable: " + process.reset_executable);
  }
  if (::access(process.launcher.c_str(), X_OK) != 0) {
    throw std::runtime_error("Missing executable: " + process.launcher);
  }
  const auto lease_path = "/tmp/sp-vision-hikrobot-" + std::to_string(::getuid()) + "-" +
                          serial_key(serial_number) + ".lock";
  process.lease_fd = above_reserved_fds(
    ::open(lease_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (process.lease_fd < 0) {
    const int error = errno;
    throw std::system_error(
      error, std::generic_category(), "HikRobot lease open failed: serial=" + serial_number +
        " path=" + lease_path + " errno=" + std::to_string(error));
  }
  if (::flock(process.lease_fd, LOCK_EX | LOCK_NB) != 0) {
    const int error = errno;
    const auto reason = error == EWOULDBLOCK || error == EAGAIN
                          ? "HikRobot lease busy: serial="
                          : "HikRobot lease flock failed: serial=";
    throw std::system_error(
      error, std::generic_category(), std::string(reason) + serial_number +
        " path=" + lease_path + " errno=" + std::to_string(error));
  }
  process.shared_fd = above_reserved_fds(::memfd_create("hikrobot-frame", MFD_CLOEXEC));
  if (process.shared_fd < 0 || ::ftruncate(process.shared_fd, sizeof(SharedMemory)) != 0) {
    throw std::runtime_error("HikRobot cannot allocate shared memory");
  }
  void * mapping = ::mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE,
                          MAP_SHARED, process.shared_fd, 0);
  if (mapping == MAP_FAILED) throw std::runtime_error("HikRobot mmap failed");
  process.shared = new (mapping) SharedMemory;
  process.shared->exposure_us.store(exposure_ms * 1e3);
  process.shared->gain.store(gain);
  process.shared->frame_rate = frame_rate;
  tools::logger()->info(
    "[HikRobot] state=starting serial='{}' owner_pid={} isolated worker",
    serial_number, ::getpid());
  std::lock_guard<std::mutex> lock(cameras_mutex);
  cameras.push_back(this);
  try {
    daemon_thread_ = std::thread([this] { supervise(); });
  } catch (...) {
    cameras.pop_back();
    throw;
  }
}

void HikRobot::request_stop() noexcept
{
  daemon_quit_.store(true, std::memory_order_release);
  process_->shared->shutdown.store(true, std::memory_order_release);
}

void HikRobot::request_stop_all()
{
  std::lock_guard<std::mutex> lock(cameras_mutex);
  for (auto * camera : cameras) camera->request_stop();
}

HikRobot::Status HikRobot::status() const noexcept
{
  return {
    process_->state.load(std::memory_order_acquire),
    process_->last_worker_pid.load(std::memory_order_acquire),
    process_->last_reset_pid.load(std::memory_order_acquire),
    process_->last_timeout_operation.load(std::memory_order_acquire)};
}

HikRobot::~HikRobot()
{
  {
    std::lock_guard<std::mutex> lock(cameras_mutex);
    cameras.erase(std::remove(cameras.begin(), cameras.end(), this), cameras.end());
  }
  request_stop();
  // This thread calls no SDK/libusb and only uses waitpid(WNOHANG).
  // Do not add logger/sink waits to the shutdown path.
  if (daemon_thread_.joinable()) daemon_thread_.join();
}

void HikRobot::supervise() noexcept
{
  auto & process = *process_;
  auto & shared = *process.shared;
  const auto stopping = [this] {
    // SIGINT/SIGTERM must reach the camera even if the application's main
    // thread is still inside inference or another synchronous operation.
    // The signal handler only stores an atomic flag; the worker owns SDK cleanup.
    if (tools::exit_requested()) request_stop();
    return daemon_quit_.load(std::memory_order_acquire);
  };
  try {
    unsigned int consecutive_failures = 0;
    while (!stopping()) {
      process.generation.fetch_add(1, std::memory_order_acq_rel);
      process.state.store(State::Starting, std::memory_order_release);
      if (stopping()) break;
      if (!process.spawn(false)) {
        process.state.store(State::Quarantined, std::memory_order_release);
        break;
      }
      auto last_progress = Clock::now();
      auto stop_started = Clock::time_point::max();
      std::uint64_t observed = shared.operation.load(std::memory_order_acquire);
      auto observed_frames = shared.frames_published.load(std::memory_order_acquire);
      int status = 0;
      bool fault = false;
      while (!reap(process.worker, status)) {
        const auto now = Clock::now();
        if (stopping() && stop_started == Clock::time_point::max()) {
          shared.shutdown.store(true, std::memory_order_release);
          stop_started = now;
        }
        const auto token = shared.operation.load(std::memory_order_acquire);
        if (token != observed) {
          observed = token;
          last_progress = now;
        }
        const auto operation = static_cast<Operation>(token & 0xff);
        const auto allowed =
          operation == Operation::EnumDevices || operation == Operation::OpenDevice ||
          operation == Operation::StartGrabbing ? 2000ms : call_timeout;
        if (now - last_progress >= allowed ||
            (stop_started != Clock::time_point::max() && now - stop_started >= shutdown_timeout)) {
          process.failure_reason = std::string(operation_name(operation)) +
            " deadline exceeded; worker=" + std::to_string(process.worker) +
            " observed_ms=" + std::to_string(
              std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count());
          process.last_timeout_operation.store(
            static_cast<unsigned int>(operation), std::memory_order_release);
          fault = true;
          break;
        }
        // Parent watchdog never logs synchronously: a full terminal/file sink can block.
        const auto frames = shared.frames_published.load(std::memory_order_acquire);
        if (frames != observed_frames) {
          observed_frames = frames;
          consecutive_failures = 0;
          process.state.store(State::Capturing, std::memory_order_release);
        }
        std::this_thread::sleep_for(5ms);
      }
      stopping();
      const bool clean = !fault && WIFEXITED(status) &&
        (WEXITSTATUS(status) == kWorkerCleanExit || WEXITSTATUS(status) == kWorkerRetryExit);
      bool recovered = clean;
      if (!clean && !fault) {
        process.failure_reason = "worker exited without clean cleanup; wait_status=" +
                                 std::to_string(status);
      }
      process.state.store(State::Recovering, std::memory_order_release);
      if (!clean) recovered = process.retire_and_reset();
      // Reset/reaping may take time; do not reuse a pre-reset stop snapshot.
      const bool stopped = stopping();
      if (!recovered || (!stopped && ++consecutive_failures >= 3)) {
        process.state.store(State::Quarantined, std::memory_order_release);
        break;
      }
      if (stopped) break;

      // Reuse only after every child was reaped. Slot 3 belongs to a concurrent reader.
      auto slot = shared.slot.load(std::memory_order_acquire);
      if (slot != 3) shared.slot.compare_exchange_strong(slot, 0, std::memory_order_acq_rel);
      mark_operation(shared, Operation::Idle);
      for (int i = 0; i < 100 && !stopping(); ++i) {
        std::this_thread::sleep_for(5ms);
      }
    }
  } catch (...) {
    process.state.store(State::Quarantined, std::memory_order_release);
    terminate(process.worker);
    terminate(process.reset_worker);
    int status = 0;
    wait_until(process.worker, status, Clock::now() + reap_timeout);
    wait_until(process.reset_worker, status, Clock::now() + reap_timeout);
  }
  if (process.state.load(std::memory_order_acquire) != State::Quarantined) {
    process.state.store(State::Stopped, std::memory_order_release);
  }
  retired_.store(true, std::memory_order_release);
}

bool HikRobot::try_read(cv::Mat & img, Clock::time_point & timestamp)
{
  if (daemon_quit_.load(std::memory_order_acquire) || retired_.load(std::memory_order_acquire)) {
    return false;
  }
  const auto generation = process_->generation.load(std::memory_order_acquire);
  if (process_->state.load(std::memory_order_acquire) != State::Capturing) return false;
  auto & shared = *process_->shared;
  std::uint32_t expected = 2;
  if (!shared.slot.compare_exchange_strong(expected, 3, std::memory_order_acq_rel)) return false;
  try {
    const auto bytes = static_cast<std::uint64_t>(shared.width) * shared.height * 3;
    if (shared.width == 0 || shared.height == 0 || shared.width > 65535 ||
        shared.height > 65535 || bytes != shared.bytes || bytes > kFrameCapacity) {
      shared.slot.store(0, std::memory_order_release);
      return false;
    }
    img = cv::Mat(static_cast<int>(shared.height), static_cast<int>(shared.width),
                  CV_8UC3, shared.pixels).clone();
    timestamp = Clock::time_point(
      std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(shared.timestamp_ns)));
  } catch (...) {
    shared.slot.store(0, std::memory_order_release);
    throw;
  }
  shared.slot.store(0, std::memory_order_release);
  if (process_->generation.load(std::memory_order_acquire) != generation ||
      process_->state.load(std::memory_order_acquire) != State::Capturing ||
      daemon_quit_.load(std::memory_order_acquire)) {
    img.release();
    timestamp = {};
    return false;
  }
  return true;
}

void HikRobot::read(cv::Mat & img, Clock::time_point & timestamp)
{
  const auto deadline = Clock::now() + 100ms;
  do {
    if (try_read(img, timestamp)) return;
    if (daemon_quit_.load(std::memory_order_acquire)) break;
    std::this_thread::sleep_for(2ms);
  } while (Clock::now() < deadline);
  img.release();
  timestamp = {};
}

void HikRobot::update_params(double exposure_ms, double gain)
{
  if (!std::isfinite(exposure_ms) || exposure_ms <= 0 || !std::isfinite(gain) || gain < 0) {
    throw std::invalid_argument("Invalid HikRobot exposure/gain");
  }
  process_->shared->exposure_us.store(exposure_ms * 1e3, std::memory_order_release);
  process_->shared->gain.store(gain, std::memory_order_release);
}
}  // namespace io
