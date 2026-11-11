#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "capture_worker.hpp"
#include "tools/logger.hpp"
#include "usb_reset.hpp"
#include "worker_protocol.hpp"
#include "worker_runtime.hpp"

namespace
{
using io::hikrobot_worker::SharedMemory;

SharedMemory * map_shared_memory(int fd)
{
  struct stat status {};
  if (fstat(fd, &status) != 0) {
    throw std::system_error(errno, std::generic_category(), "fstat shared memory");
  }
  if (status.st_size != static_cast<off_t>(sizeof(SharedMemory))) {
    throw std::runtime_error("Shared-memory size does not match this worker executable");
  }
  void * mapping = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) {
    throw std::system_error(errno, std::generic_category(), "mmap shared memory");
  }
  close(fd);
  auto * shared = static_cast<SharedMemory *>(mapping);
  if (shared->version != io::hikrobot_worker::kProtocolVersion) {
    munmap(mapping, sizeof(SharedMemory));
    throw std::runtime_error("Shared-memory protocol version does not match this worker");
  }
  return shared;
}

void report_exception(const char * message) noexcept
{
  try {
    tools::logger()->error("[HikRobot worker] startup/exit failure: {}", message);
  } catch (...) {
    // A failed log sink must not turn an exception into a second exception.
    // A blocked sink is still bounded by the external process supervisor.
  }
}
}  // namespace

int main(int argc, char ** argv)
{
  using namespace io::hikrobot_worker;
  SharedMemory * shared = nullptr;
  int result = kWorkerFaultExit;
  try {
    configure_worker_signals(kWorkerFaultExit);
    // The SDK-free launcher sanitized inherited FDs before this executable
    // loaded vendor libraries. Do not close FDs created by vendor constructors.
    if (argc == 4 && std::strcmp(argv[1], "--capture") == 0) {
      const int fd = parse_worker_integer(argv[2], std::numeric_limits<int>::max());
      if (fd != 3) throw std::invalid_argument("Capture shared-memory FD must be 3");
      shared = map_shared_memory(fd);
      tools::logger();
      result = capture_start(*shared, argv[3]);
    } else if (argc == 5 && std::strcmp(argv[1], "--reset") == 0) {
      const int vid = parse_worker_integer(argv[2], 0xffff, true);
      const int pid = parse_worker_integer(argv[3], 0xffff, true);
      tools::logger();
      result = reset_usb(vid, pid, argv[4]) == 0 ? kWorkerCleanExit : kWorkerFaultExit;
    } else {
      throw std::invalid_argument(
        "Expected --capture FD SERIAL or --reset VID PID SERIAL (VID/PID decimal or 0x hex)");
    }
  } catch (const std::exception & error) {
    if (shared != nullptr) mark_operation(*shared, Operation::Exiting);
    report_exception(error.what());
  } catch (...) {
    if (shared != nullptr) mark_operation(*shared, Operation::Exiting);
    report_exception("Unknown exception");
  }
  if (shared != nullptr) {
    mark_operation(*shared, Operation::Exiting);
    munmap(shared, sizeof(SharedMemory));
  }
  // CaptureSession or reset_usb performed all explicitly safe cleanup already.
  // Do not enter vendor/static destructors or atexit hooks on a damaged process.
  _exit(result);
}