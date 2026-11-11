#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

#include "worker_runtime.hpp"

namespace
{
constexpr int fault_exit = 20;

void preserve_across_exec(int fd)
{
  const int flags = ::fcntl(fd, F_GETFD);
  if (flags < 0 || ::fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC) != 0) {
    throw std::system_error(errno, std::generic_category(), "preserve worker descriptor");
  }
}
}  // namespace

int main(int argc, char ** argv)
{
  using namespace io::hikrobot_worker;
  try {
    configure_worker_signals(fault_exit);
    const bool capture_mode =
      argc == 5 && std::strcmp(argv[2], "--capture") == 0 &&
      std::strcmp(argv[3], "3") == 0;
    const bool reset_mode =
      (argc == 6 || argc == 7) && std::strcmp(argv[2], "--reset") == 0;
    if ((!capture_mode && !reset_mode) || argv[1][0] != '/') {
      throw std::invalid_argument("Expected an absolute worker executable and valid mode");
    }

    // This tiny executable has no SDK, libusb, OpenCV, or logger dependency.
    // Remove inherited device FDs before loading the actual worker's libraries,
    // whose constructors may open new descriptors before their main() runs.
    close_unrelated_worker_fds(capture_mode);
    preserve_across_exec(4);
    if (capture_mode) preserve_across_exec(3);
    ::execv(argv[1], argv + 1);
  } catch (...) {
    // The supervisor observes the exit code. Even reporting to stderr could
    // block on a full terminal/pipe, so the launcher performs no logging.
  }
  ::_exit(fault_exit);
}