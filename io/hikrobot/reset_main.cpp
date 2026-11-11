#include <cstring>
#include <exception>
#include <stdexcept>

#include <unistd.h>

#include "tools/logger.hpp"
#include "usb_reset.hpp"
#include "worker_runtime.hpp"

namespace
{
constexpr int clean_exit = 0;
constexpr int fault_exit = 20;

void report_exception(const char * message) noexcept
{
  try {
    tools::logger()->error("[HikRobot reset] startup/exit failure: {}", message);
  } catch (...) {
    // The external supervisor also bounds a blocked or failed log sink.
  }
}
}  // namespace

int main(int argc, char ** argv)
{
  using namespace io::hikrobot_worker;
  int result = fault_exit;
  try {
    configure_worker_signals(fault_exit);
    // The launcher sanitized inherited FDs before loading this executable.
    if ((argc != 5 && argc != 6) || std::strcmp(argv[1], "--reset") != 0) {
      throw std::invalid_argument("Expected --reset VID PID SERIAL [CAUSE] (VID/PID decimal)");
    }
    const int vid = parse_worker_integer(argv[2], 0xffff);
    const int pid = parse_worker_integer(argv[3], 0xffff);
    tools::logger();
    if (argc == 6) {
      tools::logger()->warn(
        "[HikRobot reset] serial='{}' recovery cause: {}", argv[4], argv[5]);
    }
    result = reset_usb(vid, pid, argv[4]) == 0 ? clean_exit : fault_exit;
  } catch (const std::exception & error) {
    report_exception(error.what());
  } catch (...) {
    report_exception("Unknown exception");
  }
  // This executable intentionally does not link or initialize the MVS SDK.
  // Avoid static destructors/atexit hooks after the explicit libusb cleanup.
  ::_exit(result);
}