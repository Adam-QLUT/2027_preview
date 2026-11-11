#ifndef IO__HIKROBOT_WORKER_RUNTIME_HPP
#define IO__HIKROBOT_WORKER_RUNTIME_HPP

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>

#include <dirent.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace io::hikrobot_worker
{

inline int parse_worker_integer(const char * text, int maximum, bool allow_hex = false)
{
  if (text == nullptr || *text == '\0') throw std::invalid_argument("Missing integer argument");
  const char * begin = text;
  const char * end = text + std::strlen(text);
  int base = 10;
  if (allow_hex && end - begin > 2 && begin[0] == '0' &&
      (begin[1] == 'x' || begin[1] == 'X')) {
    begin += 2;
    base = 16;
  }
  unsigned int value = 0;
  const auto parsed = std::from_chars(begin, end, value, base);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      value > static_cast<unsigned int>(maximum)) {
    throw std::invalid_argument("Invalid or out-of-range integer argument");
  }
  return static_cast<int>(value);
}

inline void configure_worker_signals(int fault_exit_status)
{
  if (std::signal(SIGINT, SIG_IGN) == SIG_ERR ||
      std::signal(SIGTERM, SIG_DFL) == SIG_ERR) {
    throw std::system_error(errno, std::generic_category(), "signal setup");
  }
  const pid_t parent = ::getppid();
  if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
    throw std::system_error(errno, std::generic_category(), "PR_SET_PDEATHSIG");
  }
  if (parent == 1 || ::getppid() != parent) ::_exit(fault_exit_status);
}

inline void close_unrelated_worker_fds(bool keep_shared_memory)
{
  // FD 4 retains the per-camera quarantine lease for this helper's lifetime.
  // A reset helper has no shared-memory mapping and must not inherit FD 3.
  if (!keep_shared_memory) ::close(3);

#ifdef SYS_close_range
  if (::syscall(SYS_close_range, 5U, std::numeric_limits<unsigned int>::max(), 0U) == 0) {
    return;
  }
#endif

  // Compatibility path for kernels without close_range. It runs immediately
  // after exec, before application logging/SDK threads create additional FDs.
  struct DirectoryCloser
  {
    void operator()(DIR * directory) const { ::closedir(directory); }
  };
  std::unique_ptr<DIR, DirectoryCloser> directory(::opendir("/proc/self/fd"));
  if (!directory) {
    throw std::system_error(errno, std::generic_category(), "opendir /proc/self/fd");
  }
  const int own_fd = ::dirfd(directory.get());
  for (;;) {
    errno = 0;
    const auto * entry = ::readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) throw std::system_error(errno, std::generic_category(), "readdir FDs");
      break;
    }
    int fd = -1;
    const auto * end = entry->d_name + std::strlen(entry->d_name);
    const auto parsed = std::from_chars(entry->d_name, end, fd);
    if (parsed.ec != std::errc{} || parsed.ptr != end || fd < 5 || fd == own_fd) continue;
    // On Linux a close error does not justify retrying the numeric descriptor:
    // the number may already have been released and reused.
    ::close(fd);
  }
}

}  // namespace io::hikrobot_worker

#endif  // IO__HIKROBOT_WORKER_RUNTIME_HPP