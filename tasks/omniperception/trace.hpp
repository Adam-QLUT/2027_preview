#ifndef OMNIPERCEPTION__TRACE_HPP
#define OMNIPERCEPTION__TRACE_HPP

#include <fmt/format.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <thread>
#include <utility>

#include "tools/logger.hpp"

namespace omniperception::trace
{
// 0: off (default), 1: heartbeat/config, 2: every pipeline boundary.
inline int level()
{
  static const int value = [] {
    const char * setting = std::getenv("SP_OMNI_TRACE");
    return setting == nullptr ? 0 : (setting[0] == '2' ? 2 : (setting[0] == '1' ? 1 : 0));
  }();
  return value;
}

inline long long time_us(std::chrono::steady_clock::time_point time)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count();
}

inline double age_ms(std::chrono::steady_clock::time_point time)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - time).count();
}

template <typename... Args>
void event(int required_level, fmt::format_string<Args...> format, Args &&... args)
{
  if (level() < required_level) return;
  // debug avoids the logger's flush_on(info) for every inference frame.
  tools::logger()->debug(
    "[OMNI] mono_us={} thread={} {}", time_us(std::chrono::steady_clock::now()),
    std::hash<std::thread::id>{}(std::this_thread::get_id()),
    fmt::format(format, std::forward<Args>(args)...));
}
}  // namespace omniperception::trace

#endif