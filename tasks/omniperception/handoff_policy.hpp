#ifndef OMNIPERCEPTION__HANDOFF_POLICY_HPP
#define OMNIPERCEPTION__HANDOFF_POLICY_HPP

#include <chrono>
#include <cmath>
#include <algorithm>

namespace omniperception::handoff
{
using Clock = std::chrono::steady_clock;

inline double angle_error(double a, double b)
{
  return std::abs(std::remainder(a - b, 2.0 * 3.14159265358979323846));
}

inline bool fresh(Clock::time_point stamp, Clock::time_point now, Clock::duration max_age)
{
  return stamp != Clock::time_point{} && stamp <= now && now - stamp <= max_age;
}

// Called AFTER updating all camera slots. Never keep a pointer across slot eviction.
template <typename Slots>
auto select_candidate(Slots & slots, Clock::time_point now, Clock::duration max_age,
                      double main_yaw, int minimum_count = 2,
                      Clock::duration confirmation_gap = Clock::duration::zero())
  -> typename Slots::value_type *
{
  typename Slots::value_type * selected = nullptr;
  for (auto & candidate : slots) {
    if (!candidate.valid) continue;
    if (!fresh(candidate.timestamp, now, std::max(max_age, confirmation_gap)) ||
        !std::isfinite(candidate.world_yaw) || !std::isfinite(candidate.pitch)) {
      candidate = {};
      continue;
    }
    // Old data may be retained solely as confirmation history, never as a command.
    if (!fresh(candidate.timestamp, now, max_age)) continue;
    if (candidate.confirm_count < minimum_count) continue;
    if (selected == nullptr || candidate.priority < selected->priority ||
        (candidate.priority == selected->priority &&
         angle_error(candidate.world_yaw, main_yaw) < angle_error(selected->world_yaw, main_yaw))) {
      selected = &candidate;
    }
  }
  return selected;
}

enum class AcquireAction { timeout, complete, follow, reacquire };

// The deadline is absolute since entering acquiring, including visible-but-unstable frames.
inline AcquireAction acquisition_action(Clock::time_point now, Clock::time_point started,
                                        Clock::duration timeout, bool tracking,
                                        bool visible, bool has_target)
{
  if (now - started > timeout) return AcquireAction::timeout;
  if (tracking && visible && has_target) return AcquireAction::complete;
  if (visible && has_target) return AcquireAction::follow;
  return AcquireAction::reacquire;
}
}  // namespace omniperception::handoff
#endif