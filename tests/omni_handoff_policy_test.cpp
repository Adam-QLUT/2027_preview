#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "tasks/omniperception/handoff_policy.hpp"

namespace h = omniperception::handoff;
using namespace std::chrono_literals;
namespace
{
class Candidate
{
public:
  bool valid = false;
  int confirm_count = 0;
  int priority = 5;
  double world_yaw = 0;
  double pitch = 0;
  h::Clock::time_point timestamp{};
};
void require(bool value, const char * message)
{
  if (!value) throw std::runtime_error(message);
}
}
int main()
{
  try {
    const auto now = h::Clock::time_point{} + 10s;
    std::array<Candidate, 2> slots{};
    slots[0] = {true, 2, 1, 1.0, 0, now - 121ms};
    require(h::select_candidate(slots, now, 120ms, 0) == nullptr,
            "expired winner must not become a zero-angle control command");
    require(!slots[0].valid, "expired slot must be cleared");
    slots[0] = {true, 2, 1, 1.0, 0, now - 150ms};
    require(h::select_candidate(slots, now, 120ms, 0, 2, 250ms) == nullptr && slots[0].valid,
            "longer confirmation history must never extend command freshness");
    slots[0] = {true, 2, 1, 1.0, 0, now + 1ms};
    require(h::select_candidate(slots, now, 120ms, 0) == nullptr, "future sample rejected");
    slots[0] = {true, 2, 1, 1.0, 0, {}};
    require(h::select_candidate(slots, now, 120ms, 0) == nullptr, "zero timestamp rejected");
    slots[0] = {true, 2, 1, std::numeric_limits<double>::quiet_NaN(), 0, now};
    require(h::select_candidate(slots, now, 120ms, 0) == nullptr, "NaN bearing rejected");
    slots[0] = {true, 1, 1, 0.0, 0, now};
    slots[1] = {true, 2, 2, 1.0, 0, now - 120ms};
    require(h::select_candidate(slots, now, 120ms, 0) == &slots[1],
            "single-frame higher-priority candidate must not steal control");
    slots[0] = {true, 2, 1, 3.0, 0, now};
    require(h::select_candidate(slots, now, 120ms, 0) == &slots[0], "priority before distance");
    slots[0] = {true, 2, 1, -3.13, 0, now};
    slots[1] = {true, 2, 1, 2.0, 0, now};
    require(h::select_candidate(slots, now, 120ms, 3.13) == &slots[0],
            "tie selection must use wrapped shortest angle");

    const auto start = now;
    // Simulate repeated visible detections that never reach tracking (e.g. dt resets).
    for (auto elapsed = 0ms; elapsed <= 1100ms; elapsed += 110ms) {
      const auto action = h::acquisition_action(start + elapsed, start, 800ms, false, true, true);
      require(action == (elapsed <= 800ms ? h::AcquireAction::follow : h::AcquireAction::timeout),
              "visible-but-unstable target must not bypass the acquisition deadline");
    }
    require(h::acquisition_action(start + 800ms, start, 800ms, true, true, true) ==
              h::AcquireAction::complete, "stable main track takes over within deadline");
    require(h::acquisition_action(start + 801ms, start, 800ms, true, true, true) ==
              h::AcquireAction::timeout, "deadline checked before completion");
    require(h::acquisition_action(start + 400ms, start, 800ms, false, false, false) ==
              h::AcquireAction::reacquire, "temporary absence allows bounded reacquisition");
    require(h::acquisition_action(start + 400ms, start, 800ms, true, true, false) ==
              h::AcquireAction::reacquire, "empty target cannot complete handoff");
    std::cout << "PASS: production handoff policy regressions\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}