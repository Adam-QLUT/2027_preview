#include "exiter.hpp"

#include <atomic>
#include <csignal>
#include <stdexcept>

namespace tools
{
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> exit_{false};
bool exiter_inited_ = false;

Exiter::Exiter()
{
  if (exiter_inited_) throw std::runtime_error("Multiple Exiter instances!");
  const auto request_exit = [](int) { exit_.store(true, std::memory_order_relaxed); };
  std::signal(SIGINT, request_exit);
  std::signal(SIGTERM, request_exit);
  exiter_inited_ = true;
}

bool exit_requested() noexcept { return exit_.load(std::memory_order_relaxed); }

bool Exiter::exit() const { return exit_requested(); }

}  // namespace tools