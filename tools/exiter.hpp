#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

namespace tools
{
// Read the process-wide signal request without waiting for the main loop.
// This does not install handlers; the application still owns one Exiter.
bool exit_requested() noexcept;

class Exiter
{
public:
  Exiter();

  bool exit() const;
};

}  // namespace tools

#endif  // TOOLS__EXITER_HPP