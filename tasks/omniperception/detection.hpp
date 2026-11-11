#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <cstdint>
#include <list>
#include <string>
#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;
  std::chrono::steady_clock::time_point timestamp{};
  double delta_yaw = 0;    // rad; auxiliary-local bearing, NOT world yaw.
  double delta_pitch = 0;  // rad; geometric pitch, NOT ballistic pitch.
  std::string camera;      // "left" or "right".
  std::uint64_t generation = 0;
  // Compiler-generated copy/move operations copy camera and generation as well.
};
}  // namespace omniperception

#endif