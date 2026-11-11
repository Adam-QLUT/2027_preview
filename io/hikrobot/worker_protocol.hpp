#ifndef IO__HIKROBOT_WORKER_PROTOCOL_HPP
#define IO__HIKROBOT_WORKER_PROTOCOL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <opencv2/core.hpp>
#include <utility>

namespace io::hikrobot_worker
{
constexpr int kWorkerCleanExit = 0;
constexpr int kWorkerRetryExit = 10;
constexpr int kWorkerFaultExit = 20;
constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::size_t kFrameCapacity = 32 * 1024 * 1024;

enum class Operation : std::uint8_t
{
  Idle, EnumDevices, CreateHandle, OpenDevice, SetEnum, SetFloat, SetFrameRate,
  StartGrabbing, GetImageBuffer, FreeImageBuffer, StopGrabbing, CloseDevice,
  DestroyHandle, Exiting
};

// This is a Linux same-binary/same-architecture IPC ABI, not a serialized wire format.
// Lock-free atomics avoid inheriting a mutex held by a dead worker.
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<double>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);

struct SharedMemory
{
  std::uint32_t version = kProtocolVersion;
  std::atomic<bool> shutdown{false};
  std::atomic<double> exposure_us{0};
  std::atomic<double> gain{0};
  float frame_rate = 0;
  // Every SDK entry/return changes the upper sequence bits, even for repeated Gets.
  std::atomic<std::uint64_t> operation{0};
  std::atomic<std::uint64_t> frames_published{0};
  // Slot ownership: 0 empty, 1 worker writing, 2 ready, 3 parent reading.
  std::atomic<std::uint32_t> slot{0};
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bytes = 0;
  std::int64_t timestamp_ns = 0;
  alignas(64) unsigned char pixels[kFrameCapacity];
};

inline std::int64_t monotonic_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void mark_operation(SharedMemory & shared, Operation operation)
{
  const auto previous = shared.operation.load(std::memory_order_relaxed);
  shared.operation.store(
    ((previous + 0x100) & ~std::uint64_t{0xff}) | static_cast<std::uint8_t>(operation),
    std::memory_order_release);
}

template <typename Function>
int invoke(SharedMemory & shared, Operation operation, Function && function)
{
  mark_operation(shared, operation);
  const int result = std::forward<Function>(function)();
  mark_operation(shared, Operation::Idle);
  return result;
}

inline bool publish_frame(
  SharedMemory & shared, const cv::Mat & image, std::int64_t timestamp_ns)
{
  if (image.empty() || image.type() != CV_8UC3 || !image.isContinuous()) return false;
  const auto bytes = image.total() * image.elemSize();
  if (bytes > kFrameCapacity) return false;
  auto expected = shared.slot.load(std::memory_order_acquire);
  if (expected != 0 && expected != 2) return false;
  if (!shared.slot.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return false;
  std::memcpy(shared.pixels, image.data, bytes);
  shared.width = static_cast<std::uint32_t>(image.cols);
  shared.height = static_cast<std::uint32_t>(image.rows);
  shared.bytes = static_cast<std::uint32_t>(bytes);
  shared.timestamp_ns = timestamp_ns;
  shared.slot.store(2, std::memory_order_release);
  shared.frames_published.fetch_add(1, std::memory_order_release);
  return true;
}

inline const char * operation_name(Operation operation)
{
  switch (operation) {
    case Operation::Idle: return "idle/processing";
    case Operation::EnumDevices: return "MV_CC_EnumDevices";
    case Operation::CreateHandle: return "MV_CC_CreateHandle";
    case Operation::OpenDevice: return "MV_CC_OpenDevice";
    case Operation::SetEnum: return "MV_CC_SetEnumValue";
    case Operation::SetFloat: return "MV_CC_SetFloatValue";
    case Operation::SetFrameRate: return "MV_CC_SetFrameRate";
    case Operation::StartGrabbing: return "MV_CC_StartGrabbing";
    case Operation::GetImageBuffer: return "MV_CC_GetImageBuffer";
    case Operation::FreeImageBuffer: return "MV_CC_FreeImageBuffer";
    case Operation::StopGrabbing: return "MV_CC_StopGrabbing";
    case Operation::CloseDevice: return "MV_CC_CloseDevice";
    case Operation::DestroyHandle: return "MV_CC_DestroyHandle";
    case Operation::Exiting: return "process-exit";
  }
  return "unknown";
}
}  // namespace io::hikrobot_worker

#endif  // IO__HIKROBOT_WORKER_PROTOCOL_HPP