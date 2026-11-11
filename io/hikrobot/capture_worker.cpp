#include "capture_worker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

#include "MvCameraControl.h"
#include "tools/logger.hpp"

namespace io
{
namespace hikrobot_worker
{
namespace
{
constexpr unsigned int image_wait_timeout_ms = 100;
constexpr int max_consecutive_no_data = 10;

// Observe host-side timings without additional SDK calls or a shared IPC change.
// A frame-number discontinuity is not by itself a count of lost USB frames.
class CaptureTelemetry
{
public:
  explicit CaptureTelemetry(int64_t started_ns) : started_ns_(started_ns), window_ns_(started_ns) {}

  void record_get(int64_t begin_ns, int64_t end_ns, int result)
  {
    if (last_get_end_ns_ != 0) {
      max_between_get_ms_ = std::max(max_between_get_ms_, ms(begin_ns - last_get_end_ns_));
    }
    last_get_end_ns_ = end_ns;
    max_get_ms_ = std::max(max_get_ms_, ms(end_ns - begin_ns));
    if (result == MV_OK) ++get_ok_;
    if (result == MV_E_NODATA) ++no_data_;
  }

  bool has_frame() const { return has_frame_; }
  unsigned int last_frame_number() const { return last_frame_number_; }
  double frame_age_ms(int64_t now_ns) const
  {
    return ms(now_ns - (has_frame_ ? last_frame_ns_ : started_ns_));
  }

  void record_frame(unsigned int number, int64_t timestamp_ns, double convert_ms, double free_ms)
  {
    if (has_frame_ && number != static_cast<unsigned int>(last_frame_number_ + 1U)) {
      ++frame_number_discontinuities_;
    }
    has_frame_ = true;
    last_frame_number_ = number;
    last_frame_ns_ = timestamp_ns;
    max_convert_ms_ = std::max(max_convert_ms_, convert_ms);
    max_free_ms_ = std::max(max_free_ms_, free_ms);
  }

  void record_publish(bool published)
  {
    if (published) ++published_;
    else ++ipc_busy_;
  }

  void report(const std::string & serial, bool force = false)
  {
    const auto now_ns = monotonic_ns();
    if (!force && now_ns - window_ns_ < 5000000000LL) return;
    const double window_s = static_cast<double>(now_ns - window_ns_) / 1e9;
    tools::logger()->info(
      "[HikRobot worker] serial='{}' pid={} ppid={} stats window_s={:.2f} get_ok={} nodata={} "
      "get_fps={:.1f} published={} ipc_busy={} has_frame={} last_frame={} "
      "frame_age_ms={:.1f} frame_number_discontinuities={} "
      "max_get_ms={:.1f} max_between_get_ms={:.1f} max_convert_ms={:.1f} max_free_ms={:.1f}",
      serial, ::getpid(), ::getppid(), window_s, get_ok_, no_data_,
      window_s > 0 ? get_ok_ / window_s : 0,
      published_, ipc_busy_, has_frame_, last_frame_number_, frame_age_ms(now_ns),
      frame_number_discontinuities_, max_get_ms_, max_between_get_ms_, max_convert_ms_, max_free_ms_);
    window_ns_ = now_ns;
    get_ok_ = no_data_ = published_ = ipc_busy_ = frame_number_discontinuities_ = 0;
    max_get_ms_ = max_between_get_ms_ = max_convert_ms_ = max_free_ms_ = 0;
  }

  static double ms(int64_t duration_ns) { return static_cast<double>(duration_ns) / 1e6; }

private:
  int64_t started_ns_;
  int64_t window_ns_;
  int64_t last_get_end_ns_ = 0;
  int64_t last_frame_ns_ = 0;
  unsigned int last_frame_number_ = 0;
  bool has_frame_ = false;
  uint64_t get_ok_ = 0, no_data_ = 0, published_ = 0, ipc_busy_ = 0;
  uint64_t frame_number_discontinuities_ = 0;
  double max_get_ms_ = 0, max_between_get_ms_ = 0, max_convert_ms_ = 0, max_free_ms_ = 0;
};

template <size_t N>
std::string fixed_string(const unsigned char (&value)[N])
{
  const auto end = std::find(value, value + N, static_cast<unsigned char>(0));
  return std::string(reinterpret_cast<const char *>(value), end - value);
}

void log_sdk_error(const std::string & serial, const char * operation, int ret)
{
  tools::logger()->error(
    "[HikRobot worker] serial='{}' {} failed: {:#x}", serial, operation,
    static_cast<unsigned int>(ret));
}

void log_capture_exception(const std::string & serial, const char * message) noexcept
{
  try {
    tools::logger()->error(
      "[HikRobot worker] serial='{}' capture failed: {}", serial, message);
  } catch (...) {
    // Preserve the explicit cleanup attempt even when a log sink throws.
  }
}

// These wrappers execute serially on the sole SDK owner. Their 500 ms deadline
// is enforced by the separate supervisor, which terminates this worker on a
// timeout. A thread timeout cannot cancel either of these SDK functions safely.
bool safe_stop_grabbing(SharedMemory & shared, void * handle, const std::string & serial)
{
  const int ret = invoke(shared, Operation::StopGrabbing, [&] {
    tools::logger()->info("[HikRobot worker] serial='{}' stopping stream", serial);
    return MV_CC_StopGrabbing(handle);
  });
  if (ret != MV_OK) log_sdk_error(serial, "MV_CC_StopGrabbing", ret);
  return ret == MV_OK;
}

bool safe_close_device(SharedMemory & shared, void * handle, const std::string & serial)
{
  const int ret = invoke(shared, Operation::CloseDevice, [&] {
    tools::logger()->info("[HikRobot worker] serial='{}' closing device", serial);
    return MV_CC_CloseDevice(handle);
  });
  if (ret != MV_OK) log_sdk_error(serial, "MV_CC_CloseDevice", ret);
  return ret == MV_OK;
}

cv::Mat owned_bgr_image(const MV_FRAME_OUT & raw)
{
  const auto & info = raw.stFrameInfo;
  const auto width = info.nWidth != 0 ? info.nWidth : info.nExtendWidth;
  const auto height = info.nHeight != 0 ? info.nHeight : info.nExtendHeight;
  if (
    raw.pBufAddr == nullptr || width == 0 || height == 0 ||
    width > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
    height > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Invalid SDK frame pointer or dimensions");
  }

  const bool packed_color =
    info.enPixelType == PixelType_Gvsp_BGR8_Packed ||
    info.enPixelType == PixelType_Gvsp_RGB8_Packed;
  const uint64_t required_bytes =
    static_cast<uint64_t>(width) * height * (packed_color ? 3U : 1U);
  if (static_cast<uint64_t>(width) * height * 3U > kFrameCapacity) {
    throw std::runtime_error("Converted image exceeds the shared-memory frame capacity");
  }
  if (required_bytes > info.nFrameLen) {
    throw std::runtime_error("SDK frame is shorter than its image dimensions");
  }

  const cv::Mat source(
    static_cast<int>(height), static_cast<int>(width), packed_color ? CV_8UC3 : CV_8UC1,
    raw.pBufAddr);
  cv::Mat result;
  switch (info.enPixelType) {
    // Preserve the Bayer conversion convention of the existing implementation.
    case PixelType_Gvsp_BayerGR8:
      cv::cvtColor(source, result, cv::COLOR_BayerGR2RGB);
      break;
    case PixelType_Gvsp_BayerRG8:
      cv::cvtColor(source, result, cv::COLOR_BayerRG2RGB);
      break;
    case PixelType_Gvsp_BayerGB8:
      cv::cvtColor(source, result, cv::COLOR_BayerGB2RGB);
      break;
    case PixelType_Gvsp_BayerBG8:
      cv::cvtColor(source, result, cv::COLOR_BayerBG2RGB);
      break;
    case PixelType_Gvsp_Mono8:
      cv::cvtColor(source, result, cv::COLOR_GRAY2BGR);
      break;
    case PixelType_Gvsp_RGB8_Packed:
      cv::cvtColor(source, result, cv::COLOR_RGB2BGR);
      break;
    case PixelType_Gvsp_BGR8_Packed:
      result = source.clone();
      break;
    default:
      throw std::runtime_error(
        "Unsupported SDK pixel format " +
        std::to_string(static_cast<unsigned int>(info.enPixelType)));
  }
  if (result.empty() || result.type() != CV_8UC3 || !result.isContinuous()) {
    throw std::runtime_error("Image conversion did not produce a contiguous color frame");
  }
  return result;
}

class CaptureSession
{
public:
  CaptureSession(SharedMemory & shared, const std::string & serial)
  : shared_(shared), serial_(serial)
  {
  }

  int run()
  {
    if (stopping()) return kWorkerCleanExit;
    tools::logger()->info(
      "[HikRobot worker] serial='{}' pid={} ppid={} enumerating",
      serial_, ::getpid(), ::getppid());

    MV_CC_DEVICE_INFO_LIST devices{};
    int ret = invoke(shared_, Operation::EnumDevices, [&] {
      return MV_CC_EnumDevices(MV_USB_DEVICE, &devices);
    });
    if (ret != MV_OK) {
      log_sdk_error(serial_, "MV_CC_EnumDevices", ret);
      return kWorkerFaultExit;
    }
    if (stopping()) return kWorkerCleanExit;
    if (devices.nDeviceNum > MV_MAX_DEVICE_NUM) {
      tools::logger()->error(
        "[HikRobot worker] SDK returned an invalid device count: {}", devices.nDeviceNum);
      return kWorkerFaultExit;
    }

    MV_CC_DEVICE_INFO selected{};
    unsigned int matches = 0;
    for (unsigned int i = 0; i < devices.nDeviceNum; ++i) {
      const auto * device = devices.pDeviceInfo[i];
      if (device == nullptr || device->nTLayerType != MV_USB_DEVICE) continue;
      const auto serial = fixed_string(device->SpecialInfo.stUsb3VInfo.chSerialNumber);
      if (serial_.empty() || serial == serial_) {
        selected = *device;
        ++matches;
      }
    }
    if (matches != 1) {
      tools::logger()->error(
        "[HikRobot worker] serial='{}' expected exactly one matching USB camera, found {}",
        serial_, matches);
      return kWorkerFaultExit;
    }
    serial_ = fixed_string(selected.SpecialInfo.stUsb3VInfo.chSerialNumber);
    tools::logger()->info("[HikRobot worker] selected serial='{}'", serial_);

    ret = invoke(shared_, Operation::CreateHandle, [&] {
      return MV_CC_CreateHandle(&handle_, &selected);
    });
    if (ret != MV_OK) {
      // A failed creation does not establish ownership of a valid SDK handle.
      handle_ = nullptr;
      log_sdk_error(serial_, "MV_CC_CreateHandle", ret);
      return kWorkerFaultExit;
    }
    if (handle_ == nullptr) {
      tools::logger()->error("[HikRobot worker] CreateHandle returned a null handle");
      return kWorkerFaultExit;
    }
    if (stopping()) return kWorkerCleanExit;

    ret = invoke(shared_, Operation::OpenDevice, [&] { return MV_CC_OpenDevice(handle_); });
    if (ret != MV_OK) {
      log_sdk_error(serial_, "MV_CC_OpenDevice", ret);
      return kWorkerFaultExit;
    }
    opened_ = true;
    if (stopping()) return kWorkerCleanExit;

    if (!set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)) return kWorkerFaultExit;
    if (stopping()) return kWorkerCleanExit;
    if (!set_enum("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS)) return kWorkerFaultExit;
    if (stopping()) return kWorkerCleanExit;
    // Mono cameras may not expose white balance; it is deliberately optional.
    set_enum("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS, false);
    if (stopping()) return kWorkerCleanExit;
    if (!set_enum("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)) return kWorkerFaultExit;
    if (stopping()) return kWorkerCleanExit;
    if (!set_enum("GainAuto", MV_GAIN_MODE_OFF)) return kWorkerFaultExit;
    if (stopping()) return kWorkerCleanExit;
    if (!update_params()) return kWorkerFaultExit;
    if (stopping()) return kWorkerCleanExit;

    if (!std::isfinite(shared_.frame_rate) || shared_.frame_rate <= 0) {
      tools::logger()->error("[HikRobot worker] serial='{}' invalid frame rate", serial_);
      return kWorkerFaultExit;
    }
    ret = invoke(shared_, Operation::SetFrameRate, [&] {
      return MV_CC_SetFrameRate(handle_, shared_.frame_rate);
    });
    if (ret != MV_OK) {
      log_sdk_error(serial_, "MV_CC_SetFrameRate", ret);
      return kWorkerFaultExit;
    }
    if (stopping()) return kWorkerCleanExit;

    ret = invoke(shared_, Operation::StartGrabbing, [&] {
      tools::logger()->info("[HikRobot worker] serial='{}' starting stream", serial_);
      return MV_CC_StartGrabbing(handle_);
    });
    if (ret != MV_OK) {
      // Start may have changed internal state before returning failure. Do not
      // guess whether CloseDevice can safely run against a half-started stream.
      cleanup_allowed_ = false;
      log_sdk_error(serial_, "MV_CC_StartGrabbing", ret);
      return kWorkerFaultExit;
    }
    grabbing_ = true;
    CaptureTelemetry telemetry(monotonic_ns());
    tools::logger()->info(
      "[HikRobot worker] serial='{}' stream started; awaiting first frame; "
      "requested_fps={:.1f} exposure_us={:.0f} gain={:.1f}",
      serial_, shared_.frame_rate, applied_exposure_, applied_gain_);

    int consecutive_no_data = 0;
    while (!stopping()) {
      if (!update_params()) return kWorkerFaultExit;
      if (stopping()) break;

      MV_FRAME_OUT raw{};
      const auto get_begin_ns = monotonic_ns();
      ret = invoke(shared_, Operation::GetImageBuffer, [&] {
        return MV_CC_GetImageBuffer(handle_, &raw, image_wait_timeout_ms);
      });
      const auto get_end_ns = monotonic_ns();
      telemetry.record_get(get_begin_ns, get_end_ns, ret);
      if (ret == MV_E_NODATA) {
        if (stopping()) break;
        ++consecutive_no_data;
        if (consecutive_no_data == 1 || consecutive_no_data % 5 == 0) {
          tools::logger()->warn(
            "[HikRobot worker] serial='{}' no data: {}/{} consecutive timeouts; "
            "get_ms={:.1f} frame_age_ms={:.1f} has_frame={} last_frame={}",
            serial_, consecutive_no_data, max_consecutive_no_data,
            CaptureTelemetry::ms(get_end_ns - get_begin_ns), telemetry.frame_age_ms(get_end_ns),
            telemetry.has_frame(), telemetry.last_frame_number());
        }
        if (consecutive_no_data >= max_consecutive_no_data) {
          tools::logger()->warn(
            "[HikRobot worker] serial='{}' retiring stream after no-data threshold", serial_);
          telemetry.report(serial_, true);
          return kWorkerRetryExit;
        }
        telemetry.report(serial_);
        continue;
      }
      if (ret != MV_OK) {
        log_sdk_error(serial_, "MV_CC_GetImageBuffer", ret);
        return kWorkerFaultExit;
      }

      // Once Get succeeds, the only route out of this block returns its loan
      // exactly once, including conversion/allocation exceptions and shutdown.
      const int64_t timestamp_ns = get_end_ns;
      // Copy metadata before Free; do not read raw after returning the SDK loan.
      const auto frame_number = raw.stFrameInfo.nFrameNum;
      const auto pixel_type = static_cast<unsigned int>(raw.stFrameInfo.enPixelType);
      const auto width = raw.stFrameInfo.nWidth != 0
        ? raw.stFrameInfo.nWidth : raw.stFrameInfo.nExtendWidth;
      const auto height = raw.stFrameInfo.nHeight != 0
        ? raw.stFrameInfo.nHeight : raw.stFrameInfo.nExtendHeight;
      const auto frame_bytes = raw.stFrameInfo.nFrameLen;
      cv::Mat image;
      std::exception_ptr conversion_error;
      const auto convert_begin_ns = monotonic_ns();
      try {
        if (!stopping()) image = owned_bgr_image(raw);
      } catch (...) {
        conversion_error = std::current_exception();
      }
      const auto convert_end_ns = monotonic_ns();
      const auto free_begin_ns = monotonic_ns();
      ret = invoke(shared_, Operation::FreeImageBuffer, [&] {
        return MV_CC_FreeImageBuffer(handle_, &raw);
      });
      const auto free_end_ns = monotonic_ns();
      if (ret != MV_OK) {
        // The loan may still be owned by the SDK. Retire the process; never
        // retry Free or destroy a handle while ownership is ambiguous.
        cleanup_allowed_ = false;
        log_sdk_error(serial_, "MV_CC_FreeImageBuffer", ret);
        return kWorkerFaultExit;
      }
      if (conversion_error) std::rethrow_exception(conversion_error);
      if (stopping()) break;

      const bool first_frame = !telemetry.has_frame();
      const auto previous_frame = telemetry.last_frame_number();
      const double gap_ms = telemetry.frame_age_ms(timestamp_ns);
      telemetry.record_frame(
        frame_number, timestamp_ns, CaptureTelemetry::ms(convert_end_ns - convert_begin_ns),
        CaptureTelemetry::ms(free_end_ns - free_begin_ns));
      // Release the SDK buffer and publish before any diagnostic logging.
      const bool published = publish_frame(shared_, image, timestamp_ns);
      telemetry.record_publish(published);
      if (first_frame) {
        tools::logger()->info(
          "[HikRobot worker] serial='{}' first frame: number={} size={}x{} "
          "pixel_type={:#x} bytes={} stream_to_frame_ms={:.1f} published={}",
          serial_, frame_number, width, height, pixel_type, frame_bytes, gap_ms, published);
      }
      if (consecutive_no_data != 0) {
        tools::logger()->info(
          "[HikRobot worker] serial='{}' recovered after {} timeouts; "
          "gap_ms={:.1f} previous_frame={} current_frame={} had_previous_frame={}",
          serial_, consecutive_no_data, gap_ms, previous_frame, frame_number, !first_frame);
        consecutive_no_data = 0;
      }
      telemetry.report(serial_);
    }
    return kWorkerCleanExit;
  }

  bool cleanup()
  {
    if (!cleanup_allowed_) return false;
    if (handle_ == nullptr) return true;
    if (grabbing_) {
      if (!safe_stop_grabbing(shared_, handle_, serial_)) return false;
      grabbing_ = false;
    }
    if (opened_) {
      if (!safe_close_device(shared_, handle_, serial_)) return false;
      opened_ = false;
    }
    const int ret = invoke(shared_, Operation::DestroyHandle, [&] {
      tools::logger()->info("[HikRobot worker] serial='{}' destroying handle", serial_);
      return MV_CC_DestroyHandle(handle_);
    });
    if (ret != MV_OK) {
      log_sdk_error(serial_, "MV_CC_DestroyHandle", ret);
      return false;
    }
    handle_ = nullptr;
    return true;
  }

private:
  bool stopping() const { return shared_.shutdown.load(std::memory_order_acquire); }

  bool set_enum(const char * name, unsigned int value, bool required = true)
  {
    const int ret = invoke(shared_, Operation::SetEnum, [&] {
      return MV_CC_SetEnumValue(handle_, name, value);
    });
    if (ret == MV_OK) return true;
    if (required) {
      log_sdk_error(serial_, name, ret);
    } else {
      tools::logger()->warn(
        "[HikRobot worker] serial='{}' optional {} unsupported/failed: {:#x}", serial_, name,
        static_cast<unsigned int>(ret));
    }
    return false;
  }

  bool set_float(const char * name, double value)
  {
    const int ret = invoke(shared_, Operation::SetFloat, [&] {
      return MV_CC_SetFloatValue(handle_, name, static_cast<float>(value));
    });
    if (ret != MV_OK) log_sdk_error(serial_, name, ret);
    return ret == MV_OK;
  }

  bool update_params()
  {
    const double exposure = shared_.exposure_us.load(std::memory_order_acquire);
    const double gain = shared_.gain.load(std::memory_order_acquire);
    if (
      !std::isfinite(exposure) || exposure <= 0 ||
      exposure > std::numeric_limits<float>::max() || !std::isfinite(gain) || gain < 0 ||
      gain > std::numeric_limits<float>::max()) {
      tools::logger()->error("[HikRobot worker] serial='{}' invalid camera parameters", serial_);
      return false;
    }
    if (exposure != applied_exposure_) {
      if (!set_float("ExposureTime", exposure)) return false;
      applied_exposure_ = exposure;
    }
    if (stopping()) return true;
    if (gain != applied_gain_) {
      if (!set_float("Gain", gain)) return false;
      applied_gain_ = gain;
    }
    return true;
  }

  SharedMemory & shared_;
  std::string serial_;
  void * handle_ = nullptr;
  bool opened_ = false;
  bool grabbing_ = false;
  bool cleanup_allowed_ = true;
  double applied_exposure_ = std::numeric_limits<double>::quiet_NaN();
  double applied_gain_ = std::numeric_limits<double>::quiet_NaN();
};
}  // namespace

int capture_start(SharedMemory & shared, const std::string & serial_number)
{
  CaptureSession session(shared, serial_number);
  int result = kWorkerFaultExit;
  try {
    result = session.run();
  } catch (const std::exception & error) {
    log_capture_exception(serial_number, error.what());
  } catch (...) {
    log_capture_exception(serial_number, "Unknown exception");
  }

  try {
    if (!session.cleanup()) result = kWorkerFaultExit;
  } catch (...) {
    // A failed cleanup is process retirement, never an in-process restart.
    result = kWorkerFaultExit;
  }
  return invoke(shared, Operation::Exiting, [&] {
    tools::logger()->info(
      "[HikRobot worker] serial='{}' exiting with status {}", serial_number, result);
    return result;
  });
}

}  // namespace hikrobot_worker
}  // namespace io
