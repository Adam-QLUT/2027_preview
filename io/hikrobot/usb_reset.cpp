#include "usb_reset.hpp"

#include <array>
#include <cstdint>
#include <libusb-1.0/libusb.h>
#include <memory>
#include <utility>

#include "tools/logger.hpp"

namespace io::hikrobot_worker
{
namespace
{
struct ContextDeleter
{
  void operator()(libusb_context * context) const { libusb_exit(context); }
};

struct HandleDeleter
{
  void operator()(libusb_device_handle * handle) const { libusb_close(handle); }
};

struct DeviceList
{
  libusb_device ** devices = nullptr;

  ~DeviceList()
  {
    if (devices != nullptr) libusb_free_device_list(devices, 1);
  }

  DeviceList() = default;
  DeviceList(const DeviceList &) = delete;
  DeviceList & operator=(const DeviceList &) = delete;
};
}  // namespace

int reset_usb(int vid, int pid, const std::string & serial_number)
{
  if (vid <= 0 || vid > 0xffff || pid < 0 || pid > 0xffff || serial_number.empty()) {
    tools::logger()->error(
      "[HikRobot reset] Refusing reset: valid VID/PID and a nonempty serial are required");
    return LIBUSB_ERROR_INVALID_PARAM;
  }

  tools::logger()->info(
    "[HikRobot reset] state=enumerating serial='{}' vid={:#06x} pid={:#06x}",
    serial_number, vid, pid);

  libusb_context * raw_context = nullptr;
  const auto init_ret = libusb_init(&raw_context);
  if (init_ret != LIBUSB_SUCCESS) {
    tools::logger()->error(
      "[HikRobot reset] libusb_init failed: {}", libusb_error_name(init_ret));
    return init_ret;
  }
  std::unique_ptr<libusb_context, ContextDeleter> context(raw_context);

  DeviceList device_list;
  const auto count = libusb_get_device_list(context.get(), &device_list.devices);
  if (count < 0) {
    tools::logger()->error(
      "[HikRobot reset] libusb_get_device_list failed: {}",
      libusb_error_name(static_cast<int>(count)));
    return static_cast<int>(count);
  }

  std::unique_ptr<libusb_device_handle, HandleDeleter> selected;
  unsigned int matches = 0;
  int identification_error = LIBUSB_SUCCESS;

  for (ssize_t i = 0; i < count; ++i) {
    libusb_device_descriptor descriptor{};
    const auto descriptor_ret =
      libusb_get_device_descriptor(device_list.devices[i], &descriptor);
    if (descriptor_ret != LIBUSB_SUCCESS) {
      // Without this descriptor we cannot rule out another matching camera.
      identification_error = descriptor_ret;
      tools::logger()->warn(
        "[HikRobot reset] Unable to identify USB device: {}",
        libusb_error_name(descriptor_ret));
      continue;
    }
    if (
      descriptor.idVendor != static_cast<std::uint16_t>(vid) ||
      descriptor.idProduct != static_cast<std::uint16_t>(pid))
      continue;

    if (descriptor.iSerialNumber == 0) {
      identification_error = LIBUSB_ERROR_NOT_FOUND;
      tools::logger()->warn(
        "[HikRobot reset] A matching VID/PID device has no serial descriptor");
      continue;
    }

    libusb_device_handle * raw_handle = nullptr;
    const auto open_ret = libusb_open(device_list.devices[i], &raw_handle);
    if (open_ret != LIBUSB_SUCCESS) {
      identification_error = open_ret;
      tools::logger()->warn(
        "[HikRobot reset] Cannot open a matching VID/PID device: {}",
        libusb_error_name(open_ret));
      continue;
    }
    std::unique_ptr<libusb_device_handle, HandleDeleter> candidate(raw_handle);

    std::array<unsigned char, 256> serial_buffer{};
    const auto serial_length = libusb_get_string_descriptor_ascii(
      candidate.get(), descriptor.iSerialNumber, serial_buffer.data(),
      static_cast<int>(serial_buffer.size()));
    if (serial_length <= 0) {
      identification_error = serial_length < 0 ? serial_length : LIBUSB_ERROR_IO;
      tools::logger()->warn(
        "[HikRobot reset] Cannot read a matching VID/PID device serial: {}",
        libusb_error_name(identification_error));
      continue;
    }

    const std::string candidate_serial(
      reinterpret_cast<const char *>(serial_buffer.data()),
      static_cast<std::size_t>(serial_length));
    if (candidate_serial != serial_number) continue;

    ++matches;
    if (matches == 1) selected = std::move(candidate);
  }

  // Finish identifying every candidate before doing anything to the device.
  // Never fall back to the first VID/PID match, including for an unreadable SN.
  if (identification_error != LIBUSB_SUCCESS) {
    tools::logger()->error(
      "[HikRobot reset] state=refused serial='{}': device identity is incomplete",
      serial_number);
    return identification_error;
  }
  if (matches != 1) {
    tools::logger()->error(
      "[HikRobot reset] state=refused serial='{}': expected one exact match, found {}",
      serial_number, matches);
    return matches == 0 ? LIBUSB_ERROR_NOT_FOUND : LIBUSB_ERROR_INVALID_PARAM;
  }

  tools::logger()->warn(
    "[HikRobot reset] state=resetting serial='{}': issuing one USB port reset",
    serial_number);
  const auto reset_ret = libusb_reset_device(selected.get());
  if (reset_ret == LIBUSB_SUCCESS) {
    tools::logger()->info(
      "[HikRobot reset] state=completed serial='{}'; reopen with a new SDK handle",
      serial_number);
  } else if (reset_ret == LIBUSB_ERROR_NOT_FOUND) {
    tools::logger()->warn(
      "[HikRobot reset] state=rediscovery_required serial='{}': device disconnected "
      "or must be re-enumerated; reset completion is unconfirmed",
      serial_number);
  } else {
    tools::logger()->error(
      "[HikRobot reset] state=failed serial='{}': {}", serial_number,
      libusb_error_name(reset_ret));
  }
  return reset_ret;
}

}  // namespace io::hikrobot_worker
