#ifndef IO__HIKROBOT__USB_RESET_HPP
#define IO__HIKROBOT__USB_RESET_HPP

#include <string>

namespace io::hikrobot_worker
{

// Blocking operation: call only in an exec-isolated helper process whose entire
// lifetime is bounded by its supervisor. libusb reset has no timeout argument.
// Returns a libusb status code; LIBUSB_ERROR_NOT_FOUND after reset means that the
// device must be rediscovered, not that reset completion has been confirmed.
int reset_usb(int vid, int pid, const std::string & serial_number);

}  // namespace io::hikrobot_worker

#endif  // IO__HIKROBOT__USB_RESET_HPP