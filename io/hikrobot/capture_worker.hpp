#ifndef IO__HIKROBOT__CAPTURE_WORKER_HPP
#define IO__HIKROBOT__CAPTURE_WORKER_HPP

#include <string>

#include "worker_protocol.hpp"

namespace io
{
namespace hikrobot_worker
{

// Called only by the separately exec'd worker. The supervisor applies deadlines
// to invoke() operations and retires the complete process when an SDK call hangs.
int capture_start(SharedMemory & shared, const std::string & serial_number);

}  // namespace hikrobot_worker
}  // namespace io

#endif  // IO__HIKROBOT__CAPTURE_WORKER_HPP
