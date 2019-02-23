#include "gpu/cldrive/profiling_data.h"

namespace gpu {
namespace cldrive {

phd::int64 GetElapsedNanoseconds(const cl::Event& event) {
  event.wait();
  cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
  cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return end - start;
}

}  // namespace cldrive
}  // namespace gpu
