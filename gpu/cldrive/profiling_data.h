#pragma once

#include "phd/port.h"

#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

phd::int64 GetElapsedNanoseconds(const cl::Event& event);

class ProfilingData {
 public:
  ProfilingData() : elapsed_nanoseconds(0), transferred_bytes(0) {}
  phd::int64 elapsed_nanoseconds;
  phd::int64 transferred_bytes;
};

}  // namespace cldrive
}  // namespace gpu
