#pragma once

#include "gpu/cldrive/opencl_util.h"
#include "gpu/cldrive/profiling_data.h"
#include "third_party/opencl/include/cl.hpp"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// Abstract base class.
class KernelArgValue {
 public:
  virtual ~KernelArgValue(){};

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) = 0;

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) = 0;

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) = 0;

  virtual bool operator==(const KernelArgValue *const rhs) const = 0;

  virtual bool operator!=(const KernelArgValue *const rhs) const = 0;

  virtual string ToString() const = 0;
};

}  // namespace cldrive
}  // namespace gpu
