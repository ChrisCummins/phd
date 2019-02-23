#pragma once

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArgValuesSet {
 public:
  bool operator==(const KernelArgValuesSet &rhs) const;

  bool operator!=(const KernelArgValuesSet &rhs) const;

  void CopyToDevice(const cl::CommandQueue &queue,
                    ProfilingData *profiling) const;

  void CopyFromDeviceToNewValueSet(const cl::CommandQueue &queue,
                                   KernelArgValuesSet *new_values,
                                   ProfilingData *profiling) const;

  void AddKernelArgValue(std::unique_ptr<KernelArgValue> value);

  void SetAsArgs(cl::Kernel *kernel);

  void Clear();

  string ToString() const;

 private:
  std::vector<std::unique_ptr<KernelArgValue>> values_;
};

}  // namespace cldrive
}  // namespace gpu
