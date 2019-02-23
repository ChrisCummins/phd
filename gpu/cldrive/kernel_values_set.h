#pragma once

#include "gpu/cldrive/kernel_values.h"

#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelValuesSet {
 public:
  bool operator==(const KernelValuesSet &rhs) const;

  bool operator!=(const KernelValuesSet &rhs) const;

  void CopyToDevice(const cl::CommandQueue &queue,
                    ProfilingData *profiling) const;

  void CopyFromDeviceToNewValueSet(const cl::CommandQueue &queue,
                                   KernelValuesSet *new_values,
                                   ProfilingData *profiling) const;

  void AddKernelValue(std::unique_ptr<KernelValue> value);

  void SetAsArgs(cl::Kernel *kernel);

  void Clear();

  string ToString() const;

 private:
  std::vector<std::unique_ptr<KernelValue>> values_;
};

}  // namespace cldrive
}  // namespace gpu
