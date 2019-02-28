#pragma once

#include "gpu/cldrive/kernel_arg_value.h"

#include "gpu/cldrive/profiling_data.h"

#include "phd/string.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

template <typename T>
class LocalMemoryArgValue : public KernelArgValue {
 public:
  LocalMemoryArgValue(size_t size) : size_(size){};

  virtual bool operator==(const KernelArgValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const LocalMemoryArgValue *const>(rhs);
    // We can't check the concrete values of local memory.
    return rhs_ptr->SizeInBytes() == SizeInBytes();
  }

  virtual bool operator!=(const KernelArgValue *const rhs) const override {
    return !(*this == rhs);
  };

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, SizeInBytes(), nullptr);
  };

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override{};

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    return std::make_unique<LocalMemoryArgValue<T>>(size_);
  }

  virtual string ToString() const override { return "[local memory]"; }

  virtual size_t SizeInBytes() const override { return sizeof(T) * size_; }

 private:
  const size_t size_;
};
}
}
