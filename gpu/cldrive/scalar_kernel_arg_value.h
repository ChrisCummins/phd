#pragma once

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// A scalar argument.
template <typename T>
class ScalarKernelArgValue : public KernelArgValue {
 public:
  ScalarKernelArgValue(const T &value) : value_(value) {}

  virtual bool operator==(const KernelArgValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ScalarKernelArgValue *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    return value() == rhs_ptr->value();
  }

  virtual bool operator!=(const KernelArgValue *const rhs) const override {
    return !(*this == rhs);
  };

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, value());
  };

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override{};

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    return std::make_unique<ScalarKernelArgValue>(value());
  }

  const T &value() const { return value_; }
  T &value() { return value_; }

  virtual string ToString() const override;

 private:
  T value_;
};

}  // namespace cldrive
}  // namespace gpu
