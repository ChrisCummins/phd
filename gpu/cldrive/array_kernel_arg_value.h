#pragma once

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// An array argument.
template <typename T>
class ArrayKernelArgValue : public KernelArgValue {
 public:
  template <typename... Args>
  ArrayKernelArgValue(size_t size, Args &&... args) : vector_(size, args...) {}

  virtual bool operator==(const KernelArgValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ArrayKernelArgValue *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    if (vector().size() != rhs_ptr->vector().size()) {
      return false;
    }

    for (size_t i = 0; i < vector().size(); ++i) {
      if (vector()[i] != rhs_ptr->vector()[i]) {
        return false;
      }
    }

    return true;
  }

  virtual bool operator!=(const KernelArgValue *const rhs) const override {
    return !(*this == rhs);
  }

  std::vector<T> &vector() { return vector_; }

  const std::vector<T> &vector() const { return vector_; }

  size_t size() const { return vector().size(); }

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override {
    CHECK(false);
  }

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    CHECK(false);
  }

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    CHECK(false);
  }

  virtual string ToString() const override {
    string s = "";
    for (auto &value : vector()) {
      absl::StrAppend(&s, value);
      absl::StrAppend(&s, ", ");
    }
    return s;
  }

 protected:
  std::vector<T> vector_;
};

// An array value with a device-side buffer.
template <typename T>
class ArrayKernelArgValueWithBuffer : public ArrayKernelArgValue<T> {
 public:
  template <typename... Args>
  ArrayKernelArgValueWithBuffer(const cl::Context &context, size_t size,
                                Args &&... args)
      : ArrayKernelArgValue<T>(size, args...),
        buffer_(context, /*flags=*/CL_MEM_READ_WRITE,
                /*size=*/sizeof(T) * size) {}

  cl::Buffer &buffer() { return buffer_; }

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, buffer());
  }

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override {
    CopyHostToDevice(queue, this->vector().begin(), this->vector().end(),
                     buffer(), profiling);
  }

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    auto new_arg =
        std::make_unique<ArrayKernelArgValue<T>>(this->vector().size());
    CopyDeviceToHost(queue, buffer(), new_arg->vector().begin(),
                     new_arg->vector().end(), profiling);
    return std::move(new_arg);
  }

 private:
  cl::Buffer buffer_;
};

}  // namespace cldrive
}  // namespace gpu
