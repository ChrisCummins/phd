#pragma once

#include "gpu/cldrive/profiling_data.h"
#include "third_party/opencl/include/cl.hpp"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
template <typename IteratorType>
void CopyHostToDevice(const cl::CommandQueue &queue, IteratorType startIterator,
                      IteratorType endIterator, const cl::Buffer &buffer,
                      ProfilingData *profiling) {
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;
  size_t length = endIterator - startIterator;
  size_t byteLength = length * sizeof(ValueType);

  cl::Event event;
  ValueType *pointer = static_cast<ValueType *>(queue.enqueueMapBuffer(
      buffer, /*blocking=*/true, /*flags=*/CL_MAP_WRITE, /*offset=*/0,
      /*size=*/byteLength, /*events=*/nullptr, /*event=*/&event,
      /*error=*/nullptr));
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  std::copy(startIterator, endIterator, pointer);

  queue.enqueueUnmapMemObject(buffer, pointer, /*events=*/nullptr, &event);

  // Set profiling data.
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += byteLength;
}

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
template <typename IteratorType>
void CopyDeviceToHost(const cl::CommandQueue &queue, const cl::Buffer &buffer,
                      IteratorType startIterator,
                      const IteratorType endIterator,
                      ProfilingData *profiling) {
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;
  size_t length = endIterator - startIterator;
  size_t byteLength = length * sizeof(ValueType);

  cl::Event event;
  ValueType *pointer = static_cast<ValueType *>(queue.enqueueMapBuffer(
      buffer, /*blocking=*/true, /*flags=*/CL_MAP_READ, /*offset=*/0,
      /*size=*/byteLength, /*events=*/nullptr, /*event=*/&event,
      /*error=*/nullptr));
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  std::copy(pointer, pointer + length, startIterator);

  queue.enqueueUnmapMemObject(buffer, pointer, /*events=*/nullptr, &event);

  // Set profiling data.
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += byteLength;
}

// Abstract base class.
class KernelValue {
 public:
  virtual ~KernelValue(){};

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) = 0;

  virtual std::unique_ptr<KernelValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) = 0;

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) = 0;

  virtual bool operator==(const KernelValue *const rhs) const = 0;

  virtual bool operator!=(const KernelValue *const rhs) const = 0;

  virtual string ToString() const = 0;
};

// An array argument.
template <typename T>
class ArrayValue : public KernelValue {
 public:
  template <typename... Args>
  ArrayValue(size_t size, Args &&... args) : vector_(size, args...) {}

  virtual bool operator==(const KernelValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ArrayValue *const>(rhs);
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

  virtual bool operator!=(const KernelValue *const rhs) const override {
    return !(*this == rhs);
  }

  std::vector<T> &vector() { return vector_; }

  const std::vector<T> &vector() const { return vector_; }

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override {
    CHECK(false);
  }

  virtual std::unique_ptr<KernelValue> CopyFromDevice(
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
class ArrayValueWithBuffer : public ArrayValue<T> {
 public:
  template <typename... Args>
  ArrayValueWithBuffer(const cl::Context &context, size_t size, Args &&... args)
      : ArrayValue<T>(size, args...),
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

  virtual std::unique_ptr<KernelValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    auto new_arg = std::make_unique<ArrayValue<T>>(this->vector().size());
    CopyDeviceToHost(queue, buffer(), new_arg->vector().begin(),
                     new_arg->vector().end(), profiling);
    return std::move(new_arg);
  }

 private:
  cl::Buffer buffer_;
};

// A scalar argument.
template <typename T>
class ScalarKernelArg : public KernelValue {
 public:
  ScalarKernelArg(const T &value) : value_(value) {}

  virtual bool operator==(const KernelValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ScalarKernelArg *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    return value() == rhs_ptr->value();
  }

  virtual bool operator!=(const KernelValue *const rhs) const override {
    return !(*this == rhs);
  };

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, value());
  };

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override{};

  virtual std::unique_ptr<KernelValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) override {
    return std::make_unique<ScalarKernelArg>(value());
  }

  const T &value() const { return value_; }
  T &value() { return value_; }

  virtual string ToString() const override {
    return absl::StrFormat("%d", value_);
  }

 private:
  T value_;
};

}  // namespace cldrive
}  // namespace gpu
