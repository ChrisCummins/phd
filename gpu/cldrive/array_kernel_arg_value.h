// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#pragma once

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// TODO(cldrive): Work in progress!
template <typename T>
bool ScalarEquality(const T &lhs, const T &rhs) {
  return lhs == rhs;
}

// An array argument.
template <typename T>
class ArrayKernelArgValue : public KernelArgValue {
 public:
  template <typename... Args>
  ArrayKernelArgValue(size_t size, Args &&... args) : vector_(size, args...) {}

  virtual bool operator==(const KernelArgValue *const rhs) const override {
    auto array_ptr = dynamic_cast<const ArrayKernelArgValue *const>(rhs);
    if (!array_ptr) {
      return false;
    }

    if (vector().size() != array_ptr->vector().size()) {
      return false;
    }

    for (size_t i = 0; i < vector().size(); ++i) {
      if (!ElementEquality(vector()[i], array_ptr->vector()[i])) {
        return false;
      }
    }

    return true;
  }

  bool ElementEquality(const T &lhs, const T &rhs) const {
    return ScalarEquality(lhs, rhs);
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
    return std::unique_ptr<KernelArgValue>(nullptr);
  }

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    CHECK(false);
  }

  virtual string ToString() const override;

 protected:
  std::vector<T> vector_;
};

// TODO(cldrive): Work in progress!
template <typename T>
/*virtual*/ string ArrayKernelArgValue<T>::ToString() const {
  string s = "";
  for (auto &value : vector()) {
    absl::StrAppend(&s, value);
    absl::StrAppend(&s, ", ");
  }
  return s;
}

template <>
/*virtual*/ string ArrayKernelArgValue<cl_char2>::ToString() const;

template <>
/*virtual*/ bool ArrayKernelArgValue<cl_char2>::ElementEquality(
    const cl_char2 &lhs, const cl_char2 &rhs) const;

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
    auto new_arg = std::make_unique<ArrayKernelArgValue<T>>(this->size());
    CopyDeviceToHost(queue, buffer(), new_arg->vector().begin(),
                     new_arg->vector().end(), profiling);
    return std::move(new_arg);
  }

 private:
  cl::Buffer buffer_;
};

}  // namespace cldrive
}  // namespace gpu
