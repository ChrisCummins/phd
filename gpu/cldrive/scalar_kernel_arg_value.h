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

#include "opencl_type.h"
#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// A scalar argument.
template <typename T>
class ScalarKernelArgValue : public KernelArgValue {
 public:
  ScalarKernelArgValue(const OpenClType &type, const T &value)
      : KernelArgValue(type), value_(value) {}

  virtual bool operator==(const KernelArgValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ScalarKernelArgValue *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    return type().ElementsAreEqual(value(), rhs_ptr->value());
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

  virtual string ToString() const override {
    return type().FormatToString(value());
  }

  const T &value() const { return value_; }
  T &value() { return value_; }

 private:
  T value_;
};

}  // namespace cldrive
}  // namespace gpu
