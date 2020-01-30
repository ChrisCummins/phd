// Copyright (c) 2016-2020 Chris Cummins.
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

#include "gpu/cldrive/profiling_data.h"

#include "labm8/cpp/string.h"
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
}  // namespace cldrive
}  // namespace gpu
