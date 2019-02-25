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
