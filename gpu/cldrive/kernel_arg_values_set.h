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
