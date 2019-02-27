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
#include "gpu/cldrive/opencl_type.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "opencl_type.h"
#include "phd/status.h"
#include "phd/statusor.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArg {
 public:
  KernelArg() : type_(OpenClType::DEFAULT_UNKNOWN) {}

  phd::Status Init(cl::Kernel *kernel, size_t arg_index);

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateRandomValue(
      const cl::Context &context, const DynamicParams &dynamic_params) const;

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateOnesValue(
      const cl::Context &context, const DynamicParams &dynamic_params) const;

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

  const OpenClType &type() const;

 private:
  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValue(
      const cl::Context &context, const DynamicParams &dynamic_params,
      bool rand_values) const;

  OpenClType type_;
  cl_kernel_arg_address_qualifier address_;
  bool is_pointer_;
};

}  // namespace cldrive
}  // namespace gpu
