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
#include "gpu/cldrive/kernel_arg.h"

#include "gpu/cldrive/global_memory_arg_value.h"
#include "gpu/cldrive/kernel_arg_util.h"
#include "gpu/cldrive/opencl_type.h"
#include "gpu/cldrive/opencl_util.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "phd/status_macros.h"

#include <cstdlib>

namespace gpu {
namespace cldrive {

phd::Status KernelArg::Init(cl::Kernel* kernel, size_t arg_index) {
  address_ = kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index);
  CHECK(IsGlobal() || IsLocal() || IsConstant() || IsPrivate());

  // Address qualifier is one of:
  //   CL_KERNEL_ARG_ACCESS_READ_ONLY
  //   CL_KERNEL_ARG_ACCESS_WRITE_ONLY
  //   CL_KERNEL_ARG_ACCESS_READ_WRITE
  //   CL_KERNEL_ARG_ACCESS_NONE
  //
  // If argument is not an image type, CL_KERNEL_ARG_ACCESS_NONE is returned.
  // If argument is an image type, the access qualifier specified or the
  // default access qualifier is returned.
  auto access_qualifier =
      kernel->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index);
  if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
    LOG(WARNING) << "Argument " << arg_index << " is an unsupported image type";
    return phd::Status(phd::error::Code::INVALID_ARGUMENT,
                       "Unsupported argument");
  }

  string full_type_name =
      kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
  CHECK(full_type_name.size());

  is_pointer_ = full_type_name[full_type_name.size() - 2] == '*';

  // Strip the trailing '*' on pointer types, and the trailing null char on
  // both.
  string type_name = full_type_name;
  if (is_pointer_) {
    type_name = full_type_name.substr(0, full_type_name.size() - 2);
  } else {
    type_name = full_type_name.substr(0, full_type_name.size() - 1);
  }

  auto type_or = OpenClTypeFromString(type_name);
  if (!type_or.ok()) {
    LOG(ERROR) << "Argument " << arg_index << " of kernel '"
               << GetOpenClKernelName(kernel)
               << "' is of unknown type: " << full_type_name;
    return type_or.status();
  }
  type_ = type_or.ValueOrDie();

  // Check for invalid private pointer arguments.
  if (is_pointer_ && IsPrivate()) {
    LOG(WARNING) << "Pointer to private argument is not allowed";
    return phd::Status(phd::error::Code::INVALID_ARGUMENT,
                       "Unsupported argument");
  }

  return phd::Status::OK;
}

const OpenClType& KernelArg::type() const { return type_; }

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateRandomValue(
    const cl::Context& context, const DynamicParams& dynamic_params) const {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/true);
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateOnesValue(
    const cl::Context& context, const DynamicParams& dynamic_params) const {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/false);
}

bool KernelArg::IsGlobal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL;
}

bool KernelArg::IsLocal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_LOCAL;
}

bool KernelArg::IsConstant() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_CONSTANT;
}

bool KernelArg::IsPrivate() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

bool KernelArg::IsPointer() const { return is_pointer_; }

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateKernelArgValue(
    const cl::Context& context, const DynamicParams& dynamic_params,
    bool rand_values) const {
  CHECK(type() != OpenClType::DEFAULT_UNKNOWN);

  if (IsPointer() && IsGlobal()) {
    return CreateArrayArgValue(type(), context,
                               /*size=*/dynamic_params.global_size_x(),
                               /*value=*/1, rand_values);
  } else if (IsPointer() && IsLocal()) {
    return CreateLocalMemoryValue(type(),
                                  /*size=*/dynamic_params.global_size_x());
  } else if (!IsPointer()) {
    return CreateScalarArgValue(type(),
                                /*value=*/dynamic_params.global_size_x());
  } else {
    return std::unique_ptr<KernelArgValue>(nullptr);
  }
}

}  // namespace cldrive
}  // namespace gpu
