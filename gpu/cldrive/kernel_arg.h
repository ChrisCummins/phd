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
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/status.h"
#include "phd/statusor.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

// A list of supported OpenCL types.
enum OpenClArgType {
  DEFAULT_UNKNOWN,  // Used as default constructor value.
  // Scalar data types. See:
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/scalarDataTypes.html
  BOOL,
  CHAR,
  UCHAR,
  SHORT,
  USHORT,
  INT,
  UINT,
  LONG,
  ULONG,
  FLOAT,
  DOUBLE,
  HALF,
  // Vector data types. See:
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/vectorDataTypes.html
  CHAR2,
  CHAR3,
  CHAR4,
  CHAR8,
  CHAR16,
  UCHAR2,
  UCHAR3,
  UCHAR4,
  UCHAR8,
  UCHAR16,
  SHORT2,
  SHORT3,
  SHORT4,
  SHORT8,
  SHORT16,
  USHORT2,
  USHORT3,
  USHORT4,
  USHORT8,
  USHORT16,
  INT2,
  INT3,
  INT4,
  INT8,
  INT16,
  UINT2,
  UINT3,
  UINT4,
  UINT8,
  UINT16,
  LONG2,
  LONG3,
  LONG4,
  LONG8,
  LONG16,
  ULONG2,
  ULONG3,
  ULONG4,
  ULONG8,
  ULONG16,
  FLOAT2,
  FLOAT3,
  FLOAT4,
  FLOAT8,
  FLOAT16,
  DOUBLE2,
  DOUBLE3,
  DOUBLE4,
  DOUBLE8,
  DOUBLE16,
  //  HALF2,
  //  HALF3,
  //  HALF4,
  //  HALF8,
  //  HALF16
};

// Look up a string type name and return the OpenClArgType. If not found,
// an error status is returned.
phd::StatusOr<OpenClArgType> OpenClArgTypeFromString(const string &type_name);

class KernelArg {
 public:
  KernelArg(cl::Kernel *kernel, size_t arg_index);

  phd::Status Init();

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateRandomValue(
      const cl::Context &context, const DynamicParams &dynamic_params);

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateOnesValue(
      const cl::Context &context, const DynamicParams &dynamic_params);

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

 private:
  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValue(
      const cl::Context &context, const DynamicParams &dynamic_params,
      bool rand_values);

  cl::Kernel *kernel_;
  size_t arg_index_;

  cl_kernel_arg_address_qualifier address_;
  OpenClArgType type_;
  bool is_pointer_;
};

}  // namespace cldrive
}  // namespace gpu
