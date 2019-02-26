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

#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "phd/status_macros.h"

namespace gpu {
namespace cldrive {

phd::StatusOr<OpenClArgType> OpenClArgTypeFromString(const string& type_name) {
  // Scalar types.
  if (!type_name.compare("bool")) {
    return OpenClArgType::BOOL;
  } else if (!type_name.compare("char")) {
    return OpenClArgType::CHAR;
  } else if (!type_name.compare("unsigned char") ||
             !type_name.compare("uchar")) {
    return OpenClArgType::UCHAR;
  } else if (!type_name.compare("short")) {
    return OpenClArgType::SHORT;
  } else if (!type_name.compare("unsigned short") ||
             !type_name.compare("ushort")) {
    return OpenClArgType::USHORT;
  } else if (!type_name.compare("int")) {
    return OpenClArgType::INT;
  } else if (!type_name.compare("unsigned int") || !type_name.compare("uint")) {
    return OpenClArgType::UINT;
  } else if (!type_name.compare("long")) {
    return OpenClArgType::LONG;
  } else if (!type_name.compare("unsigned long") ||
             !type_name.compare("ulong")) {
    return OpenClArgType::ULONG;
  } else if (!type_name.compare("float")) {
    return OpenClArgType::FLOAT;
  } else if (!type_name.compare("double")) {
    return OpenClArgType::DOUBLE;
  } else if (!type_name.compare("half")) {
    return OpenClArgType::HALF;
    // Vector types.
  } else if (!type_name.compare("char2")) {
    return OpenClArgType::CHAR2;
  } else if (!type_name.compare("char3")) {
    return OpenClArgType::CHAR3;
  } else if (!type_name.compare("char4")) {
    return OpenClArgType::CHAR4;
  } else if (!type_name.compare("char8")) {
    return OpenClArgType::CHAR8;
  } else if (!type_name.compare("char16")) {
    return OpenClArgType::CHAR16;
  } else if (!type_name.compare("uchar2")) {
    return OpenClArgType::UCHAR2;
  } else if (!type_name.compare("uchar3")) {
    return OpenClArgType::UCHAR3;
  } else if (!type_name.compare("uchar4")) {
    return OpenClArgType::UCHAR4;
  } else if (!type_name.compare("uchar8")) {
    return OpenClArgType::UCHAR8;
  } else if (!type_name.compare("uchar16")) {
    return OpenClArgType::UCHAR16;
  } else if (!type_name.compare("short2")) {
    return OpenClArgType::SHORT2;
  } else if (!type_name.compare("short3")) {
    return OpenClArgType::SHORT3;
  } else if (!type_name.compare("short4")) {
    return OpenClArgType::SHORT4;
  } else if (!type_name.compare("short8")) {
    return OpenClArgType::SHORT8;
  } else if (!type_name.compare("short16")) {
    return OpenClArgType::SHORT16;
  } else if (!type_name.compare("ushort2")) {
    return OpenClArgType::USHORT2;
  } else if (!type_name.compare("ushort3")) {
    return OpenClArgType::USHORT3;
  } else if (!type_name.compare("ushort4")) {
    return OpenClArgType::USHORT4;
  } else if (!type_name.compare("ushort8")) {
    return OpenClArgType::USHORT8;
  } else if (!type_name.compare("ushort16")) {
    return OpenClArgType::USHORT16;
  } else if (!type_name.compare("int2")) {
    return OpenClArgType::INT2;
  } else if (!type_name.compare("int3")) {
    return OpenClArgType::INT3;
  } else if (!type_name.compare("int4")) {
    return OpenClArgType::INT4;
  } else if (!type_name.compare("int8")) {
    return OpenClArgType::INT8;
  } else if (!type_name.compare("int16")) {
    return OpenClArgType::INT16;
  } else if (!type_name.compare("uint2")) {
    return OpenClArgType::UINT2;
  } else if (!type_name.compare("uint3")) {
    return OpenClArgType::UINT3;
  } else if (!type_name.compare("uint4")) {
    return OpenClArgType::UINT4;
  } else if (!type_name.compare("uint8")) {
    return OpenClArgType::UINT8;
  } else if (!type_name.compare("uint16")) {
    return OpenClArgType::UINT16;
  } else if (!type_name.compare("long2")) {
    return OpenClArgType::LONG2;
  } else if (!type_name.compare("long3")) {
    return OpenClArgType::LONG3;
  } else if (!type_name.compare("long4")) {
    return OpenClArgType::LONG4;
  } else if (!type_name.compare("long8")) {
    return OpenClArgType::LONG8;
  } else if (!type_name.compare("long16")) {
    return OpenClArgType::LONG16;
  } else if (!type_name.compare("ulong2")) {
    return OpenClArgType::ULONG2;
  } else if (!type_name.compare("ulong3")) {
    return OpenClArgType::ULONG3;
  } else if (!type_name.compare("ulong4")) {
    return OpenClArgType::ULONG4;
  } else if (!type_name.compare("ulong8")) {
    return OpenClArgType::ULONG8;
  } else if (!type_name.compare("ulong16")) {
    return OpenClArgType::ULONG16;
  } else if (!type_name.compare("float2")) {
    return OpenClArgType::FLOAT2;
  } else if (!type_name.compare("float3")) {
    return OpenClArgType::FLOAT3;
  } else if (!type_name.compare("float4")) {
    return OpenClArgType::FLOAT4;
  } else if (!type_name.compare("float8")) {
    return OpenClArgType::FLOAT8;
  } else if (!type_name.compare("float16")) {
    return OpenClArgType::FLOAT16;
  } else if (!type_name.compare("double2")) {
    return OpenClArgType::DOUBLE2;
  } else if (!type_name.compare("double3")) {
    return OpenClArgType::DOUBLE3;
  } else if (!type_name.compare("double4")) {
    return OpenClArgType::DOUBLE4;
  } else if (!type_name.compare("double8")) {
    return OpenClArgType::DOUBLE8;
  } else if (!type_name.compare("double16")) {
    return OpenClArgType::DOUBLE16;
    //  } else if (!type_name.compare("half2")) {
    //    return OpenClArgType::HALF2;
    //  } else if (!type_name.compare("half3")) {
    //    return OpenClArgType::HALF3;
    //  } else if (!type_name.compare("half4")) {
    //    return OpenClArgType::HALF4;
    //  } else if (!type_name.compare("half8")) {
    //    return OpenClArgType::HALF8;
    //  } else if (!type_name.compare("half16")) {
    //    return OpenClArgType::HALF16;
  } else {
    return phd::Status(phd::error::Code::INVALID_ARGUMENT, type_name);
  }
}

KernelArg::KernelArg(cl::Kernel* kernel, size_t arg_index)
    : kernel_(kernel),
      arg_index_(arg_index),
      address_(kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index)),
      type_(OpenClArgType::DEFAULT_UNKNOWN) {
  CHECK(IsGlobal() || IsLocal() || IsConstant() || IsPrivate());
}

phd::Status KernelArg::Init() {
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
      kernel_->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index_);
  if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
    LOG(ERROR) << "Argument " << arg_index_ << " is an unsupported image type";
    return phd::Status::UNKNOWN;
  }

  string full_type_name =
      kernel_->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index_);
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

  auto type_or = OpenClArgTypeFromString(type_name);
  if (!type_or.ok()) {
    LOG(ERROR) << "Argument " << arg_index_ << " of kernel '"
               << kernel_->getInfo<CL_KERNEL_FUNCTION_NAME>()
               << "' is of unknown type: " << full_type_name;
    return type_or.status();
  }
  type_ = type_or.ValueOrDie();

  return phd::Status::OK;
}

namespace {

template <typename T>
std::unique_ptr<ArrayKernelArgValueWithBuffer<T>> CreateArrayArgValue(
    size_t size, bool rand_values, const cl::Context& context) {
  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<T>>(
      context, size, /*value=*/static_cast<T>(1));
  if (rand_values) {
    for (size_t i = 0; i < size; ++i) {
      arg_value->vector()[i] = rand();
    }
  }
  return arg_value;
}

// FIXME:
template <>
std::unique_ptr<ArrayKernelArgValueWithBuffer<cl_char2>> CreateArrayArgValue(
    size_t size, bool rand_values, const cl::Context& context) {
  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<cl_char2>>(
      context, size /*, value={1, 1}*/);
  if (rand_values) {
    // for (size_t i = 0; i < size; ++i) {
    // arg_value->vector()[i] = rand();
    //}
  }
  return arg_value;
}

std::unique_ptr<KernelArgValue> CreateArrayArgValue(
    const OpenClArgType& type, size_t size, bool rand_values,
    const cl::Context& context) {
  switch (type) {
    // TODO(cec): Fix compilation when bool enabled:
    // case OpenClArgType::BOOL: {
    //   return CreateArrayArgValue<bool>(size, rand_values, context);
    // }
    case OpenClArgType::CHAR: {
      return CreateArrayArgValue<cl_char>(size, rand_values, context);
    }
    case OpenClArgType::UCHAR: {
      return CreateArrayArgValue<cl_uchar>(size, rand_values, context);
    }
    case OpenClArgType::SHORT: {
      return CreateArrayArgValue<cl_short>(size, rand_values, context);
    }
    case OpenClArgType::USHORT: {
      return CreateArrayArgValue<cl_ushort>(size, rand_values, context);
    }
    case OpenClArgType::INT: {
      return CreateArrayArgValue<cl_int>(size, rand_values, context);
    }
    case OpenClArgType::UINT: {
      return CreateArrayArgValue<cl_uint>(size, rand_values, context);
    }
    case OpenClArgType::LONG: {
      return CreateArrayArgValue<cl_long>(size, rand_values, context);
    }
    case OpenClArgType::ULONG: {
      return CreateArrayArgValue<cl_ulong>(size, rand_values, context);
    }
    case OpenClArgType::FLOAT: {
      return CreateArrayArgValue<cl_float>(size, rand_values, context);
    }
    case OpenClArgType::DOUBLE: {
      return CreateArrayArgValue<cl_double>(size, rand_values, context);
    }
    case OpenClArgType::HALF: {
      return CreateArrayArgValue<cl_half>(size, rand_values, context);
    }
      //    case OpenClArgType::CHAR2: {
      //      return CreateArrayArgValue<cl_char2>(size, rand_values, context);
      //    }

      //    case OpenClArgType::CHAR3: {
      //      return CreateArrayArgValue<cl_char3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::CHAR4: {
      //      return CreateArrayArgValue<cl_char4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::CHAR8: {
      //      return CreateArrayArgValue<cl_char8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::CHAR16: {
      //      return CreateArrayArgValue<cl_char16>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UCHAR2: {
      //      return CreateArrayArgValue<cl_uchar2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UCHAR3: {
      //      return CreateArrayArgValue<cl_uchar3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UCHAR4: {
      //      return CreateArrayArgValue<cl_uchar4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UCHAR8: {
      //      return CreateArrayArgValue<cl_uchar8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UCHAR16: {
      //      return CreateArrayArgValue<cl_uchar16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::SHORT2: {
      //      return CreateArrayArgValue<cl_short2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::SHORT3: {
      //      return CreateArrayArgValue<cl_short3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::SHORT4: {
      //      return CreateArrayArgValue<cl_short4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::SHORT8: {
      //      return CreateArrayArgValue<cl_short8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::SHORT16: {
      //      return CreateArrayArgValue<cl_short16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::USHORT2: {
      //      return CreateArrayArgValue<cl_ushort2>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::USHORT3: {
      //      return CreateArrayArgValue<cl_ushort3>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::USHORT4: {
      //      return CreateArrayArgValue<cl_ushort4>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::USHORT8: {
      //      return CreateArrayArgValue<cl_ushort8>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::USHORT16: {
      //      return CreateArrayArgValue<cl_ushort16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::INT2: {
      //      return CreateArrayArgValue<cl_int2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::INT3: {
      //      return CreateArrayArgValue<cl_int3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::INT4: {
      //      return CreateArrayArgValue<cl_int4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::INT8: {
      //      return CreateArrayArgValue<cl_int8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::INT16: {
      //      return CreateArrayArgValue<cl_int16>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UINT2: {
      //      return CreateArrayArgValue<cl_uint2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UINT3: {
      //      return CreateArrayArgValue<cl_uint3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UINT4: {
      //      return CreateArrayArgValue<cl_uint4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UINT8: {
      //      return CreateArrayArgValue<cl_uint8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::UINT16: {
      //      return CreateArrayArgValue<cl_uint16>(size, rand_values, context);
      //    }
      //    case OpenClArgType::LONG2: {
      //      return CreateArrayArgValue<cl_long2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::LONG3: {
      //      return CreateArrayArgValue<cl_long3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::LONG4: {
      //      return CreateArrayArgValue<cl_long4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::LONG8: {
      //      return CreateArrayArgValue<cl_long8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::LONG16: {
      //      return CreateArrayArgValue<cl_long16>(size, rand_values, context);
      //    }
      //    case OpenClArgType::ULONG2: {
      //      return CreateArrayArgValue<cl_ulong2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::ULONG3: {
      //      return CreateArrayArgValue<cl_ulong3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::ULONG4: {
      //      return CreateArrayArgValue<cl_ulong4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::ULONG8: {
      //      return CreateArrayArgValue<cl_ulong8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::ULONG16: {
      //      return CreateArrayArgValue<cl_ulong16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::FLOAT2: {
      //      return CreateArrayArgValue<cl_float2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::FLOAT3: {
      //      return CreateArrayArgValue<cl_float3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::FLOAT4: {
      //      return CreateArrayArgValue<cl_float4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::FLOAT8: {
      //      return CreateArrayArgValue<cl_float8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::FLOAT16: {
      //      return CreateArrayArgValue<cl_float16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::DOUBLE2: {
      //      return CreateArrayArgValue<cl_double2>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::DOUBLE3: {
      //      return CreateArrayArgValue<cl_double3>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::DOUBLE4: {
      //      return CreateArrayArgValue<cl_double4>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::DOUBLE8: {
      //      return CreateArrayArgValue<cl_double8>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::DOUBLE16: {
      //      return CreateArrayArgValue<cl_double16>(size, rand_values,
      //      context);
      //    }
      //    case OpenClArgType::HALF2: {
      //      return CreateArrayArgValue<cl_half2>(size, rand_values, context);
      //    }
      //    case OpenClArgType::HALF3: {
      //      return CreateArrayArgValue<cl_half3>(size, rand_values, context);
      //    }
      //    case OpenClArgType::HALF4: {
      //      return CreateArrayArgValue<cl_half4>(size, rand_values, context);
      //    }
      //    case OpenClArgType::HALF8: {
      //      return CreateArrayArgValue<cl_half8>(size, rand_values, context);
      //    }
      //    case OpenClArgType::HALF16: {
      //      return CreateArrayArgValue<cl_half16>(size, rand_values, context);
      //    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClArgType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

template <typename T, typename Y>
std::unique_ptr<KernelArgValue> CreateScalarArgValue(const Y& value) {
  return std::make_unique<ScalarKernelArgValue<T>>(static_cast<T>(value));
}

std::unique_ptr<KernelArgValue> CreateScalarArgValue(const OpenClArgType& type,
                                                     const size_t value) {
  switch (type) {
    case OpenClArgType::BOOL: {
      return CreateScalarArgValue<bool>(value);
    }
    case OpenClArgType::CHAR: {
      return CreateScalarArgValue<cl_char>(value);
    }
    case OpenClArgType::UCHAR: {
      return CreateScalarArgValue<cl_uchar>(value);
    }
    case OpenClArgType::SHORT: {
      return CreateScalarArgValue<cl_short>(value);
    }
    case OpenClArgType::USHORT: {
      return CreateScalarArgValue<cl_ushort>(value);
    }
    case OpenClArgType::INT: {
      return CreateScalarArgValue<cl_int>(value);
    }
    case OpenClArgType::UINT: {
      return CreateScalarArgValue<cl_uint>(value);
    }
    case OpenClArgType::LONG: {
      return CreateScalarArgValue<cl_long>(value);
    }
    case OpenClArgType::ULONG: {
      return CreateScalarArgValue<cl_ulong>(value);
    }
    case OpenClArgType::FLOAT: {
      return CreateScalarArgValue<cl_float>(value);
    }
    case OpenClArgType::DOUBLE: {
      return CreateScalarArgValue<cl_double>(value);
    }
    case OpenClArgType::HALF: {
      return CreateScalarArgValue<cl_half>(value);
    }
      //    case OpenClArgType::CHAR2: {
      //      return CreateScalarArgValue<cl_char2>(value);
      //    }
      //    case OpenClArgType::CHAR3: {
      //      return CreateScalarArgValue<cl_char3>(value);
      //    }
      //    case OpenClArgType::CHAR4: {
      //      return CreateScalarArgValue<cl_char4>(value);
      //    }
      //    case OpenClArgType::CHAR8: {
      //      return CreateScalarArgValue<cl_char8>(value);
      //    }
      //    case OpenClArgType::CHAR16: {
      //      return CreateScalarArgValue<cl_char16>(value);
      //    }
      //    case OpenClArgType::UCHAR2: {
      //      return CreateScalarArgValue<cl_uchar2>(value);
      //    }
      //    case OpenClArgType::UCHAR3: {
      //      return CreateScalarArgValue<cl_uchar3>(value);
      //    }
      //    case OpenClArgType::UCHAR4: {
      //      return CreateScalarArgValue<cl_uchar4>(value);
      //    }
      //    case OpenClArgType::UCHAR8: {
      //      return CreateScalarArgValue<cl_uchar8>(value);
      //    }
      //    case OpenClArgType::UCHAR16: {
      //      return CreateScalarArgValue<cl_uchar16>(value);
      //    }
      //    case OpenClArgType::SHORT2: {
      //      return CreateScalarArgValue<cl_short2>(value);
      //    }
      //    case OpenClArgType::SHORT3: {
      //      return CreateScalarArgValue<cl_short3>(value);
      //    }
      //    case OpenClArgType::SHORT4: {
      //      return CreateScalarArgValue<cl_short4>(value);
      //    }
      //    case OpenClArgType::SHORT8: {
      //      return CreateScalarArgValue<cl_short8>(value);
      //    }
      //    case OpenClArgType::SHORT16: {
      //      return CreateScalarArgValue<cl_short16>(value);
      //    }
      //    case OpenClArgType::USHORT2: {
      //      return CreateScalarArgValue<cl_ushort2>(value);
      //    }
      //    case OpenClArgType::USHORT3: {
      //      return CreateScalarArgValue<cl_ushort3>(value);
      //    }
      //    case OpenClArgType::USHORT4: {
      //      return CreateScalarArgValue<cl_ushort4>(value);
      //    }
      //    case OpenClArgType::USHORT8: {
      //      return CreateScalarArgValue<cl_ushort8>(value);
      //    }
      //    case OpenClArgType::USHORT16: {
      //      return CreateScalarArgValue<cl_ushort16>(value);
      //    }
      //    case OpenClArgType::INT2: {
      //      return CreateScalarArgValue<cl_int2>(value);
      //    }
      //    case OpenClArgType::INT3: {
      //      return CreateScalarArgValue<cl_int3>(value);
      //    }
      //    case OpenClArgType::INT4: {
      //      return CreateScalarArgValue<cl_int4>(value);
      //    }
      //    case OpenClArgType::INT8: {
      //      return CreateScalarArgValue<cl_int8>(value);
      //    }
      //    case OpenClArgType::INT16: {
      //      return CreateScalarArgValue<cl_int16>(value);
      //    }
      //    case OpenClArgType::UINT2: {
      //      return CreateScalarArgValue<cl_uint2>(value);
      //    }
      //    case OpenClArgType::UINT3: {
      //      return CreateScalarArgValue<cl_uint3>(value);
      //    }
      //    case OpenClArgType::UINT4: {
      //      return CreateScalarArgValue<cl_uint4>(value);
      //    }
      //    case OpenClArgType::UINT8: {
      //      return CreateScalarArgValue<cl_uint8>(value);
      //    }
      //    case OpenClArgType::UINT16: {
      //      return CreateScalarArgValue<cl_uint16>(value);
      //    }
      //    case OpenClArgType::LONG2: {
      //      return CreateScalarArgValue<cl_long2>(value);
      //    }
      //    case OpenClArgType::LONG3: {
      //      return CreateScalarArgValue<cl_long3>(value);
      //    }
      //    case OpenClArgType::LONG4: {
      //      return CreateScalarArgValue<cl_long4>(value);
      //    }
      //    case OpenClArgType::LONG8: {
      //      return CreateScalarArgValue<cl_long8>(value);
      //    }
      //    case OpenClArgType::LONG16: {
      //      return CreateScalarArgValue<cl_long16>(value);
      //    }
      //    case OpenClArgType::ULONG2: {
      //      return CreateScalarArgValue<cl_ulong2>(value);
      //    }
      //    case OpenClArgType::ULONG3: {
      //      return CreateScalarArgValue<cl_ulong3>(value);
      //    }
      //    case OpenClArgType::ULONG4: {
      //      return CreateScalarArgValue<cl_ulong4>(value);
      //    }
      //    case OpenClArgType::ULONG8: {
      //      return CreateScalarArgValue<cl_ulong8>(value);
      //    }
      //    case OpenClArgType::ULONG16: {
      //      return CreateScalarArgValue<cl_ulong16>(value);
      //    }
      //    case OpenClArgType::FLOAT2: {
      //      return CreateScalarArgValue<cl_float2>(value);
      //    }
      //    case OpenClArgType::FLOAT3: {
      //      return CreateScalarArgValue<cl_float3>(value);
      //    }
      //    case OpenClArgType::FLOAT4: {
      //      return CreateScalarArgValue<cl_float4>(value);
      //    }
      //    case OpenClArgType::FLOAT8: {
      //      return CreateScalarArgValue<cl_float8>(value);
      //    }
      //    case OpenClArgType::FLOAT16: {
      //      return CreateScalarArgValue<cl_float16>(value);
      //    }
      //    case OpenClArgType::DOUBLE2: {
      //      return CreateScalarArgValue<cl_double2>(value);
      //    }
      //    case OpenClArgType::DOUBLE3: {
      //      return CreateScalarArgValue<cl_double3>(value);
      //    }
      //    case OpenClArgType::DOUBLE4: {
      //      return CreateScalarArgValue<cl_double4>(value);
      //    }
      //    case OpenClArgType::DOUBLE8: {
      //      return CreateScalarArgValue<cl_double8>(value);
      //    }
      //    case OpenClArgType::DOUBLE16: {
      //      return CreateScalarArgValue<cl_double16>(value);
      //    }
      //    case OpenClArgType::HALF2: {
      //      return CreateScalarArgValue<cl_half2>(value);
      //    }
      //    case OpenClArgType::HALF3: {
      //      return CreateScalarArgValue<cl_half3>(value);
      //    }
      //    case OpenClArgType::HALF4: {
      //      return CreateScalarArgValue<cl_half4>(value);
      //    }
      //    case OpenClArgType::HALF8: {
      //      return CreateScalarArgValue<cl_half8>(value);
      //    }
      //    case OpenClArgType::HALF16: {
      //      return CreateScalarArgValue<cl_half16>(value);
      //    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClArgType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

}  // anonymous namespace

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateKernelArgValue(
    const cl::Context& context, const DynamicParams& dynamic_params,
    bool rand_values) {
  CHECK(type_ != OpenClArgType::DEFAULT_UNKNOWN) << "Init() not called";
  if (IsPointer() && IsGlobal()) {
    return CreateArrayArgValue(type_, dynamic_params.global_size_x(),
                               rand_values, context);
  } else if (!IsPointer()) {
    return CreateScalarArgValue(type_, dynamic_params.global_size_x());
  } else {
    return std::unique_ptr<KernelArgValue>(nullptr);
  }
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateRandomValue(
    const cl::Context& context, const DynamicParams& dynamic_params) {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/true);
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateOnesValue(
    const cl::Context& context, const DynamicParams& dynamic_params) {
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

}  // namespace cldrive
}  // namespace gpu
