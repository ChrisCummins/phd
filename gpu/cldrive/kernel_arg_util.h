#pragma once

#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "gpu/cldrive/opencl_type.h"
#include "phd/status.h"
#include "phd/status_macros.h"
#include "third_party/opencl/cl.hpp"

#include <cstdlib>

namespace gpu {
namespace cldrive {

template <typename T>
T MakeScalar(const int& value) {
  return T(value);
}

template <typename T>
std::unique_ptr<ArrayKernelArgValueWithBuffer<T>> CreateArrayArgValue(
    size_t size, bool rand_values, const cl::Context& context) {
  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<T>>(
      context, size, /*value=*/MakeScalar<T>(1));
  if (rand_values) {
    for (size_t i = 0; i < size; ++i) {
      arg_value->vector()[i] = MakeScalar<T>(rand());
    }
  }
  return arg_value;
}

// FIXME:
// template <>
// std::unique_ptr<ArrayKernelArgValueWithBuffer<cl_char2>> CreateArrayArgValue(
//    size_t size, bool rand_values, const cl::Context& context) {
//  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<cl_char2>>(
//      context, size  //, /*value=*/MakeScalar(1)
//  );
//  // if (rand_values) {
//  // for (size_t i = 0; i < size; ++i) {
//  // arg_value->vector()[i] = MakeScalar<T>(rand());
//  //}
//  //}
//  return arg_value;
//}

std::unique_ptr<KernelArgValue> CreateArrayArgValue(
    const OpenClType& type, size_t size, bool rand_values,
    const cl::Context& context) {
  switch (type) {
    // TODO(cec): Fix compilation when bool enabled:
    // case OpenClType::BOOL: {
    //   return CreateArrayArgValue<bool>(size, rand_values, context);
    // }
    case OpenClType::CHAR: {
      return CreateArrayArgValue<cl_char>(size, rand_values, context);
    }
    case OpenClType::UCHAR: {
      return CreateArrayArgValue<cl_uchar>(size, rand_values, context);
    }
    case OpenClType::SHORT: {
      return CreateArrayArgValue<cl_short>(size, rand_values, context);
    }
    case OpenClType::USHORT: {
      return CreateArrayArgValue<cl_ushort>(size, rand_values, context);
    }
    case OpenClType::INT: {
      return CreateArrayArgValue<cl_int>(size, rand_values, context);
    }
    case OpenClType::UINT: {
      return CreateArrayArgValue<cl_uint>(size, rand_values, context);
    }
    case OpenClType::LONG: {
      return CreateArrayArgValue<cl_long>(size, rand_values, context);
    }
    case OpenClType::ULONG: {
      return CreateArrayArgValue<cl_ulong>(size, rand_values, context);
    }
    case OpenClType::FLOAT: {
      return CreateArrayArgValue<cl_float>(size, rand_values, context);
    }
    case OpenClType::DOUBLE: {
      return CreateArrayArgValue<cl_double>(size, rand_values, context);
    }
    case OpenClType::HALF: {
      return CreateArrayArgValue<cl_half>(size, rand_values, context);
    }
    //    case OpenClType::CHAR2: {
    //      return CreateArrayArgValue<cl_char2>(size, rand_values, context);
    //    }

    //    case OpenClType::CHAR3: {
    //      return CreateArrayArgValue<cl_char3>(size, rand_values, context);
    //    }
    //    case OpenClType::CHAR4: {
    //      return CreateArrayArgValue<cl_char4>(size, rand_values, context);
    //    }
    //    case OpenClType::CHAR8: {
    //      return CreateArrayArgValue<cl_char8>(size, rand_values, context);
    //    }
    //    case OpenClType::CHAR16: {
    //      return CreateArrayArgValue<cl_char16>(size, rand_values, context);
    //    }
    //    case OpenClType::UCHAR2: {
    //      return CreateArrayArgValue<cl_uchar2>(size, rand_values, context);
    //    }
    //    case OpenClType::UCHAR3: {
    //      return CreateArrayArgValue<cl_uchar3>(size, rand_values, context);
    //    }
    //    case OpenClType::UCHAR4: {
    //      return CreateArrayArgValue<cl_uchar4>(size, rand_values, context);
    //    }
    //    case OpenClType::UCHAR8: {
    //      return CreateArrayArgValue<cl_uchar8>(size, rand_values, context);
    //    }
    //    case OpenClType::UCHAR16: {
    //      return CreateArrayArgValue<cl_uchar16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::SHORT2: {
    //      return CreateArrayArgValue<cl_short2>(size, rand_values, context);
    //    }
    //    case OpenClType::SHORT3: {
    //      return CreateArrayArgValue<cl_short3>(size, rand_values, context);
    //    }
    //    case OpenClType::SHORT4: {
    //      return CreateArrayArgValue<cl_short4>(size, rand_values, context);
    //    }
    //    case OpenClType::SHORT8: {
    //      return CreateArrayArgValue<cl_short8>(size, rand_values, context);
    //    }
    //    case OpenClType::SHORT16: {
    //      return CreateArrayArgValue<cl_short16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::USHORT2: {
    //      return CreateArrayArgValue<cl_ushort2>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::USHORT3: {
    //      return CreateArrayArgValue<cl_ushort3>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::USHORT4: {
    //      return CreateArrayArgValue<cl_ushort4>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::USHORT8: {
    //      return CreateArrayArgValue<cl_ushort8>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::USHORT16: {
    //      return CreateArrayArgValue<cl_ushort16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::INT2: {
    //      return CreateArrayArgValue<cl_int2>(size, rand_values, context);
    //    }
    //    case OpenClType::INT3: {
    //      return CreateArrayArgValue<cl_int3>(size, rand_values, context);
    //    }
    //    case OpenClType::INT4: {
    //      return CreateArrayArgValue<cl_int4>(size, rand_values, context);
    //    }
    //    case OpenClType::INT8: {
    //      return CreateArrayArgValue<cl_int8>(size, rand_values, context);
    //    }
    //    case OpenClType::INT16: {
    //      return CreateArrayArgValue<cl_int16>(size, rand_values, context);
    //    }
    //    case OpenClType::UINT2: {
    //      return CreateArrayArgValue<cl_uint2>(size, rand_values, context);
    //    }
    //    case OpenClType::UINT3: {
    //      return CreateArrayArgValue<cl_uint3>(size, rand_values, context);
    //    }
    //    case OpenClType::UINT4: {
    //      return CreateArrayArgValue<cl_uint4>(size, rand_values, context);
    //    }
    //    case OpenClType::UINT8: {
    //      return CreateArrayArgValue<cl_uint8>(size, rand_values, context);
    //    }
    //    case OpenClType::UINT16: {
    //      return CreateArrayArgValue<cl_uint16>(size, rand_values, context);
    //    }
    //    case OpenClType::LONG2: {
    //      return CreateArrayArgValue<cl_long2>(size, rand_values, context);
    //    }
    //    case OpenClType::LONG3: {
    //      return CreateArrayArgValue<cl_long3>(size, rand_values, context);
    //    }
    //    case OpenClType::LONG4: {
    //      return CreateArrayArgValue<cl_long4>(size, rand_values, context);
    //    }
    //    case OpenClType::LONG8: {
    //      return CreateArrayArgValue<cl_long8>(size, rand_values, context);
    //    }
    //    case OpenClType::LONG16: {
    //      return CreateArrayArgValue<cl_long16>(size, rand_values, context);
    //    }
    //    case OpenClType::ULONG2: {
    //      return CreateArrayArgValue<cl_ulong2>(size, rand_values, context);
    //    }
    //    case OpenClType::ULONG3: {
    //      return CreateArrayArgValue<cl_ulong3>(size, rand_values, context);
    //    }
    //    case OpenClType::ULONG4: {
    //      return CreateArrayArgValue<cl_ulong4>(size, rand_values, context);
    //    }
    //    case OpenClType::ULONG8: {
    //      return CreateArrayArgValue<cl_ulong8>(size, rand_values, context);
    //    }
    //    case OpenClType::ULONG16: {
    //      return CreateArrayArgValue<cl_ulong16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::FLOAT2: {
    //      return CreateArrayArgValue<cl_float2>(size, rand_values, context);
    //    }
    //    case OpenClType::FLOAT3: {
    //      return CreateArrayArgValue<cl_float3>(size, rand_values, context);
    //    }
    //    case OpenClType::FLOAT4: {
    //      return CreateArrayArgValue<cl_float4>(size, rand_values, context);
    //    }
    //    case OpenClType::FLOAT8: {
    //      return CreateArrayArgValue<cl_float8>(size, rand_values, context);
    //    }
    //    case OpenClType::FLOAT16: {
    //      return CreateArrayArgValue<cl_float16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::DOUBLE2: {
    //      return CreateArrayArgValue<cl_double2>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::DOUBLE3: {
    //      return CreateArrayArgValue<cl_double3>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::DOUBLE4: {
    //      return CreateArrayArgValue<cl_double4>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::DOUBLE8: {
    //      return CreateArrayArgValue<cl_double8>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::DOUBLE16: {
    //      return CreateArrayArgValue<cl_double16>(size, rand_values,
    //      context);
    //    }
    //    case OpenClType::HALF2: {
    //      return CreateArrayArgValue<cl_half2>(size, rand_values, context);
    //    }
    //    case OpenClType::HALF3: {
    //      return CreateArrayArgValue<cl_half3>(size, rand_values, context);
    //    }
    //    case OpenClType::HALF4: {
    //      return CreateArrayArgValue<cl_half4>(size, rand_values, context);
    //    }
    //    case OpenClType::HALF8: {
    //      return CreateArrayArgValue<cl_half8>(size, rand_values, context);
    //    }
    //    case OpenClType::HALF16: {
    //      return CreateArrayArgValue<cl_half16>(size, rand_values, context);
    //    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

template <typename T>
std::unique_ptr<KernelArgValue> CreateScalarArgValue(const int& value) {
  return std::make_unique<ScalarKernelArgValue<T>>(MakeScalar<T>(value));
}

std::unique_ptr<KernelArgValue> CreateScalarArgValue(const OpenClType& type,
                                                     const int& value) {
  switch (type) {
    case OpenClType::BOOL: {
      return CreateScalarArgValue<bool>(value);
    }
    case OpenClType::CHAR: {
      return CreateScalarArgValue<cl_char>(value);
    }
    case OpenClType::UCHAR: {
      return CreateScalarArgValue<cl_uchar>(value);
    }
    case OpenClType::SHORT: {
      return CreateScalarArgValue<cl_short>(value);
    }
    case OpenClType::USHORT: {
      return CreateScalarArgValue<cl_ushort>(value);
    }
    case OpenClType::INT: {
      return CreateScalarArgValue<cl_int>(value);
    }
    case OpenClType::UINT: {
      return CreateScalarArgValue<cl_uint>(value);
    }
    case OpenClType::LONG: {
      return CreateScalarArgValue<cl_long>(value);
    }
    case OpenClType::ULONG: {
      return CreateScalarArgValue<cl_ulong>(value);
    }
    case OpenClType::FLOAT: {
      return CreateScalarArgValue<cl_float>(value);
    }
    case OpenClType::DOUBLE: {
      return CreateScalarArgValue<cl_double>(value);
    }
    case OpenClType::HALF: {
      return CreateScalarArgValue<cl_half>(value);
    }
    //    case OpenClType::CHAR2: {
    //      return CreateScalarArgValue<cl_char2>(value);
    //    }
    //    case OpenClType::CHAR3: {
    //      return CreateScalarArgValue<cl_char3>(value);
    //    }
    //    case OpenClType::CHAR4: {
    //      return CreateScalarArgValue<cl_char4>(value);
    //    }
    //    case OpenClType::CHAR8: {
    //      return CreateScalarArgValue<cl_char8>(value);
    //    }
    //    case OpenClType::CHAR16: {
    //      return CreateScalarArgValue<cl_char16>(value);
    //    }
    //    case OpenClType::UCHAR2: {
    //      return CreateScalarArgValue<cl_uchar2>(value);
    //    }
    //    case OpenClType::UCHAR3: {
    //      return CreateScalarArgValue<cl_uchar3>(value);
    //    }
    //    case OpenClType::UCHAR4: {
    //      return CreateScalarArgValue<cl_uchar4>(value);
    //    }
    //    case OpenClType::UCHAR8: {
    //      return CreateScalarArgValue<cl_uchar8>(value);
    //    }
    //    case OpenClType::UCHAR16: {
    //      return CreateScalarArgValue<cl_uchar16>(value);
    //    }
    //    case OpenClType::SHORT2: {
    //      return CreateScalarArgValue<cl_short2>(value);
    //    }
    //    case OpenClType::SHORT3: {
    //      return CreateScalarArgValue<cl_short3>(value);
    //    }
    //    case OpenClType::SHORT4: {
    //      return CreateScalarArgValue<cl_short4>(value);
    //    }
    //    case OpenClType::SHORT8: {
    //      return CreateScalarArgValue<cl_short8>(value);
    //    }
    //    case OpenClType::SHORT16: {
    //      return CreateScalarArgValue<cl_short16>(value);
    //    }
    //    case OpenClType::USHORT2: {
    //      return CreateScalarArgValue<cl_ushort2>(value);
    //    }
    //    case OpenClType::USHORT3: {
    //      return CreateScalarArgValue<cl_ushort3>(value);
    //    }
    //    case OpenClType::USHORT4: {
    //      return CreateScalarArgValue<cl_ushort4>(value);
    //    }
    //    case OpenClType::USHORT8: {
    //      return CreateScalarArgValue<cl_ushort8>(value);
    //    }
    //    case OpenClType::USHORT16: {
    //      return CreateScalarArgValue<cl_ushort16>(value);
    //    }
    //    case OpenClType::INT2: {
    //      return CreateScalarArgValue<cl_int2>(value);
    //    }
    //    case OpenClType::INT3: {
    //      return CreateScalarArgValue<cl_int3>(value);
    //    }
    //    case OpenClType::INT4: {
    //      return CreateScalarArgValue<cl_int4>(value);
    //    }
    //    case OpenClType::INT8: {
    //      return CreateScalarArgValue<cl_int8>(value);
    //    }
    //    case OpenClType::INT16: {
    //      return CreateScalarArgValue<cl_int16>(value);
    //    }
    //    case OpenClType::UINT2: {
    //      return CreateScalarArgValue<cl_uint2>(value);
    //    }
    //    case OpenClType::UINT3: {
    //      return CreateScalarArgValue<cl_uint3>(value);
    //    }
    //    case OpenClType::UINT4: {
    //      return CreateScalarArgValue<cl_uint4>(value);
    //    }
    //    case OpenClType::UINT8: {
    //      return CreateScalarArgValue<cl_uint8>(value);
    //    }
    //    case OpenClType::UINT16: {
    //      return CreateScalarArgValue<cl_uint16>(value);
    //    }
    //    case OpenClType::LONG2: {
    //      return CreateScalarArgValue<cl_long2>(value);
    //    }
    //    case OpenClType::LONG3: {
    //      return CreateScalarArgValue<cl_long3>(value);
    //    }
    //    case OpenClType::LONG4: {
    //      return CreateScalarArgValue<cl_long4>(value);
    //    }
    //    case OpenClType::LONG8: {
    //      return CreateScalarArgValue<cl_long8>(value);
    //    }
    //    case OpenClType::LONG16: {
    //      return CreateScalarArgValue<cl_long16>(value);
    //    }
    //    case OpenClType::ULONG2: {
    //      return CreateScalarArgValue<cl_ulong2>(value);
    //    }
    //    case OpenClType::ULONG3: {
    //      return CreateScalarArgValue<cl_ulong3>(value);
    //    }
    //    case OpenClType::ULONG4: {
    //      return CreateScalarArgValue<cl_ulong4>(value);
    //    }
    //    case OpenClType::ULONG8: {
    //      return CreateScalarArgValue<cl_ulong8>(value);
    //    }
    //    case OpenClType::ULONG16: {
    //      return CreateScalarArgValue<cl_ulong16>(value);
    //    }
    //    case OpenClType::FLOAT2: {
    //      return CreateScalarArgValue<cl_float2>(value);
    //    }
    //    case OpenClType::FLOAT3: {
    //      return CreateScalarArgValue<cl_float3>(value);
    //    }
    //    case OpenClType::FLOAT4: {
    //      return CreateScalarArgValue<cl_float4>(value);
    //    }
    //    case OpenClType::FLOAT8: {
    //      return CreateScalarArgValue<cl_float8>(value);
    //    }
    //    case OpenClType::FLOAT16: {
    //      return CreateScalarArgValue<cl_float16>(value);
    //    }
    //    case OpenClType::DOUBLE2: {
    //      return CreateScalarArgValue<cl_double2>(value);
    //    }
    //    case OpenClType::DOUBLE3: {
    //      return CreateScalarArgValue<cl_double3>(value);
    //    }
    //    case OpenClType::DOUBLE4: {
    //      return CreateScalarArgValue<cl_double4>(value);
    //    }
    //    case OpenClType::DOUBLE8: {
    //      return CreateScalarArgValue<cl_double8>(value);
    //    }
    //    case OpenClType::DOUBLE16: {
    //      return CreateScalarArgValue<cl_double16>(value);
    //    }
    //    case OpenClType::HALF2: {
    //      return CreateScalarArgValue<cl_half2>(value);
    //    }
    //    case OpenClType::HALF3: {
    //      return CreateScalarArgValue<cl_half3>(value);
    //    }
    //    case OpenClType::HALF4: {
    //      return CreateScalarArgValue<cl_half4>(value);
    //    }
    //    case OpenClType::HALF8: {
    //      return CreateScalarArgValue<cl_half8>(value);
    //    }
    //    case OpenClType::HALF16: {
    //      return CreateScalarArgValue<cl_half16>(value);
    //    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

}  // namespace cldrive
}  // namespace gpu
