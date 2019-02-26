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
std::unique_ptr<ArrayKernelArgValueWithBuffer<T>> CreateArrayArgValue(
    const cl::Context& context, size_t size, const int& value,
    bool rand_values) {
  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<T>>(
      context, size, /*value=*/opencl_type::MakeScalar<T>(value));
  if (rand_values) {
    for (size_t i = 0; i < size; ++i) {
      arg_value->vector()[i] = opencl_type::MakeScalar<T>(rand());
    }
  }
  return arg_value;
}

// FIXME:
// template <>
// std::unique_ptr<ArrayKernelArgValueWithBuffer<cl_char2>> CreateArrayArgValue(
//    size_t size, bool rand_values, const cl::Context& context) {
//  auto arg_value = std::make_unique<ArrayKernelArgValueWithBuffer<cl_char2>>(
//      context, size  //, /*value=*/opencl_type::MakeScalar(1)
//  );
//  // if (rand_values) {
//  // for (size_t i = 0; i < size; ++i) {
//  // arg_value->vector()[i] = opencl_type::MakeScalar<T>(rand());
//  //}
//  //}
//  return arg_value;
//}

std::unique_ptr<KernelArgValue> CreateArrayArgValue(const OpenClType& type,
                                                    const cl::Context& context,
                                                    size_t size,
                                                    const int& value,
                                                    bool rand_values) {
  switch (type) {
    // TODO(cec): Fix compilation when bool enabled:
    // case OpenClType::BOOL: {
    //   return CreateArrayArgValue<bool>(size, rand_values, context);
    // }
    case OpenClType::CHAR: {
      return CreateArrayArgValue<cl_char>(context, size, value, rand_values);
    }
    case OpenClType::UCHAR: {
      return CreateArrayArgValue<cl_uchar>(context, size, value, rand_values);
    }
    case OpenClType::SHORT: {
      return CreateArrayArgValue<cl_short>(context, size, value, rand_values);
    }
    case OpenClType::USHORT: {
      return CreateArrayArgValue<cl_ushort>(context, size, value, rand_values);
    }
    case OpenClType::INT: {
      return CreateArrayArgValue<cl_int>(context, size, value, rand_values);
    }
    case OpenClType::UINT: {
      return CreateArrayArgValue<cl_uint>(context, size, value, rand_values);
    }
    case OpenClType::LONG: {
      return CreateArrayArgValue<cl_long>(context, size, value, rand_values);
    }
    case OpenClType::ULONG: {
      return CreateArrayArgValue<cl_ulong>(context, size, value, rand_values);
    }
    case OpenClType::FLOAT: {
      return CreateArrayArgValue<cl_float>(context, size, value, rand_values);
    }
    case OpenClType::DOUBLE: {
      return CreateArrayArgValue<cl_double>(context, size, value, rand_values);
    }
    case OpenClType::HALF: {
      return CreateArrayArgValue<cl_half>(context, size, value, rand_values);
    }
    //    case OpenClType::CHAR2: {
    //      return CreateArrayArgValue<cl_char2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::CHAR3: {
    //      return CreateArrayArgValue<cl_char3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::CHAR4: {
    //      return CreateArrayArgValue<cl_char4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::CHAR8: {
    //      return CreateArrayArgValue<cl_char8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::CHAR16: {
    //      return CreateArrayArgValue<cl_char16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UCHAR2: {
    //      return CreateArrayArgValue<cl_uchar2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UCHAR3: {
    //      return CreateArrayArgValue<cl_uchar3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UCHAR4: {
    //      return CreateArrayArgValue<cl_uchar4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UCHAR8: {
    //      return CreateArrayArgValue<cl_uchar8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UCHAR16: {
    //      return CreateArrayArgValue<cl_uchar16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::SHORT2: {
    //      return CreateArrayArgValue<cl_short2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::SHORT3: {
    //      return CreateArrayArgValue<cl_short3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::SHORT4: {
    //      return CreateArrayArgValue<cl_short4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::SHORT8: {
    //      return CreateArrayArgValue<cl_short8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::SHORT16: {
    //      return CreateArrayArgValue<cl_short16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::USHORT2: {
    //      return CreateArrayArgValue<cl_ushort2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::USHORT3: {
    //      return CreateArrayArgValue<cl_ushort3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::USHORT4: {
    //      return CreateArrayArgValue<cl_ushort4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::USHORT8: {
    //      return CreateArrayArgValue<cl_ushort8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::USHORT16: {
    //      return CreateArrayArgValue<cl_ushort16>(context, size,
    //      value, rand_values);
    //    }
    //    case OpenClType::INT2: {
    //      return CreateArrayArgValue<cl_int2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::INT3: {
    //      return CreateArrayArgValue<cl_int3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::INT4: {
    //      return CreateArrayArgValue<cl_int4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::INT8: {
    //      return CreateArrayArgValue<cl_int8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::INT16: {
    //      return CreateArrayArgValue<cl_int16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UINT2: {
    //      return CreateArrayArgValue<cl_uint2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UINT3: {
    //      return CreateArrayArgValue<cl_uint3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UINT4: {
    //      return CreateArrayArgValue<cl_uint4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UINT8: {
    //      return CreateArrayArgValue<cl_uint8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::UINT16: {
    //      return CreateArrayArgValue<cl_uint16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::LONG2: {
    //      return CreateArrayArgValue<cl_long2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::LONG3: {
    //      return CreateArrayArgValue<cl_long3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::LONG4: {
    //      return CreateArrayArgValue<cl_long4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::LONG8: {
    //      return CreateArrayArgValue<cl_long8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::LONG16: {
    //      return CreateArrayArgValue<cl_long16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::ULONG2: {
    //      return CreateArrayArgValue<cl_ulong2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::ULONG3: {
    //      return CreateArrayArgValue<cl_ulong3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::ULONG4: {
    //      return CreateArrayArgValue<cl_ulong4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::ULONG8: {
    //      return CreateArrayArgValue<cl_ulong8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::ULONG16: {
    //      return CreateArrayArgValue<cl_ulong16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::FLOAT2: {
    //      return CreateArrayArgValue<cl_float2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::FLOAT3: {
    //      return CreateArrayArgValue<cl_float3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::FLOAT4: {
    //      return CreateArrayArgValue<cl_float4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::FLOAT8: {
    //      return CreateArrayArgValue<cl_float8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::FLOAT16: {
    //      return CreateArrayArgValue<cl_float16>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::DOUBLE2: {
    //      return CreateArrayArgValue<cl_double2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::DOUBLE3: {
    //      return CreateArrayArgValue<cl_double3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::DOUBLE4: {
    //      return CreateArrayArgValue<cl_double4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::DOUBLE8: {
    //      return CreateArrayArgValue<cl_double8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::DOUBLE16: {
    //      return CreateArrayArgValue<cl_double16>(context, size,
    //      value, rand_values);
    //    }
    //    case OpenClType::HALF2: {
    //      return CreateArrayArgValue<cl_half2>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::HALF3: {
    //      return CreateArrayArgValue<cl_half3>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::HALF4: {
    //      return CreateArrayArgValue<cl_half4>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::HALF8: {
    //      return CreateArrayArgValue<cl_half8>(context, size, value,
    //      rand_values);
    //    }
    //    case OpenClType::HALF16: {
    //      return CreateArrayArgValue<cl_half16>(context, size, value,
    //      rand_values);
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
  return std::make_unique<ScalarKernelArgValue<T>>(
      opencl_type::MakeScalar<T>(value));
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
