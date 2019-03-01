// This file performs the translation from OpenClType enum value to templated
// classes.
//
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

#include "gpu/cldrive/global_memory_arg_value.h"
#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/local_memory_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "gpu/cldrive/opencl_type.h"
#include "phd/status.h"
#include "phd/status_macros.h"
#include "third_party/opencl/cl.hpp"

#include <cstdlib>

namespace gpu {
namespace cldrive {

template <typename T>
std::unique_ptr<GlobalMemoryArgValueWithBuffer<T>> CreateGlobalMemoryArgValue(
    const cl::Context& context, size_t size, const int& value,
    bool rand_values) {
  auto arg_value = std::make_unique<GlobalMemoryArgValueWithBuffer<T>>(
      context, size, /*value=*/opencl_type::MakeScalar<T>(value));
  if (rand_values) {
    for (size_t i = 0; i < size; ++i) {
      arg_value->vector()[i] = opencl_type::MakeScalar<T>(rand());
    }
  }
  return arg_value;
}

std::unique_ptr<KernelArgValue> CreateGlobalMemoryArgValue(
    const OpenClType& type, const cl::Context& context, size_t size,
    const int& value, bool rand_values) {
  DCHECK(size) << "Cannot create array with 0 elements";
  switch (type) {
    case OpenClType::BOOL: {
      return CreateGlobalMemoryArgValue<bool>(context, size, value,
                                              rand_values);
    }
    case OpenClType::CHAR: {
      return CreateGlobalMemoryArgValue<cl_char>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::UCHAR: {
      return CreateGlobalMemoryArgValue<cl_uchar>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::SHORT: {
      return CreateGlobalMemoryArgValue<cl_short>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::USHORT: {
      return CreateGlobalMemoryArgValue<cl_ushort>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::INT: {
      return CreateGlobalMemoryArgValue<cl_int>(context, size, value,
                                                rand_values);
    }
    case OpenClType::UINT: {
      return CreateGlobalMemoryArgValue<cl_uint>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::LONG: {
      return CreateGlobalMemoryArgValue<cl_long>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::ULONG: {
      return CreateGlobalMemoryArgValue<cl_ulong>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::FLOAT: {
      return CreateGlobalMemoryArgValue<cl_float>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::DOUBLE: {
      return CreateGlobalMemoryArgValue<cl_double>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::HALF: {
      return CreateGlobalMemoryArgValue<cl_half>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::CHAR2: {
      return CreateGlobalMemoryArgValue<cl_char2>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::CHAR3: {
      return CreateGlobalMemoryArgValue<cl_char3>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::CHAR4: {
      return CreateGlobalMemoryArgValue<cl_char4>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::CHAR8: {
      return CreateGlobalMemoryArgValue<cl_char8>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::CHAR16: {
      return CreateGlobalMemoryArgValue<cl_char16>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::UCHAR2: {
      return CreateGlobalMemoryArgValue<cl_uchar2>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::UCHAR3: {
      return CreateGlobalMemoryArgValue<cl_uchar3>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::UCHAR4: {
      return CreateGlobalMemoryArgValue<cl_uchar4>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::UCHAR8: {
      return CreateGlobalMemoryArgValue<cl_uchar8>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::UCHAR16: {
      return CreateGlobalMemoryArgValue<cl_uchar16>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::SHORT2: {
      return CreateGlobalMemoryArgValue<cl_short2>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::SHORT3: {
      return CreateGlobalMemoryArgValue<cl_short3>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::SHORT4: {
      return CreateGlobalMemoryArgValue<cl_short4>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::SHORT8: {
      return CreateGlobalMemoryArgValue<cl_short8>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::SHORT16: {
      return CreateGlobalMemoryArgValue<cl_short16>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::USHORT2: {
      return CreateGlobalMemoryArgValue<cl_ushort2>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::USHORT3: {
      return CreateGlobalMemoryArgValue<cl_ushort3>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::USHORT4: {
      return CreateGlobalMemoryArgValue<cl_ushort4>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::USHORT8: {
      return CreateGlobalMemoryArgValue<cl_ushort8>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::USHORT16: {
      return CreateGlobalMemoryArgValue<cl_ushort16>(context, size, value,
                                                     rand_values);
    }
    case OpenClType::INT2: {
      return CreateGlobalMemoryArgValue<cl_int2>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::INT3: {
      return CreateGlobalMemoryArgValue<cl_int3>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::INT4: {
      return CreateGlobalMemoryArgValue<cl_int4>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::INT8: {
      return CreateGlobalMemoryArgValue<cl_int8>(context, size, value,
                                                 rand_values);
    }
    case OpenClType::INT16: {
      return CreateGlobalMemoryArgValue<cl_int16>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::UINT2: {
      return CreateGlobalMemoryArgValue<cl_uint2>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::UINT3: {
      return CreateGlobalMemoryArgValue<cl_uint3>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::UINT4: {
      return CreateGlobalMemoryArgValue<cl_uint4>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::UINT8: {
      return CreateGlobalMemoryArgValue<cl_uint8>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::UINT16: {
      return CreateGlobalMemoryArgValue<cl_uint16>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::LONG2: {
      return CreateGlobalMemoryArgValue<cl_long2>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::LONG3: {
      return CreateGlobalMemoryArgValue<cl_long3>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::LONG4: {
      return CreateGlobalMemoryArgValue<cl_long4>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::LONG8: {
      return CreateGlobalMemoryArgValue<cl_long8>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::LONG16: {
      return CreateGlobalMemoryArgValue<cl_long16>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::ULONG2: {
      return CreateGlobalMemoryArgValue<cl_ulong2>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::ULONG3: {
      return CreateGlobalMemoryArgValue<cl_ulong3>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::ULONG4: {
      return CreateGlobalMemoryArgValue<cl_ulong4>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::ULONG8: {
      return CreateGlobalMemoryArgValue<cl_ulong8>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::ULONG16: {
      return CreateGlobalMemoryArgValue<cl_ulong16>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::FLOAT2: {
      return CreateGlobalMemoryArgValue<cl_float2>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::FLOAT3: {
      return CreateGlobalMemoryArgValue<cl_float3>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::FLOAT4: {
      return CreateGlobalMemoryArgValue<cl_float4>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::FLOAT8: {
      return CreateGlobalMemoryArgValue<cl_float8>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::FLOAT16: {
      return CreateGlobalMemoryArgValue<cl_float16>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::DOUBLE2: {
      return CreateGlobalMemoryArgValue<cl_double2>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::DOUBLE3: {
      return CreateGlobalMemoryArgValue<cl_double3>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::DOUBLE4: {
      return CreateGlobalMemoryArgValue<cl_double4>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::DOUBLE8: {
      return CreateGlobalMemoryArgValue<cl_double8>(context, size, value,
                                                    rand_values);
    }
    case OpenClType::DOUBLE16: {
      return CreateGlobalMemoryArgValue<cl_double16>(context, size, value,
                                                     rand_values);
    }
    case OpenClType::HALF2: {
      return CreateGlobalMemoryArgValue<cl_half2>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::HALF3: {
      return CreateGlobalMemoryArgValue<cl_half3>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::HALF4: {
      return CreateGlobalMemoryArgValue<cl_half4>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::HALF8: {
      return CreateGlobalMemoryArgValue<cl_half8>(context, size, value,
                                                  rand_values);
    }
    case OpenClType::HALF16: {
      return CreateGlobalMemoryArgValue<cl_half16>(context, size, value,
                                                   rand_values);
    }
    case OpenClType::DEFAULT_UNKNOWN: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status if the type cannot be determined.
      LOG(FATAL) << "CreateGlobalMemoryArgValue() called with type "
                 << "OpenClType::DEFAULT_UNKNOWN";
    }
  }
  return nullptr;  // Unreachable so long as switch covers all enum values.
}

std::unique_ptr<KernelArgValue> CreateLocalMemoryArgValue(
    const OpenClType& type, size_t size) {
  DCHECK(size) << "Cannot create array with 0 elements";
  switch (type) {
    case OpenClType::BOOL: {
      return std::make_unique<LocalMemoryArgValue<bool>>(size);
    }
    case OpenClType::CHAR: {
      return std::make_unique<LocalMemoryArgValue<cl_char>>(size);
    }
    case OpenClType::UCHAR: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar>>(size);
    }
    case OpenClType::SHORT: {
      return std::make_unique<LocalMemoryArgValue<cl_short>>(size);
    }
    case OpenClType::USHORT: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort>>(size);
    }
    case OpenClType::INT: {
      return std::make_unique<LocalMemoryArgValue<cl_int>>(size);
    }
    case OpenClType::UINT: {
      return std::make_unique<LocalMemoryArgValue<cl_uint>>(size);
    }
    case OpenClType::LONG: {
      return std::make_unique<LocalMemoryArgValue<cl_long>>(size);
    }
    case OpenClType::ULONG: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong>>(size);
    }
    case OpenClType::FLOAT: {
      return std::make_unique<LocalMemoryArgValue<cl_float>>(size);
    }
    case OpenClType::DOUBLE: {
      return std::make_unique<LocalMemoryArgValue<cl_double>>(size);
    }
    case OpenClType::HALF: {
      return std::make_unique<LocalMemoryArgValue<cl_half>>(size);
    }
    case OpenClType::CHAR2: {
      return std::make_unique<LocalMemoryArgValue<cl_char2>>(size);
    }
    case OpenClType::CHAR3: {
      return std::make_unique<LocalMemoryArgValue<cl_char3>>(size);
    }
    case OpenClType::CHAR4: {
      return std::make_unique<LocalMemoryArgValue<cl_char4>>(size);
    }
    case OpenClType::CHAR8: {
      return std::make_unique<LocalMemoryArgValue<cl_char8>>(size);
    }
    case OpenClType::CHAR16: {
      return std::make_unique<LocalMemoryArgValue<cl_char16>>(size);
    }
    case OpenClType::UCHAR2: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar2>>(size);
    }
    case OpenClType::UCHAR3: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar3>>(size);
    }
    case OpenClType::UCHAR4: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar4>>(size);
    }
    case OpenClType::UCHAR8: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar8>>(size);
    }
    case OpenClType::UCHAR16: {
      return std::make_unique<LocalMemoryArgValue<cl_uchar16>>(size);
    }
    case OpenClType::SHORT2: {
      return std::make_unique<LocalMemoryArgValue<cl_short2>>(size);
    }
    case OpenClType::SHORT3: {
      return std::make_unique<LocalMemoryArgValue<cl_short3>>(size);
    }
    case OpenClType::SHORT4: {
      return std::make_unique<LocalMemoryArgValue<cl_short4>>(size);
    }
    case OpenClType::SHORT8: {
      return std::make_unique<LocalMemoryArgValue<cl_short8>>(size);
    }
    case OpenClType::SHORT16: {
      return std::make_unique<LocalMemoryArgValue<cl_short16>>(size);
    }
    case OpenClType::USHORT2: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort2>>(size);
    }
    case OpenClType::USHORT3: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort3>>(size);
    }
    case OpenClType::USHORT4: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort4>>(size);
    }
    case OpenClType::USHORT8: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort8>>(size);
    }
    case OpenClType::USHORT16: {
      return std::make_unique<LocalMemoryArgValue<cl_ushort16>>(size);
    }
    case OpenClType::INT2: {
      return std::make_unique<LocalMemoryArgValue<cl_int2>>(size);
    }
    case OpenClType::INT3: {
      return std::make_unique<LocalMemoryArgValue<cl_int3>>(size);
    }
    case OpenClType::INT4: {
      return std::make_unique<LocalMemoryArgValue<cl_int4>>(size);
    }
    case OpenClType::INT8: {
      return std::make_unique<LocalMemoryArgValue<cl_int8>>(size);
    }
    case OpenClType::INT16: {
      return std::make_unique<LocalMemoryArgValue<cl_int16>>(size);
    }
    case OpenClType::UINT2: {
      return std::make_unique<LocalMemoryArgValue<cl_uint2>>(size);
    }
    case OpenClType::UINT3: {
      return std::make_unique<LocalMemoryArgValue<cl_uint3>>(size);
    }
    case OpenClType::UINT4: {
      return std::make_unique<LocalMemoryArgValue<cl_uint4>>(size);
    }
    case OpenClType::UINT8: {
      return std::make_unique<LocalMemoryArgValue<cl_uint8>>(size);
    }
    case OpenClType::UINT16: {
      return std::make_unique<LocalMemoryArgValue<cl_uint16>>(size);
    }
    case OpenClType::LONG2: {
      return std::make_unique<LocalMemoryArgValue<cl_long2>>(size);
    }
    case OpenClType::LONG3: {
      return std::make_unique<LocalMemoryArgValue<cl_long3>>(size);
    }
    case OpenClType::LONG4: {
      return std::make_unique<LocalMemoryArgValue<cl_long4>>(size);
    }
    case OpenClType::LONG8: {
      return std::make_unique<LocalMemoryArgValue<cl_long8>>(size);
    }
    case OpenClType::LONG16: {
      return std::make_unique<LocalMemoryArgValue<cl_long16>>(size);
    }
    case OpenClType::ULONG2: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong2>>(size);
    }
    case OpenClType::ULONG3: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong3>>(size);
    }
    case OpenClType::ULONG4: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong4>>(size);
    }
    case OpenClType::ULONG8: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong8>>(size);
    }
    case OpenClType::ULONG16: {
      return std::make_unique<LocalMemoryArgValue<cl_ulong16>>(size);
    }
    case OpenClType::FLOAT2: {
      return std::make_unique<LocalMemoryArgValue<cl_float2>>(size);
    }
    case OpenClType::FLOAT3: {
      return std::make_unique<LocalMemoryArgValue<cl_float3>>(size);
    }
    case OpenClType::FLOAT4: {
      return std::make_unique<LocalMemoryArgValue<cl_float4>>(size);
    }
    case OpenClType::FLOAT8: {
      return std::make_unique<LocalMemoryArgValue<cl_float8>>(size);
    }
    case OpenClType::FLOAT16: {
      return std::make_unique<LocalMemoryArgValue<cl_float16>>(size);
    }
    case OpenClType::DOUBLE2: {
      return std::make_unique<LocalMemoryArgValue<cl_double2>>(size);
    }
    case OpenClType::DOUBLE3: {
      return std::make_unique<LocalMemoryArgValue<cl_double3>>(size);
    }
    case OpenClType::DOUBLE4: {
      return std::make_unique<LocalMemoryArgValue<cl_double4>>(size);
    }
    case OpenClType::DOUBLE8: {
      return std::make_unique<LocalMemoryArgValue<cl_double8>>(size);
    }
    case OpenClType::DOUBLE16: {
      return std::make_unique<LocalMemoryArgValue<cl_double16>>(size);
    }
    case OpenClType::HALF2: {
      return std::make_unique<LocalMemoryArgValue<cl_half2>>(size);
    }
    case OpenClType::HALF3: {
      return std::make_unique<LocalMemoryArgValue<cl_half3>>(size);
    }
    case OpenClType::HALF4: {
      return std::make_unique<LocalMemoryArgValue<cl_half4>>(size);
    }
    case OpenClType::HALF8: {
      return std::make_unique<LocalMemoryArgValue<cl_half8>>(size);
    }
    case OpenClType::HALF16: {
      return std::make_unique<LocalMemoryArgValue<cl_half16>>(size);
    }
    case OpenClType::DEFAULT_UNKNOWN: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status if the type cannot be determined.
      LOG(FATAL) << "CreateLocalMemoryArgValue() called with type "
                 << "OpenClType::DEFAULT_UNKNOWN";
    }
  }
  return nullptr;  // Unreachable so long as switch covers all enum values.
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
    case OpenClType::CHAR2: {
      return CreateScalarArgValue<cl_char2>(value);
    }
    case OpenClType::CHAR3: {
      return CreateScalarArgValue<cl_char3>(value);
    }
    case OpenClType::CHAR4: {
      return CreateScalarArgValue<cl_char4>(value);
    }
    case OpenClType::CHAR8: {
      return CreateScalarArgValue<cl_char8>(value);
    }
    case OpenClType::CHAR16: {
      return CreateScalarArgValue<cl_char16>(value);
    }
    case OpenClType::UCHAR2: {
      return CreateScalarArgValue<cl_uchar2>(value);
    }
    case OpenClType::UCHAR3: {
      return CreateScalarArgValue<cl_uchar3>(value);
    }
    case OpenClType::UCHAR4: {
      return CreateScalarArgValue<cl_uchar4>(value);
    }
    case OpenClType::UCHAR8: {
      return CreateScalarArgValue<cl_uchar8>(value);
    }
    case OpenClType::UCHAR16: {
      return CreateScalarArgValue<cl_uchar16>(value);
    }
    case OpenClType::SHORT2: {
      return CreateScalarArgValue<cl_short2>(value);
    }
    case OpenClType::SHORT3: {
      return CreateScalarArgValue<cl_short3>(value);
    }
    case OpenClType::SHORT4: {
      return CreateScalarArgValue<cl_short4>(value);
    }
    case OpenClType::SHORT8: {
      return CreateScalarArgValue<cl_short8>(value);
    }
    case OpenClType::SHORT16: {
      return CreateScalarArgValue<cl_short16>(value);
    }
    case OpenClType::USHORT2: {
      return CreateScalarArgValue<cl_ushort2>(value);
    }
    case OpenClType::USHORT3: {
      return CreateScalarArgValue<cl_ushort3>(value);
    }
    case OpenClType::USHORT4: {
      return CreateScalarArgValue<cl_ushort4>(value);
    }
    case OpenClType::USHORT8: {
      return CreateScalarArgValue<cl_ushort8>(value);
    }
    case OpenClType::USHORT16: {
      return CreateScalarArgValue<cl_ushort16>(value);
    }
    case OpenClType::INT2: {
      return CreateScalarArgValue<cl_int2>(value);
    }
    case OpenClType::INT3: {
      return CreateScalarArgValue<cl_int3>(value);
    }
    case OpenClType::INT4: {
      return CreateScalarArgValue<cl_int4>(value);
    }
    case OpenClType::INT8: {
      return CreateScalarArgValue<cl_int8>(value);
    }
    case OpenClType::INT16: {
      return CreateScalarArgValue<cl_int16>(value);
    }
    case OpenClType::UINT2: {
      return CreateScalarArgValue<cl_uint2>(value);
    }
    case OpenClType::UINT3: {
      return CreateScalarArgValue<cl_uint3>(value);
    }
    case OpenClType::UINT4: {
      return CreateScalarArgValue<cl_uint4>(value);
    }
    case OpenClType::UINT8: {
      return CreateScalarArgValue<cl_uint8>(value);
    }
    case OpenClType::UINT16: {
      return CreateScalarArgValue<cl_uint16>(value);
    }
    case OpenClType::LONG2: {
      return CreateScalarArgValue<cl_long2>(value);
    }
    case OpenClType::LONG3: {
      return CreateScalarArgValue<cl_long3>(value);
    }
    case OpenClType::LONG4: {
      return CreateScalarArgValue<cl_long4>(value);
    }
    case OpenClType::LONG8: {
      return CreateScalarArgValue<cl_long8>(value);
    }
    case OpenClType::LONG16: {
      return CreateScalarArgValue<cl_long16>(value);
    }
    case OpenClType::ULONG2: {
      return CreateScalarArgValue<cl_ulong2>(value);
    }
    case OpenClType::ULONG3: {
      return CreateScalarArgValue<cl_ulong3>(value);
    }
    case OpenClType::ULONG4: {
      return CreateScalarArgValue<cl_ulong4>(value);
    }
    case OpenClType::ULONG8: {
      return CreateScalarArgValue<cl_ulong8>(value);
    }
    case OpenClType::ULONG16: {
      return CreateScalarArgValue<cl_ulong16>(value);
    }
    case OpenClType::FLOAT2: {
      return CreateScalarArgValue<cl_float2>(value);
    }
    case OpenClType::FLOAT3: {
      return CreateScalarArgValue<cl_float3>(value);
    }
    case OpenClType::FLOAT4: {
      return CreateScalarArgValue<cl_float4>(value);
    }
    case OpenClType::FLOAT8: {
      return CreateScalarArgValue<cl_float8>(value);
    }
    case OpenClType::FLOAT16: {
      return CreateScalarArgValue<cl_float16>(value);
    }
    case OpenClType::DOUBLE2: {
      return CreateScalarArgValue<cl_double2>(value);
    }
    case OpenClType::DOUBLE3: {
      return CreateScalarArgValue<cl_double3>(value);
    }
    case OpenClType::DOUBLE4: {
      return CreateScalarArgValue<cl_double4>(value);
    }
    case OpenClType::DOUBLE8: {
      return CreateScalarArgValue<cl_double8>(value);
    }
    case OpenClType::DOUBLE16: {
      return CreateScalarArgValue<cl_double16>(value);
    }
    case OpenClType::HALF2: {
      return CreateScalarArgValue<cl_half2>(value);
    }
    case OpenClType::HALF3: {
      return CreateScalarArgValue<cl_half3>(value);
    }
    case OpenClType::HALF4: {
      return CreateScalarArgValue<cl_half4>(value);
    }
    case OpenClType::HALF8: {
      return CreateScalarArgValue<cl_half8>(value);
    }
    case OpenClType::HALF16: {
      return CreateScalarArgValue<cl_half16>(value);
    }
    case OpenClType::DEFAULT_UNKNOWN: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status if the type cannot be determined.
      LOG(FATAL) << "CreateScalarArgValue() called with type "
                 << "OpenClType::DEFAULT_UNKNOWN";
    }
  }
  return nullptr;  // Unreachable so long as switch covers all enum values.
}

}  // namespace cldrive
}  // namespace gpu
