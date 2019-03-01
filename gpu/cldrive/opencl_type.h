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

#include "absl/strings/str_cat.h"
#include "phd/statusor.h"
#include "phd/string.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

// The list of supported OpenCL types.
enum OpenClType {
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
  HALF2,
  HALF3,
  HALF4,
  HALF8,
  HALF16
};

phd::StatusOr<OpenClType> OpenClTypeFromString(const string& type_name);

namespace opencl_type {

template <typename T>
T MakeScalar(const int& value) {
  return T(value);
}

template <typename T>
bool Equal(const T& left, const T& right) {
  return left == right;
};

template <typename T>
string ToString(const T& value) {
  string s = "";
  absl::StrAppend(&s, value);
  return s;
}

// Template specializations for vector types --------------------------------

// We don't define specializations for 3-element vectors because the vector3
// family is typedef'd to their respective vector4 types on account of
// the "6.1.5 Alignment of Types" section of the OpenCL spec.

// MakeScalar() support for vector types.

template <>
cl_char2 MakeScalar(const int& value);

template <>
cl_uchar2 MakeScalar(const int& value);

template <>
cl_short2 MakeScalar(const int& value);

template <>
cl_ushort2 MakeScalar(const int& value);

template <>
cl_int2 MakeScalar(const int& value);

template <>
cl_uint2 MakeScalar(const int& value);

template <>
cl_long2 MakeScalar(const int& value);

template <>
cl_ulong2 MakeScalar(const int& value);

template <>
cl_float2 MakeScalar(const int& value);

template <>
cl_double2 MakeScalar(const int& value);

template <>
cl_half2 MakeScalar(const int& value);

template <>
cl_char4 MakeScalar(const int& value);

template <>
cl_uchar4 MakeScalar(const int& value);

template <>
cl_short4 MakeScalar(const int& value);

template <>
cl_ushort4 MakeScalar(const int& value);

template <>
cl_int4 MakeScalar(const int& value);

template <>
cl_uint4 MakeScalar(const int& value);

template <>
cl_long4 MakeScalar(const int& value);

template <>
cl_ulong4 MakeScalar(const int& value);

template <>
cl_float4 MakeScalar(const int& value);

template <>
cl_double4 MakeScalar(const int& value);

template <>
cl_half4 MakeScalar(const int& value);

template <>
cl_char8 MakeScalar(const int& value);

template <>
cl_uchar8 MakeScalar(const int& value);

template <>
cl_short8 MakeScalar(const int& value);

template <>
cl_ushort8 MakeScalar(const int& value);

template <>
cl_int8 MakeScalar(const int& value);

template <>
cl_uint8 MakeScalar(const int& value);

template <>
cl_long8 MakeScalar(const int& value);

template <>
cl_ulong8 MakeScalar(const int& value);

template <>
cl_float8 MakeScalar(const int& value);

template <>
cl_double8 MakeScalar(const int& value);

template <>
cl_half8 MakeScalar(const int& value);

template <>
cl_char16 MakeScalar(const int& value);

template <>
cl_uchar16 MakeScalar(const int& value);

template <>
cl_short16 MakeScalar(const int& value);

template <>
cl_ushort16 MakeScalar(const int& value);

template <>
cl_int16 MakeScalar(const int& value);

template <>
cl_uint16 MakeScalar(const int& value);

template <>
cl_long16 MakeScalar(const int& value);

template <>
cl_ulong16 MakeScalar(const int& value);

template <>
cl_float16 MakeScalar(const int& value);

template <>
cl_double16 MakeScalar(const int& value);

template <>
cl_half16 MakeScalar(const int& value);

// Equal() support for vector types.

template <>
bool Equal(const cl_char2& lhs, const cl_char2& rhs);

template <>
bool Equal(const cl_uchar2& lhs, const cl_uchar2& rhs);

template <>
bool Equal(const cl_short2& lhs, const cl_short2& rhs);

template <>
bool Equal(const cl_ushort2& lhs, const cl_ushort2& rhs);

template <>
bool Equal(const cl_int2& lhs, const cl_int2& rhs);

template <>
bool Equal(const cl_uint2& lhs, const cl_uint2& rhs);

template <>
bool Equal(const cl_long2& lhs, const cl_long2& rhs);

template <>
bool Equal(const cl_ulong2& lhs, const cl_ulong2& rhs);

template <>
bool Equal(const cl_float2& lhs, const cl_float2& rhs);

template <>
bool Equal(const cl_double2& lhs, const cl_double2& rhs);

template <>
bool Equal(const cl_half2& lhs, const cl_half2& rhs);

template <>
bool Equal(const cl_char4& lhs, const cl_char4& rhs);

template <>
bool Equal(const cl_uchar4& lhs, const cl_uchar4& rhs);

template <>
bool Equal(const cl_short4& lhs, const cl_short4& rhs);

template <>
bool Equal(const cl_ushort4& lhs, const cl_ushort4& rhs);

template <>
bool Equal(const cl_int4& lhs, const cl_int4& rhs);

template <>
bool Equal(const cl_uint4& lhs, const cl_uint4& rhs);

template <>
bool Equal(const cl_long4& lhs, const cl_long4& rhs);

template <>
bool Equal(const cl_ulong4& lhs, const cl_ulong4& rhs);

template <>
bool Equal(const cl_float4& lhs, const cl_float4& rhs);

template <>
bool Equal(const cl_double4& lhs, const cl_double4& rhs);

template <>
bool Equal(const cl_half4& lhs, const cl_half4& rhs);

template <>
bool Equal(const cl_char8& lhs, const cl_char8& rhs);

template <>
bool Equal(const cl_uchar8& lhs, const cl_uchar8& rhs);

template <>
bool Equal(const cl_short8& lhs, const cl_short8& rhs);

template <>
bool Equal(const cl_ushort8& lhs, const cl_ushort8& rhs);

template <>
bool Equal(const cl_int8& lhs, const cl_int8& rhs);

template <>
bool Equal(const cl_uint8& lhs, const cl_uint8& rhs);

template <>
bool Equal(const cl_long8& lhs, const cl_long8& rhs);

template <>
bool Equal(const cl_ulong8& lhs, const cl_ulong8& rhs);

template <>
bool Equal(const cl_float8& lhs, const cl_float8& rhs);

template <>
bool Equal(const cl_double8& lhs, const cl_double8& rhs);

template <>
bool Equal(const cl_half8& lhs, const cl_half8& rhs);

template <>
bool Equal(const cl_char16& lhs, const cl_char16& rhs);

template <>
bool Equal(const cl_uchar16& lhs, const cl_uchar16& rhs);

template <>
bool Equal(const cl_short16& lhs, const cl_short16& rhs);

template <>
bool Equal(const cl_ushort16& lhs, const cl_ushort16& rhs);

template <>
bool Equal(const cl_int16& lhs, const cl_int16& rhs);

template <>
bool Equal(const cl_uint16& lhs, const cl_uint16& rhs);

template <>
bool Equal(const cl_long16& lhs, const cl_long16& rhs);

template <>
bool Equal(const cl_ulong16& lhs, const cl_ulong16& rhs);

template <>
bool Equal(const cl_float16& lhs, const cl_float16& rhs);

template <>
bool Equal(const cl_double16& lhs, const cl_double16& rhs);

template <>
bool Equal(const cl_half16& lhs, const cl_half16& rhs);

// ToString() support for vector types.

template <>
string ToString(const cl_char2& value);

template <>
string ToString(const cl_uchar2& value);

template <>
string ToString(const cl_short2& value);

template <>
string ToString(const cl_ushort2& value);

template <>
string ToString(const cl_int2& value);

template <>
string ToString(const cl_uint2& value);

template <>
string ToString(const cl_long2& value);

template <>
string ToString(const cl_ulong2& value);

template <>
string ToString(const cl_float2& value);

template <>
string ToString(const cl_double2& value);

template <>
string ToString(const cl_half2& value);

template <>
string ToString(const cl_char4& value);

template <>
string ToString(const cl_uchar4& value);

template <>
string ToString(const cl_short4& value);

template <>
string ToString(const cl_ushort4& value);

template <>
string ToString(const cl_int4& value);

template <>
string ToString(const cl_uint4& value);

template <>
string ToString(const cl_long4& value);

template <>
string ToString(const cl_ulong4& value);

template <>
string ToString(const cl_float4& value);

template <>
string ToString(const cl_double4& value);

template <>
string ToString(const cl_half4& value);

template <>
string ToString(const cl_char8& value);

template <>
string ToString(const cl_uchar8& value);

template <>
string ToString(const cl_short8& value);

template <>
string ToString(const cl_ushort8& value);

template <>
string ToString(const cl_int8& value);

template <>
string ToString(const cl_uint8& value);

template <>
string ToString(const cl_long8& value);

template <>
string ToString(const cl_ulong8& value);

template <>
string ToString(const cl_float8& value);

template <>
string ToString(const cl_double8& value);

template <>
string ToString(const cl_half8& value);

template <>
string ToString(const cl_char16& value);

template <>
string ToString(const cl_uchar16& value);

template <>
string ToString(const cl_short16& value);

template <>
string ToString(const cl_ushort16& value);

template <>
string ToString(const cl_int16& value);

template <>
string ToString(const cl_uint16& value);

template <>
string ToString(const cl_long16& value);

template <>
string ToString(const cl_ulong16& value);

template <>
string ToString(const cl_float16& value);

template <>
string ToString(const cl_double16& value);

template <>
string ToString(const cl_half16& value);

}  // namespace opencl_type

}  // namespace cldrive
}  // namespace gpu
