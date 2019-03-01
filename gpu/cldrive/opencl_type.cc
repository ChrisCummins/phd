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
#include "gpu/cldrive/opencl_type.h"

#include "phd/logging.h"
#include "phd/status_macros.h"

#include "absl/strings/str_format.h"

namespace gpu {
namespace cldrive {

phd::StatusOr<OpenClType> OpenClTypeFromString(const string& type_name) {
  if (!type_name.compare("bool")) {
    return OpenClType::BOOL;
  } else if (!type_name.compare("char")) {
    return OpenClType::CHAR;
  } else if (!type_name.compare("unsigned char") ||
             !type_name.compare("uchar")) {
    return OpenClType::UCHAR;
  } else if (!type_name.compare("short")) {
    return OpenClType::SHORT;
  } else if (!type_name.compare("unsigned short") ||
             !type_name.compare("ushort")) {
    return OpenClType::USHORT;
  } else if (!type_name.compare("int")) {
    return OpenClType::INT;
  } else if (!type_name.compare("unsigned int") || !type_name.compare("uint")) {
    return OpenClType::UINT;
  } else if (!type_name.compare("long")) {
    return OpenClType::LONG;
  } else if (!type_name.compare("unsigned long") ||
             !type_name.compare("ulong")) {
    return OpenClType::ULONG;
  } else if (!type_name.compare("float")) {
    return OpenClType::FLOAT;
  } else if (!type_name.compare("double")) {
    return OpenClType::DOUBLE;
  } else if (!type_name.compare("half")) {
    return OpenClType::HALF;
    // Vector types.
  } else if (!type_name.compare("char2")) {
    return OpenClType::CHAR2;
  } else if (!type_name.compare("char3")) {
    return OpenClType::CHAR3;
  } else if (!type_name.compare("char4")) {
    return OpenClType::CHAR4;
  } else if (!type_name.compare("char8")) {
    return OpenClType::CHAR8;
  } else if (!type_name.compare("char16")) {
    return OpenClType::CHAR16;
  } else if (!type_name.compare("uchar2")) {
    return OpenClType::UCHAR2;
  } else if (!type_name.compare("uchar3")) {
    return OpenClType::UCHAR3;
  } else if (!type_name.compare("uchar4")) {
    return OpenClType::UCHAR4;
  } else if (!type_name.compare("uchar8")) {
    return OpenClType::UCHAR8;
  } else if (!type_name.compare("uchar16")) {
    return OpenClType::UCHAR16;
  } else if (!type_name.compare("short2")) {
    return OpenClType::SHORT2;
  } else if (!type_name.compare("short3")) {
    return OpenClType::SHORT3;
  } else if (!type_name.compare("short4")) {
    return OpenClType::SHORT4;
  } else if (!type_name.compare("short8")) {
    return OpenClType::SHORT8;
  } else if (!type_name.compare("short16")) {
    return OpenClType::SHORT16;
  } else if (!type_name.compare("ushort2")) {
    return OpenClType::USHORT2;
  } else if (!type_name.compare("ushort3")) {
    return OpenClType::USHORT3;
  } else if (!type_name.compare("ushort4")) {
    return OpenClType::USHORT4;
  } else if (!type_name.compare("ushort8")) {
    return OpenClType::USHORT8;
  } else if (!type_name.compare("ushort16")) {
    return OpenClType::USHORT16;
  } else if (!type_name.compare("int2")) {
    return OpenClType::INT2;
  } else if (!type_name.compare("int3")) {
    return OpenClType::INT3;
  } else if (!type_name.compare("int4")) {
    return OpenClType::INT4;
  } else if (!type_name.compare("int8")) {
    return OpenClType::INT8;
  } else if (!type_name.compare("int16")) {
    return OpenClType::INT16;
  } else if (!type_name.compare("uint2")) {
    return OpenClType::UINT2;
  } else if (!type_name.compare("uint3")) {
    return OpenClType::UINT3;
  } else if (!type_name.compare("uint4")) {
    return OpenClType::UINT4;
  } else if (!type_name.compare("uint8")) {
    return OpenClType::UINT8;
  } else if (!type_name.compare("uint16")) {
    return OpenClType::UINT16;
  } else if (!type_name.compare("long2")) {
    return OpenClType::LONG2;
  } else if (!type_name.compare("long3")) {
    return OpenClType::LONG3;
  } else if (!type_name.compare("long4")) {
    return OpenClType::LONG4;
  } else if (!type_name.compare("long8")) {
    return OpenClType::LONG8;
  } else if (!type_name.compare("long16")) {
    return OpenClType::LONG16;
  } else if (!type_name.compare("ulong2")) {
    return OpenClType::ULONG2;
  } else if (!type_name.compare("ulong3")) {
    return OpenClType::ULONG3;
  } else if (!type_name.compare("ulong4")) {
    return OpenClType::ULONG4;
  } else if (!type_name.compare("ulong8")) {
    return OpenClType::ULONG8;
  } else if (!type_name.compare("ulong16")) {
    return OpenClType::ULONG16;
  } else if (!type_name.compare("float2")) {
    return OpenClType::FLOAT2;
  } else if (!type_name.compare("float3")) {
    return OpenClType::FLOAT3;
  } else if (!type_name.compare("float4")) {
    return OpenClType::FLOAT4;
  } else if (!type_name.compare("float8")) {
    return OpenClType::FLOAT8;
  } else if (!type_name.compare("float16")) {
    return OpenClType::FLOAT16;
  } else if (!type_name.compare("double2")) {
    return OpenClType::DOUBLE2;
  } else if (!type_name.compare("double3")) {
    return OpenClType::DOUBLE3;
  } else if (!type_name.compare("double4")) {
    return OpenClType::DOUBLE4;
  } else if (!type_name.compare("double8")) {
    return OpenClType::DOUBLE8;
  } else if (!type_name.compare("double16")) {
    return OpenClType::DOUBLE16;
  } else if (!type_name.compare("half2")) {
    return OpenClType::HALF2;
  } else if (!type_name.compare("half3")) {
    return OpenClType::HALF3;
  } else if (!type_name.compare("half4")) {
    return OpenClType::HALF4;
  } else if (!type_name.compare("half8")) {
    return OpenClType::HALF8;
  } else if (!type_name.compare("half16")) {
    return OpenClType::HALF16;
  } else {
    return phd::Status(phd::error::Code::INVALID_ARGUMENT, type_name);
  }
}

namespace opencl_type {

namespace {

template <typename T, typename Y>
string Vec2ToString(const Y& value) {
  return absl::StrFormat("{%s,%s}", ToString<T>(value.s[0]),
                         ToString<T>(value.s[1]));
}

template <typename T, typename Y>
string Vec3ToString(const Y& value) {
  return absl::StrFormat("{%s,%s,%s}", ToString<T>(value.s[0]),
                         ToString<T>(value.s[1]), ToString<T>(value.s[2]));
}

template <typename T, typename Y>
string Vec4ToString(const Y& value) {
  return absl::StrFormat("{%s,%s,%s,%s}", ToString<T>(value.s[0]),
                         ToString<T>(value.s[1]), ToString<T>(value.s[2]),
                         ToString<T>(value.s[3]));
}

template <typename T, typename Y>
string Vec8ToString(const Y& value) {
  return absl::StrFormat("{%s,%s,%s,%s,%s,%s,%s,%s}", ToString<T>(value.s[0]),
                         ToString<T>(value.s[1]), ToString<T>(value.s[2]),
                         ToString<T>(value.s[3]), ToString<T>(value.s[4]),
                         ToString<T>(value.s[5]), ToString<T>(value.s[6]),
                         ToString<T>(value.s[7]));
}

template <typename T, typename Y>
string Vec16ToString(const Y& value) {
  return absl::StrFormat("{%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s}",
                         ToString<T>(value.s[0]), ToString<T>(value.s[1]),
                         ToString<T>(value.s[2]), ToString<T>(value.s[3]),
                         ToString<T>(value.s[4]), ToString<T>(value.s[5]),
                         ToString<T>(value.s[6]), ToString<T>(value.s[7]),
                         ToString<T>(value.s[8]), ToString<T>(value.s[9]),
                         ToString<T>(value.s[10]), ToString<T>(value.s[11]),
                         ToString<T>(value.s[12]), ToString<T>(value.s[13]),
                         ToString<T>(value.s[14]), ToString<T>(value.s[15]));
}

template <typename T, typename Y>
T Vec2MakeScalar(const int& value) {
  return {MakeScalar<Y>(value), MakeScalar<Y>(value)};
}

template <typename T, typename Y>
T Vec3MakeScalar(const int& value) {
  return {
      MakeScalar<Y>(value),  // s0
      MakeScalar<Y>(value),  // s1
      MakeScalar<Y>(value)   // s2
  };
}

template <typename T, typename Y>
T Vec4MakeScalar(const int& value) {
  return {
      MakeScalar<Y>(value),  // s0
      MakeScalar<Y>(value),  // s1
      MakeScalar<Y>(value),  // s2
      MakeScalar<Y>(value)   // s3
  };
}

template <typename T, typename Y>
T Vec8MakeScalar(const int& value) {
  return {
      MakeScalar<Y>(value),  // s0
      MakeScalar<Y>(value),  // s1
      MakeScalar<Y>(value),  // s2
      MakeScalar<Y>(value),  // s3
      MakeScalar<Y>(value),  // s4
      MakeScalar<Y>(value),  // s5
      MakeScalar<Y>(value),  // s6
      MakeScalar<Y>(value)   // s7
  };
}

template <typename T, typename Y>
T Vec16MakeScalar(const int& value) {
  return {
      MakeScalar<Y>(value),  // s0
      MakeScalar<Y>(value),  // s1
      MakeScalar<Y>(value),  // s2
      MakeScalar<Y>(value),  // s3
      MakeScalar<Y>(value),  // s4
      MakeScalar<Y>(value),  // s5
      MakeScalar<Y>(value),  // s6
      MakeScalar<Y>(value),  // s7
      MakeScalar<Y>(value),  // s8
      MakeScalar<Y>(value),  // s9
      MakeScalar<Y>(value),  // s10
      MakeScalar<Y>(value),  // s11
      MakeScalar<Y>(value),  // s12
      MakeScalar<Y>(value),  // s13
      MakeScalar<Y>(value),  // s14
      MakeScalar<Y>(value)   // s15
  };
}

template <typename T, typename Y>
bool Vec2Equal(const Y& lhs, const Y& rhs) {
  return Equal<T>(lhs.s[0], rhs.s[0]) && Equal<T>(lhs.s[1], rhs.s[1]);
}

template <typename T, typename Y>
bool Vec3Equal(const Y& lhs, const Y& rhs) {
  return (Equal<T>(lhs.s[0], rhs.s[0]) && Equal<T>(lhs.s[1], rhs.s[1]) &&
          Equal<T>(lhs.s[2], rhs.s[2]));
}

template <typename T, typename Y>
bool Vec4Equal(const Y& lhs, const Y& rhs) {
  return (Equal<T>(lhs.s[0], rhs.s[0]) && Equal<T>(lhs.s[1], rhs.s[1]) &&
          Equal<T>(lhs.s[2], rhs.s[2]) && Equal<T>(lhs.s[3], rhs.s[3]));
}

template <typename T, typename Y>
bool Vec8Equal(const Y& lhs, const Y& rhs) {
  return (Equal<T>(lhs.s[0], rhs.s[0]) && Equal<T>(lhs.s[1], rhs.s[1]) &&
          Equal<T>(lhs.s[2], rhs.s[2]) && Equal<T>(lhs.s[3], rhs.s[3]) &&
          Equal<T>(lhs.s[4], rhs.s[4]) && Equal<T>(lhs.s[5], rhs.s[5]) &&
          Equal<T>(lhs.s[6], rhs.s[6]) && Equal<T>(lhs.s[7], rhs.s[7]));
}

template <typename T, typename Y>
bool Vec16Equal(const Y& lhs, const Y& rhs) {
  return (Equal<T>(lhs.s[0], rhs.s[0]) && Equal<T>(lhs.s[1], rhs.s[1]) &&
          Equal<T>(lhs.s[2], rhs.s[2]) && Equal<T>(lhs.s[3], rhs.s[3]) &&
          Equal<T>(lhs.s[4], rhs.s[4]) && Equal<T>(lhs.s[5], rhs.s[5]) &&
          Equal<T>(lhs.s[6], rhs.s[6]) && Equal<T>(lhs.s[7], rhs.s[7]) &&
          Equal<T>(lhs.s[8], rhs.s[8]) && Equal<T>(lhs.s[9], rhs.s[9]) &&
          Equal<T>(lhs.s[10], rhs.s[10]) && Equal<T>(lhs.s[11], rhs.s[11]) &&
          Equal<T>(lhs.s[12], rhs.s[12]) && Equal<T>(lhs.s[13], rhs.s[13]) &&
          Equal<T>(lhs.s[14], rhs.s[14]) && Equal<T>(lhs.s[15], rhs.s[15]));
}

}  // anonymous namespace

// Vector type overloads ---------------------------------------------------

template <>
cl_char2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_char2, cl_char>(value);
}

template <>
cl_uchar2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_uchar2, cl_uchar>(value);
}

template <>
cl_short2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_short2, cl_short>(value);
}

template <>
cl_ushort2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_ushort2, cl_ushort>(value);
}

template <>
cl_int2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_int2, cl_int>(value);
}

template <>
cl_uint2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_uint2, cl_uint>(value);
}

template <>
cl_long2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_long2, cl_long>(value);
}

template <>
cl_ulong2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_ulong2, cl_ulong>(value);
}

template <>
cl_float2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_float2, cl_float>(value);
}

template <>
cl_double2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_double2, cl_double>(value);
}

template <>
cl_half2 MakeScalar(const int& value) {
  return Vec2MakeScalar<cl_half2, cl_half>(value);
}

template <>
cl_char4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_char4, cl_char>(value);
}

template <>
cl_uchar4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_uchar4, cl_uchar>(value);
}

template <>
cl_short4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_short4, cl_short>(value);
}

template <>
cl_ushort4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_ushort4, cl_ushort>(value);
}

template <>
cl_int4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_int4, cl_int>(value);
}

template <>
cl_uint4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_uint4, cl_uint>(value);
}

template <>
cl_long4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_long4, cl_long>(value);
}

template <>
cl_ulong4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_ulong4, cl_ulong>(value);
}

template <>
cl_float4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_float4, cl_float>(value);
}

template <>
cl_double4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_double4, cl_double>(value);
}

template <>
cl_half4 MakeScalar(const int& value) {
  return Vec4MakeScalar<cl_half4, cl_half>(value);
}

template <>
cl_char8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_char8, cl_char>(value);
}

template <>
cl_uchar8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_uchar8, cl_uchar>(value);
}

template <>
cl_short8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_short8, cl_short>(value);
}

template <>
cl_ushort8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_ushort8, cl_ushort>(value);
}

template <>
cl_int8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_int8, cl_int>(value);
}

template <>
cl_uint8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_uint8, cl_uint>(value);
}

template <>
cl_long8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_long8, cl_long>(value);
}

template <>
cl_ulong8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_ulong8, cl_ulong>(value);
}

template <>
cl_float8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_float8, cl_float>(value);
}

template <>
cl_double8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_double8, cl_double>(value);
}

template <>
cl_half8 MakeScalar(const int& value) {
  return Vec8MakeScalar<cl_half8, cl_half>(value);
}

template <>
cl_char16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_char16, cl_char>(value);
}

template <>
cl_uchar16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_uchar16, cl_uchar>(value);
}

template <>
cl_short16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_short16, cl_short>(value);
}

template <>
cl_ushort16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_ushort16, cl_ushort>(value);
}

template <>
cl_int16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_int16, cl_int>(value);
}

template <>
cl_uint16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_uint16, cl_uint>(value);
}

template <>
cl_long16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_long16, cl_long>(value);
}

template <>
cl_ulong16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_ulong16, cl_ulong>(value);
}

template <>
cl_float16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_float16, cl_float>(value);
}

template <>
cl_double16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_double16, cl_double>(value);
}

template <>
cl_half16 MakeScalar(const int& value) {
  return Vec16MakeScalar<cl_half16, cl_half>(value);
}

template <>
bool Equal(const cl_char2& lhs, const cl_char2& rhs) {
  return Vec2Equal<cl_char>(lhs, rhs);
}

template <>
bool Equal(const cl_uchar2& lhs, const cl_uchar2& rhs) {
  return Vec2Equal<cl_uchar>(lhs, rhs);
}

template <>
bool Equal(const cl_short2& lhs, const cl_short2& rhs) {
  return Vec2Equal<cl_short>(lhs, rhs);
}

template <>
bool Equal(const cl_ushort2& lhs, const cl_ushort2& rhs) {
  return Vec2Equal<cl_ushort>(lhs, rhs);
}

template <>
bool Equal(const cl_int2& lhs, const cl_int2& rhs) {
  return Vec2Equal<cl_int>(lhs, rhs);
}

template <>
bool Equal(const cl_uint2& lhs, const cl_uint2& rhs) {
  return Vec2Equal<cl_uint>(lhs, rhs);
}

template <>
bool Equal(const cl_long2& lhs, const cl_long2& rhs) {
  return Vec2Equal<cl_long>(lhs, rhs);
}

template <>
bool Equal(const cl_ulong2& lhs, const cl_ulong2& rhs) {
  return Vec2Equal<cl_ulong>(lhs, rhs);
}

template <>
bool Equal(const cl_float2& lhs, const cl_float2& rhs) {
  return Vec2Equal<cl_float>(lhs, rhs);
}

template <>
bool Equal(const cl_double2& lhs, const cl_double2& rhs) {
  return Vec2Equal<cl_double>(lhs, rhs);
}

template <>
bool Equal(const cl_half2& lhs, const cl_half2& rhs) {
  return Vec2Equal<cl_half>(lhs, rhs);
}

template <>
bool Equal(const cl_char4& lhs, const cl_char4& rhs) {
  return Vec4Equal<cl_char>(lhs, rhs);
}

template <>
bool Equal(const cl_uchar4& lhs, const cl_uchar4& rhs) {
  return Vec4Equal<cl_uchar>(lhs, rhs);
}

template <>
bool Equal(const cl_short4& lhs, const cl_short4& rhs) {
  return Vec4Equal<cl_short>(lhs, rhs);
}

template <>
bool Equal(const cl_ushort4& lhs, const cl_ushort4& rhs) {
  return Vec4Equal<cl_ushort>(lhs, rhs);
}

template <>
bool Equal(const cl_int4& lhs, const cl_int4& rhs) {
  return Vec4Equal<cl_int>(lhs, rhs);
}

template <>
bool Equal(const cl_uint4& lhs, const cl_uint4& rhs) {
  return Vec4Equal<cl_uint>(lhs, rhs);
}

template <>
bool Equal(const cl_long4& lhs, const cl_long4& rhs) {
  return Vec4Equal<cl_long>(lhs, rhs);
}

template <>
bool Equal(const cl_ulong4& lhs, const cl_ulong4& rhs) {
  return Vec4Equal<cl_ulong>(lhs, rhs);
}

template <>
bool Equal(const cl_float4& lhs, const cl_float4& rhs) {
  return Vec4Equal<cl_float>(lhs, rhs);
}

template <>
bool Equal(const cl_double4& lhs, const cl_double4& rhs) {
  return Vec4Equal<cl_double>(lhs, rhs);
}

template <>
bool Equal(const cl_half4& lhs, const cl_half4& rhs) {
  return Vec4Equal<cl_half>(lhs, rhs);
}

template <>
bool Equal(const cl_char8& lhs, const cl_char8& rhs) {
  return Vec8Equal<cl_char>(lhs, rhs);
}

template <>
bool Equal(const cl_uchar8& lhs, const cl_uchar8& rhs) {
  return Vec8Equal<cl_uchar>(lhs, rhs);
}

template <>
bool Equal(const cl_short8& lhs, const cl_short8& rhs) {
  return Vec8Equal<cl_short>(lhs, rhs);
}

template <>
bool Equal(const cl_ushort8& lhs, const cl_ushort8& rhs) {
  return Vec8Equal<cl_ushort>(lhs, rhs);
}

template <>
bool Equal(const cl_int8& lhs, const cl_int8& rhs) {
  return Vec8Equal<cl_int>(lhs, rhs);
}

template <>
bool Equal(const cl_uint8& lhs, const cl_uint8& rhs) {
  return Vec8Equal<cl_uint>(lhs, rhs);
}

template <>
bool Equal(const cl_long8& lhs, const cl_long8& rhs) {
  return Vec8Equal<cl_long>(lhs, rhs);
}

template <>
bool Equal(const cl_ulong8& lhs, const cl_ulong8& rhs) {
  return Vec8Equal<cl_ulong>(lhs, rhs);
}

template <>
bool Equal(const cl_float8& lhs, const cl_float8& rhs) {
  return Vec8Equal<cl_float>(lhs, rhs);
}

template <>
bool Equal(const cl_double8& lhs, const cl_double8& rhs) {
  return Vec8Equal<cl_double>(lhs, rhs);
}

template <>
bool Equal(const cl_half8& lhs, const cl_half8& rhs) {
  return Vec8Equal<cl_half>(lhs, rhs);
}

template <>
bool Equal(const cl_char16& lhs, const cl_char16& rhs) {
  return Vec16Equal<cl_char>(lhs, rhs);
}

template <>
bool Equal(const cl_uchar16& lhs, const cl_uchar16& rhs) {
  return Vec16Equal<cl_uchar>(lhs, rhs);
}

template <>
bool Equal(const cl_short16& lhs, const cl_short16& rhs) {
  return Vec16Equal<cl_short>(lhs, rhs);
}

template <>
bool Equal(const cl_ushort16& lhs, const cl_ushort16& rhs) {
  return Vec16Equal<cl_ushort>(lhs, rhs);
}

template <>
bool Equal(const cl_int16& lhs, const cl_int16& rhs) {
  return Vec16Equal<cl_int>(lhs, rhs);
}

template <>
bool Equal(const cl_uint16& lhs, const cl_uint16& rhs) {
  return Vec16Equal<cl_uint>(lhs, rhs);
}

template <>
bool Equal(const cl_long16& lhs, const cl_long16& rhs) {
  return Vec16Equal<cl_long>(lhs, rhs);
}

template <>
bool Equal(const cl_ulong16& lhs, const cl_ulong16& rhs) {
  return Vec16Equal<cl_ulong>(lhs, rhs);
}

template <>
bool Equal(const cl_float16& lhs, const cl_float16& rhs) {
  return Vec16Equal<cl_float>(lhs, rhs);
}

template <>
bool Equal(const cl_double16& lhs, const cl_double16& rhs) {
  return Vec16Equal<cl_double>(lhs, rhs);
}

template <>
bool Equal(const cl_half16& lhs, const cl_half16& rhs) {
  return Vec16Equal<cl_half>(lhs, rhs);
}

template <>
string ToString(const cl_char2& value) {
  return Vec2ToString<cl_char>(value);
}
template <>
string ToString(const cl_uchar2& value) {
  return Vec2ToString<cl_uchar>(value);
}
template <>
string ToString(const cl_short2& value) {
  return Vec2ToString<cl_short>(value);
}
template <>
string ToString(const cl_ushort2& value) {
  return Vec2ToString<cl_ushort>(value);
}
template <>
string ToString(const cl_int2& value) {
  return Vec2ToString<cl_int>(value);
}
template <>
string ToString(const cl_uint2& value) {
  return Vec2ToString<cl_uint>(value);
}
template <>
string ToString(const cl_long2& value) {
  return Vec2ToString<cl_long>(value);
}
template <>
string ToString(const cl_ulong2& value) {
  return Vec2ToString<cl_ulong>(value);
}
template <>
string ToString(const cl_float2& value) {
  return Vec2ToString<cl_float>(value);
}
template <>
string ToString(const cl_double2& value) {
  return Vec2ToString<cl_double>(value);
}
template <>
string ToString(const cl_half2& value) {
  return Vec2ToString<cl_half>(value);
}
template <>
string ToString(const cl_char4& value) {
  return Vec4ToString<cl_char>(value);
}
template <>
string ToString(const cl_uchar4& value) {
  return Vec4ToString<cl_uchar>(value);
}
template <>
string ToString(const cl_short4& value) {
  return Vec4ToString<cl_short>(value);
}
template <>
string ToString(const cl_ushort4& value) {
  return Vec4ToString<cl_ushort>(value);
}
template <>
string ToString(const cl_int4& value) {
  return Vec4ToString<cl_int>(value);
}
template <>
string ToString(const cl_uint4& value) {
  return Vec4ToString<cl_uint>(value);
}
template <>
string ToString(const cl_long4& value) {
  return Vec4ToString<cl_long>(value);
}
template <>
string ToString(const cl_ulong4& value) {
  return Vec4ToString<cl_ulong>(value);
}
template <>
string ToString(const cl_float4& value) {
  return Vec4ToString<cl_float>(value);
}
template <>
string ToString(const cl_double4& value) {
  return Vec4ToString<cl_double>(value);
}
template <>
string ToString(const cl_half4& value) {
  return Vec4ToString<cl_half>(value);
}
template <>
string ToString(const cl_char8& value) {
  return Vec8ToString<cl_char>(value);
}
template <>
string ToString(const cl_uchar8& value) {
  return Vec8ToString<cl_uchar>(value);
}
template <>
string ToString(const cl_short8& value) {
  return Vec8ToString<cl_short>(value);
}
template <>
string ToString(const cl_ushort8& value) {
  return Vec8ToString<cl_ushort>(value);
}
template <>
string ToString(const cl_int8& value) {
  return Vec8ToString<cl_int>(value);
}
template <>
string ToString(const cl_uint8& value) {
  return Vec8ToString<cl_uint>(value);
}
template <>
string ToString(const cl_long8& value) {
  return Vec8ToString<cl_long>(value);
}
template <>
string ToString(const cl_ulong8& value) {
  return Vec8ToString<cl_ulong>(value);
}
template <>
string ToString(const cl_float8& value) {
  return Vec8ToString<cl_float>(value);
}
template <>
string ToString(const cl_double8& value) {
  return Vec8ToString<cl_double>(value);
}
template <>
string ToString(const cl_half8& value) {
  return Vec8ToString<cl_half>(value);
}
template <>
string ToString(const cl_char16& value) {
  return Vec16ToString<cl_char>(value);
}
template <>
string ToString(const cl_uchar16& value) {
  return Vec16ToString<cl_uchar>(value);
}
template <>
string ToString(const cl_short16& value) {
  return Vec16ToString<cl_short>(value);
}
template <>
string ToString(const cl_ushort16& value) {
  return Vec16ToString<cl_ushort>(value);
}
template <>
string ToString(const cl_int16& value) {
  return Vec16ToString<cl_int>(value);
}
template <>
string ToString(const cl_uint16& value) {
  return Vec16ToString<cl_uint>(value);
}
template <>
string ToString(const cl_long16& value) {
  return Vec16ToString<cl_long>(value);
}
template <>
string ToString(const cl_ulong16& value) {
  return Vec16ToString<cl_ulong>(value);
}
template <>
string ToString(const cl_float16& value) {
  return Vec16ToString<cl_float>(value);
}
template <>
string ToString(const cl_double16& value) {
  return Vec16ToString<cl_double>(value);
}
template <>
string ToString(const cl_half16& value) {
  return Vec16ToString<cl_half>(value);
}

}  // namespace opencl_type

}  // namespace cldrive
}  // namespace gpu
