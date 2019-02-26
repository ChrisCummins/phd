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

template <typename T, typename U>
bool Equal(const T& left, const U& right) {
  return false;
};

template <typename T>
bool Equal(const T& left, const T& right) {
  return left == right;
};

template <typename T>
string ToString(const T& t) {
  string s = "";
  absl::StrAppend(&s, t);
  return s;
}

template <typename T>
T MakeScalar(const int& value) {
  return T(value);
}

}  // namespace opencl_type

}  // namespace cldrive
}  // namespace gpu
