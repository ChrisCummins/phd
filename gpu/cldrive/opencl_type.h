#pragma once

#include <cstdlib>

#include "phd/statusor.h"
#include "phd/string.h"
#include "third_party/opencl/cl.hpp"

#include "absl/strings/str_cat.h"
#include "boost/variant.hpp"

namespace gpu {
namespace cldrive {

// The list of supported OpenCL types.
enum OpenClTypeEnum {
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

using Scalar =
    boost::variant<cl_bool, cl_char, cl_uchar, cl_short, cl_ushort, cl_int,
                   cl_uint, cl_long, cl_ulong, cl_float, cl_double, cl_half>;
using Array = std::vector<Scalar>;

class OpenClType {
 public:
  OpenClType() : type_num_(OpenClTypeEnum::DEFAULT_UNKNOWN){};

  explicit OpenClType(const OpenClTypeEnum& type_num) : type_num_(type_num){};

  bool ElementsAreEqual(const Scalar& lhs, const Scalar& rhs) const {
    const cl_int *cl_int_lhs = boost::get<cl_int>(&lhs),
                 *cl_int_rhs = boost::get<cl_int>(&rhs);
    if (cl_int_lhs && cl_int_rhs) {
      return *cl_int_lhs == *cl_int_rhs;
    }
    return false;
  }

  string FormatToString(const Scalar& value) const {
    string s = "";
    switch (type_num()) {
      case OpenClTypeEnum::BOOL:
        absl::StrAppend(&s, *boost::get<cl_bool>(&value));
        return s;
      default:
        return s;
    }
  }

  Scalar Create(const int& value) const {
    switch (type_num()) {
      case OpenClTypeEnum::BOOL:
        return true;
      default:
        return 1;
    }
  }

  const OpenClTypeEnum& type_num() const { return type_num_; }

  static phd::StatusOr<OpenClType> FromString(const string& type_name);

 private:
  OpenClTypeEnum type_num_;
};

}  // namespace cldrive
}  // namespace gpu
