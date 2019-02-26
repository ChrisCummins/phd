#pragme once

#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

template <typename T>
class OpenClType {
  string ToString();
};

phd::StatusOr<OpenClType> OpenClTypeFromString(const string& type_name) {
  if (!type_name.compare("bool")) {
    return OpenClType<cl_bool>();
  } else if (!type_name.compare("char")) {
    return OpenClType<cl_char>();
  } else if (!type_name.compare("unsigned char") ||
             !type_name.compare("uchar")) {
    return OpenClType<cl_uchar>();
  } else if (!type_name.compare("short")) {
    return OpenClType<cl_short>();
  } else if (!type_name.compare("unsigned short") ||
             !type_name.compare("ushort")) {
    return OpenClType<cl_ushort>();
  } else if (!type_name.compare("int")) {
    return OpenClType<cl_int>();
  } else if (!type_name.compare("unsigned int") || !type_name.compare("uint")) {
    return OpenClType<cl_uint>();
  } else if (!type_name.compare("long")) {
    return OpenClType<cl_long>();
  } else if (!type_name.compare("unsigned long") ||
             !type_name.compare("ulong")) {
    return OpenClType<cl_ulong>();
  } else if (!type_name.compare("float")) {
    return OpenClType<cl_float>();
  } else if (!type_name.compare("double")) {
    return OpenClType<cl_double>();
  } else if (!type_name.compare("half")) {
    return OpenClType<cl_half>();
    // Vector types.
  } else if (!type_name.compare("char2")) {
    return OpenClType<cl_char2>();
  } else if (!type_name.compare("char3")) {
    return OpenClType<cl_char3>();
  } else if (!type_name.compare("char4")) {
    return OpenClType<cl_char4>();
  } else if (!type_name.compare("char8")) {
    return OpenClType<cl_char8>();
  } else if (!type_name.compare("char16")) {
    return OpenClType<cl_char16>();
  } else if (!type_name.compare("uchar2")) {
    return OpenClType<cl_uchar2>();
  } else if (!type_name.compare("uchar3")) {
    return OpenClType<cl_uchar3>();
  } else if (!type_name.compare("uchar4")) {
    return OpenClType<cl_uchar4>();
  } else if (!type_name.compare("uchar8")) {
    return OpenClType<cl_uchar8>();
  } else if (!type_name.compare("uchar16")) {
    return OpenClType<cl_uchar16>();
  } else if (!type_name.compare("short2")) {
    return OpenClType<cl_short2>();
  } else if (!type_name.compare("short3")) {
    return OpenClType<cl_short3>();
  } else if (!type_name.compare("short4")) {
    return OpenClType<cl_short4>();
  } else if (!type_name.compare("short8")) {
    return OpenClType<cl_short8>();
  } else if (!type_name.compare("short16")) {
    return OpenClType<cl_short16>();
  } else if (!type_name.compare("ushort2")) {
    return OpenClType<cl_ushort2>();
  } else if (!type_name.compare("ushort3")) {
    return OpenClType<cl_ushort3>();
  } else if (!type_name.compare("ushort4")) {
    return OpenClType<cl_ushort4>();
  } else if (!type_name.compare("ushort8")) {
    return OpenClType<cl_ushort8>();
  } else if (!type_name.compare("ushort16")) {
    return OpenClType<cl_ushort16>();
  } else if (!type_name.compare("int2")) {
    return OpenClType<cl_int2>();
  } else if (!type_name.compare("int3")) {
    return OpenClType<cl_int3>();
  } else if (!type_name.compare("int4")) {
    return OpenClType<cl_int4>();
  } else if (!type_name.compare("int8")) {
    return OpenClType<cl_int8>();
  } else if (!type_name.compare("int16")) {
    return OpenClType<cl_int16>();
  } else if (!type_name.compare("uint2")) {
    return OpenClType<cl_uint2>();
  } else if (!type_name.compare("uint3")) {
    return OpenClType<cl_uint3>();
  } else if (!type_name.compare("uint4")) {
    return OpenClType<cl_uint4>();
  } else if (!type_name.compare("uint8")) {
    return OpenClType<cl_uint8>();
  } else if (!type_name.compare("uint16")) {
    return OpenClType<cl_uint16>();
  } else if (!type_name.compare("long2")) {
    return OpenClType<cl_long2>();
  } else if (!type_name.compare("long3")) {
    return OpenClType<cl_long3>();
  } else if (!type_name.compare("long4")) {
    return OpenClType<cl_long4>();
  } else if (!type_name.compare("long8")) {
    return OpenClType<cl_long8>();
  } else if (!type_name.compare("long16")) {
    return OpenClType<cl_long16>();
  } else if (!type_name.compare("ulong2")) {
    return OpenClType<cl_ulong2>();
  } else if (!type_name.compare("ulong3")) {
    return OpenClType<cl_ulong3>();
  } else if (!type_name.compare("ulong4")) {
    return OpenClType<cl_ulong4>();
  } else if (!type_name.compare("ulong8")) {
    return OpenClType<cl_ulong8>();
  } else if (!type_name.compare("ulong16")) {
    return OpenClType<cl_ulong16>();
  } else if (!type_name.compare("float2")) {
    return OpenClType<cl_float2>();
  } else if (!type_name.compare("float3")) {
    return OpenClType<cl_float3>();
  } else if (!type_name.compare("float4")) {
    return OpenClType<cl_float4>();
  } else if (!type_name.compare("float8")) {
    return OpenClType<cl_float8>();
  } else if (!type_name.compare("float16")) {
    return OpenClType<cl_float16>();
  } else if (!type_name.compare("double2")) {
    return OpenClType<cl_double2>();
  } else if (!type_name.compare("double3")) {
    return OpenClType<cl_double3>();
  } else if (!type_name.compare("double4")) {
    return OpenClType<cl_double4>();
  } else if (!type_name.compare("double8")) {
    return OpenClType<cl_double8>();
  } else if (!type_name.compare("double16")) {
    return OpenClType<cl_double16>();
    //  } else if (!type_name.compare("half2")) {
    //    return OpenClType<cl_half2>();
    //  } else if (!type_name.compare("half3")) {
    //    return OpenClType<cl_half3>();
    //  } else if (!type_name.compare("half4")) {
    //    return OpenClType<cl_half4>();
    //  } else if (!type_name.compare("half8")) {
    //    return OpenClType<cl_half8>();
    //  } else if (!type_name.compare("half16")) {
    //    return OpenClType<cl_half16>();
  } else {
    return phd::Status(phd::error::Code::INVALID_ARGUMENT, type_name);
  }
}

}  // namespace cldrive
}  // namespace gpu
