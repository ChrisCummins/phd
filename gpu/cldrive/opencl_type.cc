#include "gpu/cldrive/opencl_type.h"

#include "phd/logging.h"
#include "phd/status_macros.h"

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
    //  } else if (!type_name.compare("char2")) {
    //    return OpenClType::CHAR2;
    //  } else if (!type_name.compare("char3")) {
    //    return OpenClType::CHAR3;
    //  } else if (!type_name.compare("char4")) {
    //    return OpenClType::CHAR4;
    //  } else if (!type_name.compare("char8")) {
    //    return OpenClType::CHAR8;
    //  } else if (!type_name.compare("char16")) {
    //    return OpenClType::CHAR16;
    //  } else if (!type_name.compare("uchar2")) {
    //    return OpenClType::UCHAR2;
    //  } else if (!type_name.compare("uchar3")) {
    //    return OpenClType::UCHAR3;
    //  } else if (!type_name.compare("uchar4")) {
    //    return OpenClType::UCHAR4;
    //  } else if (!type_name.compare("uchar8")) {
    //    return OpenClType::UCHAR8;
    //  } else if (!type_name.compare("uchar16")) {
    //    return OpenClType::UCHAR16;
    //  } else if (!type_name.compare("short2")) {
    //    return OpenClType::SHORT2;
    //  } else if (!type_name.compare("short3")) {
    //    return OpenClType::SHORT3;
    //  } else if (!type_name.compare("short4")) {
    //    return OpenClType::SHORT4;
    //  } else if (!type_name.compare("short8")) {
    //    return OpenClType::SHORT8;
    //  } else if (!type_name.compare("short16")) {
    //    return OpenClType::SHORT16;
    //  } else if (!type_name.compare("ushort2")) {
    //    return OpenClType::USHORT2;
    //  } else if (!type_name.compare("ushort3")) {
    //    return OpenClType::USHORT3;
    //  } else if (!type_name.compare("ushort4")) {
    //    return OpenClType::USHORT4;
    //  } else if (!type_name.compare("ushort8")) {
    //    return OpenClType::USHORT8;
    //  } else if (!type_name.compare("ushort16")) {
    //    return OpenClType::USHORT16;
    //  } else if (!type_name.compare("int2")) {
    //    return OpenClType::INT2;
    //  } else if (!type_name.compare("int3")) {
    //    return OpenClType::INT3;
    //  } else if (!type_name.compare("int4")) {
    //    return OpenClType::INT4;
    //  } else if (!type_name.compare("int8")) {
    //    return OpenClType::INT8;
    //  } else if (!type_name.compare("int16")) {
    //    return OpenClType::INT16;
    //  } else if (!type_name.compare("uint2")) {
    //    return OpenClType::UINT2;
    //  } else if (!type_name.compare("uint3")) {
    //    return OpenClType::UINT3;
    //  } else if (!type_name.compare("uint4")) {
    //    return OpenClType::UINT4;
    //  } else if (!type_name.compare("uint8")) {
    //    return OpenClType::UINT8;
    //  } else if (!type_name.compare("uint16")) {
    //    return OpenClType::UINT16;
    //  } else if (!type_name.compare("long2")) {
    //    return OpenClType::LONG2;
    //  } else if (!type_name.compare("long3")) {
    //    return OpenClType::LONG3;
    //  } else if (!type_name.compare("long4")) {
    //    return OpenClType::LONG4;
    //  } else if (!type_name.compare("long8")) {
    //    return OpenClType::LONG8;
    //  } else if (!type_name.compare("long16")) {
    //    return OpenClType::LONG16;
    //  } else if (!type_name.compare("ulong2")) {
    //    return OpenClType::ULONG2;
    //  } else if (!type_name.compare("ulong3")) {
    //    return OpenClType::ULONG3;
    //  } else if (!type_name.compare("ulong4")) {
    //    return OpenClType::ULONG4;
    //  } else if (!type_name.compare("ulong8")) {
    //    return OpenClType::ULONG8;
    //  } else if (!type_name.compare("ulong16")) {
    //    return OpenClType::ULONG16;
    //  } else if (!type_name.compare("float2")) {
    //    return OpenClType::FLOAT2;
    //  } else if (!type_name.compare("float3")) {
    //    return OpenClType::FLOAT3;
    //  } else if (!type_name.compare("float4")) {
    //    return OpenClType::FLOAT4;
    //  } else if (!type_name.compare("float8")) {
    //    return OpenClType::FLOAT8;
    //  } else if (!type_name.compare("float16")) {
    //    return OpenClType::FLOAT16;
    //  } else if (!type_name.compare("double2")) {
    //    return OpenClType::DOUBLE2;
    //  } else if (!type_name.compare("double3")) {
    //    return OpenClType::DOUBLE3;
    //  } else if (!type_name.compare("double4")) {
    //    return OpenClType::DOUBLE4;
    //  } else if (!type_name.compare("double8")) {
    //    return OpenClType::DOUBLE8;
    //  } else if (!type_name.compare("double16")) {
    //    return OpenClType::DOUBLE16;
    //  } else if (!type_name.compare("half2")) {
    //    return OpenClType::HALF2;
    //  } else if (!type_name.compare("half3")) {
    //    return OpenClType::HALF3;
    //  } else if (!type_name.compare("half4")) {
    //    return OpenClType::HALF4;
    //  } else if (!type_name.compare("half8")) {
    //    return OpenClType::HALF8;
    //  } else if (!type_name.compare("half16")) {
    //    return OpenClType::HALF16;
  } else {
    return phd::Status(phd::error::Code::INVALID_ARGUMENT, type_name);
  }
}

}  // namespace cldrive
}  // namespace gpu
