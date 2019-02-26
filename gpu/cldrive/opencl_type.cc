#include "gpu/cldrive/opencl_type.h"
#include "opencl_type.h"

#include "phd/status_macros.h"

namespace gpu {
namespace cldrive {

phd::StatusOr<OpenClTypeEnum> OpenClTypeEnumFromString(
    const string& type_name) {
  if (!type_name.compare("bool")) {
    return OpenClTypeEnum::BOOL;
  } else if (!type_name.compare("char")) {
    return OpenClTypeEnum::CHAR;
  } else if (!type_name.compare("unsigned char") ||
             !type_name.compare("uchar")) {
    return OpenClTypeEnum::UCHAR;
  } else if (!type_name.compare("short")) {
    return OpenClTypeEnum::SHORT;
  } else if (!type_name.compare("unsigned short") ||
             !type_name.compare("ushort")) {
    return OpenClTypeEnum::USHORT;
  } else if (!type_name.compare("int")) {
    return OpenClTypeEnum::INT;
  } else if (!type_name.compare("unsigned int") || !type_name.compare("uint")) {
    return OpenClTypeEnum::UINT;
  } else if (!type_name.compare("long")) {
    return OpenClTypeEnum::LONG;
  } else if (!type_name.compare("unsigned long") ||
             !type_name.compare("ulong")) {
    return OpenClTypeEnum::ULONG;
  } else if (!type_name.compare("float")) {
    return OpenClTypeEnum::FLOAT;
  } else if (!type_name.compare("double")) {
    return OpenClTypeEnum::DOUBLE;
  } else if (!type_name.compare("half")) {
    return OpenClTypeEnum::HALF;
    // Vector types.
  } else if (!type_name.compare("char2")) {
    return OpenClTypeEnum::CHAR2;
    //  } else if (!type_name.compare("char3")) {
    //    return OpenClTypeEnum::CHAR3;
    //  } else if (!type_name.compare("char4")) {
    //    return OpenClTypeEnum::CHAR4;
    //  } else if (!type_name.compare("char8")) {
    //    return OpenClTypeEnum::CHAR8;
    //  } else if (!type_name.compare("char16")) {
    //    return OpenClTypeEnum::CHAR16;
    //  } else if (!type_name.compare("uchar2")) {
    //    return OpenClTypeEnum::UCHAR2;
    //  } else if (!type_name.compare("uchar3")) {
    //    return OpenClTypeEnum::UCHAR3;
    //  } else if (!type_name.compare("uchar4")) {
    //    return OpenClTypeEnum::UCHAR4;
    //  } else if (!type_name.compare("uchar8")) {
    //    return OpenClTypeEnum::UCHAR8;
    //  } else if (!type_name.compare("uchar16")) {
    //    return OpenClTypeEnum::UCHAR16;
    //  } else if (!type_name.compare("short2")) {
    //    return OpenClTypeEnum::SHORT2;
    //  } else if (!type_name.compare("short3")) {
    //    return OpenClTypeEnum::SHORT3;
    //  } else if (!type_name.compare("short4")) {
    //    return OpenClTypeEnum::SHORT4;
    //  } else if (!type_name.compare("short8")) {
    //    return OpenClTypeEnum::SHORT8;
    //  } else if (!type_name.compare("short16")) {
    //    return OpenClTypeEnum::SHORT16;
    //  } else if (!type_name.compare("ushort2")) {
    //    return OpenClTypeEnum::USHORT2;
    //  } else if (!type_name.compare("ushort3")) {
    //    return OpenClTypeEnum::USHORT3;
    //  } else if (!type_name.compare("ushort4")) {
    //    return OpenClTypeEnum::USHORT4;
    //  } else if (!type_name.compare("ushort8")) {
    //    return OpenClTypeEnum::USHORT8;
    //  } else if (!type_name.compare("ushort16")) {
    //    return OpenClTypeEnum::USHORT16;
    //  } else if (!type_name.compare("int2")) {
    //    return OpenClTypeEnum::INT2;
    //  } else if (!type_name.compare("int3")) {
    //    return OpenClTypeEnum::INT3;
    //  } else if (!type_name.compare("int4")) {
    //    return OpenClTypeEnum::INT4;
    //  } else if (!type_name.compare("int8")) {
    //    return OpenClTypeEnum::INT8;
    //  } else if (!type_name.compare("int16")) {
    //    return OpenClTypeEnum::INT16;
    //  } else if (!type_name.compare("uint2")) {
    //    return OpenClTypeEnum::UINT2;
    //  } else if (!type_name.compare("uint3")) {
    //    return OpenClTypeEnum::UINT3;
    //  } else if (!type_name.compare("uint4")) {
    //    return OpenClTypeEnum::UINT4;
    //  } else if (!type_name.compare("uint8")) {
    //    return OpenClTypeEnum::UINT8;
    //  } else if (!type_name.compare("uint16")) {
    //    return OpenClTypeEnum::UINT16;
    //  } else if (!type_name.compare("long2")) {
    //    return OpenClTypeEnum::LONG2;
    //  } else if (!type_name.compare("long3")) {
    //    return OpenClTypeEnum::LONG3;
    //  } else if (!type_name.compare("long4")) {
    //    return OpenClTypeEnum::LONG4;
    //  } else if (!type_name.compare("long8")) {
    //    return OpenClTypeEnum::LONG8;
    //  } else if (!type_name.compare("long16")) {
    //    return OpenClTypeEnum::LONG16;
    //  } else if (!type_name.compare("ulong2")) {
    //    return OpenClTypeEnum::ULONG2;
    //  } else if (!type_name.compare("ulong3")) {
    //    return OpenClTypeEnum::ULONG3;
    //  } else if (!type_name.compare("ulong4")) {
    //    return OpenClTypeEnum::ULONG4;
    //  } else if (!type_name.compare("ulong8")) {
    //    return OpenClTypeEnum::ULONG8;
    //  } else if (!type_name.compare("ulong16")) {
    //    return OpenClTypeEnum::ULONG16;
    //  } else if (!type_name.compare("float2")) {
    //    return OpenClTypeEnum::FLOAT2;
    //  } else if (!type_name.compare("float3")) {
    //    return OpenClTypeEnum::FLOAT3;
    //  } else if (!type_name.compare("float4")) {
    //    return OpenClTypeEnum::FLOAT4;
    //  } else if (!type_name.compare("float8")) {
    //    return OpenClTypeEnum::FLOAT8;
    //  } else if (!type_name.compare("float16")) {
    //    return OpenClTypeEnum::FLOAT16;
    //  } else if (!type_name.compare("double2")) {
    //    return OpenClTypeEnum::DOUBLE2;
    //  } else if (!type_name.compare("double3")) {
    //    return OpenClTypeEnum::DOUBLE3;
    //  } else if (!type_name.compare("double4")) {
    //    return OpenClTypeEnum::DOUBLE4;
    //  } else if (!type_name.compare("double8")) {
    //    return OpenClTypeEnum::DOUBLE8;
    //  } else if (!type_name.compare("double16")) {
    //    return OpenClTypeEnum::DOUBLE16;
    //  } else if (!type_name.compare("half2")) {
    //    return OpenClTypeEnum::HALF2;
    //  } else if (!type_name.compare("half3")) {
    //    return OpenClTypeEnum::HALF3;
    //  } else if (!type_name.compare("half4")) {
    //    return OpenClTypeEnum::HALF4;
    //  } else if (!type_name.compare("half8")) {
    //    return OpenClTypeEnum::HALF8;
    //  } else if (!type_name.compare("half16")) {
    //    return OpenClTypeEnum::HALF16;
  } else {
    return phd::Status(phd::error::Code::INVALID_ARGUMENT, type_name);
  }
}

phd::StatusOr<OpenClType> OpenClType::FromString(const string& type_name) {
  OpenClTypeEnum num;
  ASSIGN_OR_RETURN(num, OpenClTypeEnumFromString(type_name));
  return OpenClType(num);
}

}  // namespace cldrive
}  // namespace gpu
