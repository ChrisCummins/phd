/*******************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef long unsigned int size_t;
typedef long int ptrdiff_t;
typedef long int intptr_t;
typedef unsigned long int uintptr_t;
typedef __attribute__((ext_vector_type(2))) char char2;
typedef __attribute__((ext_vector_type(3))) char char3;
typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(8))) char char8;
typedef __attribute__((ext_vector_type(16))) char char16;
typedef __attribute__((ext_vector_type(2))) uchar uchar2;
typedef __attribute__((ext_vector_type(3))) uchar uchar3;
typedef __attribute__((ext_vector_type(4))) uchar uchar4;
typedef __attribute__((ext_vector_type(8))) uchar uchar8;
typedef __attribute__((ext_vector_type(16))) uchar uchar16;
typedef __attribute__((ext_vector_type(2))) short short2;
typedef __attribute__((ext_vector_type(3))) short short3;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(8))) short short8;
typedef __attribute__((ext_vector_type(16))) short short16;
typedef __attribute__((ext_vector_type(2))) ushort ushort2;
typedef __attribute__((ext_vector_type(3))) ushort ushort3;
typedef __attribute__((ext_vector_type(4))) ushort ushort4;
typedef __attribute__((ext_vector_type(8))) ushort ushort8;
typedef __attribute__((ext_vector_type(16))) ushort ushort16;
typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(3))) int int3;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(8))) int int8;
typedef __attribute__((ext_vector_type(16))) int int16;
typedef __attribute__((ext_vector_type(2))) uint uint2;
typedef __attribute__((ext_vector_type(3))) uint uint3;
typedef __attribute__((ext_vector_type(4))) uint uint4;
typedef __attribute__((ext_vector_type(8))) uint uint8;
typedef __attribute__((ext_vector_type(16))) uint uint16;
typedef __attribute__((ext_vector_type(2))) long long2;
typedef __attribute__((ext_vector_type(3))) long long3;
typedef __attribute__((ext_vector_type(4))) long long4;
typedef __attribute__((ext_vector_type(8))) long long8;
typedef __attribute__((ext_vector_type(16))) long long16;
typedef __attribute__((ext_vector_type(2))) ulong ulong2;
typedef __attribute__((ext_vector_type(3))) ulong ulong3;
typedef __attribute__((ext_vector_type(4))) ulong ulong4;
typedef __attribute__((ext_vector_type(8))) ulong ulong8;
typedef __attribute__((ext_vector_type(16))) ulong ulong16;
typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(3))) float float3;
typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(8))) float float8;
typedef __attribute__((ext_vector_type(16))) float float16;
typedef __attribute__((ext_vector_type(2))) double double2;
typedef __attribute__((ext_vector_type(3))) double double3;
typedef __attribute__((ext_vector_type(4))) double double4;
typedef __attribute__((ext_vector_type(8))) double double8;
typedef __attribute__((ext_vector_type(16))) double double16;
__attribute__((overloadable)) char convert_char_sat_rtn(char x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(char2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(char3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(char4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(char8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(char16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(char x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(char16 x); __attribute__((overloadable)) int convert_int_sat_rtn(char x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(char2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(char3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(char4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(char8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(char16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(char x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(char2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(char3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(char4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(char8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(char16 x); __attribute__((overloadable)) short convert_short_sat_rtn(char x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(char2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(char3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(char4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(char8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(char16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(char x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(char16 x); __attribute__((overloadable)) long convert_long_sat_rtn(char x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(char2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(char3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(char4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(char8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(char16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(char x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(char16 x); __attribute__((overloadable)) float convert_float_sat_rtn(char x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(char2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(char3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(char4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(char8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(char16 x); __attribute__((overloadable)) double convert_double_sat_rtn(char x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(char2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(char3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(char4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(char8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(char16 x); __attribute__((overloadable)) char convert_char_sat_rtn(uchar x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(uchar2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(uchar3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(uchar4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(uchar8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(uchar16 x); __attribute__((overloadable)) int convert_int_sat_rtn(uchar x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(uchar2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(uchar3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(uchar4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(uchar8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(uchar16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(uchar x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(uchar16 x); __attribute__((overloadable)) short convert_short_sat_rtn(uchar x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(uchar2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(uchar3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(uchar4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(uchar8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(uchar16 x); __attribute__((overloadable)) long convert_long_sat_rtn(uchar x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(uchar2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(uchar3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(uchar4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(uchar8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(uchar16 x); __attribute__((overloadable)) float convert_float_sat_rtn(uchar x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(uchar2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(uchar3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(uchar4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(uchar8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(uchar16 x); __attribute__((overloadable)) double convert_double_sat_rtn(uchar x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(uchar2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(uchar3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(uchar4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(uchar8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(uchar16 x); __attribute__((overloadable)) char convert_char_sat_rtn(int x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(int2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(int3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(int4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(int8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(int16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(int x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(int16 x); __attribute__((overloadable)) int convert_int_sat_rtn(int x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(int2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(int3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(int4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(int8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(int16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(int x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(int2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(int3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(int4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(int8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(int16 x); __attribute__((overloadable)) short convert_short_sat_rtn(int x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(int2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(int3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(int4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(int8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(int16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(int x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(int16 x); __attribute__((overloadable)) long convert_long_sat_rtn(int x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(int2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(int3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(int4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(int8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(int16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(int x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(int16 x); __attribute__((overloadable)) float convert_float_sat_rtn(int x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(int2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(int3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(int4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(int8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(int16 x); __attribute__((overloadable)) double convert_double_sat_rtn(int x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(int2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(int3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(int4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(int8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(int16 x); __attribute__((overloadable)) char convert_char_sat_rtn(uint x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(uint2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(uint3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(uint4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(uint8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(uint16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(uint16 x); __attribute__((overloadable)) int convert_int_sat_rtn(uint x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(uint2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(uint3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(uint4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(uint8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(uint16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(uint x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(uint16 x); __attribute__((overloadable)) short convert_short_sat_rtn(uint x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(uint2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(uint3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(uint4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(uint8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(uint16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(uint16 x); __attribute__((overloadable)) long convert_long_sat_rtn(uint x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(uint2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(uint3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(uint4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(uint8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(uint16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(uint16 x); __attribute__((overloadable)) float convert_float_sat_rtn(uint x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(uint2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(uint3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(uint4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(uint8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(uint16 x); __attribute__((overloadable)) double convert_double_sat_rtn(uint x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(uint2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(uint3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(uint4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(uint8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(uint16 x); __attribute__((overloadable)) char convert_char_sat_rtn(short x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(short2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(short3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(short4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(short8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(short16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(short x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(short16 x); __attribute__((overloadable)) int convert_int_sat_rtn(short x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(short2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(short3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(short4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(short8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(short16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(short x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(short2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(short3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(short4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(short8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(short16 x); __attribute__((overloadable)) short convert_short_sat_rtn(short x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(short2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(short3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(short4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(short8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(short16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(short x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(short16 x); __attribute__((overloadable)) long convert_long_sat_rtn(short x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(short2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(short3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(short4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(short8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(short16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(short x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(short16 x); __attribute__((overloadable)) float convert_float_sat_rtn(short x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(short2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(short3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(short4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(short8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(short16 x); __attribute__((overloadable)) double convert_double_sat_rtn(short x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(short2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(short3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(short4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(short8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(short16 x); __attribute__((overloadable)) char convert_char_sat_rtn(ushort x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(ushort2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(ushort3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(ushort4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(ushort8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(ushort16 x); __attribute__((overloadable)) int convert_int_sat_rtn(ushort x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(ushort2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(ushort3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(ushort4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(ushort8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(ushort16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(ushort x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(ushort16 x); __attribute__((overloadable)) short convert_short_sat_rtn(ushort x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(ushort2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(ushort3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(ushort4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(ushort8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(ushort16 x); __attribute__((overloadable)) long convert_long_sat_rtn(ushort x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(ushort2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(ushort3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(ushort4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(ushort8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(ushort16 x); __attribute__((overloadable)) float convert_float_sat_rtn(ushort x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(ushort2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(ushort3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(ushort4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(ushort8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(ushort16 x); __attribute__((overloadable)) double convert_double_sat_rtn(ushort x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(ushort2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(ushort3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(ushort4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(ushort8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(ushort16 x); __attribute__((overloadable)) char convert_char_sat_rtn(long x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(long2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(long3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(long4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(long8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(long16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(long x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(long16 x); __attribute__((overloadable)) int convert_int_sat_rtn(long x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(long2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(long3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(long4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(long8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(long16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(long x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(long2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(long3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(long4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(long8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(long16 x); __attribute__((overloadable)) short convert_short_sat_rtn(long x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(long2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(long3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(long4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(long8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(long16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(long x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(long16 x); __attribute__((overloadable)) long convert_long_sat_rtn(long x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(long2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(long3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(long4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(long8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(long16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(long x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(long16 x); __attribute__((overloadable)) float convert_float_sat_rtn(long x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(long2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(long3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(long4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(long8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(long16 x); __attribute__((overloadable)) double convert_double_sat_rtn(long x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(long2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(long3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(long4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(long8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(long16 x); __attribute__((overloadable)) char convert_char_sat_rtn(ulong x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(ulong2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(ulong3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(ulong4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(ulong8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(ulong16 x); __attribute__((overloadable)) int convert_int_sat_rtn(ulong x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(ulong2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(ulong3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(ulong4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(ulong8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(ulong16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(ulong x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(ulong16 x); __attribute__((overloadable)) short convert_short_sat_rtn(ulong x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(ulong2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(ulong3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(ulong4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(ulong8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(ulong16 x); __attribute__((overloadable)) long convert_long_sat_rtn(ulong x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(ulong2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(ulong3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(ulong4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(ulong8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(ulong16 x); __attribute__((overloadable)) float convert_float_sat_rtn(ulong x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(ulong2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(ulong3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(ulong4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(ulong8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(ulong16 x); __attribute__((overloadable)) double convert_double_sat_rtn(ulong x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(ulong2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(ulong3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(ulong4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(ulong8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(ulong16 x); __attribute__((overloadable)) char convert_char_sat_rtn(float x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(float2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(float3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(float4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(float8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(float16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(float x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(float16 x); __attribute__((overloadable)) int convert_int_sat_rtn(float x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(float2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(float3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(float4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(float8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(float16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(float x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(float2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(float3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(float4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(float8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(float16 x); __attribute__((overloadable)) short convert_short_sat_rtn(float x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(float2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(float3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(float4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(float8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(float16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(float x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(float16 x); __attribute__((overloadable)) long convert_long_sat_rtn(float x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(float2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(float3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(float4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(float8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(float16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(float x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(float16 x); __attribute__((overloadable)) float convert_float_sat_rtn(float x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(float2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(float3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(float4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(float8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(float16 x); __attribute__((overloadable)) double convert_double_sat_rtn(float x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(float2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(float3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(float4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(float8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(float16 x); __attribute__((overloadable)) char convert_char_sat_rtn(double x); __attribute__((overloadable)) char2 convert_char2_sat_rtn(double2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtn(double3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtn(double4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtn(double8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtn(double16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtn(double x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtn(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtn(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtn(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtn(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtn(double16 x); __attribute__((overloadable)) int convert_int_sat_rtn(double x); __attribute__((overloadable)) int2 convert_int2_sat_rtn(double2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtn(double3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtn(double4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtn(double8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtn(double16 x); __attribute__((overloadable)) uint convert_uint_sat_rtn(double x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtn(double2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtn(double3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtn(double4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtn(double8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtn(double16 x); __attribute__((overloadable)) short convert_short_sat_rtn(double x); __attribute__((overloadable)) short2 convert_short2_sat_rtn(double2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtn(double3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtn(double4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtn(double8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtn(double16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtn(double x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtn(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtn(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtn(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtn(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtn(double16 x); __attribute__((overloadable)) long convert_long_sat_rtn(double x); __attribute__((overloadable)) long2 convert_long2_sat_rtn(double2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtn(double3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtn(double4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtn(double8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtn(double16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtn(double x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtn(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtn(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtn(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtn(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtn(double16 x); __attribute__((overloadable)) float convert_float_sat_rtn(double x); __attribute__((overloadable)) float2 convert_float2_sat_rtn(double2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtn(double3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtn(double4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtn(double8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtn(double16 x); __attribute__((overloadable)) double convert_double_sat_rtn(double x); __attribute__((overloadable)) double2 convert_double2_sat_rtn(double2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtn(double3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtn(double4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtn(double8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtn(double16 x); __attribute__((overloadable)) char convert_char_rtn(char x); __attribute__((overloadable)) char2 convert_char2_rtn(char2 x); __attribute__((overloadable)) char3 convert_char3_rtn(char3 x); __attribute__((overloadable)) char4 convert_char4_rtn(char4 x); __attribute__((overloadable)) char8 convert_char8_rtn(char8 x); __attribute__((overloadable)) char16 convert_char16_rtn(char16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(char x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(char16 x); __attribute__((overloadable)) int convert_int_rtn(char x); __attribute__((overloadable)) int2 convert_int2_rtn(char2 x); __attribute__((overloadable)) int3 convert_int3_rtn(char3 x); __attribute__((overloadable)) int4 convert_int4_rtn(char4 x); __attribute__((overloadable)) int8 convert_int8_rtn(char8 x); __attribute__((overloadable)) int16 convert_int16_rtn(char16 x); __attribute__((overloadable)) uint convert_uint_rtn(char x); __attribute__((overloadable)) uint2 convert_uint2_rtn(char2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(char3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(char4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(char8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(char16 x); __attribute__((overloadable)) short convert_short_rtn(char x); __attribute__((overloadable)) short2 convert_short2_rtn(char2 x); __attribute__((overloadable)) short3 convert_short3_rtn(char3 x); __attribute__((overloadable)) short4 convert_short4_rtn(char4 x); __attribute__((overloadable)) short8 convert_short8_rtn(char8 x); __attribute__((overloadable)) short16 convert_short16_rtn(char16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(char x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(char16 x); __attribute__((overloadable)) long convert_long_rtn(char x); __attribute__((overloadable)) long2 convert_long2_rtn(char2 x); __attribute__((overloadable)) long3 convert_long3_rtn(char3 x); __attribute__((overloadable)) long4 convert_long4_rtn(char4 x); __attribute__((overloadable)) long8 convert_long8_rtn(char8 x); __attribute__((overloadable)) long16 convert_long16_rtn(char16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(char x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(char16 x); __attribute__((overloadable)) float convert_float_rtn(char x); __attribute__((overloadable)) float2 convert_float2_rtn(char2 x); __attribute__((overloadable)) float3 convert_float3_rtn(char3 x); __attribute__((overloadable)) float4 convert_float4_rtn(char4 x); __attribute__((overloadable)) float8 convert_float8_rtn(char8 x); __attribute__((overloadable)) float16 convert_float16_rtn(char16 x); __attribute__((overloadable)) double convert_double_rtn(char x); __attribute__((overloadable)) double2 convert_double2_rtn(char2 x); __attribute__((overloadable)) double3 convert_double3_rtn(char3 x); __attribute__((overloadable)) double4 convert_double4_rtn(char4 x); __attribute__((overloadable)) double8 convert_double8_rtn(char8 x); __attribute__((overloadable)) double16 convert_double16_rtn(char16 x); __attribute__((overloadable)) char convert_char_rtn(uchar x); __attribute__((overloadable)) char2 convert_char2_rtn(uchar2 x); __attribute__((overloadable)) char3 convert_char3_rtn(uchar3 x); __attribute__((overloadable)) char4 convert_char4_rtn(uchar4 x); __attribute__((overloadable)) char8 convert_char8_rtn(uchar8 x); __attribute__((overloadable)) char16 convert_char16_rtn(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(uchar16 x); __attribute__((overloadable)) int convert_int_rtn(uchar x); __attribute__((overloadable)) int2 convert_int2_rtn(uchar2 x); __attribute__((overloadable)) int3 convert_int3_rtn(uchar3 x); __attribute__((overloadable)) int4 convert_int4_rtn(uchar4 x); __attribute__((overloadable)) int8 convert_int8_rtn(uchar8 x); __attribute__((overloadable)) int16 convert_int16_rtn(uchar16 x); __attribute__((overloadable)) uint convert_uint_rtn(uchar x); __attribute__((overloadable)) uint2 convert_uint2_rtn(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(uchar16 x); __attribute__((overloadable)) short convert_short_rtn(uchar x); __attribute__((overloadable)) short2 convert_short2_rtn(uchar2 x); __attribute__((overloadable)) short3 convert_short3_rtn(uchar3 x); __attribute__((overloadable)) short4 convert_short4_rtn(uchar4 x); __attribute__((overloadable)) short8 convert_short8_rtn(uchar8 x); __attribute__((overloadable)) short16 convert_short16_rtn(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(uchar16 x); __attribute__((overloadable)) long convert_long_rtn(uchar x); __attribute__((overloadable)) long2 convert_long2_rtn(uchar2 x); __attribute__((overloadable)) long3 convert_long3_rtn(uchar3 x); __attribute__((overloadable)) long4 convert_long4_rtn(uchar4 x); __attribute__((overloadable)) long8 convert_long8_rtn(uchar8 x); __attribute__((overloadable)) long16 convert_long16_rtn(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(uchar16 x); __attribute__((overloadable)) float convert_float_rtn(uchar x); __attribute__((overloadable)) float2 convert_float2_rtn(uchar2 x); __attribute__((overloadable)) float3 convert_float3_rtn(uchar3 x); __attribute__((overloadable)) float4 convert_float4_rtn(uchar4 x); __attribute__((overloadable)) float8 convert_float8_rtn(uchar8 x); __attribute__((overloadable)) float16 convert_float16_rtn(uchar16 x); __attribute__((overloadable)) double convert_double_rtn(uchar x); __attribute__((overloadable)) double2 convert_double2_rtn(uchar2 x); __attribute__((overloadable)) double3 convert_double3_rtn(uchar3 x); __attribute__((overloadable)) double4 convert_double4_rtn(uchar4 x); __attribute__((overloadable)) double8 convert_double8_rtn(uchar8 x); __attribute__((overloadable)) double16 convert_double16_rtn(uchar16 x); __attribute__((overloadable)) char convert_char_rtn(int x); __attribute__((overloadable)) char2 convert_char2_rtn(int2 x); __attribute__((overloadable)) char3 convert_char3_rtn(int3 x); __attribute__((overloadable)) char4 convert_char4_rtn(int4 x); __attribute__((overloadable)) char8 convert_char8_rtn(int8 x); __attribute__((overloadable)) char16 convert_char16_rtn(int16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(int x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(int16 x); __attribute__((overloadable)) int convert_int_rtn(int x); __attribute__((overloadable)) int2 convert_int2_rtn(int2 x); __attribute__((overloadable)) int3 convert_int3_rtn(int3 x); __attribute__((overloadable)) int4 convert_int4_rtn(int4 x); __attribute__((overloadable)) int8 convert_int8_rtn(int8 x); __attribute__((overloadable)) int16 convert_int16_rtn(int16 x); __attribute__((overloadable)) uint convert_uint_rtn(int x); __attribute__((overloadable)) uint2 convert_uint2_rtn(int2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(int3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(int4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(int8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(int16 x); __attribute__((overloadable)) short convert_short_rtn(int x); __attribute__((overloadable)) short2 convert_short2_rtn(int2 x); __attribute__((overloadable)) short3 convert_short3_rtn(int3 x); __attribute__((overloadable)) short4 convert_short4_rtn(int4 x); __attribute__((overloadable)) short8 convert_short8_rtn(int8 x); __attribute__((overloadable)) short16 convert_short16_rtn(int16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(int x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(int16 x); __attribute__((overloadable)) long convert_long_rtn(int x); __attribute__((overloadable)) long2 convert_long2_rtn(int2 x); __attribute__((overloadable)) long3 convert_long3_rtn(int3 x); __attribute__((overloadable)) long4 convert_long4_rtn(int4 x); __attribute__((overloadable)) long8 convert_long8_rtn(int8 x); __attribute__((overloadable)) long16 convert_long16_rtn(int16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(int x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(int16 x); __attribute__((overloadable)) float convert_float_rtn(int x); __attribute__((overloadable)) float2 convert_float2_rtn(int2 x); __attribute__((overloadable)) float3 convert_float3_rtn(int3 x); __attribute__((overloadable)) float4 convert_float4_rtn(int4 x); __attribute__((overloadable)) float8 convert_float8_rtn(int8 x); __attribute__((overloadable)) float16 convert_float16_rtn(int16 x); __attribute__((overloadable)) double convert_double_rtn(int x); __attribute__((overloadable)) double2 convert_double2_rtn(int2 x); __attribute__((overloadable)) double3 convert_double3_rtn(int3 x); __attribute__((overloadable)) double4 convert_double4_rtn(int4 x); __attribute__((overloadable)) double8 convert_double8_rtn(int8 x); __attribute__((overloadable)) double16 convert_double16_rtn(int16 x); __attribute__((overloadable)) char convert_char_rtn(uint x); __attribute__((overloadable)) char2 convert_char2_rtn(uint2 x); __attribute__((overloadable)) char3 convert_char3_rtn(uint3 x); __attribute__((overloadable)) char4 convert_char4_rtn(uint4 x); __attribute__((overloadable)) char8 convert_char8_rtn(uint8 x); __attribute__((overloadable)) char16 convert_char16_rtn(uint16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(uint16 x); __attribute__((overloadable)) int convert_int_rtn(uint x); __attribute__((overloadable)) int2 convert_int2_rtn(uint2 x); __attribute__((overloadable)) int3 convert_int3_rtn(uint3 x); __attribute__((overloadable)) int4 convert_int4_rtn(uint4 x); __attribute__((overloadable)) int8 convert_int8_rtn(uint8 x); __attribute__((overloadable)) int16 convert_int16_rtn(uint16 x); __attribute__((overloadable)) uint convert_uint_rtn(uint x); __attribute__((overloadable)) uint2 convert_uint2_rtn(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(uint16 x); __attribute__((overloadable)) short convert_short_rtn(uint x); __attribute__((overloadable)) short2 convert_short2_rtn(uint2 x); __attribute__((overloadable)) short3 convert_short3_rtn(uint3 x); __attribute__((overloadable)) short4 convert_short4_rtn(uint4 x); __attribute__((overloadable)) short8 convert_short8_rtn(uint8 x); __attribute__((overloadable)) short16 convert_short16_rtn(uint16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(uint16 x); __attribute__((overloadable)) long convert_long_rtn(uint x); __attribute__((overloadable)) long2 convert_long2_rtn(uint2 x); __attribute__((overloadable)) long3 convert_long3_rtn(uint3 x); __attribute__((overloadable)) long4 convert_long4_rtn(uint4 x); __attribute__((overloadable)) long8 convert_long8_rtn(uint8 x); __attribute__((overloadable)) long16 convert_long16_rtn(uint16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(uint16 x); __attribute__((overloadable)) float convert_float_rtn(uint x); __attribute__((overloadable)) float2 convert_float2_rtn(uint2 x); __attribute__((overloadable)) float3 convert_float3_rtn(uint3 x); __attribute__((overloadable)) float4 convert_float4_rtn(uint4 x); __attribute__((overloadable)) float8 convert_float8_rtn(uint8 x); __attribute__((overloadable)) float16 convert_float16_rtn(uint16 x); __attribute__((overloadable)) double convert_double_rtn(uint x); __attribute__((overloadable)) double2 convert_double2_rtn(uint2 x); __attribute__((overloadable)) double3 convert_double3_rtn(uint3 x); __attribute__((overloadable)) double4 convert_double4_rtn(uint4 x); __attribute__((overloadable)) double8 convert_double8_rtn(uint8 x); __attribute__((overloadable)) double16 convert_double16_rtn(uint16 x); __attribute__((overloadable)) char convert_char_rtn(short x); __attribute__((overloadable)) char2 convert_char2_rtn(short2 x); __attribute__((overloadable)) char3 convert_char3_rtn(short3 x); __attribute__((overloadable)) char4 convert_char4_rtn(short4 x); __attribute__((overloadable)) char8 convert_char8_rtn(short8 x); __attribute__((overloadable)) char16 convert_char16_rtn(short16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(short x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(short16 x); __attribute__((overloadable)) int convert_int_rtn(short x); __attribute__((overloadable)) int2 convert_int2_rtn(short2 x); __attribute__((overloadable)) int3 convert_int3_rtn(short3 x); __attribute__((overloadable)) int4 convert_int4_rtn(short4 x); __attribute__((overloadable)) int8 convert_int8_rtn(short8 x); __attribute__((overloadable)) int16 convert_int16_rtn(short16 x); __attribute__((overloadable)) uint convert_uint_rtn(short x); __attribute__((overloadable)) uint2 convert_uint2_rtn(short2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(short3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(short4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(short8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(short16 x); __attribute__((overloadable)) short convert_short_rtn(short x); __attribute__((overloadable)) short2 convert_short2_rtn(short2 x); __attribute__((overloadable)) short3 convert_short3_rtn(short3 x); __attribute__((overloadable)) short4 convert_short4_rtn(short4 x); __attribute__((overloadable)) short8 convert_short8_rtn(short8 x); __attribute__((overloadable)) short16 convert_short16_rtn(short16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(short x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(short16 x); __attribute__((overloadable)) long convert_long_rtn(short x); __attribute__((overloadable)) long2 convert_long2_rtn(short2 x); __attribute__((overloadable)) long3 convert_long3_rtn(short3 x); __attribute__((overloadable)) long4 convert_long4_rtn(short4 x); __attribute__((overloadable)) long8 convert_long8_rtn(short8 x); __attribute__((overloadable)) long16 convert_long16_rtn(short16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(short x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(short16 x); __attribute__((overloadable)) float convert_float_rtn(short x); __attribute__((overloadable)) float2 convert_float2_rtn(short2 x); __attribute__((overloadable)) float3 convert_float3_rtn(short3 x); __attribute__((overloadable)) float4 convert_float4_rtn(short4 x); __attribute__((overloadable)) float8 convert_float8_rtn(short8 x); __attribute__((overloadable)) float16 convert_float16_rtn(short16 x); __attribute__((overloadable)) double convert_double_rtn(short x); __attribute__((overloadable)) double2 convert_double2_rtn(short2 x); __attribute__((overloadable)) double3 convert_double3_rtn(short3 x); __attribute__((overloadable)) double4 convert_double4_rtn(short4 x); __attribute__((overloadable)) double8 convert_double8_rtn(short8 x); __attribute__((overloadable)) double16 convert_double16_rtn(short16 x); __attribute__((overloadable)) char convert_char_rtn(ushort x); __attribute__((overloadable)) char2 convert_char2_rtn(ushort2 x); __attribute__((overloadable)) char3 convert_char3_rtn(ushort3 x); __attribute__((overloadable)) char4 convert_char4_rtn(ushort4 x); __attribute__((overloadable)) char8 convert_char8_rtn(ushort8 x); __attribute__((overloadable)) char16 convert_char16_rtn(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(ushort16 x); __attribute__((overloadable)) int convert_int_rtn(ushort x); __attribute__((overloadable)) int2 convert_int2_rtn(ushort2 x); __attribute__((overloadable)) int3 convert_int3_rtn(ushort3 x); __attribute__((overloadable)) int4 convert_int4_rtn(ushort4 x); __attribute__((overloadable)) int8 convert_int8_rtn(ushort8 x); __attribute__((overloadable)) int16 convert_int16_rtn(ushort16 x); __attribute__((overloadable)) uint convert_uint_rtn(ushort x); __attribute__((overloadable)) uint2 convert_uint2_rtn(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(ushort16 x); __attribute__((overloadable)) short convert_short_rtn(ushort x); __attribute__((overloadable)) short2 convert_short2_rtn(ushort2 x); __attribute__((overloadable)) short3 convert_short3_rtn(ushort3 x); __attribute__((overloadable)) short4 convert_short4_rtn(ushort4 x); __attribute__((overloadable)) short8 convert_short8_rtn(ushort8 x); __attribute__((overloadable)) short16 convert_short16_rtn(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(ushort16 x); __attribute__((overloadable)) long convert_long_rtn(ushort x); __attribute__((overloadable)) long2 convert_long2_rtn(ushort2 x); __attribute__((overloadable)) long3 convert_long3_rtn(ushort3 x); __attribute__((overloadable)) long4 convert_long4_rtn(ushort4 x); __attribute__((overloadable)) long8 convert_long8_rtn(ushort8 x); __attribute__((overloadable)) long16 convert_long16_rtn(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(ushort16 x); __attribute__((overloadable)) float convert_float_rtn(ushort x); __attribute__((overloadable)) float2 convert_float2_rtn(ushort2 x); __attribute__((overloadable)) float3 convert_float3_rtn(ushort3 x); __attribute__((overloadable)) float4 convert_float4_rtn(ushort4 x); __attribute__((overloadable)) float8 convert_float8_rtn(ushort8 x); __attribute__((overloadable)) float16 convert_float16_rtn(ushort16 x); __attribute__((overloadable)) double convert_double_rtn(ushort x); __attribute__((overloadable)) double2 convert_double2_rtn(ushort2 x); __attribute__((overloadable)) double3 convert_double3_rtn(ushort3 x); __attribute__((overloadable)) double4 convert_double4_rtn(ushort4 x); __attribute__((overloadable)) double8 convert_double8_rtn(ushort8 x); __attribute__((overloadable)) double16 convert_double16_rtn(ushort16 x); __attribute__((overloadable)) char convert_char_rtn(long x); __attribute__((overloadable)) char2 convert_char2_rtn(long2 x); __attribute__((overloadable)) char3 convert_char3_rtn(long3 x); __attribute__((overloadable)) char4 convert_char4_rtn(long4 x); __attribute__((overloadable)) char8 convert_char8_rtn(long8 x); __attribute__((overloadable)) char16 convert_char16_rtn(long16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(long x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(long16 x); __attribute__((overloadable)) int convert_int_rtn(long x); __attribute__((overloadable)) int2 convert_int2_rtn(long2 x); __attribute__((overloadable)) int3 convert_int3_rtn(long3 x); __attribute__((overloadable)) int4 convert_int4_rtn(long4 x); __attribute__((overloadable)) int8 convert_int8_rtn(long8 x); __attribute__((overloadable)) int16 convert_int16_rtn(long16 x); __attribute__((overloadable)) uint convert_uint_rtn(long x); __attribute__((overloadable)) uint2 convert_uint2_rtn(long2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(long3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(long4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(long8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(long16 x); __attribute__((overloadable)) short convert_short_rtn(long x); __attribute__((overloadable)) short2 convert_short2_rtn(long2 x); __attribute__((overloadable)) short3 convert_short3_rtn(long3 x); __attribute__((overloadable)) short4 convert_short4_rtn(long4 x); __attribute__((overloadable)) short8 convert_short8_rtn(long8 x); __attribute__((overloadable)) short16 convert_short16_rtn(long16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(long x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(long16 x); __attribute__((overloadable)) long convert_long_rtn(long x); __attribute__((overloadable)) long2 convert_long2_rtn(long2 x); __attribute__((overloadable)) long3 convert_long3_rtn(long3 x); __attribute__((overloadable)) long4 convert_long4_rtn(long4 x); __attribute__((overloadable)) long8 convert_long8_rtn(long8 x); __attribute__((overloadable)) long16 convert_long16_rtn(long16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(long x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(long16 x); __attribute__((overloadable)) float convert_float_rtn(long x); __attribute__((overloadable)) float2 convert_float2_rtn(long2 x); __attribute__((overloadable)) float3 convert_float3_rtn(long3 x); __attribute__((overloadable)) float4 convert_float4_rtn(long4 x); __attribute__((overloadable)) float8 convert_float8_rtn(long8 x); __attribute__((overloadable)) float16 convert_float16_rtn(long16 x); __attribute__((overloadable)) double convert_double_rtn(long x); __attribute__((overloadable)) double2 convert_double2_rtn(long2 x); __attribute__((overloadable)) double3 convert_double3_rtn(long3 x); __attribute__((overloadable)) double4 convert_double4_rtn(long4 x); __attribute__((overloadable)) double8 convert_double8_rtn(long8 x); __attribute__((overloadable)) double16 convert_double16_rtn(long16 x); __attribute__((overloadable)) char convert_char_rtn(ulong x); __attribute__((overloadable)) char2 convert_char2_rtn(ulong2 x); __attribute__((overloadable)) char3 convert_char3_rtn(ulong3 x); __attribute__((overloadable)) char4 convert_char4_rtn(ulong4 x); __attribute__((overloadable)) char8 convert_char8_rtn(ulong8 x); __attribute__((overloadable)) char16 convert_char16_rtn(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(ulong16 x); __attribute__((overloadable)) int convert_int_rtn(ulong x); __attribute__((overloadable)) int2 convert_int2_rtn(ulong2 x); __attribute__((overloadable)) int3 convert_int3_rtn(ulong3 x); __attribute__((overloadable)) int4 convert_int4_rtn(ulong4 x); __attribute__((overloadable)) int8 convert_int8_rtn(ulong8 x); __attribute__((overloadable)) int16 convert_int16_rtn(ulong16 x); __attribute__((overloadable)) uint convert_uint_rtn(ulong x); __attribute__((overloadable)) uint2 convert_uint2_rtn(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(ulong16 x); __attribute__((overloadable)) short convert_short_rtn(ulong x); __attribute__((overloadable)) short2 convert_short2_rtn(ulong2 x); __attribute__((overloadable)) short3 convert_short3_rtn(ulong3 x); __attribute__((overloadable)) short4 convert_short4_rtn(ulong4 x); __attribute__((overloadable)) short8 convert_short8_rtn(ulong8 x); __attribute__((overloadable)) short16 convert_short16_rtn(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(ulong16 x); __attribute__((overloadable)) long convert_long_rtn(ulong x); __attribute__((overloadable)) long2 convert_long2_rtn(ulong2 x); __attribute__((overloadable)) long3 convert_long3_rtn(ulong3 x); __attribute__((overloadable)) long4 convert_long4_rtn(ulong4 x); __attribute__((overloadable)) long8 convert_long8_rtn(ulong8 x); __attribute__((overloadable)) long16 convert_long16_rtn(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(ulong16 x); __attribute__((overloadable)) float convert_float_rtn(ulong x); __attribute__((overloadable)) float2 convert_float2_rtn(ulong2 x); __attribute__((overloadable)) float3 convert_float3_rtn(ulong3 x); __attribute__((overloadable)) float4 convert_float4_rtn(ulong4 x); __attribute__((overloadable)) float8 convert_float8_rtn(ulong8 x); __attribute__((overloadable)) float16 convert_float16_rtn(ulong16 x); __attribute__((overloadable)) double convert_double_rtn(ulong x); __attribute__((overloadable)) double2 convert_double2_rtn(ulong2 x); __attribute__((overloadable)) double3 convert_double3_rtn(ulong3 x); __attribute__((overloadable)) double4 convert_double4_rtn(ulong4 x); __attribute__((overloadable)) double8 convert_double8_rtn(ulong8 x); __attribute__((overloadable)) double16 convert_double16_rtn(ulong16 x); __attribute__((overloadable)) char convert_char_rtn(float x); __attribute__((overloadable)) char2 convert_char2_rtn(float2 x); __attribute__((overloadable)) char3 convert_char3_rtn(float3 x); __attribute__((overloadable)) char4 convert_char4_rtn(float4 x); __attribute__((overloadable)) char8 convert_char8_rtn(float8 x); __attribute__((overloadable)) char16 convert_char16_rtn(float16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(float x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(float16 x); __attribute__((overloadable)) int convert_int_rtn(float x); __attribute__((overloadable)) int2 convert_int2_rtn(float2 x); __attribute__((overloadable)) int3 convert_int3_rtn(float3 x); __attribute__((overloadable)) int4 convert_int4_rtn(float4 x); __attribute__((overloadable)) int8 convert_int8_rtn(float8 x); __attribute__((overloadable)) int16 convert_int16_rtn(float16 x); __attribute__((overloadable)) uint convert_uint_rtn(float x); __attribute__((overloadable)) uint2 convert_uint2_rtn(float2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(float3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(float4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(float8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(float16 x); __attribute__((overloadable)) short convert_short_rtn(float x); __attribute__((overloadable)) short2 convert_short2_rtn(float2 x); __attribute__((overloadable)) short3 convert_short3_rtn(float3 x); __attribute__((overloadable)) short4 convert_short4_rtn(float4 x); __attribute__((overloadable)) short8 convert_short8_rtn(float8 x); __attribute__((overloadable)) short16 convert_short16_rtn(float16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(float x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(float16 x); __attribute__((overloadable)) long convert_long_rtn(float x); __attribute__((overloadable)) long2 convert_long2_rtn(float2 x); __attribute__((overloadable)) long3 convert_long3_rtn(float3 x); __attribute__((overloadable)) long4 convert_long4_rtn(float4 x); __attribute__((overloadable)) long8 convert_long8_rtn(float8 x); __attribute__((overloadable)) long16 convert_long16_rtn(float16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(float x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(float16 x); __attribute__((overloadable)) float convert_float_rtn(float x); __attribute__((overloadable)) float2 convert_float2_rtn(float2 x); __attribute__((overloadable)) float3 convert_float3_rtn(float3 x); __attribute__((overloadable)) float4 convert_float4_rtn(float4 x); __attribute__((overloadable)) float8 convert_float8_rtn(float8 x); __attribute__((overloadable)) float16 convert_float16_rtn(float16 x); __attribute__((overloadable)) double convert_double_rtn(float x); __attribute__((overloadable)) double2 convert_double2_rtn(float2 x); __attribute__((overloadable)) double3 convert_double3_rtn(float3 x); __attribute__((overloadable)) double4 convert_double4_rtn(float4 x); __attribute__((overloadable)) double8 convert_double8_rtn(float8 x); __attribute__((overloadable)) double16 convert_double16_rtn(float16 x); __attribute__((overloadable)) char convert_char_rtn(double x); __attribute__((overloadable)) char2 convert_char2_rtn(double2 x); __attribute__((overloadable)) char3 convert_char3_rtn(double3 x); __attribute__((overloadable)) char4 convert_char4_rtn(double4 x); __attribute__((overloadable)) char8 convert_char8_rtn(double8 x); __attribute__((overloadable)) char16 convert_char16_rtn(double16 x); __attribute__((overloadable)) uchar convert_uchar_rtn(double x); __attribute__((overloadable)) uchar2 convert_uchar2_rtn(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtn(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtn(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtn(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtn(double16 x); __attribute__((overloadable)) int convert_int_rtn(double x); __attribute__((overloadable)) int2 convert_int2_rtn(double2 x); __attribute__((overloadable)) int3 convert_int3_rtn(double3 x); __attribute__((overloadable)) int4 convert_int4_rtn(double4 x); __attribute__((overloadable)) int8 convert_int8_rtn(double8 x); __attribute__((overloadable)) int16 convert_int16_rtn(double16 x); __attribute__((overloadable)) uint convert_uint_rtn(double x); __attribute__((overloadable)) uint2 convert_uint2_rtn(double2 x); __attribute__((overloadable)) uint3 convert_uint3_rtn(double3 x); __attribute__((overloadable)) uint4 convert_uint4_rtn(double4 x); __attribute__((overloadable)) uint8 convert_uint8_rtn(double8 x); __attribute__((overloadable)) uint16 convert_uint16_rtn(double16 x); __attribute__((overloadable)) short convert_short_rtn(double x); __attribute__((overloadable)) short2 convert_short2_rtn(double2 x); __attribute__((overloadable)) short3 convert_short3_rtn(double3 x); __attribute__((overloadable)) short4 convert_short4_rtn(double4 x); __attribute__((overloadable)) short8 convert_short8_rtn(double8 x); __attribute__((overloadable)) short16 convert_short16_rtn(double16 x); __attribute__((overloadable)) ushort convert_ushort_rtn(double x); __attribute__((overloadable)) ushort2 convert_ushort2_rtn(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtn(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtn(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtn(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtn(double16 x); __attribute__((overloadable)) long convert_long_rtn(double x); __attribute__((overloadable)) long2 convert_long2_rtn(double2 x); __attribute__((overloadable)) long3 convert_long3_rtn(double3 x); __attribute__((overloadable)) long4 convert_long4_rtn(double4 x); __attribute__((overloadable)) long8 convert_long8_rtn(double8 x); __attribute__((overloadable)) long16 convert_long16_rtn(double16 x); __attribute__((overloadable)) ulong convert_ulong_rtn(double x); __attribute__((overloadable)) ulong2 convert_ulong2_rtn(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtn(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtn(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtn(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtn(double16 x); __attribute__((overloadable)) float convert_float_rtn(double x); __attribute__((overloadable)) float2 convert_float2_rtn(double2 x); __attribute__((overloadable)) float3 convert_float3_rtn(double3 x); __attribute__((overloadable)) float4 convert_float4_rtn(double4 x); __attribute__((overloadable)) float8 convert_float8_rtn(double8 x); __attribute__((overloadable)) float16 convert_float16_rtn(double16 x); __attribute__((overloadable)) double convert_double_rtn(double x); __attribute__((overloadable)) double2 convert_double2_rtn(double2 x); __attribute__((overloadable)) double3 convert_double3_rtn(double3 x); __attribute__((overloadable)) double4 convert_double4_rtn(double4 x); __attribute__((overloadable)) double8 convert_double8_rtn(double8 x); __attribute__((overloadable)) double16 convert_double16_rtn(double16 x);
// __attribute__((overloadable)) char convert_char_sat_rte(char x); __attribute__((overloadable)) char2 convert_char2_sat_rte(char2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(char3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(char4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(char8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(char16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(char x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(char16 x); __attribute__((overloadable)) int convert_int_sat_rte(char x); __attribute__((overloadable)) int2 convert_int2_sat_rte(char2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(char3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(char4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(char8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(char16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(char x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(char2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(char3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(char4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(char8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(char16 x); __attribute__((overloadable)) short convert_short_sat_rte(char x); __attribute__((overloadable)) short2 convert_short2_sat_rte(char2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(char3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(char4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(char8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(char16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(char x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(char16 x); __attribute__((overloadable)) long convert_long_sat_rte(char x); __attribute__((overloadable)) long2 convert_long2_sat_rte(char2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(char3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(char4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(char8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(char16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(char x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(char16 x); __attribute__((overloadable)) float convert_float_sat_rte(char x); __attribute__((overloadable)) float2 convert_float2_sat_rte(char2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(char3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(char4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(char8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(char16 x); __attribute__((overloadable)) double convert_double_sat_rte(char x); __attribute__((overloadable)) double2 convert_double2_sat_rte(char2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(char3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(char4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(char8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(char16 x); __attribute__((overloadable)) char convert_char_sat_rte(uchar x); __attribute__((overloadable)) char2 convert_char2_sat_rte(uchar2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(uchar3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(uchar4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(uchar8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(uchar16 x); __attribute__((overloadable)) int convert_int_sat_rte(uchar x); __attribute__((overloadable)) int2 convert_int2_sat_rte(uchar2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(uchar3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(uchar4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(uchar8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(uchar16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(uchar x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(uchar16 x); __attribute__((overloadable)) short convert_short_sat_rte(uchar x); __attribute__((overloadable)) short2 convert_short2_sat_rte(uchar2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(uchar3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(uchar4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(uchar8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(uchar16 x); __attribute__((overloadable)) long convert_long_sat_rte(uchar x); __attribute__((overloadable)) long2 convert_long2_sat_rte(uchar2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(uchar3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(uchar4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(uchar8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(uchar16 x); __attribute__((overloadable)) float convert_float_sat_rte(uchar x); __attribute__((overloadable)) float2 convert_float2_sat_rte(uchar2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(uchar3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(uchar4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(uchar8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(uchar16 x); __attribute__((overloadable)) double convert_double_sat_rte(uchar x); __attribute__((overloadable)) double2 convert_double2_sat_rte(uchar2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(uchar3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(uchar4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(uchar8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(uchar16 x); __attribute__((overloadable)) char convert_char_sat_rte(int x); __attribute__((overloadable)) char2 convert_char2_sat_rte(int2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(int3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(int4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(int8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(int16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(int x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(int16 x); __attribute__((overloadable)) int convert_int_sat_rte(int x); __attribute__((overloadable)) int2 convert_int2_sat_rte(int2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(int3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(int4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(int8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(int16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(int x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(int2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(int3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(int4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(int8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(int16 x); __attribute__((overloadable)) short convert_short_sat_rte(int x); __attribute__((overloadable)) short2 convert_short2_sat_rte(int2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(int3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(int4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(int8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(int16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(int x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(int16 x); __attribute__((overloadable)) long convert_long_sat_rte(int x); __attribute__((overloadable)) long2 convert_long2_sat_rte(int2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(int3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(int4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(int8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(int16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(int x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(int16 x); __attribute__((overloadable)) float convert_float_sat_rte(int x); __attribute__((overloadable)) float2 convert_float2_sat_rte(int2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(int3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(int4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(int8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(int16 x); __attribute__((overloadable)) double convert_double_sat_rte(int x); __attribute__((overloadable)) double2 convert_double2_sat_rte(int2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(int3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(int4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(int8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(int16 x); __attribute__((overloadable)) char convert_char_sat_rte(uint x); __attribute__((overloadable)) char2 convert_char2_sat_rte(uint2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(uint3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(uint4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(uint8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(uint16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(uint16 x); __attribute__((overloadable)) int convert_int_sat_rte(uint x); __attribute__((overloadable)) int2 convert_int2_sat_rte(uint2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(uint3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(uint4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(uint8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(uint16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(uint x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(uint16 x); __attribute__((overloadable)) short convert_short_sat_rte(uint x); __attribute__((overloadable)) short2 convert_short2_sat_rte(uint2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(uint3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(uint4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(uint8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(uint16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(uint16 x); __attribute__((overloadable)) long convert_long_sat_rte(uint x); __attribute__((overloadable)) long2 convert_long2_sat_rte(uint2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(uint3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(uint4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(uint8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(uint16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(uint16 x); __attribute__((overloadable)) float convert_float_sat_rte(uint x); __attribute__((overloadable)) float2 convert_float2_sat_rte(uint2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(uint3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(uint4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(uint8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(uint16 x); __attribute__((overloadable)) double convert_double_sat_rte(uint x); __attribute__((overloadable)) double2 convert_double2_sat_rte(uint2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(uint3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(uint4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(uint8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(uint16 x); __attribute__((overloadable)) char convert_char_sat_rte(short x); __attribute__((overloadable)) char2 convert_char2_sat_rte(short2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(short3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(short4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(short8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(short16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(short x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(short16 x); __attribute__((overloadable)) int convert_int_sat_rte(short x); __attribute__((overloadable)) int2 convert_int2_sat_rte(short2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(short3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(short4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(short8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(short16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(short x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(short2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(short3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(short4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(short8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(short16 x); __attribute__((overloadable)) short convert_short_sat_rte(short x); __attribute__((overloadable)) short2 convert_short2_sat_rte(short2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(short3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(short4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(short8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(short16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(short x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(short16 x); __attribute__((overloadable)) long convert_long_sat_rte(short x); __attribute__((overloadable)) long2 convert_long2_sat_rte(short2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(short3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(short4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(short8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(short16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(short x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(short16 x); __attribute__((overloadable)) float convert_float_sat_rte(short x); __attribute__((overloadable)) float2 convert_float2_sat_rte(short2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(short3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(short4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(short8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(short16 x); __attribute__((overloadable)) double convert_double_sat_rte(short x); __attribute__((overloadable)) double2 convert_double2_sat_rte(short2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(short3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(short4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(short8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(short16 x); __attribute__((overloadable)) char convert_char_sat_rte(ushort x); __attribute__((overloadable)) char2 convert_char2_sat_rte(ushort2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(ushort3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(ushort4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(ushort8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(ushort16 x); __attribute__((overloadable)) int convert_int_sat_rte(ushort x); __attribute__((overloadable)) int2 convert_int2_sat_rte(ushort2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(ushort3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(ushort4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(ushort8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(ushort16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(ushort x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(ushort16 x); __attribute__((overloadable)) short convert_short_sat_rte(ushort x); __attribute__((overloadable)) short2 convert_short2_sat_rte(ushort2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(ushort3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(ushort4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(ushort8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(ushort16 x); __attribute__((overloadable)) long convert_long_sat_rte(ushort x); __attribute__((overloadable)) long2 convert_long2_sat_rte(ushort2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(ushort3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(ushort4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(ushort8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(ushort16 x); __attribute__((overloadable)) float convert_float_sat_rte(ushort x); __attribute__((overloadable)) float2 convert_float2_sat_rte(ushort2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(ushort3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(ushort4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(ushort8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(ushort16 x); __attribute__((overloadable)) double convert_double_sat_rte(ushort x); __attribute__((overloadable)) double2 convert_double2_sat_rte(ushort2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(ushort3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(ushort4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(ushort8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(ushort16 x); __attribute__((overloadable)) char convert_char_sat_rte(long x); __attribute__((overloadable)) char2 convert_char2_sat_rte(long2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(long3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(long4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(long8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(long16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(long x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(long16 x); __attribute__((overloadable)) int convert_int_sat_rte(long x); __attribute__((overloadable)) int2 convert_int2_sat_rte(long2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(long3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(long4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(long8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(long16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(long x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(long2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(long3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(long4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(long8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(long16 x); __attribute__((overloadable)) short convert_short_sat_rte(long x); __attribute__((overloadable)) short2 convert_short2_sat_rte(long2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(long3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(long4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(long8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(long16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(long x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(long16 x); __attribute__((overloadable)) long convert_long_sat_rte(long x); __attribute__((overloadable)) long2 convert_long2_sat_rte(long2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(long3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(long4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(long8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(long16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(long x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(long16 x); __attribute__((overloadable)) float convert_float_sat_rte(long x); __attribute__((overloadable)) float2 convert_float2_sat_rte(long2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(long3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(long4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(long8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(long16 x); __attribute__((overloadable)) double convert_double_sat_rte(long x); __attribute__((overloadable)) double2 convert_double2_sat_rte(long2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(long3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(long4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(long8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(long16 x); __attribute__((overloadable)) char convert_char_sat_rte(ulong x); __attribute__((overloadable)) char2 convert_char2_sat_rte(ulong2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(ulong3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(ulong4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(ulong8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(ulong16 x); __attribute__((overloadable)) int convert_int_sat_rte(ulong x); __attribute__((overloadable)) int2 convert_int2_sat_rte(ulong2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(ulong3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(ulong4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(ulong8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(ulong16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(ulong x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(ulong16 x); __attribute__((overloadable)) short convert_short_sat_rte(ulong x); __attribute__((overloadable)) short2 convert_short2_sat_rte(ulong2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(ulong3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(ulong4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(ulong8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(ulong16 x); __attribute__((overloadable)) long convert_long_sat_rte(ulong x); __attribute__((overloadable)) long2 convert_long2_sat_rte(ulong2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(ulong3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(ulong4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(ulong8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(ulong16 x); __attribute__((overloadable)) float convert_float_sat_rte(ulong x); __attribute__((overloadable)) float2 convert_float2_sat_rte(ulong2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(ulong3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(ulong4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(ulong8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(ulong16 x); __attribute__((overloadable)) double convert_double_sat_rte(ulong x); __attribute__((overloadable)) double2 convert_double2_sat_rte(ulong2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(ulong3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(ulong4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(ulong8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(ulong16 x); __attribute__((overloadable)) char convert_char_sat_rte(float x); __attribute__((overloadable)) char2 convert_char2_sat_rte(float2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(float3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(float4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(float8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(float16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(float x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(float16 x); __attribute__((overloadable)) int convert_int_sat_rte(float x); __attribute__((overloadable)) int2 convert_int2_sat_rte(float2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(float3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(float4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(float8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(float16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(float x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(float2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(float3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(float4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(float8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(float16 x); __attribute__((overloadable)) short convert_short_sat_rte(float x); __attribute__((overloadable)) short2 convert_short2_sat_rte(float2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(float3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(float4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(float8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(float16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(float x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(float16 x); __attribute__((overloadable)) long convert_long_sat_rte(float x); __attribute__((overloadable)) long2 convert_long2_sat_rte(float2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(float3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(float4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(float8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(float16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(float x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(float16 x); __attribute__((overloadable)) float convert_float_sat_rte(float x); __attribute__((overloadable)) float2 convert_float2_sat_rte(float2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(float3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(float4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(float8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(float16 x); __attribute__((overloadable)) double convert_double_sat_rte(float x); __attribute__((overloadable)) double2 convert_double2_sat_rte(float2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(float3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(float4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(float8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(float16 x); __attribute__((overloadable)) char convert_char_sat_rte(double x); __attribute__((overloadable)) char2 convert_char2_sat_rte(double2 x); __attribute__((overloadable)) char3 convert_char3_sat_rte(double3 x); __attribute__((overloadable)) char4 convert_char4_sat_rte(double4 x); __attribute__((overloadable)) char8 convert_char8_sat_rte(double8 x); __attribute__((overloadable)) char16 convert_char16_sat_rte(double16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rte(double x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rte(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rte(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rte(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rte(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rte(double16 x); __attribute__((overloadable)) int convert_int_sat_rte(double x); __attribute__((overloadable)) int2 convert_int2_sat_rte(double2 x); __attribute__((overloadable)) int3 convert_int3_sat_rte(double3 x); __attribute__((overloadable)) int4 convert_int4_sat_rte(double4 x); __attribute__((overloadable)) int8 convert_int8_sat_rte(double8 x); __attribute__((overloadable)) int16 convert_int16_sat_rte(double16 x); __attribute__((overloadable)) uint convert_uint_sat_rte(double x); __attribute__((overloadable)) uint2 convert_uint2_sat_rte(double2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rte(double3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rte(double4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rte(double8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rte(double16 x); __attribute__((overloadable)) short convert_short_sat_rte(double x); __attribute__((overloadable)) short2 convert_short2_sat_rte(double2 x); __attribute__((overloadable)) short3 convert_short3_sat_rte(double3 x); __attribute__((overloadable)) short4 convert_short4_sat_rte(double4 x); __attribute__((overloadable)) short8 convert_short8_sat_rte(double8 x); __attribute__((overloadable)) short16 convert_short16_sat_rte(double16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rte(double x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rte(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rte(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rte(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rte(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rte(double16 x); __attribute__((overloadable)) long convert_long_sat_rte(double x); __attribute__((overloadable)) long2 convert_long2_sat_rte(double2 x); __attribute__((overloadable)) long3 convert_long3_sat_rte(double3 x); __attribute__((overloadable)) long4 convert_long4_sat_rte(double4 x); __attribute__((overloadable)) long8 convert_long8_sat_rte(double8 x); __attribute__((overloadable)) long16 convert_long16_sat_rte(double16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rte(double x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rte(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rte(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rte(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rte(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rte(double16 x); __attribute__((overloadable)) float convert_float_sat_rte(double x); __attribute__((overloadable)) float2 convert_float2_sat_rte(double2 x); __attribute__((overloadable)) float3 convert_float3_sat_rte(double3 x); __attribute__((overloadable)) float4 convert_float4_sat_rte(double4 x); __attribute__((overloadable)) float8 convert_float8_sat_rte(double8 x); __attribute__((overloadable)) float16 convert_float16_sat_rte(double16 x); __attribute__((overloadable)) double convert_double_sat_rte(double x); __attribute__((overloadable)) double2 convert_double2_sat_rte(double2 x); __attribute__((overloadable)) double3 convert_double3_sat_rte(double3 x); __attribute__((overloadable)) double4 convert_double4_sat_rte(double4 x); __attribute__((overloadable)) double8 convert_double8_sat_rte(double8 x); __attribute__((overloadable)) double16 convert_double16_sat_rte(double16 x); __attribute__((overloadable)) char convert_char_rte(char x); __attribute__((overloadable)) char2 convert_char2_rte(char2 x); __attribute__((overloadable)) char3 convert_char3_rte(char3 x); __attribute__((overloadable)) char4 convert_char4_rte(char4 x); __attribute__((overloadable)) char8 convert_char8_rte(char8 x); __attribute__((overloadable)) char16 convert_char16_rte(char16 x); __attribute__((overloadable)) uchar convert_uchar_rte(char x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(char16 x); __attribute__((overloadable)) int convert_int_rte(char x); __attribute__((overloadable)) int2 convert_int2_rte(char2 x); __attribute__((overloadable)) int3 convert_int3_rte(char3 x); __attribute__((overloadable)) int4 convert_int4_rte(char4 x); __attribute__((overloadable)) int8 convert_int8_rte(char8 x); __attribute__((overloadable)) int16 convert_int16_rte(char16 x); __attribute__((overloadable)) uint convert_uint_rte(char x); __attribute__((overloadable)) uint2 convert_uint2_rte(char2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(char3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(char4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(char8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(char16 x); __attribute__((overloadable)) short convert_short_rte(char x); __attribute__((overloadable)) short2 convert_short2_rte(char2 x); __attribute__((overloadable)) short3 convert_short3_rte(char3 x); __attribute__((overloadable)) short4 convert_short4_rte(char4 x); __attribute__((overloadable)) short8 convert_short8_rte(char8 x); __attribute__((overloadable)) short16 convert_short16_rte(char16 x); __attribute__((overloadable)) ushort convert_ushort_rte(char x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(char16 x); __attribute__((overloadable)) long convert_long_rte(char x); __attribute__((overloadable)) long2 convert_long2_rte(char2 x); __attribute__((overloadable)) long3 convert_long3_rte(char3 x); __attribute__((overloadable)) long4 convert_long4_rte(char4 x); __attribute__((overloadable)) long8 convert_long8_rte(char8 x); __attribute__((overloadable)) long16 convert_long16_rte(char16 x); __attribute__((overloadable)) ulong convert_ulong_rte(char x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(char16 x); __attribute__((overloadable)) float convert_float_rte(char x); __attribute__((overloadable)) float2 convert_float2_rte(char2 x); __attribute__((overloadable)) float3 convert_float3_rte(char3 x); __attribute__((overloadable)) float4 convert_float4_rte(char4 x); __attribute__((overloadable)) float8 convert_float8_rte(char8 x); __attribute__((overloadable)) float16 convert_float16_rte(char16 x); __attribute__((overloadable)) double convert_double_rte(char x); __attribute__((overloadable)) double2 convert_double2_rte(char2 x); __attribute__((overloadable)) double3 convert_double3_rte(char3 x); __attribute__((overloadable)) double4 convert_double4_rte(char4 x); __attribute__((overloadable)) double8 convert_double8_rte(char8 x); __attribute__((overloadable)) double16 convert_double16_rte(char16 x); __attribute__((overloadable)) char convert_char_rte(uchar x); __attribute__((overloadable)) char2 convert_char2_rte(uchar2 x); __attribute__((overloadable)) char3 convert_char3_rte(uchar3 x); __attribute__((overloadable)) char4 convert_char4_rte(uchar4 x); __attribute__((overloadable)) char8 convert_char8_rte(uchar8 x); __attribute__((overloadable)) char16 convert_char16_rte(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_rte(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(uchar16 x); __attribute__((overloadable)) int convert_int_rte(uchar x); __attribute__((overloadable)) int2 convert_int2_rte(uchar2 x); __attribute__((overloadable)) int3 convert_int3_rte(uchar3 x); __attribute__((overloadable)) int4 convert_int4_rte(uchar4 x); __attribute__((overloadable)) int8 convert_int8_rte(uchar8 x); __attribute__((overloadable)) int16 convert_int16_rte(uchar16 x); __attribute__((overloadable)) uint convert_uint_rte(uchar x); __attribute__((overloadable)) uint2 convert_uint2_rte(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(uchar16 x); __attribute__((overloadable)) short convert_short_rte(uchar x); __attribute__((overloadable)) short2 convert_short2_rte(uchar2 x); __attribute__((overloadable)) short3 convert_short3_rte(uchar3 x); __attribute__((overloadable)) short4 convert_short4_rte(uchar4 x); __attribute__((overloadable)) short8 convert_short8_rte(uchar8 x); __attribute__((overloadable)) short16 convert_short16_rte(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_rte(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(uchar16 x); __attribute__((overloadable)) long convert_long_rte(uchar x); __attribute__((overloadable)) long2 convert_long2_rte(uchar2 x); __attribute__((overloadable)) long3 convert_long3_rte(uchar3 x); __attribute__((overloadable)) long4 convert_long4_rte(uchar4 x); __attribute__((overloadable)) long8 convert_long8_rte(uchar8 x); __attribute__((overloadable)) long16 convert_long16_rte(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_rte(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(uchar16 x); __attribute__((overloadable)) float convert_float_rte(uchar x); __attribute__((overloadable)) float2 convert_float2_rte(uchar2 x); __attribute__((overloadable)) float3 convert_float3_rte(uchar3 x); __attribute__((overloadable)) float4 convert_float4_rte(uchar4 x); __attribute__((overloadable)) float8 convert_float8_rte(uchar8 x); __attribute__((overloadable)) float16 convert_float16_rte(uchar16 x); __attribute__((overloadable)) double convert_double_rte(uchar x); __attribute__((overloadable)) double2 convert_double2_rte(uchar2 x); __attribute__((overloadable)) double3 convert_double3_rte(uchar3 x); __attribute__((overloadable)) double4 convert_double4_rte(uchar4 x); __attribute__((overloadable)) double8 convert_double8_rte(uchar8 x); __attribute__((overloadable)) double16 convert_double16_rte(uchar16 x); __attribute__((overloadable)) char convert_char_rte(int x); __attribute__((overloadable)) char2 convert_char2_rte(int2 x); __attribute__((overloadable)) char3 convert_char3_rte(int3 x); __attribute__((overloadable)) char4 convert_char4_rte(int4 x); __attribute__((overloadable)) char8 convert_char8_rte(int8 x); __attribute__((overloadable)) char16 convert_char16_rte(int16 x); __attribute__((overloadable)) uchar convert_uchar_rte(int x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(int16 x); __attribute__((overloadable)) int convert_int_rte(int x); __attribute__((overloadable)) int2 convert_int2_rte(int2 x); __attribute__((overloadable)) int3 convert_int3_rte(int3 x); __attribute__((overloadable)) int4 convert_int4_rte(int4 x); __attribute__((overloadable)) int8 convert_int8_rte(int8 x); __attribute__((overloadable)) int16 convert_int16_rte(int16 x); __attribute__((overloadable)) uint convert_uint_rte(int x); __attribute__((overloadable)) uint2 convert_uint2_rte(int2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(int3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(int4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(int8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(int16 x); __attribute__((overloadable)) short convert_short_rte(int x); __attribute__((overloadable)) short2 convert_short2_rte(int2 x); __attribute__((overloadable)) short3 convert_short3_rte(int3 x); __attribute__((overloadable)) short4 convert_short4_rte(int4 x); __attribute__((overloadable)) short8 convert_short8_rte(int8 x); __attribute__((overloadable)) short16 convert_short16_rte(int16 x); __attribute__((overloadable)) ushort convert_ushort_rte(int x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(int16 x); __attribute__((overloadable)) long convert_long_rte(int x); __attribute__((overloadable)) long2 convert_long2_rte(int2 x); __attribute__((overloadable)) long3 convert_long3_rte(int3 x); __attribute__((overloadable)) long4 convert_long4_rte(int4 x); __attribute__((overloadable)) long8 convert_long8_rte(int8 x); __attribute__((overloadable)) long16 convert_long16_rte(int16 x); __attribute__((overloadable)) ulong convert_ulong_rte(int x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(int16 x); __attribute__((overloadable)) float convert_float_rte(int x); __attribute__((overloadable)) float2 convert_float2_rte(int2 x); __attribute__((overloadable)) float3 convert_float3_rte(int3 x); __attribute__((overloadable)) float4 convert_float4_rte(int4 x); __attribute__((overloadable)) float8 convert_float8_rte(int8 x); __attribute__((overloadable)) float16 convert_float16_rte(int16 x); __attribute__((overloadable)) double convert_double_rte(int x); __attribute__((overloadable)) double2 convert_double2_rte(int2 x); __attribute__((overloadable)) double3 convert_double3_rte(int3 x); __attribute__((overloadable)) double4 convert_double4_rte(int4 x); __attribute__((overloadable)) double8 convert_double8_rte(int8 x); __attribute__((overloadable)) double16 convert_double16_rte(int16 x); __attribute__((overloadable)) char convert_char_rte(uint x); __attribute__((overloadable)) char2 convert_char2_rte(uint2 x); __attribute__((overloadable)) char3 convert_char3_rte(uint3 x); __attribute__((overloadable)) char4 convert_char4_rte(uint4 x); __attribute__((overloadable)) char8 convert_char8_rte(uint8 x); __attribute__((overloadable)) char16 convert_char16_rte(uint16 x); __attribute__((overloadable)) uchar convert_uchar_rte(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(uint16 x); __attribute__((overloadable)) int convert_int_rte(uint x); __attribute__((overloadable)) int2 convert_int2_rte(uint2 x); __attribute__((overloadable)) int3 convert_int3_rte(uint3 x); __attribute__((overloadable)) int4 convert_int4_rte(uint4 x); __attribute__((overloadable)) int8 convert_int8_rte(uint8 x); __attribute__((overloadable)) int16 convert_int16_rte(uint16 x); __attribute__((overloadable)) uint convert_uint_rte(uint x); __attribute__((overloadable)) uint2 convert_uint2_rte(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(uint16 x); __attribute__((overloadable)) short convert_short_rte(uint x); __attribute__((overloadable)) short2 convert_short2_rte(uint2 x); __attribute__((overloadable)) short3 convert_short3_rte(uint3 x); __attribute__((overloadable)) short4 convert_short4_rte(uint4 x); __attribute__((overloadable)) short8 convert_short8_rte(uint8 x); __attribute__((overloadable)) short16 convert_short16_rte(uint16 x); __attribute__((overloadable)) ushort convert_ushort_rte(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(uint16 x); __attribute__((overloadable)) long convert_long_rte(uint x); __attribute__((overloadable)) long2 convert_long2_rte(uint2 x); __attribute__((overloadable)) long3 convert_long3_rte(uint3 x); __attribute__((overloadable)) long4 convert_long4_rte(uint4 x); __attribute__((overloadable)) long8 convert_long8_rte(uint8 x); __attribute__((overloadable)) long16 convert_long16_rte(uint16 x); __attribute__((overloadable)) ulong convert_ulong_rte(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(uint16 x); __attribute__((overloadable)) float convert_float_rte(uint x); __attribute__((overloadable)) float2 convert_float2_rte(uint2 x); __attribute__((overloadable)) float3 convert_float3_rte(uint3 x); __attribute__((overloadable)) float4 convert_float4_rte(uint4 x); __attribute__((overloadable)) float8 convert_float8_rte(uint8 x); __attribute__((overloadable)) float16 convert_float16_rte(uint16 x); __attribute__((overloadable)) double convert_double_rte(uint x); __attribute__((overloadable)) double2 convert_double2_rte(uint2 x); __attribute__((overloadable)) double3 convert_double3_rte(uint3 x); __attribute__((overloadable)) double4 convert_double4_rte(uint4 x); __attribute__((overloadable)) double8 convert_double8_rte(uint8 x); __attribute__((overloadable)) double16 convert_double16_rte(uint16 x); __attribute__((overloadable)) char convert_char_rte(short x); __attribute__((overloadable)) char2 convert_char2_rte(short2 x); __attribute__((overloadable)) char3 convert_char3_rte(short3 x); __attribute__((overloadable)) char4 convert_char4_rte(short4 x); __attribute__((overloadable)) char8 convert_char8_rte(short8 x); __attribute__((overloadable)) char16 convert_char16_rte(short16 x); __attribute__((overloadable)) uchar convert_uchar_rte(short x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(short16 x); __attribute__((overloadable)) int convert_int_rte(short x); __attribute__((overloadable)) int2 convert_int2_rte(short2 x); __attribute__((overloadable)) int3 convert_int3_rte(short3 x); __attribute__((overloadable)) int4 convert_int4_rte(short4 x); __attribute__((overloadable)) int8 convert_int8_rte(short8 x); __attribute__((overloadable)) int16 convert_int16_rte(short16 x); __attribute__((overloadable)) uint convert_uint_rte(short x); __attribute__((overloadable)) uint2 convert_uint2_rte(short2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(short3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(short4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(short8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(short16 x); __attribute__((overloadable)) short convert_short_rte(short x); __attribute__((overloadable)) short2 convert_short2_rte(short2 x); __attribute__((overloadable)) short3 convert_short3_rte(short3 x); __attribute__((overloadable)) short4 convert_short4_rte(short4 x); __attribute__((overloadable)) short8 convert_short8_rte(short8 x); __attribute__((overloadable)) short16 convert_short16_rte(short16 x); __attribute__((overloadable)) ushort convert_ushort_rte(short x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(short16 x); __attribute__((overloadable)) long convert_long_rte(short x); __attribute__((overloadable)) long2 convert_long2_rte(short2 x); __attribute__((overloadable)) long3 convert_long3_rte(short3 x); __attribute__((overloadable)) long4 convert_long4_rte(short4 x); __attribute__((overloadable)) long8 convert_long8_rte(short8 x); __attribute__((overloadable)) long16 convert_long16_rte(short16 x); __attribute__((overloadable)) ulong convert_ulong_rte(short x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(short16 x); __attribute__((overloadable)) float convert_float_rte(short x); __attribute__((overloadable)) float2 convert_float2_rte(short2 x); __attribute__((overloadable)) float3 convert_float3_rte(short3 x); __attribute__((overloadable)) float4 convert_float4_rte(short4 x); __attribute__((overloadable)) float8 convert_float8_rte(short8 x); __attribute__((overloadable)) float16 convert_float16_rte(short16 x); __attribute__((overloadable)) double convert_double_rte(short x); __attribute__((overloadable)) double2 convert_double2_rte(short2 x); __attribute__((overloadable)) double3 convert_double3_rte(short3 x); __attribute__((overloadable)) double4 convert_double4_rte(short4 x); __attribute__((overloadable)) double8 convert_double8_rte(short8 x); __attribute__((overloadable)) double16 convert_double16_rte(short16 x); __attribute__((overloadable)) char convert_char_rte(ushort x); __attribute__((overloadable)) char2 convert_char2_rte(ushort2 x); __attribute__((overloadable)) char3 convert_char3_rte(ushort3 x); __attribute__((overloadable)) char4 convert_char4_rte(ushort4 x); __attribute__((overloadable)) char8 convert_char8_rte(ushort8 x); __attribute__((overloadable)) char16 convert_char16_rte(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_rte(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(ushort16 x); __attribute__((overloadable)) int convert_int_rte(ushort x); __attribute__((overloadable)) int2 convert_int2_rte(ushort2 x); __attribute__((overloadable)) int3 convert_int3_rte(ushort3 x); __attribute__((overloadable)) int4 convert_int4_rte(ushort4 x); __attribute__((overloadable)) int8 convert_int8_rte(ushort8 x); __attribute__((overloadable)) int16 convert_int16_rte(ushort16 x); __attribute__((overloadable)) uint convert_uint_rte(ushort x); __attribute__((overloadable)) uint2 convert_uint2_rte(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(ushort16 x); __attribute__((overloadable)) short convert_short_rte(ushort x); __attribute__((overloadable)) short2 convert_short2_rte(ushort2 x); __attribute__((overloadable)) short3 convert_short3_rte(ushort3 x); __attribute__((overloadable)) short4 convert_short4_rte(ushort4 x); __attribute__((overloadable)) short8 convert_short8_rte(ushort8 x); __attribute__((overloadable)) short16 convert_short16_rte(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_rte(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(ushort16 x); __attribute__((overloadable)) long convert_long_rte(ushort x); __attribute__((overloadable)) long2 convert_long2_rte(ushort2 x); __attribute__((overloadable)) long3 convert_long3_rte(ushort3 x); __attribute__((overloadable)) long4 convert_long4_rte(ushort4 x); __attribute__((overloadable)) long8 convert_long8_rte(ushort8 x); __attribute__((overloadable)) long16 convert_long16_rte(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_rte(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(ushort16 x); __attribute__((overloadable)) float convert_float_rte(ushort x); __attribute__((overloadable)) float2 convert_float2_rte(ushort2 x); __attribute__((overloadable)) float3 convert_float3_rte(ushort3 x); __attribute__((overloadable)) float4 convert_float4_rte(ushort4 x); __attribute__((overloadable)) float8 convert_float8_rte(ushort8 x); __attribute__((overloadable)) float16 convert_float16_rte(ushort16 x); __attribute__((overloadable)) double convert_double_rte(ushort x); __attribute__((overloadable)) double2 convert_double2_rte(ushort2 x); __attribute__((overloadable)) double3 convert_double3_rte(ushort3 x); __attribute__((overloadable)) double4 convert_double4_rte(ushort4 x); __attribute__((overloadable)) double8 convert_double8_rte(ushort8 x); __attribute__((overloadable)) double16 convert_double16_rte(ushort16 x); __attribute__((overloadable)) char convert_char_rte(long x); __attribute__((overloadable)) char2 convert_char2_rte(long2 x); __attribute__((overloadable)) char3 convert_char3_rte(long3 x); __attribute__((overloadable)) char4 convert_char4_rte(long4 x); __attribute__((overloadable)) char8 convert_char8_rte(long8 x); __attribute__((overloadable)) char16 convert_char16_rte(long16 x); __attribute__((overloadable)) uchar convert_uchar_rte(long x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(long16 x); __attribute__((overloadable)) int convert_int_rte(long x); __attribute__((overloadable)) int2 convert_int2_rte(long2 x); __attribute__((overloadable)) int3 convert_int3_rte(long3 x); __attribute__((overloadable)) int4 convert_int4_rte(long4 x); __attribute__((overloadable)) int8 convert_int8_rte(long8 x); __attribute__((overloadable)) int16 convert_int16_rte(long16 x); __attribute__((overloadable)) uint convert_uint_rte(long x); __attribute__((overloadable)) uint2 convert_uint2_rte(long2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(long3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(long4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(long8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(long16 x); __attribute__((overloadable)) short convert_short_rte(long x); __attribute__((overloadable)) short2 convert_short2_rte(long2 x); __attribute__((overloadable)) short3 convert_short3_rte(long3 x); __attribute__((overloadable)) short4 convert_short4_rte(long4 x); __attribute__((overloadable)) short8 convert_short8_rte(long8 x); __attribute__((overloadable)) short16 convert_short16_rte(long16 x); __attribute__((overloadable)) ushort convert_ushort_rte(long x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(long16 x); __attribute__((overloadable)) long convert_long_rte(long x); __attribute__((overloadable)) long2 convert_long2_rte(long2 x); __attribute__((overloadable)) long3 convert_long3_rte(long3 x); __attribute__((overloadable)) long4 convert_long4_rte(long4 x); __attribute__((overloadable)) long8 convert_long8_rte(long8 x); __attribute__((overloadable)) long16 convert_long16_rte(long16 x); __attribute__((overloadable)) ulong convert_ulong_rte(long x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(long16 x); __attribute__((overloadable)) float convert_float_rte(long x); __attribute__((overloadable)) float2 convert_float2_rte(long2 x); __attribute__((overloadable)) float3 convert_float3_rte(long3 x); __attribute__((overloadable)) float4 convert_float4_rte(long4 x); __attribute__((overloadable)) float8 convert_float8_rte(long8 x); __attribute__((overloadable)) float16 convert_float16_rte(long16 x); __attribute__((overloadable)) double convert_double_rte(long x); __attribute__((overloadable)) double2 convert_double2_rte(long2 x); __attribute__((overloadable)) double3 convert_double3_rte(long3 x); __attribute__((overloadable)) double4 convert_double4_rte(long4 x); __attribute__((overloadable)) double8 convert_double8_rte(long8 x); __attribute__((overloadable)) double16 convert_double16_rte(long16 x); __attribute__((overloadable)) char convert_char_rte(ulong x); __attribute__((overloadable)) char2 convert_char2_rte(ulong2 x); __attribute__((overloadable)) char3 convert_char3_rte(ulong3 x); __attribute__((overloadable)) char4 convert_char4_rte(ulong4 x); __attribute__((overloadable)) char8 convert_char8_rte(ulong8 x); __attribute__((overloadable)) char16 convert_char16_rte(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_rte(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(ulong16 x); __attribute__((overloadable)) int convert_int_rte(ulong x); __attribute__((overloadable)) int2 convert_int2_rte(ulong2 x); __attribute__((overloadable)) int3 convert_int3_rte(ulong3 x); __attribute__((overloadable)) int4 convert_int4_rte(ulong4 x); __attribute__((overloadable)) int8 convert_int8_rte(ulong8 x); __attribute__((overloadable)) int16 convert_int16_rte(ulong16 x); __attribute__((overloadable)) uint convert_uint_rte(ulong x); __attribute__((overloadable)) uint2 convert_uint2_rte(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(ulong16 x); __attribute__((overloadable)) short convert_short_rte(ulong x); __attribute__((overloadable)) short2 convert_short2_rte(ulong2 x); __attribute__((overloadable)) short3 convert_short3_rte(ulong3 x); __attribute__((overloadable)) short4 convert_short4_rte(ulong4 x); __attribute__((overloadable)) short8 convert_short8_rte(ulong8 x); __attribute__((overloadable)) short16 convert_short16_rte(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_rte(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(ulong16 x); __attribute__((overloadable)) long convert_long_rte(ulong x); __attribute__((overloadable)) long2 convert_long2_rte(ulong2 x); __attribute__((overloadable)) long3 convert_long3_rte(ulong3 x); __attribute__((overloadable)) long4 convert_long4_rte(ulong4 x); __attribute__((overloadable)) long8 convert_long8_rte(ulong8 x); __attribute__((overloadable)) long16 convert_long16_rte(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_rte(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(ulong16 x); __attribute__((overloadable)) float convert_float_rte(ulong x); __attribute__((overloadable)) float2 convert_float2_rte(ulong2 x); __attribute__((overloadable)) float3 convert_float3_rte(ulong3 x); __attribute__((overloadable)) float4 convert_float4_rte(ulong4 x); __attribute__((overloadable)) float8 convert_float8_rte(ulong8 x); __attribute__((overloadable)) float16 convert_float16_rte(ulong16 x); __attribute__((overloadable)) double convert_double_rte(ulong x); __attribute__((overloadable)) double2 convert_double2_rte(ulong2 x); __attribute__((overloadable)) double3 convert_double3_rte(ulong3 x); __attribute__((overloadable)) double4 convert_double4_rte(ulong4 x); __attribute__((overloadable)) double8 convert_double8_rte(ulong8 x); __attribute__((overloadable)) double16 convert_double16_rte(ulong16 x); __attribute__((overloadable)) char convert_char_rte(float x); __attribute__((overloadable)) char2 convert_char2_rte(float2 x); __attribute__((overloadable)) char3 convert_char3_rte(float3 x); __attribute__((overloadable)) char4 convert_char4_rte(float4 x); __attribute__((overloadable)) char8 convert_char8_rte(float8 x); __attribute__((overloadable)) char16 convert_char16_rte(float16 x); __attribute__((overloadable)) uchar convert_uchar_rte(float x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(float16 x); __attribute__((overloadable)) int convert_int_rte(float x); __attribute__((overloadable)) int2 convert_int2_rte(float2 x); __attribute__((overloadable)) int3 convert_int3_rte(float3 x); __attribute__((overloadable)) int4 convert_int4_rte(float4 x); __attribute__((overloadable)) int8 convert_int8_rte(float8 x); __attribute__((overloadable)) int16 convert_int16_rte(float16 x); __attribute__((overloadable)) uint convert_uint_rte(float x); __attribute__((overloadable)) uint2 convert_uint2_rte(float2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(float3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(float4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(float8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(float16 x); __attribute__((overloadable)) short convert_short_rte(float x); __attribute__((overloadable)) short2 convert_short2_rte(float2 x); __attribute__((overloadable)) short3 convert_short3_rte(float3 x); __attribute__((overloadable)) short4 convert_short4_rte(float4 x); __attribute__((overloadable)) short8 convert_short8_rte(float8 x); __attribute__((overloadable)) short16 convert_short16_rte(float16 x); __attribute__((overloadable)) ushort convert_ushort_rte(float x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(float16 x); __attribute__((overloadable)) long convert_long_rte(float x); __attribute__((overloadable)) long2 convert_long2_rte(float2 x); __attribute__((overloadable)) long3 convert_long3_rte(float3 x); __attribute__((overloadable)) long4 convert_long4_rte(float4 x); __attribute__((overloadable)) long8 convert_long8_rte(float8 x); __attribute__((overloadable)) long16 convert_long16_rte(float16 x); __attribute__((overloadable)) ulong convert_ulong_rte(float x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(float16 x); __attribute__((overloadable)) float convert_float_rte(float x); __attribute__((overloadable)) float2 convert_float2_rte(float2 x); __attribute__((overloadable)) float3 convert_float3_rte(float3 x); __attribute__((overloadable)) float4 convert_float4_rte(float4 x); __attribute__((overloadable)) float8 convert_float8_rte(float8 x); __attribute__((overloadable)) float16 convert_float16_rte(float16 x); __attribute__((overloadable)) double convert_double_rte(float x); __attribute__((overloadable)) double2 convert_double2_rte(float2 x); __attribute__((overloadable)) double3 convert_double3_rte(float3 x); __attribute__((overloadable)) double4 convert_double4_rte(float4 x); __attribute__((overloadable)) double8 convert_double8_rte(float8 x); __attribute__((overloadable)) double16 convert_double16_rte(float16 x); __attribute__((overloadable)) char convert_char_rte(double x); __attribute__((overloadable)) char2 convert_char2_rte(double2 x); __attribute__((overloadable)) char3 convert_char3_rte(double3 x); __attribute__((overloadable)) char4 convert_char4_rte(double4 x); __attribute__((overloadable)) char8 convert_char8_rte(double8 x); __attribute__((overloadable)) char16 convert_char16_rte(double16 x); __attribute__((overloadable)) uchar convert_uchar_rte(double x); __attribute__((overloadable)) uchar2 convert_uchar2_rte(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rte(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rte(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rte(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rte(double16 x); __attribute__((overloadable)) int convert_int_rte(double x); __attribute__((overloadable)) int2 convert_int2_rte(double2 x); __attribute__((overloadable)) int3 convert_int3_rte(double3 x); __attribute__((overloadable)) int4 convert_int4_rte(double4 x); __attribute__((overloadable)) int8 convert_int8_rte(double8 x); __attribute__((overloadable)) int16 convert_int16_rte(double16 x); __attribute__((overloadable)) uint convert_uint_rte(double x); __attribute__((overloadable)) uint2 convert_uint2_rte(double2 x); __attribute__((overloadable)) uint3 convert_uint3_rte(double3 x); __attribute__((overloadable)) uint4 convert_uint4_rte(double4 x); __attribute__((overloadable)) uint8 convert_uint8_rte(double8 x); __attribute__((overloadable)) uint16 convert_uint16_rte(double16 x); __attribute__((overloadable)) short convert_short_rte(double x); __attribute__((overloadable)) short2 convert_short2_rte(double2 x); __attribute__((overloadable)) short3 convert_short3_rte(double3 x); __attribute__((overloadable)) short4 convert_short4_rte(double4 x); __attribute__((overloadable)) short8 convert_short8_rte(double8 x); __attribute__((overloadable)) short16 convert_short16_rte(double16 x); __attribute__((overloadable)) ushort convert_ushort_rte(double x); __attribute__((overloadable)) ushort2 convert_ushort2_rte(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rte(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rte(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rte(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rte(double16 x); __attribute__((overloadable)) long convert_long_rte(double x); __attribute__((overloadable)) long2 convert_long2_rte(double2 x); __attribute__((overloadable)) long3 convert_long3_rte(double3 x); __attribute__((overloadable)) long4 convert_long4_rte(double4 x); __attribute__((overloadable)) long8 convert_long8_rte(double8 x); __attribute__((overloadable)) long16 convert_long16_rte(double16 x); __attribute__((overloadable)) ulong convert_ulong_rte(double x); __attribute__((overloadable)) ulong2 convert_ulong2_rte(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rte(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rte(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rte(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rte(double16 x); __attribute__((overloadable)) float convert_float_rte(double x); __attribute__((overloadable)) float2 convert_float2_rte(double2 x); __attribute__((overloadable)) float3 convert_float3_rte(double3 x); __attribute__((overloadable)) float4 convert_float4_rte(double4 x); __attribute__((overloadable)) float8 convert_float8_rte(double8 x); __attribute__((overloadable)) float16 convert_float16_rte(double16 x); __attribute__((overloadable)) double convert_double_rte(double x); __attribute__((overloadable)) double2 convert_double2_rte(double2 x); __attribute__((overloadable)) double3 convert_double3_rte(double3 x); __attribute__((overloadable)) double4 convert_double4_rte(double4 x); __attribute__((overloadable)) double8 convert_double8_rte(double8 x); __attribute__((overloadable)) double16 convert_double16_rte(double16 x);
__attribute__((overloadable)) char convert_char_sat_rtz(char x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(char2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(char3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(char4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(char8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(char16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(char x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(char16 x); __attribute__((overloadable)) int convert_int_sat_rtz(char x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(char2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(char3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(char4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(char8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(char16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(char x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(char2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(char3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(char4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(char8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(char16 x); __attribute__((overloadable)) short convert_short_sat_rtz(char x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(char2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(char3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(char4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(char8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(char16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(char x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(char16 x); __attribute__((overloadable)) long convert_long_sat_rtz(char x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(char2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(char3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(char4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(char8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(char16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(char x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(char16 x); __attribute__((overloadable)) float convert_float_sat_rtz(char x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(char2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(char3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(char4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(char8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(char16 x); __attribute__((overloadable)) double convert_double_sat_rtz(char x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(char2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(char3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(char4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(char8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(char16 x); __attribute__((overloadable)) char convert_char_sat_rtz(uchar x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(uchar2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(uchar3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(uchar4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(uchar8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(uchar16 x); __attribute__((overloadable)) int convert_int_sat_rtz(uchar x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(uchar2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(uchar3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(uchar4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(uchar8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(uchar16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(uchar x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(uchar16 x); __attribute__((overloadable)) short convert_short_sat_rtz(uchar x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(uchar2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(uchar3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(uchar4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(uchar8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(uchar16 x); __attribute__((overloadable)) long convert_long_sat_rtz(uchar x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(uchar2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(uchar3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(uchar4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(uchar8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(uchar16 x); __attribute__((overloadable)) float convert_float_sat_rtz(uchar x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(uchar2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(uchar3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(uchar4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(uchar8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(uchar16 x); __attribute__((overloadable)) double convert_double_sat_rtz(uchar x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(uchar2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(uchar3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(uchar4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(uchar8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(uchar16 x); __attribute__((overloadable)) char convert_char_sat_rtz(int x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(int2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(int3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(int4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(int8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(int16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(int x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(int16 x); __attribute__((overloadable)) int convert_int_sat_rtz(int x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(int2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(int3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(int4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(int8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(int16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(int x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(int2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(int3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(int4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(int8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(int16 x); __attribute__((overloadable)) short convert_short_sat_rtz(int x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(int2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(int3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(int4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(int8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(int16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(int x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(int16 x); __attribute__((overloadable)) long convert_long_sat_rtz(int x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(int2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(int3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(int4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(int8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(int16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(int x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(int16 x); __attribute__((overloadable)) float convert_float_sat_rtz(int x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(int2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(int3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(int4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(int8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(int16 x); __attribute__((overloadable)) double convert_double_sat_rtz(int x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(int2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(int3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(int4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(int8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(int16 x); __attribute__((overloadable)) char convert_char_sat_rtz(uint x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(uint2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(uint3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(uint4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(uint8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(uint16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(uint16 x); __attribute__((overloadable)) int convert_int_sat_rtz(uint x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(uint2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(uint3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(uint4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(uint8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(uint16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(uint x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(uint16 x); __attribute__((overloadable)) short convert_short_sat_rtz(uint x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(uint2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(uint3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(uint4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(uint8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(uint16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(uint16 x); __attribute__((overloadable)) long convert_long_sat_rtz(uint x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(uint2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(uint3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(uint4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(uint8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(uint16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(uint16 x); __attribute__((overloadable)) float convert_float_sat_rtz(uint x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(uint2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(uint3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(uint4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(uint8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(uint16 x); __attribute__((overloadable)) double convert_double_sat_rtz(uint x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(uint2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(uint3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(uint4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(uint8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(uint16 x); __attribute__((overloadable)) char convert_char_sat_rtz(short x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(short2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(short3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(short4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(short8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(short16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(short x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(short16 x); __attribute__((overloadable)) int convert_int_sat_rtz(short x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(short2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(short3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(short4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(short8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(short16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(short x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(short2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(short3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(short4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(short8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(short16 x); __attribute__((overloadable)) short convert_short_sat_rtz(short x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(short2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(short3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(short4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(short8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(short16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(short x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(short16 x); __attribute__((overloadable)) long convert_long_sat_rtz(short x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(short2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(short3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(short4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(short8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(short16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(short x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(short16 x); __attribute__((overloadable)) float convert_float_sat_rtz(short x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(short2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(short3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(short4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(short8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(short16 x); __attribute__((overloadable)) double convert_double_sat_rtz(short x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(short2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(short3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(short4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(short8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(short16 x); __attribute__((overloadable)) char convert_char_sat_rtz(ushort x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(ushort2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(ushort3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(ushort4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(ushort8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(ushort16 x); __attribute__((overloadable)) int convert_int_sat_rtz(ushort x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(ushort2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(ushort3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(ushort4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(ushort8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(ushort16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(ushort x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(ushort16 x); __attribute__((overloadable)) short convert_short_sat_rtz(ushort x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(ushort2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(ushort3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(ushort4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(ushort8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(ushort16 x); __attribute__((overloadable)) long convert_long_sat_rtz(ushort x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(ushort2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(ushort3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(ushort4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(ushort8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(ushort16 x); __attribute__((overloadable)) float convert_float_sat_rtz(ushort x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(ushort2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(ushort3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(ushort4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(ushort8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(ushort16 x); __attribute__((overloadable)) double convert_double_sat_rtz(ushort x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(ushort2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(ushort3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(ushort4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(ushort8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(ushort16 x); __attribute__((overloadable)) char convert_char_sat_rtz(long x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(long2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(long3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(long4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(long8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(long16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(long x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(long16 x); __attribute__((overloadable)) int convert_int_sat_rtz(long x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(long2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(long3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(long4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(long8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(long16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(long x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(long2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(long3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(long4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(long8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(long16 x); __attribute__((overloadable)) short convert_short_sat_rtz(long x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(long2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(long3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(long4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(long8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(long16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(long x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(long16 x); __attribute__((overloadable)) long convert_long_sat_rtz(long x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(long2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(long3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(long4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(long8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(long16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(long x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(long16 x); __attribute__((overloadable)) float convert_float_sat_rtz(long x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(long2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(long3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(long4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(long8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(long16 x); __attribute__((overloadable)) double convert_double_sat_rtz(long x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(long2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(long3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(long4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(long8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(long16 x); __attribute__((overloadable)) char convert_char_sat_rtz(ulong x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(ulong2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(ulong3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(ulong4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(ulong8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(ulong16 x); __attribute__((overloadable)) int convert_int_sat_rtz(ulong x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(ulong2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(ulong3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(ulong4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(ulong8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(ulong16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(ulong x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(ulong16 x); __attribute__((overloadable)) short convert_short_sat_rtz(ulong x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(ulong2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(ulong3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(ulong4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(ulong8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(ulong16 x); __attribute__((overloadable)) long convert_long_sat_rtz(ulong x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(ulong2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(ulong3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(ulong4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(ulong8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(ulong16 x); __attribute__((overloadable)) float convert_float_sat_rtz(ulong x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(ulong2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(ulong3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(ulong4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(ulong8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(ulong16 x); __attribute__((overloadable)) double convert_double_sat_rtz(ulong x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(ulong2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(ulong3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(ulong4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(ulong8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(ulong16 x); __attribute__((overloadable)) char convert_char_sat_rtz(float x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(float2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(float3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(float4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(float8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(float16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(float x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(float16 x); __attribute__((overloadable)) int convert_int_sat_rtz(float x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(float2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(float3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(float4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(float8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(float16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(float x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(float2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(float3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(float4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(float8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(float16 x); __attribute__((overloadable)) short convert_short_sat_rtz(float x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(float2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(float3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(float4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(float8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(float16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(float x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(float16 x); __attribute__((overloadable)) long convert_long_sat_rtz(float x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(float2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(float3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(float4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(float8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(float16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(float x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(float16 x); __attribute__((overloadable)) float convert_float_sat_rtz(float x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(float2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(float3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(float4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(float8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(float16 x); __attribute__((overloadable)) double convert_double_sat_rtz(float x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(float2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(float3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(float4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(float8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(float16 x); __attribute__((overloadable)) char convert_char_sat_rtz(double x); __attribute__((overloadable)) char2 convert_char2_sat_rtz(double2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtz(double3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtz(double4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtz(double8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtz(double16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtz(double x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtz(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtz(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtz(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtz(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtz(double16 x); __attribute__((overloadable)) int convert_int_sat_rtz(double x); __attribute__((overloadable)) int2 convert_int2_sat_rtz(double2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtz(double3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtz(double4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtz(double8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtz(double16 x); __attribute__((overloadable)) uint convert_uint_sat_rtz(double x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtz(double2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtz(double3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtz(double4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtz(double8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtz(double16 x); __attribute__((overloadable)) short convert_short_sat_rtz(double x); __attribute__((overloadable)) short2 convert_short2_sat_rtz(double2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtz(double3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtz(double4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtz(double8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtz(double16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtz(double x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtz(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtz(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtz(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtz(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtz(double16 x); __attribute__((overloadable)) long convert_long_sat_rtz(double x); __attribute__((overloadable)) long2 convert_long2_sat_rtz(double2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtz(double3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtz(double4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtz(double8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtz(double16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtz(double x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtz(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtz(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtz(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtz(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtz(double16 x); __attribute__((overloadable)) float convert_float_sat_rtz(double x); __attribute__((overloadable)) float2 convert_float2_sat_rtz(double2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtz(double3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtz(double4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtz(double8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtz(double16 x); __attribute__((overloadable)) double convert_double_sat_rtz(double x); __attribute__((overloadable)) double2 convert_double2_sat_rtz(double2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtz(double3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtz(double4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtz(double8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtz(double16 x); __attribute__((overloadable)) char convert_char_rtz(char x); __attribute__((overloadable)) char2 convert_char2_rtz(char2 x); __attribute__((overloadable)) char3 convert_char3_rtz(char3 x); __attribute__((overloadable)) char4 convert_char4_rtz(char4 x); __attribute__((overloadable)) char8 convert_char8_rtz(char8 x); __attribute__((overloadable)) char16 convert_char16_rtz(char16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(char x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(char16 x); __attribute__((overloadable)) int convert_int_rtz(char x); __attribute__((overloadable)) int2 convert_int2_rtz(char2 x); __attribute__((overloadable)) int3 convert_int3_rtz(char3 x); __attribute__((overloadable)) int4 convert_int4_rtz(char4 x); __attribute__((overloadable)) int8 convert_int8_rtz(char8 x); __attribute__((overloadable)) int16 convert_int16_rtz(char16 x); __attribute__((overloadable)) uint convert_uint_rtz(char x); __attribute__((overloadable)) uint2 convert_uint2_rtz(char2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(char3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(char4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(char8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(char16 x); __attribute__((overloadable)) short convert_short_rtz(char x); __attribute__((overloadable)) short2 convert_short2_rtz(char2 x); __attribute__((overloadable)) short3 convert_short3_rtz(char3 x); __attribute__((overloadable)) short4 convert_short4_rtz(char4 x); __attribute__((overloadable)) short8 convert_short8_rtz(char8 x); __attribute__((overloadable)) short16 convert_short16_rtz(char16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(char x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(char16 x); __attribute__((overloadable)) long convert_long_rtz(char x); __attribute__((overloadable)) long2 convert_long2_rtz(char2 x); __attribute__((overloadable)) long3 convert_long3_rtz(char3 x); __attribute__((overloadable)) long4 convert_long4_rtz(char4 x); __attribute__((overloadable)) long8 convert_long8_rtz(char8 x); __attribute__((overloadable)) long16 convert_long16_rtz(char16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(char x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(char16 x); __attribute__((overloadable)) float convert_float_rtz(char x); __attribute__((overloadable)) float2 convert_float2_rtz(char2 x); __attribute__((overloadable)) float3 convert_float3_rtz(char3 x); __attribute__((overloadable)) float4 convert_float4_rtz(char4 x); __attribute__((overloadable)) float8 convert_float8_rtz(char8 x); __attribute__((overloadable)) float16 convert_float16_rtz(char16 x); __attribute__((overloadable)) double convert_double_rtz(char x); __attribute__((overloadable)) double2 convert_double2_rtz(char2 x); __attribute__((overloadable)) double3 convert_double3_rtz(char3 x); __attribute__((overloadable)) double4 convert_double4_rtz(char4 x); __attribute__((overloadable)) double8 convert_double8_rtz(char8 x); __attribute__((overloadable)) double16 convert_double16_rtz(char16 x); __attribute__((overloadable)) char convert_char_rtz(uchar x); __attribute__((overloadable)) char2 convert_char2_rtz(uchar2 x); __attribute__((overloadable)) char3 convert_char3_rtz(uchar3 x); __attribute__((overloadable)) char4 convert_char4_rtz(uchar4 x); __attribute__((overloadable)) char8 convert_char8_rtz(uchar8 x); __attribute__((overloadable)) char16 convert_char16_rtz(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(uchar16 x); __attribute__((overloadable)) int convert_int_rtz(uchar x); __attribute__((overloadable)) int2 convert_int2_rtz(uchar2 x); __attribute__((overloadable)) int3 convert_int3_rtz(uchar3 x); __attribute__((overloadable)) int4 convert_int4_rtz(uchar4 x); __attribute__((overloadable)) int8 convert_int8_rtz(uchar8 x); __attribute__((overloadable)) int16 convert_int16_rtz(uchar16 x); __attribute__((overloadable)) uint convert_uint_rtz(uchar x); __attribute__((overloadable)) uint2 convert_uint2_rtz(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(uchar16 x); __attribute__((overloadable)) short convert_short_rtz(uchar x); __attribute__((overloadable)) short2 convert_short2_rtz(uchar2 x); __attribute__((overloadable)) short3 convert_short3_rtz(uchar3 x); __attribute__((overloadable)) short4 convert_short4_rtz(uchar4 x); __attribute__((overloadable)) short8 convert_short8_rtz(uchar8 x); __attribute__((overloadable)) short16 convert_short16_rtz(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(uchar16 x); __attribute__((overloadable)) long convert_long_rtz(uchar x); __attribute__((overloadable)) long2 convert_long2_rtz(uchar2 x); __attribute__((overloadable)) long3 convert_long3_rtz(uchar3 x); __attribute__((overloadable)) long4 convert_long4_rtz(uchar4 x); __attribute__((overloadable)) long8 convert_long8_rtz(uchar8 x); __attribute__((overloadable)) long16 convert_long16_rtz(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(uchar16 x); __attribute__((overloadable)) float convert_float_rtz(uchar x); __attribute__((overloadable)) float2 convert_float2_rtz(uchar2 x); __attribute__((overloadable)) float3 convert_float3_rtz(uchar3 x); __attribute__((overloadable)) float4 convert_float4_rtz(uchar4 x); __attribute__((overloadable)) float8 convert_float8_rtz(uchar8 x); __attribute__((overloadable)) float16 convert_float16_rtz(uchar16 x); __attribute__((overloadable)) double convert_double_rtz(uchar x); __attribute__((overloadable)) double2 convert_double2_rtz(uchar2 x); __attribute__((overloadable)) double3 convert_double3_rtz(uchar3 x); __attribute__((overloadable)) double4 convert_double4_rtz(uchar4 x); __attribute__((overloadable)) double8 convert_double8_rtz(uchar8 x); __attribute__((overloadable)) double16 convert_double16_rtz(uchar16 x); __attribute__((overloadable)) char convert_char_rtz(int x); __attribute__((overloadable)) char2 convert_char2_rtz(int2 x); __attribute__((overloadable)) char3 convert_char3_rtz(int3 x); __attribute__((overloadable)) char4 convert_char4_rtz(int4 x); __attribute__((overloadable)) char8 convert_char8_rtz(int8 x); __attribute__((overloadable)) char16 convert_char16_rtz(int16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(int x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(int16 x); __attribute__((overloadable)) int convert_int_rtz(int x); __attribute__((overloadable)) int2 convert_int2_rtz(int2 x); __attribute__((overloadable)) int3 convert_int3_rtz(int3 x); __attribute__((overloadable)) int4 convert_int4_rtz(int4 x); __attribute__((overloadable)) int8 convert_int8_rtz(int8 x); __attribute__((overloadable)) int16 convert_int16_rtz(int16 x); __attribute__((overloadable)) uint convert_uint_rtz(int x); __attribute__((overloadable)) uint2 convert_uint2_rtz(int2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(int3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(int4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(int8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(int16 x); __attribute__((overloadable)) short convert_short_rtz(int x); __attribute__((overloadable)) short2 convert_short2_rtz(int2 x); __attribute__((overloadable)) short3 convert_short3_rtz(int3 x); __attribute__((overloadable)) short4 convert_short4_rtz(int4 x); __attribute__((overloadable)) short8 convert_short8_rtz(int8 x); __attribute__((overloadable)) short16 convert_short16_rtz(int16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(int x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(int16 x); __attribute__((overloadable)) long convert_long_rtz(int x); __attribute__((overloadable)) long2 convert_long2_rtz(int2 x); __attribute__((overloadable)) long3 convert_long3_rtz(int3 x); __attribute__((overloadable)) long4 convert_long4_rtz(int4 x); __attribute__((overloadable)) long8 convert_long8_rtz(int8 x); __attribute__((overloadable)) long16 convert_long16_rtz(int16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(int x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(int16 x); __attribute__((overloadable)) float convert_float_rtz(int x); __attribute__((overloadable)) float2 convert_float2_rtz(int2 x); __attribute__((overloadable)) float3 convert_float3_rtz(int3 x); __attribute__((overloadable)) float4 convert_float4_rtz(int4 x); __attribute__((overloadable)) float8 convert_float8_rtz(int8 x); __attribute__((overloadable)) float16 convert_float16_rtz(int16 x); __attribute__((overloadable)) double convert_double_rtz(int x); __attribute__((overloadable)) double2 convert_double2_rtz(int2 x); __attribute__((overloadable)) double3 convert_double3_rtz(int3 x); __attribute__((overloadable)) double4 convert_double4_rtz(int4 x); __attribute__((overloadable)) double8 convert_double8_rtz(int8 x); __attribute__((overloadable)) double16 convert_double16_rtz(int16 x); __attribute__((overloadable)) char convert_char_rtz(uint x); __attribute__((overloadable)) char2 convert_char2_rtz(uint2 x); __attribute__((overloadable)) char3 convert_char3_rtz(uint3 x); __attribute__((overloadable)) char4 convert_char4_rtz(uint4 x); __attribute__((overloadable)) char8 convert_char8_rtz(uint8 x); __attribute__((overloadable)) char16 convert_char16_rtz(uint16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(uint16 x); __attribute__((overloadable)) int convert_int_rtz(uint x); __attribute__((overloadable)) int2 convert_int2_rtz(uint2 x); __attribute__((overloadable)) int3 convert_int3_rtz(uint3 x); __attribute__((overloadable)) int4 convert_int4_rtz(uint4 x); __attribute__((overloadable)) int8 convert_int8_rtz(uint8 x); __attribute__((overloadable)) int16 convert_int16_rtz(uint16 x); __attribute__((overloadable)) uint convert_uint_rtz(uint x); __attribute__((overloadable)) uint2 convert_uint2_rtz(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(uint16 x); __attribute__((overloadable)) short convert_short_rtz(uint x); __attribute__((overloadable)) short2 convert_short2_rtz(uint2 x); __attribute__((overloadable)) short3 convert_short3_rtz(uint3 x); __attribute__((overloadable)) short4 convert_short4_rtz(uint4 x); __attribute__((overloadable)) short8 convert_short8_rtz(uint8 x); __attribute__((overloadable)) short16 convert_short16_rtz(uint16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(uint16 x); __attribute__((overloadable)) long convert_long_rtz(uint x); __attribute__((overloadable)) long2 convert_long2_rtz(uint2 x); __attribute__((overloadable)) long3 convert_long3_rtz(uint3 x); __attribute__((overloadable)) long4 convert_long4_rtz(uint4 x); __attribute__((overloadable)) long8 convert_long8_rtz(uint8 x); __attribute__((overloadable)) long16 convert_long16_rtz(uint16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(uint16 x); __attribute__((overloadable)) float convert_float_rtz(uint x); __attribute__((overloadable)) float2 convert_float2_rtz(uint2 x); __attribute__((overloadable)) float3 convert_float3_rtz(uint3 x); __attribute__((overloadable)) float4 convert_float4_rtz(uint4 x); __attribute__((overloadable)) float8 convert_float8_rtz(uint8 x); __attribute__((overloadable)) float16 convert_float16_rtz(uint16 x); __attribute__((overloadable)) double convert_double_rtz(uint x); __attribute__((overloadable)) double2 convert_double2_rtz(uint2 x); __attribute__((overloadable)) double3 convert_double3_rtz(uint3 x); __attribute__((overloadable)) double4 convert_double4_rtz(uint4 x); __attribute__((overloadable)) double8 convert_double8_rtz(uint8 x); __attribute__((overloadable)) double16 convert_double16_rtz(uint16 x); __attribute__((overloadable)) char convert_char_rtz(short x); __attribute__((overloadable)) char2 convert_char2_rtz(short2 x); __attribute__((overloadable)) char3 convert_char3_rtz(short3 x); __attribute__((overloadable)) char4 convert_char4_rtz(short4 x); __attribute__((overloadable)) char8 convert_char8_rtz(short8 x); __attribute__((overloadable)) char16 convert_char16_rtz(short16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(short x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(short16 x); __attribute__((overloadable)) int convert_int_rtz(short x); __attribute__((overloadable)) int2 convert_int2_rtz(short2 x); __attribute__((overloadable)) int3 convert_int3_rtz(short3 x); __attribute__((overloadable)) int4 convert_int4_rtz(short4 x); __attribute__((overloadable)) int8 convert_int8_rtz(short8 x); __attribute__((overloadable)) int16 convert_int16_rtz(short16 x); __attribute__((overloadable)) uint convert_uint_rtz(short x); __attribute__((overloadable)) uint2 convert_uint2_rtz(short2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(short3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(short4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(short8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(short16 x); __attribute__((overloadable)) short convert_short_rtz(short x); __attribute__((overloadable)) short2 convert_short2_rtz(short2 x); __attribute__((overloadable)) short3 convert_short3_rtz(short3 x); __attribute__((overloadable)) short4 convert_short4_rtz(short4 x); __attribute__((overloadable)) short8 convert_short8_rtz(short8 x); __attribute__((overloadable)) short16 convert_short16_rtz(short16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(short x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(short16 x); __attribute__((overloadable)) long convert_long_rtz(short x); __attribute__((overloadable)) long2 convert_long2_rtz(short2 x); __attribute__((overloadable)) long3 convert_long3_rtz(short3 x); __attribute__((overloadable)) long4 convert_long4_rtz(short4 x); __attribute__((overloadable)) long8 convert_long8_rtz(short8 x); __attribute__((overloadable)) long16 convert_long16_rtz(short16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(short x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(short16 x); __attribute__((overloadable)) float convert_float_rtz(short x); __attribute__((overloadable)) float2 convert_float2_rtz(short2 x); __attribute__((overloadable)) float3 convert_float3_rtz(short3 x); __attribute__((overloadable)) float4 convert_float4_rtz(short4 x); __attribute__((overloadable)) float8 convert_float8_rtz(short8 x); __attribute__((overloadable)) float16 convert_float16_rtz(short16 x); __attribute__((overloadable)) double convert_double_rtz(short x); __attribute__((overloadable)) double2 convert_double2_rtz(short2 x); __attribute__((overloadable)) double3 convert_double3_rtz(short3 x); __attribute__((overloadable)) double4 convert_double4_rtz(short4 x); __attribute__((overloadable)) double8 convert_double8_rtz(short8 x); __attribute__((overloadable)) double16 convert_double16_rtz(short16 x); __attribute__((overloadable)) char convert_char_rtz(ushort x); __attribute__((overloadable)) char2 convert_char2_rtz(ushort2 x); __attribute__((overloadable)) char3 convert_char3_rtz(ushort3 x); __attribute__((overloadable)) char4 convert_char4_rtz(ushort4 x); __attribute__((overloadable)) char8 convert_char8_rtz(ushort8 x); __attribute__((overloadable)) char16 convert_char16_rtz(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(ushort16 x); __attribute__((overloadable)) int convert_int_rtz(ushort x); __attribute__((overloadable)) int2 convert_int2_rtz(ushort2 x); __attribute__((overloadable)) int3 convert_int3_rtz(ushort3 x); __attribute__((overloadable)) int4 convert_int4_rtz(ushort4 x); __attribute__((overloadable)) int8 convert_int8_rtz(ushort8 x); __attribute__((overloadable)) int16 convert_int16_rtz(ushort16 x); __attribute__((overloadable)) uint convert_uint_rtz(ushort x); __attribute__((overloadable)) uint2 convert_uint2_rtz(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(ushort16 x); __attribute__((overloadable)) short convert_short_rtz(ushort x); __attribute__((overloadable)) short2 convert_short2_rtz(ushort2 x); __attribute__((overloadable)) short3 convert_short3_rtz(ushort3 x); __attribute__((overloadable)) short4 convert_short4_rtz(ushort4 x); __attribute__((overloadable)) short8 convert_short8_rtz(ushort8 x); __attribute__((overloadable)) short16 convert_short16_rtz(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(ushort16 x); __attribute__((overloadable)) long convert_long_rtz(ushort x); __attribute__((overloadable)) long2 convert_long2_rtz(ushort2 x); __attribute__((overloadable)) long3 convert_long3_rtz(ushort3 x); __attribute__((overloadable)) long4 convert_long4_rtz(ushort4 x); __attribute__((overloadable)) long8 convert_long8_rtz(ushort8 x); __attribute__((overloadable)) long16 convert_long16_rtz(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(ushort16 x); __attribute__((overloadable)) float convert_float_rtz(ushort x); __attribute__((overloadable)) float2 convert_float2_rtz(ushort2 x); __attribute__((overloadable)) float3 convert_float3_rtz(ushort3 x); __attribute__((overloadable)) float4 convert_float4_rtz(ushort4 x); __attribute__((overloadable)) float8 convert_float8_rtz(ushort8 x); __attribute__((overloadable)) float16 convert_float16_rtz(ushort16 x); __attribute__((overloadable)) double convert_double_rtz(ushort x); __attribute__((overloadable)) double2 convert_double2_rtz(ushort2 x); __attribute__((overloadable)) double3 convert_double3_rtz(ushort3 x); __attribute__((overloadable)) double4 convert_double4_rtz(ushort4 x); __attribute__((overloadable)) double8 convert_double8_rtz(ushort8 x); __attribute__((overloadable)) double16 convert_double16_rtz(ushort16 x); __attribute__((overloadable)) char convert_char_rtz(long x); __attribute__((overloadable)) char2 convert_char2_rtz(long2 x); __attribute__((overloadable)) char3 convert_char3_rtz(long3 x); __attribute__((overloadable)) char4 convert_char4_rtz(long4 x); __attribute__((overloadable)) char8 convert_char8_rtz(long8 x); __attribute__((overloadable)) char16 convert_char16_rtz(long16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(long x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(long16 x); __attribute__((overloadable)) int convert_int_rtz(long x); __attribute__((overloadable)) int2 convert_int2_rtz(long2 x); __attribute__((overloadable)) int3 convert_int3_rtz(long3 x); __attribute__((overloadable)) int4 convert_int4_rtz(long4 x); __attribute__((overloadable)) int8 convert_int8_rtz(long8 x); __attribute__((overloadable)) int16 convert_int16_rtz(long16 x); __attribute__((overloadable)) uint convert_uint_rtz(long x); __attribute__((overloadable)) uint2 convert_uint2_rtz(long2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(long3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(long4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(long8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(long16 x); __attribute__((overloadable)) short convert_short_rtz(long x); __attribute__((overloadable)) short2 convert_short2_rtz(long2 x); __attribute__((overloadable)) short3 convert_short3_rtz(long3 x); __attribute__((overloadable)) short4 convert_short4_rtz(long4 x); __attribute__((overloadable)) short8 convert_short8_rtz(long8 x); __attribute__((overloadable)) short16 convert_short16_rtz(long16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(long x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(long16 x); __attribute__((overloadable)) long convert_long_rtz(long x); __attribute__((overloadable)) long2 convert_long2_rtz(long2 x); __attribute__((overloadable)) long3 convert_long3_rtz(long3 x); __attribute__((overloadable)) long4 convert_long4_rtz(long4 x); __attribute__((overloadable)) long8 convert_long8_rtz(long8 x); __attribute__((overloadable)) long16 convert_long16_rtz(long16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(long x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(long16 x); __attribute__((overloadable)) float convert_float_rtz(long x); __attribute__((overloadable)) float2 convert_float2_rtz(long2 x); __attribute__((overloadable)) float3 convert_float3_rtz(long3 x); __attribute__((overloadable)) float4 convert_float4_rtz(long4 x); __attribute__((overloadable)) float8 convert_float8_rtz(long8 x); __attribute__((overloadable)) float16 convert_float16_rtz(long16 x); __attribute__((overloadable)) double convert_double_rtz(long x); __attribute__((overloadable)) double2 convert_double2_rtz(long2 x); __attribute__((overloadable)) double3 convert_double3_rtz(long3 x); __attribute__((overloadable)) double4 convert_double4_rtz(long4 x); __attribute__((overloadable)) double8 convert_double8_rtz(long8 x); __attribute__((overloadable)) double16 convert_double16_rtz(long16 x); __attribute__((overloadable)) char convert_char_rtz(ulong x); __attribute__((overloadable)) char2 convert_char2_rtz(ulong2 x); __attribute__((overloadable)) char3 convert_char3_rtz(ulong3 x); __attribute__((overloadable)) char4 convert_char4_rtz(ulong4 x); __attribute__((overloadable)) char8 convert_char8_rtz(ulong8 x); __attribute__((overloadable)) char16 convert_char16_rtz(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(ulong16 x); __attribute__((overloadable)) int convert_int_rtz(ulong x); __attribute__((overloadable)) int2 convert_int2_rtz(ulong2 x); __attribute__((overloadable)) int3 convert_int3_rtz(ulong3 x); __attribute__((overloadable)) int4 convert_int4_rtz(ulong4 x); __attribute__((overloadable)) int8 convert_int8_rtz(ulong8 x); __attribute__((overloadable)) int16 convert_int16_rtz(ulong16 x); __attribute__((overloadable)) uint convert_uint_rtz(ulong x); __attribute__((overloadable)) uint2 convert_uint2_rtz(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(ulong16 x); __attribute__((overloadable)) short convert_short_rtz(ulong x); __attribute__((overloadable)) short2 convert_short2_rtz(ulong2 x); __attribute__((overloadable)) short3 convert_short3_rtz(ulong3 x); __attribute__((overloadable)) short4 convert_short4_rtz(ulong4 x); __attribute__((overloadable)) short8 convert_short8_rtz(ulong8 x); __attribute__((overloadable)) short16 convert_short16_rtz(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(ulong16 x); __attribute__((overloadable)) long convert_long_rtz(ulong x); __attribute__((overloadable)) long2 convert_long2_rtz(ulong2 x); __attribute__((overloadable)) long3 convert_long3_rtz(ulong3 x); __attribute__((overloadable)) long4 convert_long4_rtz(ulong4 x); __attribute__((overloadable)) long8 convert_long8_rtz(ulong8 x); __attribute__((overloadable)) long16 convert_long16_rtz(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(ulong16 x); __attribute__((overloadable)) float convert_float_rtz(ulong x); __attribute__((overloadable)) float2 convert_float2_rtz(ulong2 x); __attribute__((overloadable)) float3 convert_float3_rtz(ulong3 x); __attribute__((overloadable)) float4 convert_float4_rtz(ulong4 x); __attribute__((overloadable)) float8 convert_float8_rtz(ulong8 x); __attribute__((overloadable)) float16 convert_float16_rtz(ulong16 x); __attribute__((overloadable)) double convert_double_rtz(ulong x); __attribute__((overloadable)) double2 convert_double2_rtz(ulong2 x); __attribute__((overloadable)) double3 convert_double3_rtz(ulong3 x); __attribute__((overloadable)) double4 convert_double4_rtz(ulong4 x); __attribute__((overloadable)) double8 convert_double8_rtz(ulong8 x); __attribute__((overloadable)) double16 convert_double16_rtz(ulong16 x); __attribute__((overloadable)) char convert_char_rtz(float x); __attribute__((overloadable)) char2 convert_char2_rtz(float2 x); __attribute__((overloadable)) char3 convert_char3_rtz(float3 x); __attribute__((overloadable)) char4 convert_char4_rtz(float4 x); __attribute__((overloadable)) char8 convert_char8_rtz(float8 x); __attribute__((overloadable)) char16 convert_char16_rtz(float16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(float x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(float16 x); __attribute__((overloadable)) int convert_int_rtz(float x); __attribute__((overloadable)) int2 convert_int2_rtz(float2 x); __attribute__((overloadable)) int3 convert_int3_rtz(float3 x); __attribute__((overloadable)) int4 convert_int4_rtz(float4 x); __attribute__((overloadable)) int8 convert_int8_rtz(float8 x); __attribute__((overloadable)) int16 convert_int16_rtz(float16 x); __attribute__((overloadable)) uint convert_uint_rtz(float x); __attribute__((overloadable)) uint2 convert_uint2_rtz(float2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(float3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(float4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(float8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(float16 x); __attribute__((overloadable)) short convert_short_rtz(float x); __attribute__((overloadable)) short2 convert_short2_rtz(float2 x); __attribute__((overloadable)) short3 convert_short3_rtz(float3 x); __attribute__((overloadable)) short4 convert_short4_rtz(float4 x); __attribute__((overloadable)) short8 convert_short8_rtz(float8 x); __attribute__((overloadable)) short16 convert_short16_rtz(float16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(float x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(float16 x); __attribute__((overloadable)) long convert_long_rtz(float x); __attribute__((overloadable)) long2 convert_long2_rtz(float2 x); __attribute__((overloadable)) long3 convert_long3_rtz(float3 x); __attribute__((overloadable)) long4 convert_long4_rtz(float4 x); __attribute__((overloadable)) long8 convert_long8_rtz(float8 x); __attribute__((overloadable)) long16 convert_long16_rtz(float16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(float x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(float16 x); __attribute__((overloadable)) float convert_float_rtz(float x); __attribute__((overloadable)) float2 convert_float2_rtz(float2 x); __attribute__((overloadable)) float3 convert_float3_rtz(float3 x); __attribute__((overloadable)) float4 convert_float4_rtz(float4 x); __attribute__((overloadable)) float8 convert_float8_rtz(float8 x); __attribute__((overloadable)) float16 convert_float16_rtz(float16 x); __attribute__((overloadable)) double convert_double_rtz(float x); __attribute__((overloadable)) double2 convert_double2_rtz(float2 x); __attribute__((overloadable)) double3 convert_double3_rtz(float3 x); __attribute__((overloadable)) double4 convert_double4_rtz(float4 x); __attribute__((overloadable)) double8 convert_double8_rtz(float8 x); __attribute__((overloadable)) double16 convert_double16_rtz(float16 x); __attribute__((overloadable)) char convert_char_rtz(double x); __attribute__((overloadable)) char2 convert_char2_rtz(double2 x); __attribute__((overloadable)) char3 convert_char3_rtz(double3 x); __attribute__((overloadable)) char4 convert_char4_rtz(double4 x); __attribute__((overloadable)) char8 convert_char8_rtz(double8 x); __attribute__((overloadable)) char16 convert_char16_rtz(double16 x); __attribute__((overloadable)) uchar convert_uchar_rtz(double x); __attribute__((overloadable)) uchar2 convert_uchar2_rtz(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtz(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtz(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtz(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtz(double16 x); __attribute__((overloadable)) int convert_int_rtz(double x); __attribute__((overloadable)) int2 convert_int2_rtz(double2 x); __attribute__((overloadable)) int3 convert_int3_rtz(double3 x); __attribute__((overloadable)) int4 convert_int4_rtz(double4 x); __attribute__((overloadable)) int8 convert_int8_rtz(double8 x); __attribute__((overloadable)) int16 convert_int16_rtz(double16 x); __attribute__((overloadable)) uint convert_uint_rtz(double x); __attribute__((overloadable)) uint2 convert_uint2_rtz(double2 x); __attribute__((overloadable)) uint3 convert_uint3_rtz(double3 x); __attribute__((overloadable)) uint4 convert_uint4_rtz(double4 x); __attribute__((overloadable)) uint8 convert_uint8_rtz(double8 x); __attribute__((overloadable)) uint16 convert_uint16_rtz(double16 x); __attribute__((overloadable)) short convert_short_rtz(double x); __attribute__((overloadable)) short2 convert_short2_rtz(double2 x); __attribute__((overloadable)) short3 convert_short3_rtz(double3 x); __attribute__((overloadable)) short4 convert_short4_rtz(double4 x); __attribute__((overloadable)) short8 convert_short8_rtz(double8 x); __attribute__((overloadable)) short16 convert_short16_rtz(double16 x); __attribute__((overloadable)) ushort convert_ushort_rtz(double x); __attribute__((overloadable)) ushort2 convert_ushort2_rtz(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtz(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtz(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtz(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtz(double16 x); __attribute__((overloadable)) long convert_long_rtz(double x); __attribute__((overloadable)) long2 convert_long2_rtz(double2 x); __attribute__((overloadable)) long3 convert_long3_rtz(double3 x); __attribute__((overloadable)) long4 convert_long4_rtz(double4 x); __attribute__((overloadable)) long8 convert_long8_rtz(double8 x); __attribute__((overloadable)) long16 convert_long16_rtz(double16 x); __attribute__((overloadable)) ulong convert_ulong_rtz(double x); __attribute__((overloadable)) ulong2 convert_ulong2_rtz(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtz(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtz(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtz(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtz(double16 x); __attribute__((overloadable)) float convert_float_rtz(double x); __attribute__((overloadable)) float2 convert_float2_rtz(double2 x); __attribute__((overloadable)) float3 convert_float3_rtz(double3 x); __attribute__((overloadable)) float4 convert_float4_rtz(double4 x); __attribute__((overloadable)) float8 convert_float8_rtz(double8 x); __attribute__((overloadable)) float16 convert_float16_rtz(double16 x); __attribute__((overloadable)) double convert_double_rtz(double x); __attribute__((overloadable)) double2 convert_double2_rtz(double2 x); __attribute__((overloadable)) double3 convert_double3_rtz(double3 x); __attribute__((overloadable)) double4 convert_double4_rtz(double4 x); __attribute__((overloadable)) double8 convert_double8_rtz(double8 x); __attribute__((overloadable)) double16 convert_double16_rtz(double16 x);
__attribute__((overloadable)) char convert_char_sat_rtp(char x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(char2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(char3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(char4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(char8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(char16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(char x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(char16 x); __attribute__((overloadable)) int convert_int_sat_rtp(char x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(char2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(char3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(char4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(char8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(char16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(char x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(char2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(char3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(char4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(char8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(char16 x); __attribute__((overloadable)) short convert_short_sat_rtp(char x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(char2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(char3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(char4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(char8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(char16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(char x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(char16 x); __attribute__((overloadable)) long convert_long_sat_rtp(char x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(char2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(char3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(char4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(char8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(char16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(char x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(char16 x); __attribute__((overloadable)) float convert_float_sat_rtp(char x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(char2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(char3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(char4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(char8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(char16 x); __attribute__((overloadable)) double convert_double_sat_rtp(char x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(char2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(char3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(char4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(char8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(char16 x); __attribute__((overloadable)) char convert_char_sat_rtp(uchar x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(uchar2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(uchar3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(uchar4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(uchar8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(uchar16 x); __attribute__((overloadable)) int convert_int_sat_rtp(uchar x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(uchar2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(uchar3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(uchar4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(uchar8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(uchar16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(uchar x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(uchar16 x); __attribute__((overloadable)) short convert_short_sat_rtp(uchar x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(uchar2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(uchar3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(uchar4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(uchar8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(uchar16 x); __attribute__((overloadable)) long convert_long_sat_rtp(uchar x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(uchar2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(uchar3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(uchar4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(uchar8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(uchar16 x); __attribute__((overloadable)) float convert_float_sat_rtp(uchar x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(uchar2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(uchar3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(uchar4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(uchar8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(uchar16 x); __attribute__((overloadable)) double convert_double_sat_rtp(uchar x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(uchar2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(uchar3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(uchar4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(uchar8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(uchar16 x); __attribute__((overloadable)) char convert_char_sat_rtp(int x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(int2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(int3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(int4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(int8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(int16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(int x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(int16 x); __attribute__((overloadable)) int convert_int_sat_rtp(int x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(int2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(int3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(int4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(int8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(int16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(int x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(int2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(int3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(int4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(int8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(int16 x); __attribute__((overloadable)) short convert_short_sat_rtp(int x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(int2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(int3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(int4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(int8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(int16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(int x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(int16 x); __attribute__((overloadable)) long convert_long_sat_rtp(int x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(int2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(int3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(int4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(int8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(int16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(int x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(int16 x); __attribute__((overloadable)) float convert_float_sat_rtp(int x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(int2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(int3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(int4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(int8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(int16 x); __attribute__((overloadable)) double convert_double_sat_rtp(int x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(int2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(int3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(int4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(int8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(int16 x); __attribute__((overloadable)) char convert_char_sat_rtp(uint x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(uint2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(uint3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(uint4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(uint8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(uint16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(uint16 x); __attribute__((overloadable)) int convert_int_sat_rtp(uint x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(uint2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(uint3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(uint4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(uint8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(uint16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(uint x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(uint16 x); __attribute__((overloadable)) short convert_short_sat_rtp(uint x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(uint2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(uint3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(uint4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(uint8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(uint16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(uint16 x); __attribute__((overloadable)) long convert_long_sat_rtp(uint x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(uint2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(uint3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(uint4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(uint8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(uint16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(uint16 x); __attribute__((overloadable)) float convert_float_sat_rtp(uint x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(uint2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(uint3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(uint4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(uint8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(uint16 x); __attribute__((overloadable)) double convert_double_sat_rtp(uint x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(uint2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(uint3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(uint4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(uint8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(uint16 x); __attribute__((overloadable)) char convert_char_sat_rtp(short x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(short2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(short3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(short4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(short8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(short16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(short x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(short16 x); __attribute__((overloadable)) int convert_int_sat_rtp(short x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(short2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(short3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(short4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(short8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(short16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(short x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(short2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(short3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(short4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(short8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(short16 x); __attribute__((overloadable)) short convert_short_sat_rtp(short x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(short2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(short3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(short4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(short8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(short16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(short x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(short16 x); __attribute__((overloadable)) long convert_long_sat_rtp(short x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(short2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(short3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(short4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(short8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(short16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(short x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(short16 x); __attribute__((overloadable)) float convert_float_sat_rtp(short x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(short2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(short3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(short4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(short8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(short16 x); __attribute__((overloadable)) double convert_double_sat_rtp(short x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(short2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(short3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(short4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(short8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(short16 x); __attribute__((overloadable)) char convert_char_sat_rtp(ushort x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(ushort2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(ushort3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(ushort4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(ushort8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(ushort16 x); __attribute__((overloadable)) int convert_int_sat_rtp(ushort x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(ushort2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(ushort3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(ushort4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(ushort8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(ushort16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(ushort x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(ushort16 x); __attribute__((overloadable)) short convert_short_sat_rtp(ushort x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(ushort2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(ushort3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(ushort4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(ushort8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(ushort16 x); __attribute__((overloadable)) long convert_long_sat_rtp(ushort x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(ushort2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(ushort3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(ushort4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(ushort8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(ushort16 x); __attribute__((overloadable)) float convert_float_sat_rtp(ushort x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(ushort2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(ushort3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(ushort4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(ushort8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(ushort16 x); __attribute__((overloadable)) double convert_double_sat_rtp(ushort x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(ushort2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(ushort3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(ushort4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(ushort8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(ushort16 x); __attribute__((overloadable)) char convert_char_sat_rtp(long x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(long2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(long3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(long4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(long8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(long16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(long x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(long16 x); __attribute__((overloadable)) int convert_int_sat_rtp(long x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(long2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(long3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(long4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(long8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(long16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(long x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(long2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(long3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(long4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(long8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(long16 x); __attribute__((overloadable)) short convert_short_sat_rtp(long x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(long2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(long3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(long4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(long8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(long16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(long x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(long16 x); __attribute__((overloadable)) long convert_long_sat_rtp(long x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(long2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(long3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(long4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(long8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(long16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(long x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(long16 x); __attribute__((overloadable)) float convert_float_sat_rtp(long x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(long2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(long3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(long4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(long8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(long16 x); __attribute__((overloadable)) double convert_double_sat_rtp(long x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(long2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(long3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(long4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(long8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(long16 x); __attribute__((overloadable)) char convert_char_sat_rtp(ulong x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(ulong2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(ulong3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(ulong4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(ulong8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(ulong16 x); __attribute__((overloadable)) int convert_int_sat_rtp(ulong x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(ulong2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(ulong3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(ulong4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(ulong8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(ulong16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(ulong x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(ulong16 x); __attribute__((overloadable)) short convert_short_sat_rtp(ulong x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(ulong2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(ulong3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(ulong4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(ulong8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(ulong16 x); __attribute__((overloadable)) long convert_long_sat_rtp(ulong x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(ulong2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(ulong3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(ulong4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(ulong8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(ulong16 x); __attribute__((overloadable)) float convert_float_sat_rtp(ulong x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(ulong2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(ulong3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(ulong4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(ulong8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(ulong16 x); __attribute__((overloadable)) double convert_double_sat_rtp(ulong x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(ulong2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(ulong3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(ulong4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(ulong8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(ulong16 x); __attribute__((overloadable)) char convert_char_sat_rtp(float x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(float2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(float3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(float4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(float8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(float16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(float x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(float16 x); __attribute__((overloadable)) int convert_int_sat_rtp(float x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(float2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(float3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(float4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(float8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(float16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(float x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(float2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(float3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(float4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(float8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(float16 x); __attribute__((overloadable)) short convert_short_sat_rtp(float x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(float2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(float3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(float4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(float8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(float16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(float x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(float16 x); __attribute__((overloadable)) long convert_long_sat_rtp(float x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(float2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(float3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(float4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(float8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(float16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(float x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(float16 x); __attribute__((overloadable)) float convert_float_sat_rtp(float x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(float2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(float3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(float4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(float8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(float16 x); __attribute__((overloadable)) double convert_double_sat_rtp(float x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(float2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(float3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(float4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(float8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(float16 x); __attribute__((overloadable)) char convert_char_sat_rtp(double x); __attribute__((overloadable)) char2 convert_char2_sat_rtp(double2 x); __attribute__((overloadable)) char3 convert_char3_sat_rtp(double3 x); __attribute__((overloadable)) char4 convert_char4_sat_rtp(double4 x); __attribute__((overloadable)) char8 convert_char8_sat_rtp(double8 x); __attribute__((overloadable)) char16 convert_char16_sat_rtp(double16 x); __attribute__((overloadable)) uchar convert_uchar_sat_rtp(double x); __attribute__((overloadable)) uchar2 convert_uchar2_sat_rtp(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat_rtp(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat_rtp(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat_rtp(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat_rtp(double16 x); __attribute__((overloadable)) int convert_int_sat_rtp(double x); __attribute__((overloadable)) int2 convert_int2_sat_rtp(double2 x); __attribute__((overloadable)) int3 convert_int3_sat_rtp(double3 x); __attribute__((overloadable)) int4 convert_int4_sat_rtp(double4 x); __attribute__((overloadable)) int8 convert_int8_sat_rtp(double8 x); __attribute__((overloadable)) int16 convert_int16_sat_rtp(double16 x); __attribute__((overloadable)) uint convert_uint_sat_rtp(double x); __attribute__((overloadable)) uint2 convert_uint2_sat_rtp(double2 x); __attribute__((overloadable)) uint3 convert_uint3_sat_rtp(double3 x); __attribute__((overloadable)) uint4 convert_uint4_sat_rtp(double4 x); __attribute__((overloadable)) uint8 convert_uint8_sat_rtp(double8 x); __attribute__((overloadable)) uint16 convert_uint16_sat_rtp(double16 x); __attribute__((overloadable)) short convert_short_sat_rtp(double x); __attribute__((overloadable)) short2 convert_short2_sat_rtp(double2 x); __attribute__((overloadable)) short3 convert_short3_sat_rtp(double3 x); __attribute__((overloadable)) short4 convert_short4_sat_rtp(double4 x); __attribute__((overloadable)) short8 convert_short8_sat_rtp(double8 x); __attribute__((overloadable)) short16 convert_short16_sat_rtp(double16 x); __attribute__((overloadable)) ushort convert_ushort_sat_rtp(double x); __attribute__((overloadable)) ushort2 convert_ushort2_sat_rtp(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat_rtp(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat_rtp(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat_rtp(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat_rtp(double16 x); __attribute__((overloadable)) long convert_long_sat_rtp(double x); __attribute__((overloadable)) long2 convert_long2_sat_rtp(double2 x); __attribute__((overloadable)) long3 convert_long3_sat_rtp(double3 x); __attribute__((overloadable)) long4 convert_long4_sat_rtp(double4 x); __attribute__((overloadable)) long8 convert_long8_sat_rtp(double8 x); __attribute__((overloadable)) long16 convert_long16_sat_rtp(double16 x); __attribute__((overloadable)) ulong convert_ulong_sat_rtp(double x); __attribute__((overloadable)) ulong2 convert_ulong2_sat_rtp(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat_rtp(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat_rtp(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat_rtp(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat_rtp(double16 x); __attribute__((overloadable)) float convert_float_sat_rtp(double x); __attribute__((overloadable)) float2 convert_float2_sat_rtp(double2 x); __attribute__((overloadable)) float3 convert_float3_sat_rtp(double3 x); __attribute__((overloadable)) float4 convert_float4_sat_rtp(double4 x); __attribute__((overloadable)) float8 convert_float8_sat_rtp(double8 x); __attribute__((overloadable)) float16 convert_float16_sat_rtp(double16 x); __attribute__((overloadable)) double convert_double_sat_rtp(double x); __attribute__((overloadable)) double2 convert_double2_sat_rtp(double2 x); __attribute__((overloadable)) double3 convert_double3_sat_rtp(double3 x); __attribute__((overloadable)) double4 convert_double4_sat_rtp(double4 x); __attribute__((overloadable)) double8 convert_double8_sat_rtp(double8 x); __attribute__((overloadable)) double16 convert_double16_sat_rtp(double16 x); __attribute__((overloadable)) char convert_char_rtp(char x); __attribute__((overloadable)) char2 convert_char2_rtp(char2 x); __attribute__((overloadable)) char3 convert_char3_rtp(char3 x); __attribute__((overloadable)) char4 convert_char4_rtp(char4 x); __attribute__((overloadable)) char8 convert_char8_rtp(char8 x); __attribute__((overloadable)) char16 convert_char16_rtp(char16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(char x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(char16 x); __attribute__((overloadable)) int convert_int_rtp(char x); __attribute__((overloadable)) int2 convert_int2_rtp(char2 x); __attribute__((overloadable)) int3 convert_int3_rtp(char3 x); __attribute__((overloadable)) int4 convert_int4_rtp(char4 x); __attribute__((overloadable)) int8 convert_int8_rtp(char8 x); __attribute__((overloadable)) int16 convert_int16_rtp(char16 x); __attribute__((overloadable)) uint convert_uint_rtp(char x); __attribute__((overloadable)) uint2 convert_uint2_rtp(char2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(char3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(char4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(char8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(char16 x); __attribute__((overloadable)) short convert_short_rtp(char x); __attribute__((overloadable)) short2 convert_short2_rtp(char2 x); __attribute__((overloadable)) short3 convert_short3_rtp(char3 x); __attribute__((overloadable)) short4 convert_short4_rtp(char4 x); __attribute__((overloadable)) short8 convert_short8_rtp(char8 x); __attribute__((overloadable)) short16 convert_short16_rtp(char16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(char x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(char16 x); __attribute__((overloadable)) long convert_long_rtp(char x); __attribute__((overloadable)) long2 convert_long2_rtp(char2 x); __attribute__((overloadable)) long3 convert_long3_rtp(char3 x); __attribute__((overloadable)) long4 convert_long4_rtp(char4 x); __attribute__((overloadable)) long8 convert_long8_rtp(char8 x); __attribute__((overloadable)) long16 convert_long16_rtp(char16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(char x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(char16 x); __attribute__((overloadable)) float convert_float_rtp(char x); __attribute__((overloadable)) float2 convert_float2_rtp(char2 x); __attribute__((overloadable)) float3 convert_float3_rtp(char3 x); __attribute__((overloadable)) float4 convert_float4_rtp(char4 x); __attribute__((overloadable)) float8 convert_float8_rtp(char8 x); __attribute__((overloadable)) float16 convert_float16_rtp(char16 x); __attribute__((overloadable)) double convert_double_rtp(char x); __attribute__((overloadable)) double2 convert_double2_rtp(char2 x); __attribute__((overloadable)) double3 convert_double3_rtp(char3 x); __attribute__((overloadable)) double4 convert_double4_rtp(char4 x); __attribute__((overloadable)) double8 convert_double8_rtp(char8 x); __attribute__((overloadable)) double16 convert_double16_rtp(char16 x); __attribute__((overloadable)) char convert_char_rtp(uchar x); __attribute__((overloadable)) char2 convert_char2_rtp(uchar2 x); __attribute__((overloadable)) char3 convert_char3_rtp(uchar3 x); __attribute__((overloadable)) char4 convert_char4_rtp(uchar4 x); __attribute__((overloadable)) char8 convert_char8_rtp(uchar8 x); __attribute__((overloadable)) char16 convert_char16_rtp(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(uchar16 x); __attribute__((overloadable)) int convert_int_rtp(uchar x); __attribute__((overloadable)) int2 convert_int2_rtp(uchar2 x); __attribute__((overloadable)) int3 convert_int3_rtp(uchar3 x); __attribute__((overloadable)) int4 convert_int4_rtp(uchar4 x); __attribute__((overloadable)) int8 convert_int8_rtp(uchar8 x); __attribute__((overloadable)) int16 convert_int16_rtp(uchar16 x); __attribute__((overloadable)) uint convert_uint_rtp(uchar x); __attribute__((overloadable)) uint2 convert_uint2_rtp(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(uchar16 x); __attribute__((overloadable)) short convert_short_rtp(uchar x); __attribute__((overloadable)) short2 convert_short2_rtp(uchar2 x); __attribute__((overloadable)) short3 convert_short3_rtp(uchar3 x); __attribute__((overloadable)) short4 convert_short4_rtp(uchar4 x); __attribute__((overloadable)) short8 convert_short8_rtp(uchar8 x); __attribute__((overloadable)) short16 convert_short16_rtp(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(uchar16 x); __attribute__((overloadable)) long convert_long_rtp(uchar x); __attribute__((overloadable)) long2 convert_long2_rtp(uchar2 x); __attribute__((overloadable)) long3 convert_long3_rtp(uchar3 x); __attribute__((overloadable)) long4 convert_long4_rtp(uchar4 x); __attribute__((overloadable)) long8 convert_long8_rtp(uchar8 x); __attribute__((overloadable)) long16 convert_long16_rtp(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(uchar16 x); __attribute__((overloadable)) float convert_float_rtp(uchar x); __attribute__((overloadable)) float2 convert_float2_rtp(uchar2 x); __attribute__((overloadable)) float3 convert_float3_rtp(uchar3 x); __attribute__((overloadable)) float4 convert_float4_rtp(uchar4 x); __attribute__((overloadable)) float8 convert_float8_rtp(uchar8 x); __attribute__((overloadable)) float16 convert_float16_rtp(uchar16 x); __attribute__((overloadable)) double convert_double_rtp(uchar x); __attribute__((overloadable)) double2 convert_double2_rtp(uchar2 x); __attribute__((overloadable)) double3 convert_double3_rtp(uchar3 x); __attribute__((overloadable)) double4 convert_double4_rtp(uchar4 x); __attribute__((overloadable)) double8 convert_double8_rtp(uchar8 x); __attribute__((overloadable)) double16 convert_double16_rtp(uchar16 x); __attribute__((overloadable)) char convert_char_rtp(int x); __attribute__((overloadable)) char2 convert_char2_rtp(int2 x); __attribute__((overloadable)) char3 convert_char3_rtp(int3 x); __attribute__((overloadable)) char4 convert_char4_rtp(int4 x); __attribute__((overloadable)) char8 convert_char8_rtp(int8 x); __attribute__((overloadable)) char16 convert_char16_rtp(int16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(int x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(int16 x); __attribute__((overloadable)) int convert_int_rtp(int x); __attribute__((overloadable)) int2 convert_int2_rtp(int2 x); __attribute__((overloadable)) int3 convert_int3_rtp(int3 x); __attribute__((overloadable)) int4 convert_int4_rtp(int4 x); __attribute__((overloadable)) int8 convert_int8_rtp(int8 x); __attribute__((overloadable)) int16 convert_int16_rtp(int16 x); __attribute__((overloadable)) uint convert_uint_rtp(int x); __attribute__((overloadable)) uint2 convert_uint2_rtp(int2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(int3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(int4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(int8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(int16 x); __attribute__((overloadable)) short convert_short_rtp(int x); __attribute__((overloadable)) short2 convert_short2_rtp(int2 x); __attribute__((overloadable)) short3 convert_short3_rtp(int3 x); __attribute__((overloadable)) short4 convert_short4_rtp(int4 x); __attribute__((overloadable)) short8 convert_short8_rtp(int8 x); __attribute__((overloadable)) short16 convert_short16_rtp(int16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(int x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(int16 x); __attribute__((overloadable)) long convert_long_rtp(int x); __attribute__((overloadable)) long2 convert_long2_rtp(int2 x); __attribute__((overloadable)) long3 convert_long3_rtp(int3 x); __attribute__((overloadable)) long4 convert_long4_rtp(int4 x); __attribute__((overloadable)) long8 convert_long8_rtp(int8 x); __attribute__((overloadable)) long16 convert_long16_rtp(int16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(int x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(int16 x); __attribute__((overloadable)) float convert_float_rtp(int x); __attribute__((overloadable)) float2 convert_float2_rtp(int2 x); __attribute__((overloadable)) float3 convert_float3_rtp(int3 x); __attribute__((overloadable)) float4 convert_float4_rtp(int4 x); __attribute__((overloadable)) float8 convert_float8_rtp(int8 x); __attribute__((overloadable)) float16 convert_float16_rtp(int16 x); __attribute__((overloadable)) double convert_double_rtp(int x); __attribute__((overloadable)) double2 convert_double2_rtp(int2 x); __attribute__((overloadable)) double3 convert_double3_rtp(int3 x); __attribute__((overloadable)) double4 convert_double4_rtp(int4 x); __attribute__((overloadable)) double8 convert_double8_rtp(int8 x); __attribute__((overloadable)) double16 convert_double16_rtp(int16 x); __attribute__((overloadable)) char convert_char_rtp(uint x); __attribute__((overloadable)) char2 convert_char2_rtp(uint2 x); __attribute__((overloadable)) char3 convert_char3_rtp(uint3 x); __attribute__((overloadable)) char4 convert_char4_rtp(uint4 x); __attribute__((overloadable)) char8 convert_char8_rtp(uint8 x); __attribute__((overloadable)) char16 convert_char16_rtp(uint16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(uint16 x); __attribute__((overloadable)) int convert_int_rtp(uint x); __attribute__((overloadable)) int2 convert_int2_rtp(uint2 x); __attribute__((overloadable)) int3 convert_int3_rtp(uint3 x); __attribute__((overloadable)) int4 convert_int4_rtp(uint4 x); __attribute__((overloadable)) int8 convert_int8_rtp(uint8 x); __attribute__((overloadable)) int16 convert_int16_rtp(uint16 x); __attribute__((overloadable)) uint convert_uint_rtp(uint x); __attribute__((overloadable)) uint2 convert_uint2_rtp(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(uint16 x); __attribute__((overloadable)) short convert_short_rtp(uint x); __attribute__((overloadable)) short2 convert_short2_rtp(uint2 x); __attribute__((overloadable)) short3 convert_short3_rtp(uint3 x); __attribute__((overloadable)) short4 convert_short4_rtp(uint4 x); __attribute__((overloadable)) short8 convert_short8_rtp(uint8 x); __attribute__((overloadable)) short16 convert_short16_rtp(uint16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(uint16 x); __attribute__((overloadable)) long convert_long_rtp(uint x); __attribute__((overloadable)) long2 convert_long2_rtp(uint2 x); __attribute__((overloadable)) long3 convert_long3_rtp(uint3 x); __attribute__((overloadable)) long4 convert_long4_rtp(uint4 x); __attribute__((overloadable)) long8 convert_long8_rtp(uint8 x); __attribute__((overloadable)) long16 convert_long16_rtp(uint16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(uint16 x); __attribute__((overloadable)) float convert_float_rtp(uint x); __attribute__((overloadable)) float2 convert_float2_rtp(uint2 x); __attribute__((overloadable)) float3 convert_float3_rtp(uint3 x); __attribute__((overloadable)) float4 convert_float4_rtp(uint4 x); __attribute__((overloadable)) float8 convert_float8_rtp(uint8 x); __attribute__((overloadable)) float16 convert_float16_rtp(uint16 x); __attribute__((overloadable)) double convert_double_rtp(uint x); __attribute__((overloadable)) double2 convert_double2_rtp(uint2 x); __attribute__((overloadable)) double3 convert_double3_rtp(uint3 x); __attribute__((overloadable)) double4 convert_double4_rtp(uint4 x); __attribute__((overloadable)) double8 convert_double8_rtp(uint8 x); __attribute__((overloadable)) double16 convert_double16_rtp(uint16 x); __attribute__((overloadable)) char convert_char_rtp(short x); __attribute__((overloadable)) char2 convert_char2_rtp(short2 x); __attribute__((overloadable)) char3 convert_char3_rtp(short3 x); __attribute__((overloadable)) char4 convert_char4_rtp(short4 x); __attribute__((overloadable)) char8 convert_char8_rtp(short8 x); __attribute__((overloadable)) char16 convert_char16_rtp(short16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(short x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(short16 x); __attribute__((overloadable)) int convert_int_rtp(short x); __attribute__((overloadable)) int2 convert_int2_rtp(short2 x); __attribute__((overloadable)) int3 convert_int3_rtp(short3 x); __attribute__((overloadable)) int4 convert_int4_rtp(short4 x); __attribute__((overloadable)) int8 convert_int8_rtp(short8 x); __attribute__((overloadable)) int16 convert_int16_rtp(short16 x); __attribute__((overloadable)) uint convert_uint_rtp(short x); __attribute__((overloadable)) uint2 convert_uint2_rtp(short2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(short3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(short4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(short8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(short16 x); __attribute__((overloadable)) short convert_short_rtp(short x); __attribute__((overloadable)) short2 convert_short2_rtp(short2 x); __attribute__((overloadable)) short3 convert_short3_rtp(short3 x); __attribute__((overloadable)) short4 convert_short4_rtp(short4 x); __attribute__((overloadable)) short8 convert_short8_rtp(short8 x); __attribute__((overloadable)) short16 convert_short16_rtp(short16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(short x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(short16 x); __attribute__((overloadable)) long convert_long_rtp(short x); __attribute__((overloadable)) long2 convert_long2_rtp(short2 x); __attribute__((overloadable)) long3 convert_long3_rtp(short3 x); __attribute__((overloadable)) long4 convert_long4_rtp(short4 x); __attribute__((overloadable)) long8 convert_long8_rtp(short8 x); __attribute__((overloadable)) long16 convert_long16_rtp(short16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(short x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(short16 x); __attribute__((overloadable)) float convert_float_rtp(short x); __attribute__((overloadable)) float2 convert_float2_rtp(short2 x); __attribute__((overloadable)) float3 convert_float3_rtp(short3 x); __attribute__((overloadable)) float4 convert_float4_rtp(short4 x); __attribute__((overloadable)) float8 convert_float8_rtp(short8 x); __attribute__((overloadable)) float16 convert_float16_rtp(short16 x); __attribute__((overloadable)) double convert_double_rtp(short x); __attribute__((overloadable)) double2 convert_double2_rtp(short2 x); __attribute__((overloadable)) double3 convert_double3_rtp(short3 x); __attribute__((overloadable)) double4 convert_double4_rtp(short4 x); __attribute__((overloadable)) double8 convert_double8_rtp(short8 x); __attribute__((overloadable)) double16 convert_double16_rtp(short16 x); __attribute__((overloadable)) char convert_char_rtp(ushort x); __attribute__((overloadable)) char2 convert_char2_rtp(ushort2 x); __attribute__((overloadable)) char3 convert_char3_rtp(ushort3 x); __attribute__((overloadable)) char4 convert_char4_rtp(ushort4 x); __attribute__((overloadable)) char8 convert_char8_rtp(ushort8 x); __attribute__((overloadable)) char16 convert_char16_rtp(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(ushort16 x); __attribute__((overloadable)) int convert_int_rtp(ushort x); __attribute__((overloadable)) int2 convert_int2_rtp(ushort2 x); __attribute__((overloadable)) int3 convert_int3_rtp(ushort3 x); __attribute__((overloadable)) int4 convert_int4_rtp(ushort4 x); __attribute__((overloadable)) int8 convert_int8_rtp(ushort8 x); __attribute__((overloadable)) int16 convert_int16_rtp(ushort16 x); __attribute__((overloadable)) uint convert_uint_rtp(ushort x); __attribute__((overloadable)) uint2 convert_uint2_rtp(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(ushort16 x); __attribute__((overloadable)) short convert_short_rtp(ushort x); __attribute__((overloadable)) short2 convert_short2_rtp(ushort2 x); __attribute__((overloadable)) short3 convert_short3_rtp(ushort3 x); __attribute__((overloadable)) short4 convert_short4_rtp(ushort4 x); __attribute__((overloadable)) short8 convert_short8_rtp(ushort8 x); __attribute__((overloadable)) short16 convert_short16_rtp(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(ushort16 x); __attribute__((overloadable)) long convert_long_rtp(ushort x); __attribute__((overloadable)) long2 convert_long2_rtp(ushort2 x); __attribute__((overloadable)) long3 convert_long3_rtp(ushort3 x); __attribute__((overloadable)) long4 convert_long4_rtp(ushort4 x); __attribute__((overloadable)) long8 convert_long8_rtp(ushort8 x); __attribute__((overloadable)) long16 convert_long16_rtp(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(ushort16 x); __attribute__((overloadable)) float convert_float_rtp(ushort x); __attribute__((overloadable)) float2 convert_float2_rtp(ushort2 x); __attribute__((overloadable)) float3 convert_float3_rtp(ushort3 x); __attribute__((overloadable)) float4 convert_float4_rtp(ushort4 x); __attribute__((overloadable)) float8 convert_float8_rtp(ushort8 x); __attribute__((overloadable)) float16 convert_float16_rtp(ushort16 x); __attribute__((overloadable)) double convert_double_rtp(ushort x); __attribute__((overloadable)) double2 convert_double2_rtp(ushort2 x); __attribute__((overloadable)) double3 convert_double3_rtp(ushort3 x); __attribute__((overloadable)) double4 convert_double4_rtp(ushort4 x); __attribute__((overloadable)) double8 convert_double8_rtp(ushort8 x); __attribute__((overloadable)) double16 convert_double16_rtp(ushort16 x); __attribute__((overloadable)) char convert_char_rtp(long x); __attribute__((overloadable)) char2 convert_char2_rtp(long2 x); __attribute__((overloadable)) char3 convert_char3_rtp(long3 x); __attribute__((overloadable)) char4 convert_char4_rtp(long4 x); __attribute__((overloadable)) char8 convert_char8_rtp(long8 x); __attribute__((overloadable)) char16 convert_char16_rtp(long16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(long x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(long16 x); __attribute__((overloadable)) int convert_int_rtp(long x); __attribute__((overloadable)) int2 convert_int2_rtp(long2 x); __attribute__((overloadable)) int3 convert_int3_rtp(long3 x); __attribute__((overloadable)) int4 convert_int4_rtp(long4 x); __attribute__((overloadable)) int8 convert_int8_rtp(long8 x); __attribute__((overloadable)) int16 convert_int16_rtp(long16 x); __attribute__((overloadable)) uint convert_uint_rtp(long x); __attribute__((overloadable)) uint2 convert_uint2_rtp(long2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(long3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(long4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(long8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(long16 x); __attribute__((overloadable)) short convert_short_rtp(long x); __attribute__((overloadable)) short2 convert_short2_rtp(long2 x); __attribute__((overloadable)) short3 convert_short3_rtp(long3 x); __attribute__((overloadable)) short4 convert_short4_rtp(long4 x); __attribute__((overloadable)) short8 convert_short8_rtp(long8 x); __attribute__((overloadable)) short16 convert_short16_rtp(long16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(long x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(long16 x); __attribute__((overloadable)) long convert_long_rtp(long x); __attribute__((overloadable)) long2 convert_long2_rtp(long2 x); __attribute__((overloadable)) long3 convert_long3_rtp(long3 x); __attribute__((overloadable)) long4 convert_long4_rtp(long4 x); __attribute__((overloadable)) long8 convert_long8_rtp(long8 x); __attribute__((overloadable)) long16 convert_long16_rtp(long16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(long x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(long16 x); __attribute__((overloadable)) float convert_float_rtp(long x); __attribute__((overloadable)) float2 convert_float2_rtp(long2 x); __attribute__((overloadable)) float3 convert_float3_rtp(long3 x); __attribute__((overloadable)) float4 convert_float4_rtp(long4 x); __attribute__((overloadable)) float8 convert_float8_rtp(long8 x); __attribute__((overloadable)) float16 convert_float16_rtp(long16 x); __attribute__((overloadable)) double convert_double_rtp(long x); __attribute__((overloadable)) double2 convert_double2_rtp(long2 x); __attribute__((overloadable)) double3 convert_double3_rtp(long3 x); __attribute__((overloadable)) double4 convert_double4_rtp(long4 x); __attribute__((overloadable)) double8 convert_double8_rtp(long8 x); __attribute__((overloadable)) double16 convert_double16_rtp(long16 x); __attribute__((overloadable)) char convert_char_rtp(ulong x); __attribute__((overloadable)) char2 convert_char2_rtp(ulong2 x); __attribute__((overloadable)) char3 convert_char3_rtp(ulong3 x); __attribute__((overloadable)) char4 convert_char4_rtp(ulong4 x); __attribute__((overloadable)) char8 convert_char8_rtp(ulong8 x); __attribute__((overloadable)) char16 convert_char16_rtp(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(ulong16 x); __attribute__((overloadable)) int convert_int_rtp(ulong x); __attribute__((overloadable)) int2 convert_int2_rtp(ulong2 x); __attribute__((overloadable)) int3 convert_int3_rtp(ulong3 x); __attribute__((overloadable)) int4 convert_int4_rtp(ulong4 x); __attribute__((overloadable)) int8 convert_int8_rtp(ulong8 x); __attribute__((overloadable)) int16 convert_int16_rtp(ulong16 x); __attribute__((overloadable)) uint convert_uint_rtp(ulong x); __attribute__((overloadable)) uint2 convert_uint2_rtp(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(ulong16 x); __attribute__((overloadable)) short convert_short_rtp(ulong x); __attribute__((overloadable)) short2 convert_short2_rtp(ulong2 x); __attribute__((overloadable)) short3 convert_short3_rtp(ulong3 x); __attribute__((overloadable)) short4 convert_short4_rtp(ulong4 x); __attribute__((overloadable)) short8 convert_short8_rtp(ulong8 x); __attribute__((overloadable)) short16 convert_short16_rtp(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(ulong16 x); __attribute__((overloadable)) long convert_long_rtp(ulong x); __attribute__((overloadable)) long2 convert_long2_rtp(ulong2 x); __attribute__((overloadable)) long3 convert_long3_rtp(ulong3 x); __attribute__((overloadable)) long4 convert_long4_rtp(ulong4 x); __attribute__((overloadable)) long8 convert_long8_rtp(ulong8 x); __attribute__((overloadable)) long16 convert_long16_rtp(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(ulong16 x); __attribute__((overloadable)) float convert_float_rtp(ulong x); __attribute__((overloadable)) float2 convert_float2_rtp(ulong2 x); __attribute__((overloadable)) float3 convert_float3_rtp(ulong3 x); __attribute__((overloadable)) float4 convert_float4_rtp(ulong4 x); __attribute__((overloadable)) float8 convert_float8_rtp(ulong8 x); __attribute__((overloadable)) float16 convert_float16_rtp(ulong16 x); __attribute__((overloadable)) double convert_double_rtp(ulong x); __attribute__((overloadable)) double2 convert_double2_rtp(ulong2 x); __attribute__((overloadable)) double3 convert_double3_rtp(ulong3 x); __attribute__((overloadable)) double4 convert_double4_rtp(ulong4 x); __attribute__((overloadable)) double8 convert_double8_rtp(ulong8 x); __attribute__((overloadable)) double16 convert_double16_rtp(ulong16 x); __attribute__((overloadable)) char convert_char_rtp(float x); __attribute__((overloadable)) char2 convert_char2_rtp(float2 x); __attribute__((overloadable)) char3 convert_char3_rtp(float3 x); __attribute__((overloadable)) char4 convert_char4_rtp(float4 x); __attribute__((overloadable)) char8 convert_char8_rtp(float8 x); __attribute__((overloadable)) char16 convert_char16_rtp(float16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(float x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(float16 x); __attribute__((overloadable)) int convert_int_rtp(float x); __attribute__((overloadable)) int2 convert_int2_rtp(float2 x); __attribute__((overloadable)) int3 convert_int3_rtp(float3 x); __attribute__((overloadable)) int4 convert_int4_rtp(float4 x); __attribute__((overloadable)) int8 convert_int8_rtp(float8 x); __attribute__((overloadable)) int16 convert_int16_rtp(float16 x); __attribute__((overloadable)) uint convert_uint_rtp(float x); __attribute__((overloadable)) uint2 convert_uint2_rtp(float2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(float3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(float4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(float8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(float16 x); __attribute__((overloadable)) short convert_short_rtp(float x); __attribute__((overloadable)) short2 convert_short2_rtp(float2 x); __attribute__((overloadable)) short3 convert_short3_rtp(float3 x); __attribute__((overloadable)) short4 convert_short4_rtp(float4 x); __attribute__((overloadable)) short8 convert_short8_rtp(float8 x); __attribute__((overloadable)) short16 convert_short16_rtp(float16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(float x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(float16 x); __attribute__((overloadable)) long convert_long_rtp(float x); __attribute__((overloadable)) long2 convert_long2_rtp(float2 x); __attribute__((overloadable)) long3 convert_long3_rtp(float3 x); __attribute__((overloadable)) long4 convert_long4_rtp(float4 x); __attribute__((overloadable)) long8 convert_long8_rtp(float8 x); __attribute__((overloadable)) long16 convert_long16_rtp(float16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(float x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(float16 x); __attribute__((overloadable)) float convert_float_rtp(float x); __attribute__((overloadable)) float2 convert_float2_rtp(float2 x); __attribute__((overloadable)) float3 convert_float3_rtp(float3 x); __attribute__((overloadable)) float4 convert_float4_rtp(float4 x); __attribute__((overloadable)) float8 convert_float8_rtp(float8 x); __attribute__((overloadable)) float16 convert_float16_rtp(float16 x); __attribute__((overloadable)) double convert_double_rtp(float x); __attribute__((overloadable)) double2 convert_double2_rtp(float2 x); __attribute__((overloadable)) double3 convert_double3_rtp(float3 x); __attribute__((overloadable)) double4 convert_double4_rtp(float4 x); __attribute__((overloadable)) double8 convert_double8_rtp(float8 x); __attribute__((overloadable)) double16 convert_double16_rtp(float16 x); __attribute__((overloadable)) char convert_char_rtp(double x); __attribute__((overloadable)) char2 convert_char2_rtp(double2 x); __attribute__((overloadable)) char3 convert_char3_rtp(double3 x); __attribute__((overloadable)) char4 convert_char4_rtp(double4 x); __attribute__((overloadable)) char8 convert_char8_rtp(double8 x); __attribute__((overloadable)) char16 convert_char16_rtp(double16 x); __attribute__((overloadable)) uchar convert_uchar_rtp(double x); __attribute__((overloadable)) uchar2 convert_uchar2_rtp(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_rtp(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_rtp(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_rtp(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_rtp(double16 x); __attribute__((overloadable)) int convert_int_rtp(double x); __attribute__((overloadable)) int2 convert_int2_rtp(double2 x); __attribute__((overloadable)) int3 convert_int3_rtp(double3 x); __attribute__((overloadable)) int4 convert_int4_rtp(double4 x); __attribute__((overloadable)) int8 convert_int8_rtp(double8 x); __attribute__((overloadable)) int16 convert_int16_rtp(double16 x); __attribute__((overloadable)) uint convert_uint_rtp(double x); __attribute__((overloadable)) uint2 convert_uint2_rtp(double2 x); __attribute__((overloadable)) uint3 convert_uint3_rtp(double3 x); __attribute__((overloadable)) uint4 convert_uint4_rtp(double4 x); __attribute__((overloadable)) uint8 convert_uint8_rtp(double8 x); __attribute__((overloadable)) uint16 convert_uint16_rtp(double16 x); __attribute__((overloadable)) short convert_short_rtp(double x); __attribute__((overloadable)) short2 convert_short2_rtp(double2 x); __attribute__((overloadable)) short3 convert_short3_rtp(double3 x); __attribute__((overloadable)) short4 convert_short4_rtp(double4 x); __attribute__((overloadable)) short8 convert_short8_rtp(double8 x); __attribute__((overloadable)) short16 convert_short16_rtp(double16 x); __attribute__((overloadable)) ushort convert_ushort_rtp(double x); __attribute__((overloadable)) ushort2 convert_ushort2_rtp(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_rtp(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_rtp(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_rtp(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_rtp(double16 x); __attribute__((overloadable)) long convert_long_rtp(double x); __attribute__((overloadable)) long2 convert_long2_rtp(double2 x); __attribute__((overloadable)) long3 convert_long3_rtp(double3 x); __attribute__((overloadable)) long4 convert_long4_rtp(double4 x); __attribute__((overloadable)) long8 convert_long8_rtp(double8 x); __attribute__((overloadable)) long16 convert_long16_rtp(double16 x); __attribute__((overloadable)) ulong convert_ulong_rtp(double x); __attribute__((overloadable)) ulong2 convert_ulong2_rtp(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_rtp(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_rtp(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_rtp(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_rtp(double16 x); __attribute__((overloadable)) float convert_float_rtp(double x); __attribute__((overloadable)) float2 convert_float2_rtp(double2 x); __attribute__((overloadable)) float3 convert_float3_rtp(double3 x); __attribute__((overloadable)) float4 convert_float4_rtp(double4 x); __attribute__((overloadable)) float8 convert_float8_rtp(double8 x); __attribute__((overloadable)) float16 convert_float16_rtp(double16 x); __attribute__((overloadable)) double convert_double_rtp(double x); __attribute__((overloadable)) double2 convert_double2_rtp(double2 x); __attribute__((overloadable)) double3 convert_double3_rtp(double3 x); __attribute__((overloadable)) double4 convert_double4_rtp(double4 x); __attribute__((overloadable)) double8 convert_double8_rtp(double8 x); __attribute__((overloadable)) double16 convert_double16_rtp(double16 x);
__attribute__((overloadable)) char convert_char_sat(char x); __attribute__((overloadable)) char2 convert_char2_sat(char2 x); __attribute__((overloadable)) char3 convert_char3_sat(char3 x); __attribute__((overloadable)) char4 convert_char4_sat(char4 x); __attribute__((overloadable)) char8 convert_char8_sat(char8 x); __attribute__((overloadable)) char16 convert_char16_sat(char16 x); __attribute__((overloadable)) uchar convert_uchar_sat(char x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(char16 x); __attribute__((overloadable)) int convert_int_sat(char x); __attribute__((overloadable)) int2 convert_int2_sat(char2 x); __attribute__((overloadable)) int3 convert_int3_sat(char3 x); __attribute__((overloadable)) int4 convert_int4_sat(char4 x); __attribute__((overloadable)) int8 convert_int8_sat(char8 x); __attribute__((overloadable)) int16 convert_int16_sat(char16 x); __attribute__((overloadable)) uint convert_uint_sat(char x); __attribute__((overloadable)) uint2 convert_uint2_sat(char2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(char3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(char4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(char8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(char16 x); __attribute__((overloadable)) short convert_short_sat(char x); __attribute__((overloadable)) short2 convert_short2_sat(char2 x); __attribute__((overloadable)) short3 convert_short3_sat(char3 x); __attribute__((overloadable)) short4 convert_short4_sat(char4 x); __attribute__((overloadable)) short8 convert_short8_sat(char8 x); __attribute__((overloadable)) short16 convert_short16_sat(char16 x); __attribute__((overloadable)) ushort convert_ushort_sat(char x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(char16 x); __attribute__((overloadable)) long convert_long_sat(char x); __attribute__((overloadable)) long2 convert_long2_sat(char2 x); __attribute__((overloadable)) long3 convert_long3_sat(char3 x); __attribute__((overloadable)) long4 convert_long4_sat(char4 x); __attribute__((overloadable)) long8 convert_long8_sat(char8 x); __attribute__((overloadable)) long16 convert_long16_sat(char16 x); __attribute__((overloadable)) ulong convert_ulong_sat(char x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(char16 x); __attribute__((overloadable)) float convert_float_sat(char x); __attribute__((overloadable)) float2 convert_float2_sat(char2 x); __attribute__((overloadable)) float3 convert_float3_sat(char3 x); __attribute__((overloadable)) float4 convert_float4_sat(char4 x); __attribute__((overloadable)) float8 convert_float8_sat(char8 x); __attribute__((overloadable)) float16 convert_float16_sat(char16 x); __attribute__((overloadable)) double convert_double_sat(char x); __attribute__((overloadable)) double2 convert_double2_sat(char2 x); __attribute__((overloadable)) double3 convert_double3_sat(char3 x); __attribute__((overloadable)) double4 convert_double4_sat(char4 x); __attribute__((overloadable)) double8 convert_double8_sat(char8 x); __attribute__((overloadable)) double16 convert_double16_sat(char16 x); __attribute__((overloadable)) char convert_char_sat(uchar x); __attribute__((overloadable)) char2 convert_char2_sat(uchar2 x); __attribute__((overloadable)) char3 convert_char3_sat(uchar3 x); __attribute__((overloadable)) char4 convert_char4_sat(uchar4 x); __attribute__((overloadable)) char8 convert_char8_sat(uchar8 x); __attribute__((overloadable)) char16 convert_char16_sat(uchar16 x); __attribute__((overloadable)) uchar convert_uchar_sat(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(uchar16 x); __attribute__((overloadable)) int convert_int_sat(uchar x); __attribute__((overloadable)) int2 convert_int2_sat(uchar2 x); __attribute__((overloadable)) int3 convert_int3_sat(uchar3 x); __attribute__((overloadable)) int4 convert_int4_sat(uchar4 x); __attribute__((overloadable)) int8 convert_int8_sat(uchar8 x); __attribute__((overloadable)) int16 convert_int16_sat(uchar16 x); __attribute__((overloadable)) uint convert_uint_sat(uchar x); __attribute__((overloadable)) uint2 convert_uint2_sat(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(uchar16 x); __attribute__((overloadable)) short convert_short_sat(uchar x); __attribute__((overloadable)) short2 convert_short2_sat(uchar2 x); __attribute__((overloadable)) short3 convert_short3_sat(uchar3 x); __attribute__((overloadable)) short4 convert_short4_sat(uchar4 x); __attribute__((overloadable)) short8 convert_short8_sat(uchar8 x); __attribute__((overloadable)) short16 convert_short16_sat(uchar16 x); __attribute__((overloadable)) ushort convert_ushort_sat(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(uchar16 x); __attribute__((overloadable)) long convert_long_sat(uchar x); __attribute__((overloadable)) long2 convert_long2_sat(uchar2 x); __attribute__((overloadable)) long3 convert_long3_sat(uchar3 x); __attribute__((overloadable)) long4 convert_long4_sat(uchar4 x); __attribute__((overloadable)) long8 convert_long8_sat(uchar8 x); __attribute__((overloadable)) long16 convert_long16_sat(uchar16 x); __attribute__((overloadable)) ulong convert_ulong_sat(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(uchar16 x); __attribute__((overloadable)) float convert_float_sat(uchar x); __attribute__((overloadable)) float2 convert_float2_sat(uchar2 x); __attribute__((overloadable)) float3 convert_float3_sat(uchar3 x); __attribute__((overloadable)) float4 convert_float4_sat(uchar4 x); __attribute__((overloadable)) float8 convert_float8_sat(uchar8 x); __attribute__((overloadable)) float16 convert_float16_sat(uchar16 x); __attribute__((overloadable)) double convert_double_sat(uchar x); __attribute__((overloadable)) double2 convert_double2_sat(uchar2 x); __attribute__((overloadable)) double3 convert_double3_sat(uchar3 x); __attribute__((overloadable)) double4 convert_double4_sat(uchar4 x); __attribute__((overloadable)) double8 convert_double8_sat(uchar8 x); __attribute__((overloadable)) double16 convert_double16_sat(uchar16 x); __attribute__((overloadable)) char convert_char_sat(int x); __attribute__((overloadable)) char2 convert_char2_sat(int2 x); __attribute__((overloadable)) char3 convert_char3_sat(int3 x); __attribute__((overloadable)) char4 convert_char4_sat(int4 x); __attribute__((overloadable)) char8 convert_char8_sat(int8 x); __attribute__((overloadable)) char16 convert_char16_sat(int16 x); __attribute__((overloadable)) uchar convert_uchar_sat(int x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(int16 x); __attribute__((overloadable)) int convert_int_sat(int x); __attribute__((overloadable)) int2 convert_int2_sat(int2 x); __attribute__((overloadable)) int3 convert_int3_sat(int3 x); __attribute__((overloadable)) int4 convert_int4_sat(int4 x); __attribute__((overloadable)) int8 convert_int8_sat(int8 x); __attribute__((overloadable)) int16 convert_int16_sat(int16 x); __attribute__((overloadable)) uint convert_uint_sat(int x); __attribute__((overloadable)) uint2 convert_uint2_sat(int2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(int3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(int4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(int8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(int16 x); __attribute__((overloadable)) short convert_short_sat(int x); __attribute__((overloadable)) short2 convert_short2_sat(int2 x); __attribute__((overloadable)) short3 convert_short3_sat(int3 x); __attribute__((overloadable)) short4 convert_short4_sat(int4 x); __attribute__((overloadable)) short8 convert_short8_sat(int8 x); __attribute__((overloadable)) short16 convert_short16_sat(int16 x); __attribute__((overloadable)) ushort convert_ushort_sat(int x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(int16 x); __attribute__((overloadable)) long convert_long_sat(int x); __attribute__((overloadable)) long2 convert_long2_sat(int2 x); __attribute__((overloadable)) long3 convert_long3_sat(int3 x); __attribute__((overloadable)) long4 convert_long4_sat(int4 x); __attribute__((overloadable)) long8 convert_long8_sat(int8 x); __attribute__((overloadable)) long16 convert_long16_sat(int16 x); __attribute__((overloadable)) ulong convert_ulong_sat(int x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(int16 x); __attribute__((overloadable)) float convert_float_sat(int x); __attribute__((overloadable)) float2 convert_float2_sat(int2 x); __attribute__((overloadable)) float3 convert_float3_sat(int3 x); __attribute__((overloadable)) float4 convert_float4_sat(int4 x); __attribute__((overloadable)) float8 convert_float8_sat(int8 x); __attribute__((overloadable)) float16 convert_float16_sat(int16 x); __attribute__((overloadable)) double convert_double_sat(int x); __attribute__((overloadable)) double2 convert_double2_sat(int2 x); __attribute__((overloadable)) double3 convert_double3_sat(int3 x); __attribute__((overloadable)) double4 convert_double4_sat(int4 x); __attribute__((overloadable)) double8 convert_double8_sat(int8 x); __attribute__((overloadable)) double16 convert_double16_sat(int16 x); __attribute__((overloadable)) char convert_char_sat(uint x); __attribute__((overloadable)) char2 convert_char2_sat(uint2 x); __attribute__((overloadable)) char3 convert_char3_sat(uint3 x); __attribute__((overloadable)) char4 convert_char4_sat(uint4 x); __attribute__((overloadable)) char8 convert_char8_sat(uint8 x); __attribute__((overloadable)) char16 convert_char16_sat(uint16 x); __attribute__((overloadable)) uchar convert_uchar_sat(uint x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(uint16 x); __attribute__((overloadable)) int convert_int_sat(uint x); __attribute__((overloadable)) int2 convert_int2_sat(uint2 x); __attribute__((overloadable)) int3 convert_int3_sat(uint3 x); __attribute__((overloadable)) int4 convert_int4_sat(uint4 x); __attribute__((overloadable)) int8 convert_int8_sat(uint8 x); __attribute__((overloadable)) int16 convert_int16_sat(uint16 x); __attribute__((overloadable)) uint convert_uint_sat(uint x); __attribute__((overloadable)) uint2 convert_uint2_sat(uint2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(uint3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(uint4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(uint8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(uint16 x); __attribute__((overloadable)) short convert_short_sat(uint x); __attribute__((overloadable)) short2 convert_short2_sat(uint2 x); __attribute__((overloadable)) short3 convert_short3_sat(uint3 x); __attribute__((overloadable)) short4 convert_short4_sat(uint4 x); __attribute__((overloadable)) short8 convert_short8_sat(uint8 x); __attribute__((overloadable)) short16 convert_short16_sat(uint16 x); __attribute__((overloadable)) ushort convert_ushort_sat(uint x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(uint16 x); __attribute__((overloadable)) long convert_long_sat(uint x); __attribute__((overloadable)) long2 convert_long2_sat(uint2 x); __attribute__((overloadable)) long3 convert_long3_sat(uint3 x); __attribute__((overloadable)) long4 convert_long4_sat(uint4 x); __attribute__((overloadable)) long8 convert_long8_sat(uint8 x); __attribute__((overloadable)) long16 convert_long16_sat(uint16 x); __attribute__((overloadable)) ulong convert_ulong_sat(uint x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(uint16 x); __attribute__((overloadable)) float convert_float_sat(uint x); __attribute__((overloadable)) float2 convert_float2_sat(uint2 x); __attribute__((overloadable)) float3 convert_float3_sat(uint3 x); __attribute__((overloadable)) float4 convert_float4_sat(uint4 x); __attribute__((overloadable)) float8 convert_float8_sat(uint8 x); __attribute__((overloadable)) float16 convert_float16_sat(uint16 x); __attribute__((overloadable)) double convert_double_sat(uint x); __attribute__((overloadable)) double2 convert_double2_sat(uint2 x); __attribute__((overloadable)) double3 convert_double3_sat(uint3 x); __attribute__((overloadable)) double4 convert_double4_sat(uint4 x); __attribute__((overloadable)) double8 convert_double8_sat(uint8 x); __attribute__((overloadable)) double16 convert_double16_sat(uint16 x); __attribute__((overloadable)) char convert_char_sat(short x); __attribute__((overloadable)) char2 convert_char2_sat(short2 x); __attribute__((overloadable)) char3 convert_char3_sat(short3 x); __attribute__((overloadable)) char4 convert_char4_sat(short4 x); __attribute__((overloadable)) char8 convert_char8_sat(short8 x); __attribute__((overloadable)) char16 convert_char16_sat(short16 x); __attribute__((overloadable)) uchar convert_uchar_sat(short x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(short16 x); __attribute__((overloadable)) int convert_int_sat(short x); __attribute__((overloadable)) int2 convert_int2_sat(short2 x); __attribute__((overloadable)) int3 convert_int3_sat(short3 x); __attribute__((overloadable)) int4 convert_int4_sat(short4 x); __attribute__((overloadable)) int8 convert_int8_sat(short8 x); __attribute__((overloadable)) int16 convert_int16_sat(short16 x); __attribute__((overloadable)) uint convert_uint_sat(short x); __attribute__((overloadable)) uint2 convert_uint2_sat(short2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(short3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(short4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(short8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(short16 x); __attribute__((overloadable)) short convert_short_sat(short x); __attribute__((overloadable)) short2 convert_short2_sat(short2 x); __attribute__((overloadable)) short3 convert_short3_sat(short3 x); __attribute__((overloadable)) short4 convert_short4_sat(short4 x); __attribute__((overloadable)) short8 convert_short8_sat(short8 x); __attribute__((overloadable)) short16 convert_short16_sat(short16 x); __attribute__((overloadable)) ushort convert_ushort_sat(short x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(short16 x); __attribute__((overloadable)) long convert_long_sat(short x); __attribute__((overloadable)) long2 convert_long2_sat(short2 x); __attribute__((overloadable)) long3 convert_long3_sat(short3 x); __attribute__((overloadable)) long4 convert_long4_sat(short4 x); __attribute__((overloadable)) long8 convert_long8_sat(short8 x); __attribute__((overloadable)) long16 convert_long16_sat(short16 x); __attribute__((overloadable)) ulong convert_ulong_sat(short x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(short16 x); __attribute__((overloadable)) float convert_float_sat(short x); __attribute__((overloadable)) float2 convert_float2_sat(short2 x); __attribute__((overloadable)) float3 convert_float3_sat(short3 x); __attribute__((overloadable)) float4 convert_float4_sat(short4 x); __attribute__((overloadable)) float8 convert_float8_sat(short8 x); __attribute__((overloadable)) float16 convert_float16_sat(short16 x); __attribute__((overloadable)) double convert_double_sat(short x); __attribute__((overloadable)) double2 convert_double2_sat(short2 x); __attribute__((overloadable)) double3 convert_double3_sat(short3 x); __attribute__((overloadable)) double4 convert_double4_sat(short4 x); __attribute__((overloadable)) double8 convert_double8_sat(short8 x); __attribute__((overloadable)) double16 convert_double16_sat(short16 x); __attribute__((overloadable)) char convert_char_sat(ushort x); __attribute__((overloadable)) char2 convert_char2_sat(ushort2 x); __attribute__((overloadable)) char3 convert_char3_sat(ushort3 x); __attribute__((overloadable)) char4 convert_char4_sat(ushort4 x); __attribute__((overloadable)) char8 convert_char8_sat(ushort8 x); __attribute__((overloadable)) char16 convert_char16_sat(ushort16 x); __attribute__((overloadable)) uchar convert_uchar_sat(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(ushort16 x); __attribute__((overloadable)) int convert_int_sat(ushort x); __attribute__((overloadable)) int2 convert_int2_sat(ushort2 x); __attribute__((overloadable)) int3 convert_int3_sat(ushort3 x); __attribute__((overloadable)) int4 convert_int4_sat(ushort4 x); __attribute__((overloadable)) int8 convert_int8_sat(ushort8 x); __attribute__((overloadable)) int16 convert_int16_sat(ushort16 x); __attribute__((overloadable)) uint convert_uint_sat(ushort x); __attribute__((overloadable)) uint2 convert_uint2_sat(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(ushort16 x); __attribute__((overloadable)) short convert_short_sat(ushort x); __attribute__((overloadable)) short2 convert_short2_sat(ushort2 x); __attribute__((overloadable)) short3 convert_short3_sat(ushort3 x); __attribute__((overloadable)) short4 convert_short4_sat(ushort4 x); __attribute__((overloadable)) short8 convert_short8_sat(ushort8 x); __attribute__((overloadable)) short16 convert_short16_sat(ushort16 x); __attribute__((overloadable)) ushort convert_ushort_sat(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(ushort16 x); __attribute__((overloadable)) long convert_long_sat(ushort x); __attribute__((overloadable)) long2 convert_long2_sat(ushort2 x); __attribute__((overloadable)) long3 convert_long3_sat(ushort3 x); __attribute__((overloadable)) long4 convert_long4_sat(ushort4 x); __attribute__((overloadable)) long8 convert_long8_sat(ushort8 x); __attribute__((overloadable)) long16 convert_long16_sat(ushort16 x); __attribute__((overloadable)) ulong convert_ulong_sat(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(ushort16 x); __attribute__((overloadable)) float convert_float_sat(ushort x); __attribute__((overloadable)) float2 convert_float2_sat(ushort2 x); __attribute__((overloadable)) float3 convert_float3_sat(ushort3 x); __attribute__((overloadable)) float4 convert_float4_sat(ushort4 x); __attribute__((overloadable)) float8 convert_float8_sat(ushort8 x); __attribute__((overloadable)) float16 convert_float16_sat(ushort16 x); __attribute__((overloadable)) double convert_double_sat(ushort x); __attribute__((overloadable)) double2 convert_double2_sat(ushort2 x); __attribute__((overloadable)) double3 convert_double3_sat(ushort3 x); __attribute__((overloadable)) double4 convert_double4_sat(ushort4 x); __attribute__((overloadable)) double8 convert_double8_sat(ushort8 x); __attribute__((overloadable)) double16 convert_double16_sat(ushort16 x); __attribute__((overloadable)) char convert_char_sat(long x); __attribute__((overloadable)) char2 convert_char2_sat(long2 x); __attribute__((overloadable)) char3 convert_char3_sat(long3 x); __attribute__((overloadable)) char4 convert_char4_sat(long4 x); __attribute__((overloadable)) char8 convert_char8_sat(long8 x); __attribute__((overloadable)) char16 convert_char16_sat(long16 x); __attribute__((overloadable)) uchar convert_uchar_sat(long x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(long16 x); __attribute__((overloadable)) int convert_int_sat(long x); __attribute__((overloadable)) int2 convert_int2_sat(long2 x); __attribute__((overloadable)) int3 convert_int3_sat(long3 x); __attribute__((overloadable)) int4 convert_int4_sat(long4 x); __attribute__((overloadable)) int8 convert_int8_sat(long8 x); __attribute__((overloadable)) int16 convert_int16_sat(long16 x); __attribute__((overloadable)) uint convert_uint_sat(long x); __attribute__((overloadable)) uint2 convert_uint2_sat(long2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(long3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(long4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(long8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(long16 x); __attribute__((overloadable)) short convert_short_sat(long x); __attribute__((overloadable)) short2 convert_short2_sat(long2 x); __attribute__((overloadable)) short3 convert_short3_sat(long3 x); __attribute__((overloadable)) short4 convert_short4_sat(long4 x); __attribute__((overloadable)) short8 convert_short8_sat(long8 x); __attribute__((overloadable)) short16 convert_short16_sat(long16 x); __attribute__((overloadable)) ushort convert_ushort_sat(long x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(long16 x); __attribute__((overloadable)) long convert_long_sat(long x); __attribute__((overloadable)) long2 convert_long2_sat(long2 x); __attribute__((overloadable)) long3 convert_long3_sat(long3 x); __attribute__((overloadable)) long4 convert_long4_sat(long4 x); __attribute__((overloadable)) long8 convert_long8_sat(long8 x); __attribute__((overloadable)) long16 convert_long16_sat(long16 x); __attribute__((overloadable)) ulong convert_ulong_sat(long x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(long16 x); __attribute__((overloadable)) float convert_float_sat(long x); __attribute__((overloadable)) float2 convert_float2_sat(long2 x); __attribute__((overloadable)) float3 convert_float3_sat(long3 x); __attribute__((overloadable)) float4 convert_float4_sat(long4 x); __attribute__((overloadable)) float8 convert_float8_sat(long8 x); __attribute__((overloadable)) float16 convert_float16_sat(long16 x); __attribute__((overloadable)) double convert_double_sat(long x); __attribute__((overloadable)) double2 convert_double2_sat(long2 x); __attribute__((overloadable)) double3 convert_double3_sat(long3 x); __attribute__((overloadable)) double4 convert_double4_sat(long4 x); __attribute__((overloadable)) double8 convert_double8_sat(long8 x); __attribute__((overloadable)) double16 convert_double16_sat(long16 x); __attribute__((overloadable)) char convert_char_sat(ulong x); __attribute__((overloadable)) char2 convert_char2_sat(ulong2 x); __attribute__((overloadable)) char3 convert_char3_sat(ulong3 x); __attribute__((overloadable)) char4 convert_char4_sat(ulong4 x); __attribute__((overloadable)) char8 convert_char8_sat(ulong8 x); __attribute__((overloadable)) char16 convert_char16_sat(ulong16 x); __attribute__((overloadable)) uchar convert_uchar_sat(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(ulong16 x); __attribute__((overloadable)) int convert_int_sat(ulong x); __attribute__((overloadable)) int2 convert_int2_sat(ulong2 x); __attribute__((overloadable)) int3 convert_int3_sat(ulong3 x); __attribute__((overloadable)) int4 convert_int4_sat(ulong4 x); __attribute__((overloadable)) int8 convert_int8_sat(ulong8 x); __attribute__((overloadable)) int16 convert_int16_sat(ulong16 x); __attribute__((overloadable)) uint convert_uint_sat(ulong x); __attribute__((overloadable)) uint2 convert_uint2_sat(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(ulong16 x); __attribute__((overloadable)) short convert_short_sat(ulong x); __attribute__((overloadable)) short2 convert_short2_sat(ulong2 x); __attribute__((overloadable)) short3 convert_short3_sat(ulong3 x); __attribute__((overloadable)) short4 convert_short4_sat(ulong4 x); __attribute__((overloadable)) short8 convert_short8_sat(ulong8 x); __attribute__((overloadable)) short16 convert_short16_sat(ulong16 x); __attribute__((overloadable)) ushort convert_ushort_sat(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(ulong16 x); __attribute__((overloadable)) long convert_long_sat(ulong x); __attribute__((overloadable)) long2 convert_long2_sat(ulong2 x); __attribute__((overloadable)) long3 convert_long3_sat(ulong3 x); __attribute__((overloadable)) long4 convert_long4_sat(ulong4 x); __attribute__((overloadable)) long8 convert_long8_sat(ulong8 x); __attribute__((overloadable)) long16 convert_long16_sat(ulong16 x); __attribute__((overloadable)) ulong convert_ulong_sat(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(ulong16 x); __attribute__((overloadable)) float convert_float_sat(ulong x); __attribute__((overloadable)) float2 convert_float2_sat(ulong2 x); __attribute__((overloadable)) float3 convert_float3_sat(ulong3 x); __attribute__((overloadable)) float4 convert_float4_sat(ulong4 x); __attribute__((overloadable)) float8 convert_float8_sat(ulong8 x); __attribute__((overloadable)) float16 convert_float16_sat(ulong16 x); __attribute__((overloadable)) double convert_double_sat(ulong x); __attribute__((overloadable)) double2 convert_double2_sat(ulong2 x); __attribute__((overloadable)) double3 convert_double3_sat(ulong3 x); __attribute__((overloadable)) double4 convert_double4_sat(ulong4 x); __attribute__((overloadable)) double8 convert_double8_sat(ulong8 x); __attribute__((overloadable)) double16 convert_double16_sat(ulong16 x); __attribute__((overloadable)) char convert_char_sat(float x); __attribute__((overloadable)) char2 convert_char2_sat(float2 x); __attribute__((overloadable)) char3 convert_char3_sat(float3 x); __attribute__((overloadable)) char4 convert_char4_sat(float4 x); __attribute__((overloadable)) char8 convert_char8_sat(float8 x); __attribute__((overloadable)) char16 convert_char16_sat(float16 x); __attribute__((overloadable)) uchar convert_uchar_sat(float x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(float16 x); __attribute__((overloadable)) int convert_int_sat(float x); __attribute__((overloadable)) int2 convert_int2_sat(float2 x); __attribute__((overloadable)) int3 convert_int3_sat(float3 x); __attribute__((overloadable)) int4 convert_int4_sat(float4 x); __attribute__((overloadable)) int8 convert_int8_sat(float8 x); __attribute__((overloadable)) int16 convert_int16_sat(float16 x); __attribute__((overloadable)) uint convert_uint_sat(float x); __attribute__((overloadable)) uint2 convert_uint2_sat(float2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(float3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(float4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(float8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(float16 x); __attribute__((overloadable)) short convert_short_sat(float x); __attribute__((overloadable)) short2 convert_short2_sat(float2 x); __attribute__((overloadable)) short3 convert_short3_sat(float3 x); __attribute__((overloadable)) short4 convert_short4_sat(float4 x); __attribute__((overloadable)) short8 convert_short8_sat(float8 x); __attribute__((overloadable)) short16 convert_short16_sat(float16 x); __attribute__((overloadable)) ushort convert_ushort_sat(float x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(float16 x); __attribute__((overloadable)) long convert_long_sat(float x); __attribute__((overloadable)) long2 convert_long2_sat(float2 x); __attribute__((overloadable)) long3 convert_long3_sat(float3 x); __attribute__((overloadable)) long4 convert_long4_sat(float4 x); __attribute__((overloadable)) long8 convert_long8_sat(float8 x); __attribute__((overloadable)) long16 convert_long16_sat(float16 x); __attribute__((overloadable)) ulong convert_ulong_sat(float x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(float16 x); __attribute__((overloadable)) float convert_float_sat(float x); __attribute__((overloadable)) float2 convert_float2_sat(float2 x); __attribute__((overloadable)) float3 convert_float3_sat(float3 x); __attribute__((overloadable)) float4 convert_float4_sat(float4 x); __attribute__((overloadable)) float8 convert_float8_sat(float8 x); __attribute__((overloadable)) float16 convert_float16_sat(float16 x); __attribute__((overloadable)) double convert_double_sat(float x); __attribute__((overloadable)) double2 convert_double2_sat(float2 x); __attribute__((overloadable)) double3 convert_double3_sat(float3 x); __attribute__((overloadable)) double4 convert_double4_sat(float4 x); __attribute__((overloadable)) double8 convert_double8_sat(float8 x); __attribute__((overloadable)) double16 convert_double16_sat(float16 x); __attribute__((overloadable)) char convert_char_sat(double x); __attribute__((overloadable)) char2 convert_char2_sat(double2 x); __attribute__((overloadable)) char3 convert_char3_sat(double3 x); __attribute__((overloadable)) char4 convert_char4_sat(double4 x); __attribute__((overloadable)) char8 convert_char8_sat(double8 x); __attribute__((overloadable)) char16 convert_char16_sat(double16 x); __attribute__((overloadable)) uchar convert_uchar_sat(double x); __attribute__((overloadable)) uchar2 convert_uchar2_sat(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3_sat(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4_sat(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8_sat(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16_sat(double16 x); __attribute__((overloadable)) int convert_int_sat(double x); __attribute__((overloadable)) int2 convert_int2_sat(double2 x); __attribute__((overloadable)) int3 convert_int3_sat(double3 x); __attribute__((overloadable)) int4 convert_int4_sat(double4 x); __attribute__((overloadable)) int8 convert_int8_sat(double8 x); __attribute__((overloadable)) int16 convert_int16_sat(double16 x); __attribute__((overloadable)) uint convert_uint_sat(double x); __attribute__((overloadable)) uint2 convert_uint2_sat(double2 x); __attribute__((overloadable)) uint3 convert_uint3_sat(double3 x); __attribute__((overloadable)) uint4 convert_uint4_sat(double4 x); __attribute__((overloadable)) uint8 convert_uint8_sat(double8 x); __attribute__((overloadable)) uint16 convert_uint16_sat(double16 x); __attribute__((overloadable)) short convert_short_sat(double x); __attribute__((overloadable)) short2 convert_short2_sat(double2 x); __attribute__((overloadable)) short3 convert_short3_sat(double3 x); __attribute__((overloadable)) short4 convert_short4_sat(double4 x); __attribute__((overloadable)) short8 convert_short8_sat(double8 x); __attribute__((overloadable)) short16 convert_short16_sat(double16 x); __attribute__((overloadable)) ushort convert_ushort_sat(double x); __attribute__((overloadable)) ushort2 convert_ushort2_sat(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3_sat(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4_sat(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8_sat(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16_sat(double16 x); __attribute__((overloadable)) long convert_long_sat(double x); __attribute__((overloadable)) long2 convert_long2_sat(double2 x); __attribute__((overloadable)) long3 convert_long3_sat(double3 x); __attribute__((overloadable)) long4 convert_long4_sat(double4 x); __attribute__((overloadable)) long8 convert_long8_sat(double8 x); __attribute__((overloadable)) long16 convert_long16_sat(double16 x); __attribute__((overloadable)) ulong convert_ulong_sat(double x); __attribute__((overloadable)) ulong2 convert_ulong2_sat(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3_sat(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4_sat(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8_sat(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16_sat(double16 x); __attribute__((overloadable)) float convert_float_sat(double x); __attribute__((overloadable)) float2 convert_float2_sat(double2 x); __attribute__((overloadable)) float3 convert_float3_sat(double3 x); __attribute__((overloadable)) float4 convert_float4_sat(double4 x); __attribute__((overloadable)) float8 convert_float8_sat(double8 x); __attribute__((overloadable)) float16 convert_float16_sat(double16 x); __attribute__((overloadable)) double convert_double_sat(double x); __attribute__((overloadable)) double2 convert_double2_sat(double2 x); __attribute__((overloadable)) double3 convert_double3_sat(double3 x); __attribute__((overloadable)) double4 convert_double4_sat(double4 x); __attribute__((overloadable)) double8 convert_double8_sat(double8 x); __attribute__((overloadable)) double16 convert_double16_sat(double16 x); __attribute__((overloadable)) char convert_char(char x); __attribute__((overloadable)) char2 convert_char2(char2 x); __attribute__((overloadable)) char3 convert_char3(char3 x); __attribute__((overloadable)) char4 convert_char4(char4 x); __attribute__((overloadable)) char8 convert_char8(char8 x); __attribute__((overloadable)) char16 convert_char16(char16 x); __attribute__((overloadable)) uchar convert_uchar(char x); __attribute__((overloadable)) uchar2 convert_uchar2(char2 x); __attribute__((overloadable)) uchar3 convert_uchar3(char3 x); __attribute__((overloadable)) uchar4 convert_uchar4(char4 x); __attribute__((overloadable)) uchar8 convert_uchar8(char8 x); __attribute__((overloadable)) uchar16 convert_uchar16(char16 x); __attribute__((overloadable)) int convert_int(char x); __attribute__((overloadable)) int2 convert_int2(char2 x); __attribute__((overloadable)) int3 convert_int3(char3 x); __attribute__((overloadable)) int4 convert_int4(char4 x); __attribute__((overloadable)) int8 convert_int8(char8 x); __attribute__((overloadable)) int16 convert_int16(char16 x); __attribute__((overloadable)) uint convert_uint(char x); __attribute__((overloadable)) uint2 convert_uint2(char2 x); __attribute__((overloadable)) uint3 convert_uint3(char3 x); __attribute__((overloadable)) uint4 convert_uint4(char4 x); __attribute__((overloadable)) uint8 convert_uint8(char8 x); __attribute__((overloadable)) uint16 convert_uint16(char16 x); __attribute__((overloadable)) short convert_short(char x); __attribute__((overloadable)) short2 convert_short2(char2 x); __attribute__((overloadable)) short3 convert_short3(char3 x); __attribute__((overloadable)) short4 convert_short4(char4 x); __attribute__((overloadable)) short8 convert_short8(char8 x); __attribute__((overloadable)) short16 convert_short16(char16 x); __attribute__((overloadable)) ushort convert_ushort(char x); __attribute__((overloadable)) ushort2 convert_ushort2(char2 x); __attribute__((overloadable)) ushort3 convert_ushort3(char3 x); __attribute__((overloadable)) ushort4 convert_ushort4(char4 x); __attribute__((overloadable)) ushort8 convert_ushort8(char8 x); __attribute__((overloadable)) ushort16 convert_ushort16(char16 x); __attribute__((overloadable)) long convert_long(char x); __attribute__((overloadable)) long2 convert_long2(char2 x); __attribute__((overloadable)) long3 convert_long3(char3 x); __attribute__((overloadable)) long4 convert_long4(char4 x); __attribute__((overloadable)) long8 convert_long8(char8 x); __attribute__((overloadable)) long16 convert_long16(char16 x); __attribute__((overloadable)) ulong convert_ulong(char x); __attribute__((overloadable)) ulong2 convert_ulong2(char2 x); __attribute__((overloadable)) ulong3 convert_ulong3(char3 x); __attribute__((overloadable)) ulong4 convert_ulong4(char4 x); __attribute__((overloadable)) ulong8 convert_ulong8(char8 x); __attribute__((overloadable)) ulong16 convert_ulong16(char16 x); __attribute__((overloadable)) float convert_float(char x); __attribute__((overloadable)) float2 convert_float2(char2 x); __attribute__((overloadable)) float3 convert_float3(char3 x); __attribute__((overloadable)) float4 convert_float4(char4 x); __attribute__((overloadable)) float8 convert_float8(char8 x); __attribute__((overloadable)) float16 convert_float16(char16 x); __attribute__((overloadable)) double convert_double(char x); __attribute__((overloadable)) double2 convert_double2(char2 x); __attribute__((overloadable)) double3 convert_double3(char3 x); __attribute__((overloadable)) double4 convert_double4(char4 x); __attribute__((overloadable)) double8 convert_double8(char8 x); __attribute__((overloadable)) double16 convert_double16(char16 x); __attribute__((overloadable)) char convert_char(uchar x); __attribute__((overloadable)) char2 convert_char2(uchar2 x); __attribute__((overloadable)) char3 convert_char3(uchar3 x); __attribute__((overloadable)) char4 convert_char4(uchar4 x); __attribute__((overloadable)) char8 convert_char8(uchar8 x); __attribute__((overloadable)) char16 convert_char16(uchar16 x); __attribute__((overloadable)) uchar convert_uchar(uchar x); __attribute__((overloadable)) uchar2 convert_uchar2(uchar2 x); __attribute__((overloadable)) uchar3 convert_uchar3(uchar3 x); __attribute__((overloadable)) uchar4 convert_uchar4(uchar4 x); __attribute__((overloadable)) uchar8 convert_uchar8(uchar8 x); __attribute__((overloadable)) uchar16 convert_uchar16(uchar16 x); __attribute__((overloadable)) int convert_int(uchar x); __attribute__((overloadable)) int2 convert_int2(uchar2 x); __attribute__((overloadable)) int3 convert_int3(uchar3 x); __attribute__((overloadable)) int4 convert_int4(uchar4 x); __attribute__((overloadable)) int8 convert_int8(uchar8 x); __attribute__((overloadable)) int16 convert_int16(uchar16 x); __attribute__((overloadable)) uint convert_uint(uchar x); __attribute__((overloadable)) uint2 convert_uint2(uchar2 x); __attribute__((overloadable)) uint3 convert_uint3(uchar3 x); __attribute__((overloadable)) uint4 convert_uint4(uchar4 x); __attribute__((overloadable)) uint8 convert_uint8(uchar8 x); __attribute__((overloadable)) uint16 convert_uint16(uchar16 x); __attribute__((overloadable)) short convert_short(uchar x); __attribute__((overloadable)) short2 convert_short2(uchar2 x); __attribute__((overloadable)) short3 convert_short3(uchar3 x); __attribute__((overloadable)) short4 convert_short4(uchar4 x); __attribute__((overloadable)) short8 convert_short8(uchar8 x); __attribute__((overloadable)) short16 convert_short16(uchar16 x); __attribute__((overloadable)) ushort convert_ushort(uchar x); __attribute__((overloadable)) ushort2 convert_ushort2(uchar2 x); __attribute__((overloadable)) ushort3 convert_ushort3(uchar3 x); __attribute__((overloadable)) ushort4 convert_ushort4(uchar4 x); __attribute__((overloadable)) ushort8 convert_ushort8(uchar8 x); __attribute__((overloadable)) ushort16 convert_ushort16(uchar16 x); __attribute__((overloadable)) long convert_long(uchar x); __attribute__((overloadable)) long2 convert_long2(uchar2 x); __attribute__((overloadable)) long3 convert_long3(uchar3 x); __attribute__((overloadable)) long4 convert_long4(uchar4 x); __attribute__((overloadable)) long8 convert_long8(uchar8 x); __attribute__((overloadable)) long16 convert_long16(uchar16 x); __attribute__((overloadable)) ulong convert_ulong(uchar x); __attribute__((overloadable)) ulong2 convert_ulong2(uchar2 x); __attribute__((overloadable)) ulong3 convert_ulong3(uchar3 x); __attribute__((overloadable)) ulong4 convert_ulong4(uchar4 x); __attribute__((overloadable)) ulong8 convert_ulong8(uchar8 x); __attribute__((overloadable)) ulong16 convert_ulong16(uchar16 x); __attribute__((overloadable)) float convert_float(uchar x); __attribute__((overloadable)) float2 convert_float2(uchar2 x); __attribute__((overloadable)) float3 convert_float3(uchar3 x); __attribute__((overloadable)) float4 convert_float4(uchar4 x); __attribute__((overloadable)) float8 convert_float8(uchar8 x); __attribute__((overloadable)) float16 convert_float16(uchar16 x); __attribute__((overloadable)) double convert_double(uchar x); __attribute__((overloadable)) double2 convert_double2(uchar2 x); __attribute__((overloadable)) double3 convert_double3(uchar3 x); __attribute__((overloadable)) double4 convert_double4(uchar4 x); __attribute__((overloadable)) double8 convert_double8(uchar8 x); __attribute__((overloadable)) double16 convert_double16(uchar16 x); __attribute__((overloadable)) char convert_char(int x); __attribute__((overloadable)) char2 convert_char2(int2 x); __attribute__((overloadable)) char3 convert_char3(int3 x); __attribute__((overloadable)) char4 convert_char4(int4 x); __attribute__((overloadable)) char8 convert_char8(int8 x); __attribute__((overloadable)) char16 convert_char16(int16 x); __attribute__((overloadable)) uchar convert_uchar(int x); __attribute__((overloadable)) uchar2 convert_uchar2(int2 x); __attribute__((overloadable)) uchar3 convert_uchar3(int3 x); __attribute__((overloadable)) uchar4 convert_uchar4(int4 x); __attribute__((overloadable)) uchar8 convert_uchar8(int8 x); __attribute__((overloadable)) uchar16 convert_uchar16(int16 x); __attribute__((overloadable)) int convert_int(int x); __attribute__((overloadable)) int2 convert_int2(int2 x); __attribute__((overloadable)) int3 convert_int3(int3 x); __attribute__((overloadable)) int4 convert_int4(int4 x); __attribute__((overloadable)) int8 convert_int8(int8 x); __attribute__((overloadable)) int16 convert_int16(int16 x); __attribute__((overloadable)) uint convert_uint(int x); __attribute__((overloadable)) uint2 convert_uint2(int2 x); __attribute__((overloadable)) uint3 convert_uint3(int3 x); __attribute__((overloadable)) uint4 convert_uint4(int4 x); __attribute__((overloadable)) uint8 convert_uint8(int8 x); __attribute__((overloadable)) uint16 convert_uint16(int16 x); __attribute__((overloadable)) short convert_short(int x); __attribute__((overloadable)) short2 convert_short2(int2 x); __attribute__((overloadable)) short3 convert_short3(int3 x); __attribute__((overloadable)) short4 convert_short4(int4 x); __attribute__((overloadable)) short8 convert_short8(int8 x); __attribute__((overloadable)) short16 convert_short16(int16 x); __attribute__((overloadable)) ushort convert_ushort(int x); __attribute__((overloadable)) ushort2 convert_ushort2(int2 x); __attribute__((overloadable)) ushort3 convert_ushort3(int3 x); __attribute__((overloadable)) ushort4 convert_ushort4(int4 x); __attribute__((overloadable)) ushort8 convert_ushort8(int8 x); __attribute__((overloadable)) ushort16 convert_ushort16(int16 x); __attribute__((overloadable)) long convert_long(int x); __attribute__((overloadable)) long2 convert_long2(int2 x); __attribute__((overloadable)) long3 convert_long3(int3 x); __attribute__((overloadable)) long4 convert_long4(int4 x); __attribute__((overloadable)) long8 convert_long8(int8 x); __attribute__((overloadable)) long16 convert_long16(int16 x); __attribute__((overloadable)) ulong convert_ulong(int x); __attribute__((overloadable)) ulong2 convert_ulong2(int2 x); __attribute__((overloadable)) ulong3 convert_ulong3(int3 x); __attribute__((overloadable)) ulong4 convert_ulong4(int4 x); __attribute__((overloadable)) ulong8 convert_ulong8(int8 x); __attribute__((overloadable)) ulong16 convert_ulong16(int16 x); __attribute__((overloadable)) float convert_float(int x); __attribute__((overloadable)) float2 convert_float2(int2 x); __attribute__((overloadable)) float3 convert_float3(int3 x); __attribute__((overloadable)) float4 convert_float4(int4 x); __attribute__((overloadable)) float8 convert_float8(int8 x); __attribute__((overloadable)) float16 convert_float16(int16 x); __attribute__((overloadable)) double convert_double(int x); __attribute__((overloadable)) double2 convert_double2(int2 x); __attribute__((overloadable)) double3 convert_double3(int3 x); __attribute__((overloadable)) double4 convert_double4(int4 x); __attribute__((overloadable)) double8 convert_double8(int8 x); __attribute__((overloadable)) double16 convert_double16(int16 x); __attribute__((overloadable)) char convert_char(uint x); __attribute__((overloadable)) char2 convert_char2(uint2 x); __attribute__((overloadable)) char3 convert_char3(uint3 x); __attribute__((overloadable)) char4 convert_char4(uint4 x); __attribute__((overloadable)) char8 convert_char8(uint8 x); __attribute__((overloadable)) char16 convert_char16(uint16 x); __attribute__((overloadable)) uchar convert_uchar(uint x); __attribute__((overloadable)) uchar2 convert_uchar2(uint2 x); __attribute__((overloadable)) uchar3 convert_uchar3(uint3 x); __attribute__((overloadable)) uchar4 convert_uchar4(uint4 x); __attribute__((overloadable)) uchar8 convert_uchar8(uint8 x); __attribute__((overloadable)) uchar16 convert_uchar16(uint16 x); __attribute__((overloadable)) int convert_int(uint x); __attribute__((overloadable)) int2 convert_int2(uint2 x); __attribute__((overloadable)) int3 convert_int3(uint3 x); __attribute__((overloadable)) int4 convert_int4(uint4 x); __attribute__((overloadable)) int8 convert_int8(uint8 x); __attribute__((overloadable)) int16 convert_int16(uint16 x); __attribute__((overloadable)) uint convert_uint(uint x); __attribute__((overloadable)) uint2 convert_uint2(uint2 x); __attribute__((overloadable)) uint3 convert_uint3(uint3 x); __attribute__((overloadable)) uint4 convert_uint4(uint4 x); __attribute__((overloadable)) uint8 convert_uint8(uint8 x); __attribute__((overloadable)) uint16 convert_uint16(uint16 x); __attribute__((overloadable)) short convert_short(uint x); __attribute__((overloadable)) short2 convert_short2(uint2 x); __attribute__((overloadable)) short3 convert_short3(uint3 x); __attribute__((overloadable)) short4 convert_short4(uint4 x); __attribute__((overloadable)) short8 convert_short8(uint8 x); __attribute__((overloadable)) short16 convert_short16(uint16 x); __attribute__((overloadable)) ushort convert_ushort(uint x); __attribute__((overloadable)) ushort2 convert_ushort2(uint2 x); __attribute__((overloadable)) ushort3 convert_ushort3(uint3 x); __attribute__((overloadable)) ushort4 convert_ushort4(uint4 x); __attribute__((overloadable)) ushort8 convert_ushort8(uint8 x); __attribute__((overloadable)) ushort16 convert_ushort16(uint16 x); __attribute__((overloadable)) long convert_long(uint x); __attribute__((overloadable)) long2 convert_long2(uint2 x); __attribute__((overloadable)) long3 convert_long3(uint3 x); __attribute__((overloadable)) long4 convert_long4(uint4 x); __attribute__((overloadable)) long8 convert_long8(uint8 x); __attribute__((overloadable)) long16 convert_long16(uint16 x); __attribute__((overloadable)) ulong convert_ulong(uint x); __attribute__((overloadable)) ulong2 convert_ulong2(uint2 x); __attribute__((overloadable)) ulong3 convert_ulong3(uint3 x); __attribute__((overloadable)) ulong4 convert_ulong4(uint4 x); __attribute__((overloadable)) ulong8 convert_ulong8(uint8 x); __attribute__((overloadable)) ulong16 convert_ulong16(uint16 x); __attribute__((overloadable)) float convert_float(uint x); __attribute__((overloadable)) float2 convert_float2(uint2 x); __attribute__((overloadable)) float3 convert_float3(uint3 x); __attribute__((overloadable)) float4 convert_float4(uint4 x); __attribute__((overloadable)) float8 convert_float8(uint8 x); __attribute__((overloadable)) float16 convert_float16(uint16 x); __attribute__((overloadable)) double convert_double(uint x); __attribute__((overloadable)) double2 convert_double2(uint2 x); __attribute__((overloadable)) double3 convert_double3(uint3 x); __attribute__((overloadable)) double4 convert_double4(uint4 x); __attribute__((overloadable)) double8 convert_double8(uint8 x); __attribute__((overloadable)) double16 convert_double16(uint16 x); __attribute__((overloadable)) char convert_char(short x); __attribute__((overloadable)) char2 convert_char2(short2 x); __attribute__((overloadable)) char3 convert_char3(short3 x); __attribute__((overloadable)) char4 convert_char4(short4 x); __attribute__((overloadable)) char8 convert_char8(short8 x); __attribute__((overloadable)) char16 convert_char16(short16 x); __attribute__((overloadable)) uchar convert_uchar(short x); __attribute__((overloadable)) uchar2 convert_uchar2(short2 x); __attribute__((overloadable)) uchar3 convert_uchar3(short3 x); __attribute__((overloadable)) uchar4 convert_uchar4(short4 x); __attribute__((overloadable)) uchar8 convert_uchar8(short8 x); __attribute__((overloadable)) uchar16 convert_uchar16(short16 x); __attribute__((overloadable)) int convert_int(short x); __attribute__((overloadable)) int2 convert_int2(short2 x); __attribute__((overloadable)) int3 convert_int3(short3 x); __attribute__((overloadable)) int4 convert_int4(short4 x); __attribute__((overloadable)) int8 convert_int8(short8 x); __attribute__((overloadable)) int16 convert_int16(short16 x); __attribute__((overloadable)) uint convert_uint(short x); __attribute__((overloadable)) uint2 convert_uint2(short2 x); __attribute__((overloadable)) uint3 convert_uint3(short3 x); __attribute__((overloadable)) uint4 convert_uint4(short4 x); __attribute__((overloadable)) uint8 convert_uint8(short8 x); __attribute__((overloadable)) uint16 convert_uint16(short16 x); __attribute__((overloadable)) short convert_short(short x); __attribute__((overloadable)) short2 convert_short2(short2 x); __attribute__((overloadable)) short3 convert_short3(short3 x); __attribute__((overloadable)) short4 convert_short4(short4 x); __attribute__((overloadable)) short8 convert_short8(short8 x); __attribute__((overloadable)) short16 convert_short16(short16 x); __attribute__((overloadable)) ushort convert_ushort(short x); __attribute__((overloadable)) ushort2 convert_ushort2(short2 x); __attribute__((overloadable)) ushort3 convert_ushort3(short3 x); __attribute__((overloadable)) ushort4 convert_ushort4(short4 x); __attribute__((overloadable)) ushort8 convert_ushort8(short8 x); __attribute__((overloadable)) ushort16 convert_ushort16(short16 x); __attribute__((overloadable)) long convert_long(short x); __attribute__((overloadable)) long2 convert_long2(short2 x); __attribute__((overloadable)) long3 convert_long3(short3 x); __attribute__((overloadable)) long4 convert_long4(short4 x); __attribute__((overloadable)) long8 convert_long8(short8 x); __attribute__((overloadable)) long16 convert_long16(short16 x); __attribute__((overloadable)) ulong convert_ulong(short x); __attribute__((overloadable)) ulong2 convert_ulong2(short2 x); __attribute__((overloadable)) ulong3 convert_ulong3(short3 x); __attribute__((overloadable)) ulong4 convert_ulong4(short4 x); __attribute__((overloadable)) ulong8 convert_ulong8(short8 x); __attribute__((overloadable)) ulong16 convert_ulong16(short16 x); __attribute__((overloadable)) float convert_float(short x); __attribute__((overloadable)) float2 convert_float2(short2 x); __attribute__((overloadable)) float3 convert_float3(short3 x); __attribute__((overloadable)) float4 convert_float4(short4 x); __attribute__((overloadable)) float8 convert_float8(short8 x); __attribute__((overloadable)) float16 convert_float16(short16 x); __attribute__((overloadable)) double convert_double(short x); __attribute__((overloadable)) double2 convert_double2(short2 x); __attribute__((overloadable)) double3 convert_double3(short3 x); __attribute__((overloadable)) double4 convert_double4(short4 x); __attribute__((overloadable)) double8 convert_double8(short8 x); __attribute__((overloadable)) double16 convert_double16(short16 x); __attribute__((overloadable)) char convert_char(ushort x); __attribute__((overloadable)) char2 convert_char2(ushort2 x); __attribute__((overloadable)) char3 convert_char3(ushort3 x); __attribute__((overloadable)) char4 convert_char4(ushort4 x); __attribute__((overloadable)) char8 convert_char8(ushort8 x); __attribute__((overloadable)) char16 convert_char16(ushort16 x); __attribute__((overloadable)) uchar convert_uchar(ushort x); __attribute__((overloadable)) uchar2 convert_uchar2(ushort2 x); __attribute__((overloadable)) uchar3 convert_uchar3(ushort3 x); __attribute__((overloadable)) uchar4 convert_uchar4(ushort4 x); __attribute__((overloadable)) uchar8 convert_uchar8(ushort8 x); __attribute__((overloadable)) uchar16 convert_uchar16(ushort16 x); __attribute__((overloadable)) int convert_int(ushort x); __attribute__((overloadable)) int2 convert_int2(ushort2 x); __attribute__((overloadable)) int3 convert_int3(ushort3 x); __attribute__((overloadable)) int4 convert_int4(ushort4 x); __attribute__((overloadable)) int8 convert_int8(ushort8 x); __attribute__((overloadable)) int16 convert_int16(ushort16 x); __attribute__((overloadable)) uint convert_uint(ushort x); __attribute__((overloadable)) uint2 convert_uint2(ushort2 x); __attribute__((overloadable)) uint3 convert_uint3(ushort3 x); __attribute__((overloadable)) uint4 convert_uint4(ushort4 x); __attribute__((overloadable)) uint8 convert_uint8(ushort8 x); __attribute__((overloadable)) uint16 convert_uint16(ushort16 x); __attribute__((overloadable)) short convert_short(ushort x); __attribute__((overloadable)) short2 convert_short2(ushort2 x); __attribute__((overloadable)) short3 convert_short3(ushort3 x); __attribute__((overloadable)) short4 convert_short4(ushort4 x); __attribute__((overloadable)) short8 convert_short8(ushort8 x); __attribute__((overloadable)) short16 convert_short16(ushort16 x); __attribute__((overloadable)) ushort convert_ushort(ushort x); __attribute__((overloadable)) ushort2 convert_ushort2(ushort2 x); __attribute__((overloadable)) ushort3 convert_ushort3(ushort3 x); __attribute__((overloadable)) ushort4 convert_ushort4(ushort4 x); __attribute__((overloadable)) ushort8 convert_ushort8(ushort8 x); __attribute__((overloadable)) ushort16 convert_ushort16(ushort16 x); __attribute__((overloadable)) long convert_long(ushort x); __attribute__((overloadable)) long2 convert_long2(ushort2 x); __attribute__((overloadable)) long3 convert_long3(ushort3 x); __attribute__((overloadable)) long4 convert_long4(ushort4 x); __attribute__((overloadable)) long8 convert_long8(ushort8 x); __attribute__((overloadable)) long16 convert_long16(ushort16 x); __attribute__((overloadable)) ulong convert_ulong(ushort x); __attribute__((overloadable)) ulong2 convert_ulong2(ushort2 x); __attribute__((overloadable)) ulong3 convert_ulong3(ushort3 x); __attribute__((overloadable)) ulong4 convert_ulong4(ushort4 x); __attribute__((overloadable)) ulong8 convert_ulong8(ushort8 x); __attribute__((overloadable)) ulong16 convert_ulong16(ushort16 x); __attribute__((overloadable)) float convert_float(ushort x); __attribute__((overloadable)) float2 convert_float2(ushort2 x); __attribute__((overloadable)) float3 convert_float3(ushort3 x); __attribute__((overloadable)) float4 convert_float4(ushort4 x); __attribute__((overloadable)) float8 convert_float8(ushort8 x); __attribute__((overloadable)) float16 convert_float16(ushort16 x); __attribute__((overloadable)) double convert_double(ushort x); __attribute__((overloadable)) double2 convert_double2(ushort2 x); __attribute__((overloadable)) double3 convert_double3(ushort3 x); __attribute__((overloadable)) double4 convert_double4(ushort4 x); __attribute__((overloadable)) double8 convert_double8(ushort8 x); __attribute__((overloadable)) double16 convert_double16(ushort16 x); __attribute__((overloadable)) char convert_char(long x); __attribute__((overloadable)) char2 convert_char2(long2 x); __attribute__((overloadable)) char3 convert_char3(long3 x); __attribute__((overloadable)) char4 convert_char4(long4 x); __attribute__((overloadable)) char8 convert_char8(long8 x); __attribute__((overloadable)) char16 convert_char16(long16 x); __attribute__((overloadable)) uchar convert_uchar(long x); __attribute__((overloadable)) uchar2 convert_uchar2(long2 x); __attribute__((overloadable)) uchar3 convert_uchar3(long3 x); __attribute__((overloadable)) uchar4 convert_uchar4(long4 x); __attribute__((overloadable)) uchar8 convert_uchar8(long8 x); __attribute__((overloadable)) uchar16 convert_uchar16(long16 x); __attribute__((overloadable)) int convert_int(long x); __attribute__((overloadable)) int2 convert_int2(long2 x); __attribute__((overloadable)) int3 convert_int3(long3 x); __attribute__((overloadable)) int4 convert_int4(long4 x); __attribute__((overloadable)) int8 convert_int8(long8 x); __attribute__((overloadable)) int16 convert_int16(long16 x); __attribute__((overloadable)) uint convert_uint(long x); __attribute__((overloadable)) uint2 convert_uint2(long2 x); __attribute__((overloadable)) uint3 convert_uint3(long3 x); __attribute__((overloadable)) uint4 convert_uint4(long4 x); __attribute__((overloadable)) uint8 convert_uint8(long8 x); __attribute__((overloadable)) uint16 convert_uint16(long16 x); __attribute__((overloadable)) short convert_short(long x); __attribute__((overloadable)) short2 convert_short2(long2 x); __attribute__((overloadable)) short3 convert_short3(long3 x); __attribute__((overloadable)) short4 convert_short4(long4 x); __attribute__((overloadable)) short8 convert_short8(long8 x); __attribute__((overloadable)) short16 convert_short16(long16 x); __attribute__((overloadable)) ushort convert_ushort(long x); __attribute__((overloadable)) ushort2 convert_ushort2(long2 x); __attribute__((overloadable)) ushort3 convert_ushort3(long3 x); __attribute__((overloadable)) ushort4 convert_ushort4(long4 x); __attribute__((overloadable)) ushort8 convert_ushort8(long8 x); __attribute__((overloadable)) ushort16 convert_ushort16(long16 x); __attribute__((overloadable)) long convert_long(long x); __attribute__((overloadable)) long2 convert_long2(long2 x); __attribute__((overloadable)) long3 convert_long3(long3 x); __attribute__((overloadable)) long4 convert_long4(long4 x); __attribute__((overloadable)) long8 convert_long8(long8 x); __attribute__((overloadable)) long16 convert_long16(long16 x); __attribute__((overloadable)) ulong convert_ulong(long x); __attribute__((overloadable)) ulong2 convert_ulong2(long2 x); __attribute__((overloadable)) ulong3 convert_ulong3(long3 x); __attribute__((overloadable)) ulong4 convert_ulong4(long4 x); __attribute__((overloadable)) ulong8 convert_ulong8(long8 x); __attribute__((overloadable)) ulong16 convert_ulong16(long16 x); __attribute__((overloadable)) float convert_float(long x); __attribute__((overloadable)) float2 convert_float2(long2 x); __attribute__((overloadable)) float3 convert_float3(long3 x); __attribute__((overloadable)) float4 convert_float4(long4 x); __attribute__((overloadable)) float8 convert_float8(long8 x); __attribute__((overloadable)) float16 convert_float16(long16 x); __attribute__((overloadable)) double convert_double(long x); __attribute__((overloadable)) double2 convert_double2(long2 x); __attribute__((overloadable)) double3 convert_double3(long3 x); __attribute__((overloadable)) double4 convert_double4(long4 x); __attribute__((overloadable)) double8 convert_double8(long8 x); __attribute__((overloadable)) double16 convert_double16(long16 x); __attribute__((overloadable)) char convert_char(ulong x); __attribute__((overloadable)) char2 convert_char2(ulong2 x); __attribute__((overloadable)) char3 convert_char3(ulong3 x); __attribute__((overloadable)) char4 convert_char4(ulong4 x); __attribute__((overloadable)) char8 convert_char8(ulong8 x); __attribute__((overloadable)) char16 convert_char16(ulong16 x); __attribute__((overloadable)) uchar convert_uchar(ulong x); __attribute__((overloadable)) uchar2 convert_uchar2(ulong2 x); __attribute__((overloadable)) uchar3 convert_uchar3(ulong3 x); __attribute__((overloadable)) uchar4 convert_uchar4(ulong4 x); __attribute__((overloadable)) uchar8 convert_uchar8(ulong8 x); __attribute__((overloadable)) uchar16 convert_uchar16(ulong16 x); __attribute__((overloadable)) int convert_int(ulong x); __attribute__((overloadable)) int2 convert_int2(ulong2 x); __attribute__((overloadable)) int3 convert_int3(ulong3 x); __attribute__((overloadable)) int4 convert_int4(ulong4 x); __attribute__((overloadable)) int8 convert_int8(ulong8 x); __attribute__((overloadable)) int16 convert_int16(ulong16 x); __attribute__((overloadable)) uint convert_uint(ulong x); __attribute__((overloadable)) uint2 convert_uint2(ulong2 x); __attribute__((overloadable)) uint3 convert_uint3(ulong3 x); __attribute__((overloadable)) uint4 convert_uint4(ulong4 x); __attribute__((overloadable)) uint8 convert_uint8(ulong8 x); __attribute__((overloadable)) uint16 convert_uint16(ulong16 x); __attribute__((overloadable)) short convert_short(ulong x); __attribute__((overloadable)) short2 convert_short2(ulong2 x); __attribute__((overloadable)) short3 convert_short3(ulong3 x); __attribute__((overloadable)) short4 convert_short4(ulong4 x); __attribute__((overloadable)) short8 convert_short8(ulong8 x); __attribute__((overloadable)) short16 convert_short16(ulong16 x); __attribute__((overloadable)) ushort convert_ushort(ulong x); __attribute__((overloadable)) ushort2 convert_ushort2(ulong2 x); __attribute__((overloadable)) ushort3 convert_ushort3(ulong3 x); __attribute__((overloadable)) ushort4 convert_ushort4(ulong4 x); __attribute__((overloadable)) ushort8 convert_ushort8(ulong8 x); __attribute__((overloadable)) ushort16 convert_ushort16(ulong16 x); __attribute__((overloadable)) long convert_long(ulong x); __attribute__((overloadable)) long2 convert_long2(ulong2 x); __attribute__((overloadable)) long3 convert_long3(ulong3 x); __attribute__((overloadable)) long4 convert_long4(ulong4 x); __attribute__((overloadable)) long8 convert_long8(ulong8 x); __attribute__((overloadable)) long16 convert_long16(ulong16 x); __attribute__((overloadable)) ulong convert_ulong(ulong x); __attribute__((overloadable)) ulong2 convert_ulong2(ulong2 x); __attribute__((overloadable)) ulong3 convert_ulong3(ulong3 x); __attribute__((overloadable)) ulong4 convert_ulong4(ulong4 x); __attribute__((overloadable)) ulong8 convert_ulong8(ulong8 x); __attribute__((overloadable)) ulong16 convert_ulong16(ulong16 x); __attribute__((overloadable)) float convert_float(ulong x); __attribute__((overloadable)) float2 convert_float2(ulong2 x); __attribute__((overloadable)) float3 convert_float3(ulong3 x); __attribute__((overloadable)) float4 convert_float4(ulong4 x); __attribute__((overloadable)) float8 convert_float8(ulong8 x); __attribute__((overloadable)) float16 convert_float16(ulong16 x); __attribute__((overloadable)) double convert_double(ulong x); __attribute__((overloadable)) double2 convert_double2(ulong2 x); __attribute__((overloadable)) double3 convert_double3(ulong3 x); __attribute__((overloadable)) double4 convert_double4(ulong4 x); __attribute__((overloadable)) double8 convert_double8(ulong8 x); __attribute__((overloadable)) double16 convert_double16(ulong16 x); __attribute__((overloadable)) char convert_char(float x); __attribute__((overloadable)) char2 convert_char2(float2 x); __attribute__((overloadable)) char3 convert_char3(float3 x); __attribute__((overloadable)) char4 convert_char4(float4 x); __attribute__((overloadable)) char8 convert_char8(float8 x); __attribute__((overloadable)) char16 convert_char16(float16 x); __attribute__((overloadable)) uchar convert_uchar(float x); __attribute__((overloadable)) uchar2 convert_uchar2(float2 x); __attribute__((overloadable)) uchar3 convert_uchar3(float3 x); __attribute__((overloadable)) uchar4 convert_uchar4(float4 x); __attribute__((overloadable)) uchar8 convert_uchar8(float8 x); __attribute__((overloadable)) uchar16 convert_uchar16(float16 x); __attribute__((overloadable)) int convert_int(float x); __attribute__((overloadable)) int2 convert_int2(float2 x); __attribute__((overloadable)) int3 convert_int3(float3 x); __attribute__((overloadable)) int4 convert_int4(float4 x); __attribute__((overloadable)) int8 convert_int8(float8 x); __attribute__((overloadable)) int16 convert_int16(float16 x); __attribute__((overloadable)) uint convert_uint(float x); __attribute__((overloadable)) uint2 convert_uint2(float2 x); __attribute__((overloadable)) uint3 convert_uint3(float3 x); __attribute__((overloadable)) uint4 convert_uint4(float4 x); __attribute__((overloadable)) uint8 convert_uint8(float8 x); __attribute__((overloadable)) uint16 convert_uint16(float16 x); __attribute__((overloadable)) short convert_short(float x); __attribute__((overloadable)) short2 convert_short2(float2 x); __attribute__((overloadable)) short3 convert_short3(float3 x); __attribute__((overloadable)) short4 convert_short4(float4 x); __attribute__((overloadable)) short8 convert_short8(float8 x); __attribute__((overloadable)) short16 convert_short16(float16 x); __attribute__((overloadable)) ushort convert_ushort(float x); __attribute__((overloadable)) ushort2 convert_ushort2(float2 x); __attribute__((overloadable)) ushort3 convert_ushort3(float3 x); __attribute__((overloadable)) ushort4 convert_ushort4(float4 x); __attribute__((overloadable)) ushort8 convert_ushort8(float8 x); __attribute__((overloadable)) ushort16 convert_ushort16(float16 x); __attribute__((overloadable)) long convert_long(float x); __attribute__((overloadable)) long2 convert_long2(float2 x); __attribute__((overloadable)) long3 convert_long3(float3 x); __attribute__((overloadable)) long4 convert_long4(float4 x); __attribute__((overloadable)) long8 convert_long8(float8 x); __attribute__((overloadable)) long16 convert_long16(float16 x); __attribute__((overloadable)) ulong convert_ulong(float x); __attribute__((overloadable)) ulong2 convert_ulong2(float2 x); __attribute__((overloadable)) ulong3 convert_ulong3(float3 x); __attribute__((overloadable)) ulong4 convert_ulong4(float4 x); __attribute__((overloadable)) ulong8 convert_ulong8(float8 x); __attribute__((overloadable)) ulong16 convert_ulong16(float16 x); __attribute__((overloadable)) float convert_float(float x); __attribute__((overloadable)) float2 convert_float2(float2 x); __attribute__((overloadable)) float3 convert_float3(float3 x); __attribute__((overloadable)) float4 convert_float4(float4 x); __attribute__((overloadable)) float8 convert_float8(float8 x); __attribute__((overloadable)) float16 convert_float16(float16 x); __attribute__((overloadable)) double convert_double(float x); __attribute__((overloadable)) double2 convert_double2(float2 x); __attribute__((overloadable)) double3 convert_double3(float3 x); __attribute__((overloadable)) double4 convert_double4(float4 x); __attribute__((overloadable)) double8 convert_double8(float8 x); __attribute__((overloadable)) double16 convert_double16(float16 x); __attribute__((overloadable)) char convert_char(double x); __attribute__((overloadable)) char2 convert_char2(double2 x); __attribute__((overloadable)) char3 convert_char3(double3 x); __attribute__((overloadable)) char4 convert_char4(double4 x); __attribute__((overloadable)) char8 convert_char8(double8 x); __attribute__((overloadable)) char16 convert_char16(double16 x); __attribute__((overloadable)) uchar convert_uchar(double x); __attribute__((overloadable)) uchar2 convert_uchar2(double2 x); __attribute__((overloadable)) uchar3 convert_uchar3(double3 x); __attribute__((overloadable)) uchar4 convert_uchar4(double4 x); __attribute__((overloadable)) uchar8 convert_uchar8(double8 x); __attribute__((overloadable)) uchar16 convert_uchar16(double16 x); __attribute__((overloadable)) int convert_int(double x); __attribute__((overloadable)) int2 convert_int2(double2 x); __attribute__((overloadable)) int3 convert_int3(double3 x); __attribute__((overloadable)) int4 convert_int4(double4 x); __attribute__((overloadable)) int8 convert_int8(double8 x); __attribute__((overloadable)) int16 convert_int16(double16 x); __attribute__((overloadable)) uint convert_uint(double x); __attribute__((overloadable)) uint2 convert_uint2(double2 x); __attribute__((overloadable)) uint3 convert_uint3(double3 x); __attribute__((overloadable)) uint4 convert_uint4(double4 x); __attribute__((overloadable)) uint8 convert_uint8(double8 x); __attribute__((overloadable)) uint16 convert_uint16(double16 x); __attribute__((overloadable)) short convert_short(double x); __attribute__((overloadable)) short2 convert_short2(double2 x); __attribute__((overloadable)) short3 convert_short3(double3 x); __attribute__((overloadable)) short4 convert_short4(double4 x); __attribute__((overloadable)) short8 convert_short8(double8 x); __attribute__((overloadable)) short16 convert_short16(double16 x); __attribute__((overloadable)) ushort convert_ushort(double x); __attribute__((overloadable)) ushort2 convert_ushort2(double2 x); __attribute__((overloadable)) ushort3 convert_ushort3(double3 x); __attribute__((overloadable)) ushort4 convert_ushort4(double4 x); __attribute__((overloadable)) ushort8 convert_ushort8(double8 x); __attribute__((overloadable)) ushort16 convert_ushort16(double16 x); __attribute__((overloadable)) long convert_long(double x); __attribute__((overloadable)) long2 convert_long2(double2 x); __attribute__((overloadable)) long3 convert_long3(double3 x); __attribute__((overloadable)) long4 convert_long4(double4 x); __attribute__((overloadable)) long8 convert_long8(double8 x); __attribute__((overloadable)) long16 convert_long16(double16 x); __attribute__((overloadable)) ulong convert_ulong(double x); __attribute__((overloadable)) ulong2 convert_ulong2(double2 x); __attribute__((overloadable)) ulong3 convert_ulong3(double3 x); __attribute__((overloadable)) ulong4 convert_ulong4(double4 x); __attribute__((overloadable)) ulong8 convert_ulong8(double8 x); __attribute__((overloadable)) ulong16 convert_ulong16(double16 x); __attribute__((overloadable)) float convert_float(double x); __attribute__((overloadable)) float2 convert_float2(double2 x); __attribute__((overloadable)) float3 convert_float3(double3 x); __attribute__((overloadable)) float4 convert_float4(double4 x); __attribute__((overloadable)) float8 convert_float8(double8 x); __attribute__((overloadable)) float16 convert_float16(double16 x); __attribute__((overloadable)) double convert_double(double x); __attribute__((overloadable)) double2 convert_double2(double2 x); __attribute__((overloadable)) double3 convert_double3(double3 x); __attribute__((overloadable)) double4 convert_double4(double4 x); __attribute__((overloadable)) double8 convert_double8(double8 x); __attribute__((overloadable)) double16 convert_double16(double16 x);
size_t get_global_size(uint dim);
size_t get_global_id(uint dim);
size_t get_local_size(uint dim);
size_t get_local_id(uint dim);
size_t get_num_groups(uint dim);
size_t get_group_id(uint dim);
__attribute__((overloadable)) float acos(float x);
__attribute__((overloadable)) float2 acos(float2 x);
__attribute__((overloadable)) float3 acos(float3 x);
__attribute__((overloadable)) float4 acos(float4 x);
__attribute__((overloadable)) float8 acos(float8 x);
__attribute__((overloadable)) float16 acos(float16 x);
__attribute__((overloadable)) double acos(double x);
__attribute__((overloadable)) double2 acos(double2 x);
__attribute__((overloadable)) double3 acos(double3 x);
__attribute__((overloadable)) double4 acos(double4 x);
__attribute__((overloadable)) double8 acos(double8 x);
__attribute__((overloadable)) double16 acos(double16 x);
__attribute__((overloadable)) float acosh(float x);
__attribute__((overloadable)) float2 acosh(float2 x);
__attribute__((overloadable)) float3 acosh(float3 x);
__attribute__((overloadable)) float4 acosh(float4 x);
__attribute__((overloadable)) float8 acosh(float8 x);
__attribute__((overloadable)) float16 acosh(float16 x);
__attribute__((overloadable)) double acosh(double x);
__attribute__((overloadable)) double2 acosh(double2 x);
__attribute__((overloadable)) double3 acosh(double3 x);
__attribute__((overloadable)) double4 acosh(double4 x);
__attribute__((overloadable)) double8 acosh(double8 x);
__attribute__((overloadable)) double16 acosh(double16 x);
__attribute__((overloadable)) float acospi(float x);
__attribute__((overloadable)) float2 acospi(float2 x);
__attribute__((overloadable)) float3 acospi(float3 x);
__attribute__((overloadable)) float4 acospi(float4 x);
__attribute__((overloadable)) float8 acospi(float8 x);
__attribute__((overloadable)) float16 acospi(float16 x);
__attribute__((overloadable)) double acospi(double x);
__attribute__((overloadable)) double2 acospi(double2 x);
__attribute__((overloadable)) double3 acospi(double3 x);
__attribute__((overloadable)) double4 acospi(double4 x);
__attribute__((overloadable)) double8 acospi(double8 x);
__attribute__((overloadable)) double16 acospi(double16 x);
__attribute__((overloadable)) float asin(float x);
__attribute__((overloadable)) float2 asin(float2 x);
__attribute__((overloadable)) float3 asin(float3 x);
__attribute__((overloadable)) float4 asin(float4 x);
__attribute__((overloadable)) float8 asin(float8 x);
__attribute__((overloadable)) float16 asin(float16 x);
__attribute__((overloadable)) double asin(double x);
__attribute__((overloadable)) double2 asin(double2 x);
__attribute__((overloadable)) double3 asin(double3 x);
__attribute__((overloadable)) double4 asin(double4 x);
__attribute__((overloadable)) double8 asin(double8 x);
__attribute__((overloadable)) double16 asin(double16 x);
__attribute__((overloadable)) float asinh(float x);
__attribute__((overloadable)) float2 asinh(float2 x);
__attribute__((overloadable)) float3 asinh(float3 x);
__attribute__((overloadable)) float4 asinh(float4 x);
__attribute__((overloadable)) float8 asinh(float8 x);
__attribute__((overloadable)) float16 asinh(float16 x);
__attribute__((overloadable)) double asinh(double x);
__attribute__((overloadable)) double2 asinh(double2 x);
__attribute__((overloadable)) double3 asinh(double3 x);
__attribute__((overloadable)) double4 asinh(double4 x);
__attribute__((overloadable)) double8 asinh(double8 x);
__attribute__((overloadable)) double16 asinh(double16 x);
__attribute__((overloadable)) float asinpi(float x);
__attribute__((overloadable)) float2 asinpi(float2 x);
__attribute__((overloadable)) float3 asinpi(float3 x);
__attribute__((overloadable)) float4 asinpi(float4 x);
__attribute__((overloadable)) float8 asinpi(float8 x);
__attribute__((overloadable)) float16 asinpi(float16 x);
__attribute__((overloadable)) double asinpi(double x);
__attribute__((overloadable)) double2 asinpi(double2 x);
__attribute__((overloadable)) double3 asinpi(double3 x);
__attribute__((overloadable)) double4 asinpi(double4 x);
__attribute__((overloadable)) double8 asinpi(double8 x);
__attribute__((overloadable)) double16 asinpi(double16 x);
__attribute__((overloadable)) float atan(float a);
__attribute__((overloadable)) float2 atan(float2 a);
__attribute__((overloadable)) float3 atan(float3 a);
__attribute__((overloadable)) float4 atan(float4 a);
__attribute__((overloadable)) float8 atan(float8 a);
__attribute__((overloadable)) float16 atan(float16 a);
__attribute__((overloadable)) double atan(double a);
__attribute__((overloadable)) double2 atan(double2 a);
__attribute__((overloadable)) double3 atan(double3 a);
__attribute__((overloadable)) double4 atan(double4 a);
__attribute__((overloadable)) double8 atan(double8 a);
__attribute__((overloadable)) double16 atan(double16 a);
__attribute__((overloadable)) float atan2(float a, float b);
__attribute__((overloadable)) float2 atan2(float2 a, float2 b);
__attribute__((overloadable)) float3 atan2(float3 a, float3 b);
__attribute__((overloadable)) float4 atan2(float4 a, float4 b);
__attribute__((overloadable)) float8 atan2(float8 a, float8 b);
__attribute__((overloadable)) float16 atan2(float16 a, float16 b);
__attribute__((overloadable)) double atan2(double a, double b);
__attribute__((overloadable)) double2 atan2(double2 a, double2 b);
__attribute__((overloadable)) double3 atan2(double3 a, double3 b);
__attribute__((overloadable)) double4 atan2(double4 a, double4 b);
__attribute__((overloadable)) double8 atan2(double8 a, double8 b);
__attribute__((overloadable)) double16 atan2(double16 a, double16 b);
__attribute__((overloadable)) float atan2pi(float x, float y);
__attribute__((overloadable)) float2 atan2pi(float2 x, float2 y);
__attribute__((overloadable)) float3 atan2pi(float3 x, float3 y);
__attribute__((overloadable)) float4 atan2pi(float4 x, float4 y);
__attribute__((overloadable)) float8 atan2pi(float8 x, float8 y);
__attribute__((overloadable)) float16 atan2pi(float16 x, float16 y);
__attribute__((overloadable)) double atan2pi(double x, double y);
__attribute__((overloadable)) double2 atan2pi(double2 x, double2 y);
__attribute__((overloadable)) double3 atan2pi(double3 x, double3 y);
__attribute__((overloadable)) double4 atan2pi(double4 x, double4 y);
__attribute__((overloadable)) double8 atan2pi(double8 x, double8 y);
__attribute__((overloadable)) double16 atan2pi(double16 x, double16 y);
__attribute__((overloadable)) float atanh(float x);
__attribute__((overloadable)) float2 atanh(float2 x);
__attribute__((overloadable)) float3 atanh(float3 x);
__attribute__((overloadable)) float4 atanh(float4 x);
__attribute__((overloadable)) float8 atanh(float8 x);
__attribute__((overloadable)) float16 atanh(float16 x);
__attribute__((overloadable)) double atanh(double x);
__attribute__((overloadable)) double2 atanh(double2 x);
__attribute__((overloadable)) double3 atanh(double3 x);
__attribute__((overloadable)) double4 atanh(double4 x);
__attribute__((overloadable)) double8 atanh(double8 x);
__attribute__((overloadable)) double16 atanh(double16 x);
__attribute__((overloadable)) float atanpi(float x);
__attribute__((overloadable)) float2 atanpi(float2 x);
__attribute__((overloadable)) float3 atanpi(float3 x);
__attribute__((overloadable)) float4 atanpi(float4 x);
__attribute__((overloadable)) float8 atanpi(float8 x);
__attribute__((overloadable)) float16 atanpi(float16 x);
__attribute__((overloadable)) double atanpi(double x);
__attribute__((overloadable)) double2 atanpi(double2 x);
__attribute__((overloadable)) double3 atanpi(double3 x);
__attribute__((overloadable)) double4 atanpi(double4 x);
__attribute__((overloadable)) double8 atanpi(double8 x);
__attribute__((overloadable)) double16 atanpi(double16 x);
__attribute__((overloadable)) float copysign(float a, float b);
__attribute__((overloadable)) float2 copysign(float2 a, float2 b);
__attribute__((overloadable)) float3 copysign(float3 a, float3 b);
__attribute__((overloadable)) float4 copysign(float4 a, float4 b);
__attribute__((overloadable)) float8 copysign(float8 a, float8 b);
__attribute__((overloadable)) float16 copysign(float16 a, float16 b);
__attribute__((overloadable)) double copysign(double a, double b);
__attribute__((overloadable)) double2 copysign(double2 a, double2 b);
__attribute__((overloadable)) double3 copysign(double3 a, double3 b);
__attribute__((overloadable)) double4 copysign(double4 a, double4 b);
__attribute__((overloadable)) double8 copysign(double8 a, double8 b);
__attribute__((overloadable)) double16 copysign(double16 a, double16 b);
__attribute__((overloadable)) float cos(float a);
__attribute__((overloadable)) float2 cos(float2 a);
__attribute__((overloadable)) float3 cos(float3 a);
__attribute__((overloadable)) float4 cos(float4 a);
__attribute__((overloadable)) float8 cos(float8 a);
__attribute__((overloadable)) float16 cos(float16 a);
__attribute__((overloadable)) double cos(double a);
__attribute__((overloadable)) double2 cos(double2 a);
__attribute__((overloadable)) double3 cos(double3 a);
__attribute__((overloadable)) double4 cos(double4 a);
__attribute__((overloadable)) double8 cos(double8 a);
__attribute__((overloadable)) double16 cos(double16 a);
__attribute__((overloadable)) float cospi(float a);
__attribute__((overloadable)) float2 cospi(float2 a);
__attribute__((overloadable)) float3 cospi(float3 a);
__attribute__((overloadable)) float4 cospi(float4 a);
__attribute__((overloadable)) float8 cospi(float8 a);
__attribute__((overloadable)) float16 cospi(float16 a);
__attribute__((overloadable)) double cospi(double a);
__attribute__((overloadable)) double2 cospi(double2 a);
__attribute__((overloadable)) double3 cospi(double3 a);
__attribute__((overloadable)) double4 cospi(double4 a);
__attribute__((overloadable)) double8 cospi(double8 a);
__attribute__((overloadable)) double16 cospi(double16 a);
__attribute__((overloadable)) float __clc_ceil(float f) __asm("llvm.ceil" ".f32");
__attribute__((overloadable)) float2 __clc_ceil(float2 f) __asm("llvm.ceil" ".v2f32");
__attribute__((overloadable)) float3 __clc_ceil(float3 f) __asm("llvm.ceil" ".v3f32");
__attribute__((overloadable)) float4 __clc_ceil(float4 f) __asm("llvm.ceil" ".v4f32");
__attribute__((overloadable)) float8 __clc_ceil(float8 f) __asm("llvm.ceil" ".v8f32");
__attribute__((overloadable)) float16 __clc_ceil(float16 f) __asm("llvm.ceil" ".v16f32");
__attribute__((overloadable)) double __clc_ceil(double d) __asm("llvm.ceil" ".f64");
__attribute__((overloadable)) double2 __clc_ceil(double2 d) __asm("llvm.ceil" ".v2f64");
__attribute__((overloadable)) double3 __clc_ceil(double3 d) __asm("llvm.ceil" ".v3f64");
__attribute__((overloadable)) double4 __clc_ceil(double4 d) __asm("llvm.ceil" ".v4f64");
__attribute__((overloadable)) double8 __clc_ceil(double8 d) __asm("llvm.ceil" ".v8f64");
__attribute__((overloadable)) double16 __clc_ceil(double16 d) __asm("llvm.ceil" ".v16f64");
__attribute__((overloadable)) float erfc(float x);
__attribute__((overloadable)) float2 erfc(float2 x);
__attribute__((overloadable)) float3 erfc(float3 x);
__attribute__((overloadable)) float4 erfc(float4 x);
__attribute__((overloadable)) float8 erfc(float8 x);
__attribute__((overloadable)) float16 erfc(float16 x);
__attribute__((overloadable)) double erfc(double x);
__attribute__((overloadable)) double2 erfc(double2 x);
__attribute__((overloadable)) double3 erfc(double3 x);
__attribute__((overloadable)) double4 erfc(double4 x);
__attribute__((overloadable)) double8 erfc(double8 x);
__attribute__((overloadable)) double16 erfc(double16 x);
__attribute__((overloadable)) float exp(float x);
__attribute__((overloadable)) float2 exp(float2 x);
__attribute__((overloadable)) float3 exp(float3 x);
__attribute__((overloadable)) float4 exp(float4 x);
__attribute__((overloadable)) float8 exp(float8 x);
__attribute__((overloadable)) float16 exp(float16 x);
__attribute__((overloadable)) double exp(double x);
__attribute__((overloadable)) double2 exp(double2 x);
__attribute__((overloadable)) double3 exp(double3 x);
__attribute__((overloadable)) double4 exp(double4 x);
__attribute__((overloadable)) double8 exp(double8 x);
__attribute__((overloadable)) double16 exp(double16 x);
__attribute__((overloadable)) float exp10(float x);
__attribute__((overloadable)) float2 exp10(float2 x);
__attribute__((overloadable)) float3 exp10(float3 x);
__attribute__((overloadable)) float4 exp10(float4 x);
__attribute__((overloadable)) float8 exp10(float8 x);
__attribute__((overloadable)) float16 exp10(float16 x);
__attribute__((overloadable)) double exp10(double x);
__attribute__((overloadable)) double2 exp10(double2 x);
__attribute__((overloadable)) double3 exp10(double3 x);
__attribute__((overloadable)) double4 exp10(double4 x);
__attribute__((overloadable)) double8 exp10(double8 x);
__attribute__((overloadable)) double16 exp10(double16 x);
__attribute__((overloadable)) float exp2(float x);
__attribute__((overloadable)) float2 exp2(float2 x);
__attribute__((overloadable)) float3 exp2(float3 x);
__attribute__((overloadable)) float4 exp2(float4 x);
__attribute__((overloadable)) float8 exp2(float8 x);
__attribute__((overloadable)) float16 exp2(float16 x);
__attribute__((overloadable)) double exp2(double x);
__attribute__((overloadable)) double2 exp2(double2 x);
__attribute__((overloadable)) double3 exp2(double3 x);
__attribute__((overloadable)) double4 exp2(double4 x);
__attribute__((overloadable)) double8 exp2(double8 x);
__attribute__((overloadable)) double16 exp2(double16 x);
__attribute__((overloadable)) float __clc_fabs(float f) __asm("llvm.fabs" ".f32");
__attribute__((overloadable)) float2 __clc_fabs(float2 f) __asm("llvm.fabs" ".v2f32");
__attribute__((overloadable)) float3 __clc_fabs(float3 f) __asm("llvm.fabs" ".v3f32");
__attribute__((overloadable)) float4 __clc_fabs(float4 f) __asm("llvm.fabs" ".v4f32");
__attribute__((overloadable)) float8 __clc_fabs(float8 f) __asm("llvm.fabs" ".v8f32");
__attribute__((overloadable)) float16 __clc_fabs(float16 f) __asm("llvm.fabs" ".v16f32");
__attribute__((overloadable)) double __clc_fabs(double d) __asm("llvm.fabs" ".f64");
__attribute__((overloadable)) double2 __clc_fabs(double2 d) __asm("llvm.fabs" ".v2f64");
__attribute__((overloadable)) double3 __clc_fabs(double3 d) __asm("llvm.fabs" ".v3f64");
__attribute__((overloadable)) double4 __clc_fabs(double4 d) __asm("llvm.fabs" ".v4f64");
__attribute__((overloadable)) double8 __clc_fabs(double8 d) __asm("llvm.fabs" ".v8f64");
__attribute__((overloadable)) double16 __clc_fabs(double16 d) __asm("llvm.fabs" ".v16f64");
__attribute__((overloadable)) float __clc_floor(float f) __asm("llvm.floor" ".f32");
__attribute__((overloadable)) float2 __clc_floor(float2 f) __asm("llvm.floor" ".v2f32");
__attribute__((overloadable)) float3 __clc_floor(float3 f) __asm("llvm.floor" ".v3f32");
__attribute__((overloadable)) float4 __clc_floor(float4 f) __asm("llvm.floor" ".v4f32");
__attribute__((overloadable)) float8 __clc_floor(float8 f) __asm("llvm.floor" ".v8f32");
__attribute__((overloadable)) float16 __clc_floor(float16 f) __asm("llvm.floor" ".v16f32");
__attribute__((overloadable)) double __clc_floor(double d) __asm("llvm.floor" ".f64");
__attribute__((overloadable)) double2 __clc_floor(double2 d) __asm("llvm.floor" ".v2f64");
__attribute__((overloadable)) double3 __clc_floor(double3 d) __asm("llvm.floor" ".v3f64");
__attribute__((overloadable)) double4 __clc_floor(double4 d) __asm("llvm.floor" ".v4f64");
__attribute__((overloadable)) double8 __clc_floor(double8 d) __asm("llvm.floor" ".v8f64");
__attribute__((overloadable)) double16 __clc_floor(double16 d) __asm("llvm.floor" ".v16f64");
// __attribute__((overloadable)) float __clc_fma(float, float, float) __asm("llvm.fma" ".f32");
// __attribute__((overloadable)) float2 __clc_fma(float2, float2, float2) __asm("llvm.fma" ".v2f32");
// __attribute__((overloadable)) float3 __clc_fma(float3, float3, float3) __asm("llvm.fma" ".v3f32");
// __attribute__((overloadable)) float4 __clc_fma(float4, float4, float4) __asm("llvm.fma" ".v4f32");
// __attribute__((overloadable)) float8 __clc_fma(float8, float8, float8) __asm("llvm.fma" ".v8f32");
// __attribute__((overloadable)) float16 __clc_fma(float16, float16, float16) __asm("llvm.fma" ".v16f32");
// __attribute__((overloadable)) double __clc_fma(double, double, double) __asm("llvm.fma" ".f64");
// __attribute__((overloadable)) double2 __clc_fma(double2, double2, double2) __asm("llvm.fma" ".v2f64");
// __attribute__((overloadable)) double3 __clc_fma(double3, double3, double3) __asm("llvm.fma" ".v3f64");
// __attribute__((overloadable)) double4 __clc_fma(double4, double4, double4) __asm("llvm.fma" ".v4f64");
// __attribute__((overloadable)) double8 __clc_fma(double8, double8, double8) __asm("llvm.fma" ".v8f64");
// __attribute__((overloadable)) double16 __clc_fma(double16, double16, double16) __asm("llvm.fma" ".v16f64");
__attribute__((overloadable)) float fmax(float a, float b);
__attribute__((overloadable)) float fmax(float a, float b);
__attribute__((overloadable)) float fmax(float a, double b);
__attribute__((overloadable)) float2 fmax(float2 a, float2 b);
__attribute__((overloadable)) float2 fmax(float2 a, float b);
__attribute__((overloadable)) float2 fmax(float2 a, double b);
__attribute__((overloadable)) float3 fmax(float3 a, float3 b);
__attribute__((overloadable)) float3 fmax(float3 a, float b);
__attribute__((overloadable)) float3 fmax(float3 a, double b);
__attribute__((overloadable)) float4 fmax(float4 a, float4 b);
__attribute__((overloadable)) float4 fmax(float4 a, float b);
__attribute__((overloadable)) float4 fmax(float4 a, double b);
__attribute__((overloadable)) float8 fmax(float8 a, float8 b);
__attribute__((overloadable)) float8 fmax(float8 a, float b);
__attribute__((overloadable)) float8 fmax(float8 a, double b);
__attribute__((overloadable)) float16 fmax(float16 a, float16 b);
__attribute__((overloadable)) float16 fmax(float16 a, float b);
__attribute__((overloadable)) float16 fmax(float16 a, double b);
__attribute__((overloadable)) double fmax(double a, double b);
__attribute__((overloadable)) double fmax(double a, float b);
__attribute__((overloadable)) double fmax(double a, double b);
__attribute__((overloadable)) double2 fmax(double2 a, double2 b);
__attribute__((overloadable)) double2 fmax(double2 a, float b);
__attribute__((overloadable)) double2 fmax(double2 a, double b);
__attribute__((overloadable)) double3 fmax(double3 a, double3 b);
__attribute__((overloadable)) double3 fmax(double3 a, float b);
__attribute__((overloadable)) double3 fmax(double3 a, double b);
__attribute__((overloadable)) double4 fmax(double4 a, double4 b);
__attribute__((overloadable)) double4 fmax(double4 a, float b);
__attribute__((overloadable)) double4 fmax(double4 a, double b);
__attribute__((overloadable)) double8 fmax(double8 a, double8 b);
__attribute__((overloadable)) double8 fmax(double8 a, float b);
__attribute__((overloadable)) double8 fmax(double8 a, double b);
__attribute__((overloadable)) double16 fmax(double16 a, double16 b);
__attribute__((overloadable)) double16 fmax(double16 a, float b);
__attribute__((overloadable)) double16 fmax(double16 a, double b);
__attribute__((overloadable)) float fmin(float a, float b);
__attribute__((overloadable)) float fmin(float a, float b);
__attribute__((overloadable)) float fmin(float a, double b);
__attribute__((overloadable)) float2 fmin(float2 a, float2 b);
__attribute__((overloadable)) float2 fmin(float2 a, float b);
__attribute__((overloadable)) float2 fmin(float2 a, double b);
__attribute__((overloadable)) float3 fmin(float3 a, float3 b);
__attribute__((overloadable)) float3 fmin(float3 a, float b);
__attribute__((overloadable)) float3 fmin(float3 a, double b);
__attribute__((overloadable)) float4 fmin(float4 a, float4 b);
__attribute__((overloadable)) float4 fmin(float4 a, float b);
__attribute__((overloadable)) float4 fmin(float4 a, double b);
__attribute__((overloadable)) float8 fmin(float8 a, float8 b);
__attribute__((overloadable)) float8 fmin(float8 a, float b);
__attribute__((overloadable)) float8 fmin(float8 a, double b);
__attribute__((overloadable)) float16 fmin(float16 a, float16 b);
__attribute__((overloadable)) float16 fmin(float16 a, float b);
__attribute__((overloadable)) float16 fmin(float16 a, double b);
__attribute__((overloadable)) double fmin(double a, double b);
__attribute__((overloadable)) double fmin(double a, float b);
__attribute__((overloadable)) double fmin(double a, double b);
__attribute__((overloadable)) double2 fmin(double2 a, double2 b);
__attribute__((overloadable)) double2 fmin(double2 a, float b);
__attribute__((overloadable)) double2 fmin(double2 a, double b);
__attribute__((overloadable)) double3 fmin(double3 a, double3 b);
__attribute__((overloadable)) double3 fmin(double3 a, float b);
__attribute__((overloadable)) double3 fmin(double3 a, double b);
__attribute__((overloadable)) double4 fmin(double4 a, double4 b);
__attribute__((overloadable)) double4 fmin(double4 a, float b);
__attribute__((overloadable)) double4 fmin(double4 a, double b);
__attribute__((overloadable)) double8 fmin(double8 a, double8 b);
__attribute__((overloadable)) double8 fmin(double8 a, float b);
__attribute__((overloadable)) double8 fmin(double8 a, double b);
__attribute__((overloadable)) double16 fmin(double16 a, double16 b);
__attribute__((overloadable)) double16 fmin(double16 a, float b);
__attribute__((overloadable)) double16 fmin(double16 a, double b);
__attribute__((overloadable)) float fmod(float a, float b);
__attribute__((overloadable)) float2 fmod(float2 a, float2 b);
__attribute__((overloadable)) float3 fmod(float3 a, float3 b);
__attribute__((overloadable)) float4 fmod(float4 a, float4 b);
__attribute__((overloadable)) float8 fmod(float8 a, float8 b);
__attribute__((overloadable)) float16 fmod(float16 a, float16 b);
__attribute__((overloadable)) double fmod(double a, double b);
__attribute__((overloadable)) double2 fmod(double2 a, double2 b);
__attribute__((overloadable)) double3 fmod(double3 a, double3 b);
__attribute__((overloadable)) double4 fmod(double4 a, double4 b);
__attribute__((overloadable)) double8 fmod(double8 a, double8 b);
__attribute__((overloadable)) double16 fmod(double16 a, double16 b);
__attribute__((overloadable)) float fract(float x, global float *iptr);
__attribute__((overloadable)) float fract(float x, local float *iptr);
__attribute__((overloadable)) float fract(float x, private float *iptr);
__attribute__((overloadable)) float2 fract(float2 x, global float2 *iptr);
__attribute__((overloadable)) float2 fract(float2 x, local float2 *iptr);
__attribute__((overloadable)) float2 fract(float2 x, private float2 *iptr);
__attribute__((overloadable)) float3 fract(float3 x, global float3 *iptr);
__attribute__((overloadable)) float3 fract(float3 x, local float3 *iptr);
__attribute__((overloadable)) float3 fract(float3 x, private float3 *iptr);
__attribute__((overloadable)) float4 fract(float4 x, global float4 *iptr);
__attribute__((overloadable)) float4 fract(float4 x, local float4 *iptr);
__attribute__((overloadable)) float4 fract(float4 x, private float4 *iptr);
__attribute__((overloadable)) float8 fract(float8 x, global float8 *iptr);
__attribute__((overloadable)) float8 fract(float8 x, local float8 *iptr);
__attribute__((overloadable)) float8 fract(float8 x, private float8 *iptr);
__attribute__((overloadable)) float16 fract(float16 x, global float16 *iptr);
__attribute__((overloadable)) float16 fract(float16 x, local float16 *iptr);
__attribute__((overloadable)) float16 fract(float16 x, private float16 *iptr);
__attribute__((overloadable)) double fract(double x, global double *iptr);
__attribute__((overloadable)) double fract(double x, local double *iptr);
__attribute__((overloadable)) double fract(double x, private double *iptr);
__attribute__((overloadable)) double2 fract(double2 x, global double2 *iptr);
__attribute__((overloadable)) double2 fract(double2 x, local double2 *iptr);
__attribute__((overloadable)) double2 fract(double2 x, private double2 *iptr);
__attribute__((overloadable)) double3 fract(double3 x, global double3 *iptr);
__attribute__((overloadable)) double3 fract(double3 x, local double3 *iptr);
__attribute__((overloadable)) double3 fract(double3 x, private double3 *iptr);
__attribute__((overloadable)) double4 fract(double4 x, global double4 *iptr);
__attribute__((overloadable)) double4 fract(double4 x, local double4 *iptr);
__attribute__((overloadable)) double4 fract(double4 x, private double4 *iptr);
__attribute__((overloadable)) double8 fract(double8 x, global double8 *iptr);
__attribute__((overloadable)) double8 fract(double8 x, local double8 *iptr);
__attribute__((overloadable)) double8 fract(double8 x, private double8 *iptr);
__attribute__((overloadable)) double16 fract(double16 x, global double16 *iptr);
__attribute__((overloadable)) double16 fract(double16 x, local double16 *iptr);
__attribute__((overloadable)) double16 fract(double16 x, private double16 *iptr);
__attribute__((overloadable)) float frexp(float x, global int *iptr);
__attribute__((overloadable)) float frexp(float x, local int *iptr);
__attribute__((overloadable)) float frexp(float x, private int *iptr);
__attribute__((overloadable)) float2 frexp(float2 x, global int2 *iptr);
__attribute__((overloadable)) float2 frexp(float2 x, local int2 *iptr);
__attribute__((overloadable)) float2 frexp(float2 x, private int2 *iptr);
__attribute__((overloadable)) float3 frexp(float3 x, global int3 *iptr);
__attribute__((overloadable)) float3 frexp(float3 x, local int3 *iptr);
__attribute__((overloadable)) float3 frexp(float3 x, private int3 *iptr);
__attribute__((overloadable)) float4 frexp(float4 x, global int4 *iptr);
__attribute__((overloadable)) float4 frexp(float4 x, local int4 *iptr);
__attribute__((overloadable)) float4 frexp(float4 x, private int4 *iptr);
__attribute__((overloadable)) float8 frexp(float8 x, global int8 *iptr);
__attribute__((overloadable)) float8 frexp(float8 x, local int8 *iptr);
__attribute__((overloadable)) float8 frexp(float8 x, private int8 *iptr);
__attribute__((overloadable)) float16 frexp(float16 x, global int16 *iptr);
__attribute__((overloadable)) float16 frexp(float16 x, local int16 *iptr);
__attribute__((overloadable)) float16 frexp(float16 x, private int16 *iptr);
__attribute__((overloadable)) double frexp(double x, global int *iptr);
__attribute__((overloadable)) double frexp(double x, local int *iptr);
__attribute__((overloadable)) double frexp(double x, private int *iptr);
__attribute__((overloadable)) double2 frexp(double2 x, global int2 *iptr);
__attribute__((overloadable)) double2 frexp(double2 x, local int2 *iptr);
__attribute__((overloadable)) double2 frexp(double2 x, private int2 *iptr);
__attribute__((overloadable)) double3 frexp(double3 x, global int3 *iptr);
__attribute__((overloadable)) double3 frexp(double3 x, local int3 *iptr);
__attribute__((overloadable)) double3 frexp(double3 x, private int3 *iptr);
__attribute__((overloadable)) double4 frexp(double4 x, global int4 *iptr);
__attribute__((overloadable)) double4 frexp(double4 x, local int4 *iptr);
__attribute__((overloadable)) double4 frexp(double4 x, private int4 *iptr);
__attribute__((overloadable)) double8 frexp(double8 x, global int8 *iptr);
__attribute__((overloadable)) double8 frexp(double8 x, local int8 *iptr);
__attribute__((overloadable)) double8 frexp(double8 x, private int8 *iptr);
__attribute__((overloadable)) double16 frexp(double16 x, global int16 *iptr);
__attribute__((overloadable)) double16 frexp(double16 x, local int16 *iptr);
__attribute__((overloadable)) double16 frexp(double16 x, private int16 *iptr);
__attribute__((overloadable)) float half_rsqrt(float x);
__attribute__((overloadable)) float2 half_rsqrt(float2 x);
__attribute__((overloadable)) float3 half_rsqrt(float3 x);
__attribute__((overloadable)) float4 half_rsqrt(float4 x);
__attribute__((overloadable)) float8 half_rsqrt(float8 x);
__attribute__((overloadable)) float16 half_rsqrt(float16 x);
__attribute__((overloadable)) float half_sqrt(float x);
__attribute__((overloadable)) float2 half_sqrt(float2 x);
__attribute__((overloadable)) float3 half_sqrt(float3 x);
__attribute__((overloadable)) float4 half_sqrt(float4 x);
__attribute__((overloadable)) float8 half_sqrt(float8 x);
__attribute__((overloadable)) float16 half_sqrt(float16 x);
__attribute__((overloadable)) float hypot(float x, float y);
__attribute__((overloadable)) float2 hypot(float2 x, float2 y);
__attribute__((overloadable)) float3 hypot(float3 x, float3 y);
__attribute__((overloadable)) float4 hypot(float4 x, float4 y);
__attribute__((overloadable)) float8 hypot(float8 x, float8 y);
__attribute__((overloadable)) float16 hypot(float16 x, float16 y);
__attribute__((overloadable)) double hypot(double x, double y);
__attribute__((overloadable)) double2 hypot(double2 x, double2 y);
__attribute__((overloadable)) double3 hypot(double3 x, double3 y);
__attribute__((overloadable)) double4 hypot(double4 x, double4 y);
__attribute__((overloadable)) double8 hypot(double8 x, double8 y);
__attribute__((overloadable)) double16 hypot(double16 x, double16 y);
__attribute__((overloadable)) int ilogb(float x);
__attribute__((overloadable)) int2 ilogb(float2 x);
__attribute__((overloadable)) int3 ilogb(float3 x);
__attribute__((overloadable)) int4 ilogb(float4 x);
__attribute__((overloadable)) int8 ilogb(float8 x);
__attribute__((overloadable)) int16 ilogb(float16 x);
__attribute__((overloadable)) int ilogb(double x);
__attribute__((overloadable)) int2 ilogb(double2 x);
__attribute__((overloadable)) int3 ilogb(double3 x);
__attribute__((overloadable)) int4 ilogb(double4 x);
__attribute__((overloadable)) int8 ilogb(double8 x);
__attribute__((overloadable)) int16 ilogb(double16 x);
__attribute__((overloadable)) float ldexp(float x, int n);
__attribute__((overloadable)) float2 ldexp(float2 x, int n);
__attribute__((overloadable)) float2 ldexp(float2 x, int2 n);
__attribute__((overloadable)) float3 ldexp(float3 x, int n);
__attribute__((overloadable)) float3 ldexp(float3 x, int3 n);
__attribute__((overloadable)) float4 ldexp(float4 x, int n);
__attribute__((overloadable)) float4 ldexp(float4 x, int4 n);
__attribute__((overloadable)) float8 ldexp(float8 x, int n);
__attribute__((overloadable)) float8 ldexp(float8 x, int8 n);
__attribute__((overloadable)) float16 ldexp(float16 x, int n);
__attribute__((overloadable)) float16 ldexp(float16 x, int16 n);
__attribute__((overloadable)) double ldexp(double x, int n);
__attribute__((overloadable)) double2 ldexp(double2 x, int n);
__attribute__((overloadable)) double2 ldexp(double2 x, int2 n);
__attribute__((overloadable)) double3 ldexp(double3 x, int n);
__attribute__((overloadable)) double3 ldexp(double3 x, int3 n);
__attribute__((overloadable)) double4 ldexp(double4 x, int n);
__attribute__((overloadable)) double4 ldexp(double4 x, int4 n);
__attribute__((overloadable)) double8 ldexp(double8 x, int n);
__attribute__((overloadable)) double8 ldexp(double8 x, int8 n);
__attribute__((overloadable)) double16 ldexp(double16 x, int n);
__attribute__((overloadable)) double16 ldexp(double16 x, int16 n);
__attribute__((overloadable)) float log(float a);
__attribute__((overloadable)) float2 log(float2 a);
__attribute__((overloadable)) float3 log(float3 a);
__attribute__((overloadable)) float4 log(float4 a);
__attribute__((overloadable)) float8 log(float8 a);
__attribute__((overloadable)) float16 log(float16 a);
__attribute__((overloadable)) double log(double a);
__attribute__((overloadable)) double2 log(double2 a);
__attribute__((overloadable)) double3 log(double3 a);
__attribute__((overloadable)) double4 log(double4 a);
__attribute__((overloadable)) double8 log(double8 a);
__attribute__((overloadable)) double16 log(double16 a);
__attribute__((overloadable)) float log10(float x);
__attribute__((overloadable)) float2 log10(float2 x);
__attribute__((overloadable)) float3 log10(float3 x);
__attribute__((overloadable)) float4 log10(float4 x);
__attribute__((overloadable)) float8 log10(float8 x);
__attribute__((overloadable)) float16 log10(float16 x);
__attribute__((overloadable)) double log10(double x);
__attribute__((overloadable)) double2 log10(double2 x);
__attribute__((overloadable)) double3 log10(double3 x);
__attribute__((overloadable)) double4 log10(double4 x);
__attribute__((overloadable)) double8 log10(double8 x);
__attribute__((overloadable)) double16 log10(double16 x);
__attribute__((overloadable)) float log1p(float a);
__attribute__((overloadable)) float2 log1p(float2 a);
__attribute__((overloadable)) float3 log1p(float3 a);
__attribute__((overloadable)) float4 log1p(float4 a);
__attribute__((overloadable)) float8 log1p(float8 a);
__attribute__((overloadable)) float16 log1p(float16 a);
__attribute__((overloadable)) double log1p(double a);
__attribute__((overloadable)) double2 log1p(double2 a);
__attribute__((overloadable)) double3 log1p(double3 a);
__attribute__((overloadable)) double4 log1p(double4 a);
__attribute__((overloadable)) double8 log1p(double8 a);
__attribute__((overloadable)) double16 log1p(double16 a);
__attribute__((overloadable)) float log2(float a);
__attribute__((overloadable)) float2 log2(float2 a);
__attribute__((overloadable)) float3 log2(float3 a);
__attribute__((overloadable)) float4 log2(float4 a);
__attribute__((overloadable)) float8 log2(float8 a);
__attribute__((overloadable)) float16 log2(float16 a);
__attribute__((overloadable)) double log2(double a);
__attribute__((overloadable)) double2 log2(double2 a);
__attribute__((overloadable)) double3 log2(double3 a);
__attribute__((overloadable)) double4 log2(double4 a);
__attribute__((overloadable)) double8 log2(double8 a);
__attribute__((overloadable)) double16 log2(double16 a);
__attribute__((overloadable)) float mad(float a, float b, float c);
__attribute__((overloadable)) float2 mad(float2 a, float2 b, float2 c);
__attribute__((overloadable)) float3 mad(float3 a, float3 b, float3 c);
__attribute__((overloadable)) float4 mad(float4 a, float4 b, float4 c);
__attribute__((overloadable)) float8 mad(float8 a, float8 b, float8 c);
__attribute__((overloadable)) float16 mad(float16 a, float16 b, float16 c);
__attribute__((overloadable)) double mad(double a, double b, double c);
__attribute__((overloadable)) double2 mad(double2 a, double2 b, double2 c);
__attribute__((overloadable)) double3 mad(double3 a, double3 b, double3 c);
__attribute__((overloadable)) double4 mad(double4 a, double4 b, double4 c);
__attribute__((overloadable)) double8 mad(double8 a, double8 b, double8 c);
__attribute__((overloadable)) double16 mad(double16 a, double16 b, double16 c);
__attribute__((overloadable)) float modf(float x, global float *iptr);
__attribute__((overloadable)) float modf(float x, local float *iptr);
__attribute__((overloadable)) float modf(float x, private float *iptr);
__attribute__((overloadable)) float2 modf(float2 x, global float2 *iptr);
__attribute__((overloadable)) float2 modf(float2 x, local float2 *iptr);
__attribute__((overloadable)) float2 modf(float2 x, private float2 *iptr);
__attribute__((overloadable)) float3 modf(float3 x, global float3 *iptr);
__attribute__((overloadable)) float3 modf(float3 x, local float3 *iptr);
__attribute__((overloadable)) float3 modf(float3 x, private float3 *iptr);
__attribute__((overloadable)) float4 modf(float4 x, global float4 *iptr);
__attribute__((overloadable)) float4 modf(float4 x, local float4 *iptr);
__attribute__((overloadable)) float4 modf(float4 x, private float4 *iptr);
__attribute__((overloadable)) float8 modf(float8 x, global float8 *iptr);
__attribute__((overloadable)) float8 modf(float8 x, local float8 *iptr);
__attribute__((overloadable)) float8 modf(float8 x, private float8 *iptr);
__attribute__((overloadable)) float16 modf(float16 x, global float16 *iptr);
__attribute__((overloadable)) float16 modf(float16 x, local float16 *iptr);
__attribute__((overloadable)) float16 modf(float16 x, private float16 *iptr);
__attribute__((overloadable)) double modf(double x, global double *iptr);
__attribute__((overloadable)) double modf(double x, local double *iptr);
__attribute__((overloadable)) double modf(double x, private double *iptr);
__attribute__((overloadable)) double2 modf(double2 x, global double2 *iptr);
__attribute__((overloadable)) double2 modf(double2 x, local double2 *iptr);
__attribute__((overloadable)) double2 modf(double2 x, private double2 *iptr);
__attribute__((overloadable)) double3 modf(double3 x, global double3 *iptr);
__attribute__((overloadable)) double3 modf(double3 x, local double3 *iptr);
__attribute__((overloadable)) double3 modf(double3 x, private double3 *iptr);
__attribute__((overloadable)) double4 modf(double4 x, global double4 *iptr);
__attribute__((overloadable)) double4 modf(double4 x, local double4 *iptr);
__attribute__((overloadable)) double4 modf(double4 x, private double4 *iptr);
__attribute__((overloadable)) double8 modf(double8 x, global double8 *iptr);
__attribute__((overloadable)) double8 modf(double8 x, local double8 *iptr);
__attribute__((overloadable)) double8 modf(double8 x, private double8 *iptr);
__attribute__((overloadable)) double16 modf(double16 x, global double16 *iptr);
__attribute__((overloadable)) double16 modf(double16 x, local double16 *iptr);
__attribute__((overloadable)) double16 modf(double16 x, private double16 *iptr);
__attribute__((overloadable)) float nextafter(float a, float b);
__attribute__((overloadable)) float nextafter(float a, float b);
__attribute__((overloadable)) float nextafter(float a, double b);
__attribute__((overloadable)) float2 nextafter(float2 a, float2 b);
__attribute__((overloadable)) float2 nextafter(float2 a, float b);
__attribute__((overloadable)) float2 nextafter(float2 a, double b);
__attribute__((overloadable)) float3 nextafter(float3 a, float3 b);
__attribute__((overloadable)) float3 nextafter(float3 a, float b);
__attribute__((overloadable)) float3 nextafter(float3 a, double b);
__attribute__((overloadable)) float4 nextafter(float4 a, float4 b);
__attribute__((overloadable)) float4 nextafter(float4 a, float b);
__attribute__((overloadable)) float4 nextafter(float4 a, double b);
__attribute__((overloadable)) float8 nextafter(float8 a, float8 b);
__attribute__((overloadable)) float8 nextafter(float8 a, float b);
__attribute__((overloadable)) float8 nextafter(float8 a, double b);
__attribute__((overloadable)) float16 nextafter(float16 a, float16 b);
__attribute__((overloadable)) float16 nextafter(float16 a, float b);
__attribute__((overloadable)) float16 nextafter(float16 a, double b);
__attribute__((overloadable)) double nextafter(double a, double b);
__attribute__((overloadable)) double nextafter(double a, float b);
__attribute__((overloadable)) double nextafter(double a, double b);
__attribute__((overloadable)) double2 nextafter(double2 a, double2 b);
__attribute__((overloadable)) double2 nextafter(double2 a, float b);
__attribute__((overloadable)) double2 nextafter(double2 a, double b);
__attribute__((overloadable)) double3 nextafter(double3 a, double3 b);
__attribute__((overloadable)) double3 nextafter(double3 a, float b);
__attribute__((overloadable)) double3 nextafter(double3 a, double b);
__attribute__((overloadable)) double4 nextafter(double4 a, double4 b);
__attribute__((overloadable)) double4 nextafter(double4 a, float b);
__attribute__((overloadable)) double4 nextafter(double4 a, double b);
__attribute__((overloadable)) double8 nextafter(double8 a, double8 b);
__attribute__((overloadable)) double8 nextafter(double8 a, float b);
__attribute__((overloadable)) double8 nextafter(double8 a, double b);
__attribute__((overloadable)) double16 nextafter(double16 a, double16 b);
__attribute__((overloadable)) double16 nextafter(double16 a, float b);
__attribute__((overloadable)) double16 nextafter(double16 a, double b);
// __attribute__((overloadable)) float __clc_pow(float, float) __asm("llvm.pow" ".f32");
// __attribute__((overloadable)) float2 __clc_pow(float2, float2) __asm("llvm.pow" ".v2f32");
// __attribute__((overloadable)) float3 __clc_pow(float3, float3) __asm("llvm.pow" ".v3f32");
// __attribute__((overloadable)) float4 __clc_pow(float4, float4) __asm("llvm.pow" ".v4f32");
// __attribute__((overloadable)) float8 __clc_pow(float8, float8) __asm("llvm.pow" ".v8f32");
// __attribute__((overloadable)) float16 __clc_pow(float16, float16) __asm("llvm.pow" ".v16f32");
// __attribute__((overloadable)) double __clc_pow(double, double) __asm("llvm.pow" ".f64");
// __attribute__((overloadable)) double2 __clc_pow(double2, double2) __asm("llvm.pow" ".v2f64");
// __attribute__((overloadable)) double3 __clc_pow(double3, double3) __asm("llvm.pow" ".v3f64");
// __attribute__((overloadable)) double4 __clc_pow(double4, double4) __asm("llvm.pow" ".v4f64");
// __attribute__((overloadable)) double8 __clc_pow(double8, double8) __asm("llvm.pow" ".v8f64");
// __attribute__((overloadable)) double16 __clc_pow(double16, double16) __asm("llvm.pow" ".v16f64");
// __attribute__((overloadable)) float pown(float x, int y) __asm("llvm.powi" ".f32");
__attribute__((overloadable)) float2 pown(float2 x, int2 y); __attribute__((overloadable)) float3 pown(float3 x, int3 y); __attribute__((overloadable)) float4 pown(float4 x, int4 y); __attribute__((overloadable)) float8 pown(float8 x, int8 y); __attribute__((overloadable)) float16 pown(float16 x, int16 y);
__attribute__((overloadable)) double pown(double x, int y) __asm("llvm.powi" ".f64");
__attribute__((overloadable)) double2 pown(double2 x, int2 y); __attribute__((overloadable)) double3 pown(double3 x, int3 y); __attribute__((overloadable)) double4 pown(double4 x, int4 y); __attribute__((overloadable)) double8 pown(double8 x, int8 y); __attribute__((overloadable)) double16 pown(double16 x, int16 y);
__attribute__((overloadable)) float __clc_rint(float f) __asm("llvm.rint" ".f32");
__attribute__((overloadable)) float2 __clc_rint(float2 f) __asm("llvm.rint" ".v2f32");
__attribute__((overloadable)) float3 __clc_rint(float3 f) __asm("llvm.rint" ".v3f32");
__attribute__((overloadable)) float4 __clc_rint(float4 f) __asm("llvm.rint" ".v4f32");
__attribute__((overloadable)) float8 __clc_rint(float8 f) __asm("llvm.rint" ".v8f32");
__attribute__((overloadable)) float16 __clc_rint(float16 f) __asm("llvm.rint" ".v16f32");
__attribute__((overloadable)) double __clc_rint(double d) __asm("llvm.rint" ".f64");
__attribute__((overloadable)) double2 __clc_rint(double2 d) __asm("llvm.rint" ".v2f64");
__attribute__((overloadable)) double3 __clc_rint(double3 d) __asm("llvm.rint" ".v3f64");
__attribute__((overloadable)) double4 __clc_rint(double4 d) __asm("llvm.rint" ".v4f64");
__attribute__((overloadable)) double8 __clc_rint(double8 d) __asm("llvm.rint" ".v8f64");
__attribute__((overloadable)) double16 __clc_rint(double16 d) __asm("llvm.rint" ".v16f64");
__attribute__((overloadable)) float __clc_round(float f) __asm("llvm.round" ".f32");
__attribute__((overloadable)) float2 __clc_round(float2 f) __asm("llvm.round" ".v2f32");
__attribute__((overloadable)) float3 __clc_round(float3 f) __asm("llvm.round" ".v3f32");
__attribute__((overloadable)) float4 __clc_round(float4 f) __asm("llvm.round" ".v4f32");
__attribute__((overloadable)) float8 __clc_round(float8 f) __asm("llvm.round" ".v8f32");
__attribute__((overloadable)) float16 __clc_round(float16 f) __asm("llvm.round" ".v16f32");
__attribute__((overloadable)) double __clc_round(double d) __asm("llvm.round" ".f64");
__attribute__((overloadable)) double2 __clc_round(double2 d) __asm("llvm.round" ".v2f64");
__attribute__((overloadable)) double3 __clc_round(double3 d) __asm("llvm.round" ".v3f64");
__attribute__((overloadable)) double4 __clc_round(double4 d) __asm("llvm.round" ".v4f64");
__attribute__((overloadable)) double8 __clc_round(double8 d) __asm("llvm.round" ".v8f64");
__attribute__((overloadable)) double16 __clc_round(double16 d) __asm("llvm.round" ".v16f64");
__attribute__((overloadable)) float sin(float a);
__attribute__((overloadable)) float2 sin(float2 a);
__attribute__((overloadable)) float3 sin(float3 a);
__attribute__((overloadable)) float4 sin(float4 a);
__attribute__((overloadable)) float8 sin(float8 a);
__attribute__((overloadable)) float16 sin(float16 a);
__attribute__((overloadable)) double sin(double a);
__attribute__((overloadable)) double2 sin(double2 a);
__attribute__((overloadable)) double3 sin(double3 a);
__attribute__((overloadable)) double4 sin(double4 a);
__attribute__((overloadable)) double8 sin(double8 a);
__attribute__((overloadable)) double16 sin(double16 a);
__attribute__((overloadable)) float sincos (float x, global float * cosval);
__attribute__((overloadable)) float sincos (float x, local float * cosval);
__attribute__((overloadable)) float sincos (float x, private float * cosval);
__attribute__((overloadable)) float2 sincos (float2 x, global float2 * cosval);
__attribute__((overloadable)) float2 sincos (float2 x, local float2 * cosval);
__attribute__((overloadable)) float2 sincos (float2 x, private float2 * cosval);
__attribute__((overloadable)) float3 sincos (float3 x, global float3 * cosval);
__attribute__((overloadable)) float3 sincos (float3 x, local float3 * cosval);
__attribute__((overloadable)) float3 sincos (float3 x, private float3 * cosval);
__attribute__((overloadable)) float4 sincos (float4 x, global float4 * cosval);
__attribute__((overloadable)) float4 sincos (float4 x, local float4 * cosval);
__attribute__((overloadable)) float4 sincos (float4 x, private float4 * cosval);
__attribute__((overloadable)) float8 sincos (float8 x, global float8 * cosval);
__attribute__((overloadable)) float8 sincos (float8 x, local float8 * cosval);
__attribute__((overloadable)) float8 sincos (float8 x, private float8 * cosval);
__attribute__((overloadable)) float16 sincos (float16 x, global float16 * cosval);
__attribute__((overloadable)) float16 sincos (float16 x, local float16 * cosval);
__attribute__((overloadable)) float16 sincos (float16 x, private float16 * cosval);
__attribute__((overloadable)) double sincos (double x, global double * cosval);
__attribute__((overloadable)) double sincos (double x, local double * cosval);
__attribute__((overloadable)) double sincos (double x, private double * cosval);
__attribute__((overloadable)) double2 sincos (double2 x, global double2 * cosval);
__attribute__((overloadable)) double2 sincos (double2 x, local double2 * cosval);
__attribute__((overloadable)) double2 sincos (double2 x, private double2 * cosval);
__attribute__((overloadable)) double3 sincos (double3 x, global double3 * cosval);
__attribute__((overloadable)) double3 sincos (double3 x, local double3 * cosval);
__attribute__((overloadable)) double3 sincos (double3 x, private double3 * cosval);
__attribute__((overloadable)) double4 sincos (double4 x, global double4 * cosval);
__attribute__((overloadable)) double4 sincos (double4 x, local double4 * cosval);
__attribute__((overloadable)) double4 sincos (double4 x, private double4 * cosval);
__attribute__((overloadable)) double8 sincos (double8 x, global double8 * cosval);
__attribute__((overloadable)) double8 sincos (double8 x, local double8 * cosval);
__attribute__((overloadable)) double8 sincos (double8 x, private double8 * cosval);
__attribute__((overloadable)) double16 sincos (double16 x, global double16 * cosval);
__attribute__((overloadable)) double16 sincos (double16 x, local double16 * cosval);
__attribute__((overloadable)) double16 sincos (double16 x, private double16 * cosval);
__attribute__((overloadable)) float sinpi(float a);
__attribute__((overloadable)) float2 sinpi(float2 a);
__attribute__((overloadable)) float3 sinpi(float3 a);
__attribute__((overloadable)) float4 sinpi(float4 a);
__attribute__((overloadable)) float8 sinpi(float8 a);
__attribute__((overloadable)) float16 sinpi(float16 a);
__attribute__((overloadable)) double sinpi(double a);
__attribute__((overloadable)) double2 sinpi(double2 a);
__attribute__((overloadable)) double3 sinpi(double3 a);
__attribute__((overloadable)) double4 sinpi(double4 a);
__attribute__((overloadable)) double8 sinpi(double8 a);
__attribute__((overloadable)) double16 sinpi(double16 a);
__attribute__((overloadable)) float sqrt(float a);
__attribute__((overloadable)) float2 sqrt(float2 a);
__attribute__((overloadable)) float3 sqrt(float3 a);
__attribute__((overloadable)) float4 sqrt(float4 a);
__attribute__((overloadable)) float8 sqrt(float8 a);
__attribute__((overloadable)) float16 sqrt(float16 a);
__attribute__((overloadable)) double sqrt(double a);
__attribute__((overloadable)) double2 sqrt(double2 a);
__attribute__((overloadable)) double3 sqrt(double3 a);
__attribute__((overloadable)) double4 sqrt(double4 a);
__attribute__((overloadable)) double8 sqrt(double8 a);
__attribute__((overloadable)) double16 sqrt(double16 a);
__attribute__((overloadable)) float tan(float x);
__attribute__((overloadable)) float2 tan(float2 x);
__attribute__((overloadable)) float3 tan(float3 x);
__attribute__((overloadable)) float4 tan(float4 x);
__attribute__((overloadable)) float8 tan(float8 x);
__attribute__((overloadable)) float16 tan(float16 x);
__attribute__((overloadable)) double tan(double x);
__attribute__((overloadable)) double2 tan(double2 x);
__attribute__((overloadable)) double3 tan(double3 x);
__attribute__((overloadable)) double4 tan(double4 x);
__attribute__((overloadable)) double8 tan(double8 x);
__attribute__((overloadable)) double16 tan(double16 x);
__attribute__((overloadable)) float tanh(float a);
__attribute__((overloadable)) float2 tanh(float2 a);
__attribute__((overloadable)) float3 tanh(float3 a);
__attribute__((overloadable)) float4 tanh(float4 a);
__attribute__((overloadable)) float8 tanh(float8 a);
__attribute__((overloadable)) float16 tanh(float16 a);
__attribute__((overloadable)) double tanh(double a);
__attribute__((overloadable)) double2 tanh(double2 a);
__attribute__((overloadable)) double3 tanh(double3 a);
__attribute__((overloadable)) double4 tanh(double4 a);
__attribute__((overloadable)) double8 tanh(double8 a);
__attribute__((overloadable)) double16 tanh(double16 a);
__attribute__((overloadable)) float __clc_trunc(float f) __asm("llvm.trunc" ".f32");
__attribute__((overloadable)) float2 __clc_trunc(float2 f) __asm("llvm.trunc" ".v2f32");
__attribute__((overloadable)) float3 __clc_trunc(float3 f) __asm("llvm.trunc" ".v3f32");
__attribute__((overloadable)) float4 __clc_trunc(float4 f) __asm("llvm.trunc" ".v4f32");
__attribute__((overloadable)) float8 __clc_trunc(float8 f) __asm("llvm.trunc" ".v8f32");
__attribute__((overloadable)) float16 __clc_trunc(float16 f) __asm("llvm.trunc" ".v16f32");
__attribute__((overloadable)) double __clc_trunc(double d) __asm("llvm.trunc" ".f64");
__attribute__((overloadable)) double2 __clc_trunc(double2 d) __asm("llvm.trunc" ".v2f64");
__attribute__((overloadable)) double3 __clc_trunc(double3 d) __asm("llvm.trunc" ".v3f64");
__attribute__((overloadable)) double4 __clc_trunc(double4 d) __asm("llvm.trunc" ".v4f64");
__attribute__((overloadable)) double8 __clc_trunc(double8 d) __asm("llvm.trunc" ".v8f64");
__attribute__((overloadable)) double16 __clc_trunc(double16 d) __asm("llvm.trunc" ".v16f64");
__attribute__((overloadable)) float native_log(float a);
__attribute__((overloadable)) float2 native_log(float2 a);
__attribute__((overloadable)) float3 native_log(float3 a);
__attribute__((overloadable)) float4 native_log(float4 a);
__attribute__((overloadable)) float8 native_log(float8 a);
__attribute__((overloadable)) float16 native_log(float16 a);
__attribute__((overloadable)) float native_log2(float a);
__attribute__((overloadable)) float2 native_log2(float2 a);
__attribute__((overloadable)) float3 native_log2(float3 a);
__attribute__((overloadable)) float4 native_log2(float4 a);
__attribute__((overloadable)) float8 native_log2(float8 a);
__attribute__((overloadable)) float16 native_log2(float16 a);
__attribute__((overloadable)) uchar abs(char x);
__attribute__((overloadable)) uchar2 abs(char2 x);
__attribute__((overloadable)) uchar3 abs(char3 x);
__attribute__((overloadable)) uchar4 abs(char4 x);
__attribute__((overloadable)) uchar8 abs(char8 x);
__attribute__((overloadable)) uchar16 abs(char16 x);
__attribute__((overloadable)) uchar abs(uchar x);
__attribute__((overloadable)) uchar2 abs(uchar2 x);
__attribute__((overloadable)) uchar3 abs(uchar3 x);
__attribute__((overloadable)) uchar4 abs(uchar4 x);
__attribute__((overloadable)) uchar8 abs(uchar8 x);
__attribute__((overloadable)) uchar16 abs(uchar16 x);
__attribute__((overloadable)) ushort abs(short x);
__attribute__((overloadable)) ushort2 abs(short2 x);
__attribute__((overloadable)) ushort3 abs(short3 x);
__attribute__((overloadable)) ushort4 abs(short4 x);
__attribute__((overloadable)) ushort8 abs(short8 x);
__attribute__((overloadable)) ushort16 abs(short16 x);
__attribute__((overloadable)) ushort abs(ushort x);
__attribute__((overloadable)) ushort2 abs(ushort2 x);
__attribute__((overloadable)) ushort3 abs(ushort3 x);
__attribute__((overloadable)) ushort4 abs(ushort4 x);
__attribute__((overloadable)) ushort8 abs(ushort8 x);
__attribute__((overloadable)) ushort16 abs(ushort16 x);
__attribute__((overloadable)) uint abs(int x);
__attribute__((overloadable)) uint2 abs(int2 x);
__attribute__((overloadable)) uint3 abs(int3 x);
__attribute__((overloadable)) uint4 abs(int4 x);
__attribute__((overloadable)) uint8 abs(int8 x);
__attribute__((overloadable)) uint16 abs(int16 x);
__attribute__((overloadable)) uint abs(uint x);
__attribute__((overloadable)) uint2 abs(uint2 x);
__attribute__((overloadable)) uint3 abs(uint3 x);
__attribute__((overloadable)) uint4 abs(uint4 x);
__attribute__((overloadable)) uint8 abs(uint8 x);
__attribute__((overloadable)) uint16 abs(uint16 x);
__attribute__((overloadable)) ulong abs(long x);
__attribute__((overloadable)) ulong2 abs(long2 x);
__attribute__((overloadable)) ulong3 abs(long3 x);
__attribute__((overloadable)) ulong4 abs(long4 x);
__attribute__((overloadable)) ulong8 abs(long8 x);
__attribute__((overloadable)) ulong16 abs(long16 x);
__attribute__((overloadable)) ulong abs(ulong x);
__attribute__((overloadable)) ulong2 abs(ulong2 x);
__attribute__((overloadable)) ulong3 abs(ulong3 x);
__attribute__((overloadable)) ulong4 abs(ulong4 x);
__attribute__((overloadable)) ulong8 abs(ulong8 x);
__attribute__((overloadable)) ulong16 abs(ulong16 x);
__attribute__((overloadable)) uchar abs_diff(char x, char y);
__attribute__((overloadable)) uchar2 abs_diff(char2 x, char2 y);
__attribute__((overloadable)) uchar3 abs_diff(char3 x, char3 y);
__attribute__((overloadable)) uchar4 abs_diff(char4 x, char4 y);
__attribute__((overloadable)) uchar8 abs_diff(char8 x, char8 y);
__attribute__((overloadable)) uchar16 abs_diff(char16 x, char16 y);
__attribute__((overloadable)) uchar abs_diff(uchar x, uchar y);
__attribute__((overloadable)) uchar2 abs_diff(uchar2 x, uchar2 y);
__attribute__((overloadable)) uchar3 abs_diff(uchar3 x, uchar3 y);
__attribute__((overloadable)) uchar4 abs_diff(uchar4 x, uchar4 y);
__attribute__((overloadable)) uchar8 abs_diff(uchar8 x, uchar8 y);
__attribute__((overloadable)) uchar16 abs_diff(uchar16 x, uchar16 y);
__attribute__((overloadable)) ushort abs_diff(short x, short y);
__attribute__((overloadable)) ushort2 abs_diff(short2 x, short2 y);
__attribute__((overloadable)) ushort3 abs_diff(short3 x, short3 y);
__attribute__((overloadable)) ushort4 abs_diff(short4 x, short4 y);
__attribute__((overloadable)) ushort8 abs_diff(short8 x, short8 y);
__attribute__((overloadable)) ushort16 abs_diff(short16 x, short16 y);
__attribute__((overloadable)) ushort abs_diff(ushort x, ushort y);
__attribute__((overloadable)) ushort2 abs_diff(ushort2 x, ushort2 y);
__attribute__((overloadable)) ushort3 abs_diff(ushort3 x, ushort3 y);
__attribute__((overloadable)) ushort4 abs_diff(ushort4 x, ushort4 y);
__attribute__((overloadable)) ushort8 abs_diff(ushort8 x, ushort8 y);
__attribute__((overloadable)) ushort16 abs_diff(ushort16 x, ushort16 y);
__attribute__((overloadable)) uint abs_diff(int x, int y);
__attribute__((overloadable)) uint2 abs_diff(int2 x, int2 y);
__attribute__((overloadable)) uint3 abs_diff(int3 x, int3 y);
__attribute__((overloadable)) uint4 abs_diff(int4 x, int4 y);
__attribute__((overloadable)) uint8 abs_diff(int8 x, int8 y);
__attribute__((overloadable)) uint16 abs_diff(int16 x, int16 y);
__attribute__((overloadable)) uint abs_diff(uint x, uint y);
__attribute__((overloadable)) uint2 abs_diff(uint2 x, uint2 y);
__attribute__((overloadable)) uint3 abs_diff(uint3 x, uint3 y);
__attribute__((overloadable)) uint4 abs_diff(uint4 x, uint4 y);
__attribute__((overloadable)) uint8 abs_diff(uint8 x, uint8 y);
__attribute__((overloadable)) uint16 abs_diff(uint16 x, uint16 y);
__attribute__((overloadable)) ulong abs_diff(long x, long y);
__attribute__((overloadable)) ulong2 abs_diff(long2 x, long2 y);
__attribute__((overloadable)) ulong3 abs_diff(long3 x, long3 y);
__attribute__((overloadable)) ulong4 abs_diff(long4 x, long4 y);
__attribute__((overloadable)) ulong8 abs_diff(long8 x, long8 y);
__attribute__((overloadable)) ulong16 abs_diff(long16 x, long16 y);
__attribute__((overloadable)) ulong abs_diff(ulong x, ulong y);
__attribute__((overloadable)) ulong2 abs_diff(ulong2 x, ulong2 y);
__attribute__((overloadable)) ulong3 abs_diff(ulong3 x, ulong3 y);
__attribute__((overloadable)) ulong4 abs_diff(ulong4 x, ulong4 y);
__attribute__((overloadable)) ulong8 abs_diff(ulong8 x, ulong8 y);
__attribute__((overloadable)) ulong16 abs_diff(ulong16 x, ulong16 y);
__attribute__((overloadable)) char add_sat(char x, char y);
__attribute__((overloadable)) char2 add_sat(char2 x, char2 y);
__attribute__((overloadable)) char3 add_sat(char3 x, char3 y);
__attribute__((overloadable)) char4 add_sat(char4 x, char4 y);
__attribute__((overloadable)) char8 add_sat(char8 x, char8 y);
__attribute__((overloadable)) char16 add_sat(char16 x, char16 y);
__attribute__((overloadable)) uchar add_sat(uchar x, uchar y);
__attribute__((overloadable)) uchar2 add_sat(uchar2 x, uchar2 y);
__attribute__((overloadable)) uchar3 add_sat(uchar3 x, uchar3 y);
__attribute__((overloadable)) uchar4 add_sat(uchar4 x, uchar4 y);
__attribute__((overloadable)) uchar8 add_sat(uchar8 x, uchar8 y);
__attribute__((overloadable)) uchar16 add_sat(uchar16 x, uchar16 y);
__attribute__((overloadable)) short add_sat(short x, short y);
__attribute__((overloadable)) short2 add_sat(short2 x, short2 y);
__attribute__((overloadable)) short3 add_sat(short3 x, short3 y);
__attribute__((overloadable)) short4 add_sat(short4 x, short4 y);
__attribute__((overloadable)) short8 add_sat(short8 x, short8 y);
__attribute__((overloadable)) short16 add_sat(short16 x, short16 y);
__attribute__((overloadable)) ushort add_sat(ushort x, ushort y);
__attribute__((overloadable)) ushort2 add_sat(ushort2 x, ushort2 y);
__attribute__((overloadable)) ushort3 add_sat(ushort3 x, ushort3 y);
__attribute__((overloadable)) ushort4 add_sat(ushort4 x, ushort4 y);
__attribute__((overloadable)) ushort8 add_sat(ushort8 x, ushort8 y);
__attribute__((overloadable)) ushort16 add_sat(ushort16 x, ushort16 y);
__attribute__((overloadable)) int add_sat(int x, int y);
__attribute__((overloadable)) int2 add_sat(int2 x, int2 y);
__attribute__((overloadable)) int3 add_sat(int3 x, int3 y);
__attribute__((overloadable)) int4 add_sat(int4 x, int4 y);
__attribute__((overloadable)) int8 add_sat(int8 x, int8 y);
__attribute__((overloadable)) int16 add_sat(int16 x, int16 y);
__attribute__((overloadable)) uint add_sat(uint x, uint y);
__attribute__((overloadable)) uint2 add_sat(uint2 x, uint2 y);
__attribute__((overloadable)) uint3 add_sat(uint3 x, uint3 y);
__attribute__((overloadable)) uint4 add_sat(uint4 x, uint4 y);
__attribute__((overloadable)) uint8 add_sat(uint8 x, uint8 y);
__attribute__((overloadable)) uint16 add_sat(uint16 x, uint16 y);
__attribute__((overloadable)) long add_sat(long x, long y);
__attribute__((overloadable)) long2 add_sat(long2 x, long2 y);
__attribute__((overloadable)) long3 add_sat(long3 x, long3 y);
__attribute__((overloadable)) long4 add_sat(long4 x, long4 y);
__attribute__((overloadable)) long8 add_sat(long8 x, long8 y);
__attribute__((overloadable)) long16 add_sat(long16 x, long16 y);
__attribute__((overloadable)) ulong add_sat(ulong x, ulong y);
__attribute__((overloadable)) ulong2 add_sat(ulong2 x, ulong2 y);
__attribute__((overloadable)) ulong3 add_sat(ulong3 x, ulong3 y);
__attribute__((overloadable)) ulong4 add_sat(ulong4 x, ulong4 y);
__attribute__((overloadable)) ulong8 add_sat(ulong8 x, ulong8 y);
__attribute__((overloadable)) ulong16 add_sat(ulong16 x, ulong16 y);
__attribute__((overloadable)) char clz(char x);
__attribute__((overloadable)) char2 clz(char2 x);
__attribute__((overloadable)) char3 clz(char3 x);
__attribute__((overloadable)) char4 clz(char4 x);
__attribute__((overloadable)) char8 clz(char8 x);
__attribute__((overloadable)) char16 clz(char16 x);
__attribute__((overloadable)) uchar clz(uchar x);
__attribute__((overloadable)) uchar2 clz(uchar2 x);
__attribute__((overloadable)) uchar3 clz(uchar3 x);
__attribute__((overloadable)) uchar4 clz(uchar4 x);
__attribute__((overloadable)) uchar8 clz(uchar8 x);
__attribute__((overloadable)) uchar16 clz(uchar16 x);
__attribute__((overloadable)) short clz(short x);
__attribute__((overloadable)) short2 clz(short2 x);
__attribute__((overloadable)) short3 clz(short3 x);
__attribute__((overloadable)) short4 clz(short4 x);
__attribute__((overloadable)) short8 clz(short8 x);
__attribute__((overloadable)) short16 clz(short16 x);
__attribute__((overloadable)) ushort clz(ushort x);
__attribute__((overloadable)) ushort2 clz(ushort2 x);
__attribute__((overloadable)) ushort3 clz(ushort3 x);
__attribute__((overloadable)) ushort4 clz(ushort4 x);
__attribute__((overloadable)) ushort8 clz(ushort8 x);
__attribute__((overloadable)) ushort16 clz(ushort16 x);
__attribute__((overloadable)) int clz(int x);
__attribute__((overloadable)) int2 clz(int2 x);
__attribute__((overloadable)) int3 clz(int3 x);
__attribute__((overloadable)) int4 clz(int4 x);
__attribute__((overloadable)) int8 clz(int8 x);
__attribute__((overloadable)) int16 clz(int16 x);
__attribute__((overloadable)) uint clz(uint x);
__attribute__((overloadable)) uint2 clz(uint2 x);
__attribute__((overloadable)) uint3 clz(uint3 x);
__attribute__((overloadable)) uint4 clz(uint4 x);
__attribute__((overloadable)) uint8 clz(uint8 x);
__attribute__((overloadable)) uint16 clz(uint16 x);
__attribute__((overloadable)) long clz(long x);
__attribute__((overloadable)) long2 clz(long2 x);
__attribute__((overloadable)) long3 clz(long3 x);
__attribute__((overloadable)) long4 clz(long4 x);
__attribute__((overloadable)) long8 clz(long8 x);
__attribute__((overloadable)) long16 clz(long16 x);
__attribute__((overloadable)) ulong clz(ulong x);
__attribute__((overloadable)) ulong2 clz(ulong2 x);
__attribute__((overloadable)) ulong3 clz(ulong3 x);
__attribute__((overloadable)) ulong4 clz(ulong4 x);
__attribute__((overloadable)) ulong8 clz(ulong8 x);
__attribute__((overloadable)) ulong16 clz(ulong16 x);
__attribute__((overloadable)) char hadd(char x, char y);
__attribute__((overloadable)) char2 hadd(char2 x, char2 y);
__attribute__((overloadable)) char3 hadd(char3 x, char3 y);
__attribute__((overloadable)) char4 hadd(char4 x, char4 y);
__attribute__((overloadable)) char8 hadd(char8 x, char8 y);
__attribute__((overloadable)) char16 hadd(char16 x, char16 y);
__attribute__((overloadable)) uchar hadd(uchar x, uchar y);
__attribute__((overloadable)) uchar2 hadd(uchar2 x, uchar2 y);
__attribute__((overloadable)) uchar3 hadd(uchar3 x, uchar3 y);
__attribute__((overloadable)) uchar4 hadd(uchar4 x, uchar4 y);
__attribute__((overloadable)) uchar8 hadd(uchar8 x, uchar8 y);
__attribute__((overloadable)) uchar16 hadd(uchar16 x, uchar16 y);
__attribute__((overloadable)) short hadd(short x, short y);
__attribute__((overloadable)) short2 hadd(short2 x, short2 y);
__attribute__((overloadable)) short3 hadd(short3 x, short3 y);
__attribute__((overloadable)) short4 hadd(short4 x, short4 y);
__attribute__((overloadable)) short8 hadd(short8 x, short8 y);
__attribute__((overloadable)) short16 hadd(short16 x, short16 y);
__attribute__((overloadable)) ushort hadd(ushort x, ushort y);
__attribute__((overloadable)) ushort2 hadd(ushort2 x, ushort2 y);
__attribute__((overloadable)) ushort3 hadd(ushort3 x, ushort3 y);
__attribute__((overloadable)) ushort4 hadd(ushort4 x, ushort4 y);
__attribute__((overloadable)) ushort8 hadd(ushort8 x, ushort8 y);
__attribute__((overloadable)) ushort16 hadd(ushort16 x, ushort16 y);
__attribute__((overloadable)) int hadd(int x, int y);
__attribute__((overloadable)) int2 hadd(int2 x, int2 y);
__attribute__((overloadable)) int3 hadd(int3 x, int3 y);
__attribute__((overloadable)) int4 hadd(int4 x, int4 y);
__attribute__((overloadable)) int8 hadd(int8 x, int8 y);
__attribute__((overloadable)) int16 hadd(int16 x, int16 y);
__attribute__((overloadable)) uint hadd(uint x, uint y);
__attribute__((overloadable)) uint2 hadd(uint2 x, uint2 y);
__attribute__((overloadable)) uint3 hadd(uint3 x, uint3 y);
__attribute__((overloadable)) uint4 hadd(uint4 x, uint4 y);
__attribute__((overloadable)) uint8 hadd(uint8 x, uint8 y);
__attribute__((overloadable)) uint16 hadd(uint16 x, uint16 y);
__attribute__((overloadable)) long hadd(long x, long y);
__attribute__((overloadable)) long2 hadd(long2 x, long2 y);
__attribute__((overloadable)) long3 hadd(long3 x, long3 y);
__attribute__((overloadable)) long4 hadd(long4 x, long4 y);
__attribute__((overloadable)) long8 hadd(long8 x, long8 y);
__attribute__((overloadable)) long16 hadd(long16 x, long16 y);
__attribute__((overloadable)) ulong hadd(ulong x, ulong y);
__attribute__((overloadable)) ulong2 hadd(ulong2 x, ulong2 y);
__attribute__((overloadable)) ulong3 hadd(ulong3 x, ulong3 y);
__attribute__((overloadable)) ulong4 hadd(ulong4 x, ulong4 y);
__attribute__((overloadable)) ulong8 hadd(ulong8 x, ulong8 y);
__attribute__((overloadable)) ulong16 hadd(ulong16 x, ulong16 y);
__attribute__((overloadable)) int mad24(int x, int y, int z);
__attribute__((overloadable)) int2 mad24(int2 x, int2 y, int2 z);
__attribute__((overloadable)) int3 mad24(int3 x, int3 y, int3 z);
__attribute__((overloadable)) int4 mad24(int4 x, int4 y, int4 z);
__attribute__((overloadable)) int8 mad24(int8 x, int8 y, int8 z);
__attribute__((overloadable)) int16 mad24(int16 x, int16 y, int16 z);
__attribute__((overloadable)) uint mad24(uint x, uint y, uint z);
__attribute__((overloadable)) uint2 mad24(uint2 x, uint2 y, uint2 z);
__attribute__((overloadable)) uint3 mad24(uint3 x, uint3 y, uint3 z);
__attribute__((overloadable)) uint4 mad24(uint4 x, uint4 y, uint4 z);
__attribute__((overloadable)) uint8 mad24(uint8 x, uint8 y, uint8 z);
__attribute__((overloadable)) uint16 mad24(uint16 x, uint16 y, uint16 z);
__attribute__((overloadable)) char mad_sat(char x, char y, char z);
__attribute__((overloadable)) char2 mad_sat(char2 x, char2 y, char2 z);
__attribute__((overloadable)) char3 mad_sat(char3 x, char3 y, char3 z);
__attribute__((overloadable)) char4 mad_sat(char4 x, char4 y, char4 z);
__attribute__((overloadable)) char8 mad_sat(char8 x, char8 y, char8 z);
__attribute__((overloadable)) char16 mad_sat(char16 x, char16 y, char16 z);
__attribute__((overloadable)) uchar mad_sat(uchar x, uchar y, uchar z);
__attribute__((overloadable)) uchar2 mad_sat(uchar2 x, uchar2 y, uchar2 z);
__attribute__((overloadable)) uchar3 mad_sat(uchar3 x, uchar3 y, uchar3 z);
__attribute__((overloadable)) uchar4 mad_sat(uchar4 x, uchar4 y, uchar4 z);
__attribute__((overloadable)) uchar8 mad_sat(uchar8 x, uchar8 y, uchar8 z);
__attribute__((overloadable)) uchar16 mad_sat(uchar16 x, uchar16 y, uchar16 z);
__attribute__((overloadable)) short mad_sat(short x, short y, short z);
__attribute__((overloadable)) short2 mad_sat(short2 x, short2 y, short2 z);
__attribute__((overloadable)) short3 mad_sat(short3 x, short3 y, short3 z);
__attribute__((overloadable)) short4 mad_sat(short4 x, short4 y, short4 z);
__attribute__((overloadable)) short8 mad_sat(short8 x, short8 y, short8 z);
__attribute__((overloadable)) short16 mad_sat(short16 x, short16 y, short16 z);
__attribute__((overloadable)) ushort mad_sat(ushort x, ushort y, ushort z);
__attribute__((overloadable)) ushort2 mad_sat(ushort2 x, ushort2 y, ushort2 z);
__attribute__((overloadable)) ushort3 mad_sat(ushort3 x, ushort3 y, ushort3 z);
__attribute__((overloadable)) ushort4 mad_sat(ushort4 x, ushort4 y, ushort4 z);
__attribute__((overloadable)) ushort8 mad_sat(ushort8 x, ushort8 y, ushort8 z);
__attribute__((overloadable)) ushort16 mad_sat(ushort16 x, ushort16 y, ushort16 z);
__attribute__((overloadable)) int mad_sat(int x, int y, int z);
__attribute__((overloadable)) int2 mad_sat(int2 x, int2 y, int2 z);
__attribute__((overloadable)) int3 mad_sat(int3 x, int3 y, int3 z);
__attribute__((overloadable)) int4 mad_sat(int4 x, int4 y, int4 z);
__attribute__((overloadable)) int8 mad_sat(int8 x, int8 y, int8 z);
__attribute__((overloadable)) int16 mad_sat(int16 x, int16 y, int16 z);
__attribute__((overloadable)) uint mad_sat(uint x, uint y, uint z);
__attribute__((overloadable)) uint2 mad_sat(uint2 x, uint2 y, uint2 z);
__attribute__((overloadable)) uint3 mad_sat(uint3 x, uint3 y, uint3 z);
__attribute__((overloadable)) uint4 mad_sat(uint4 x, uint4 y, uint4 z);
__attribute__((overloadable)) uint8 mad_sat(uint8 x, uint8 y, uint8 z);
__attribute__((overloadable)) uint16 mad_sat(uint16 x, uint16 y, uint16 z);
__attribute__((overloadable)) long mad_sat(long x, long y, long z);
__attribute__((overloadable)) long2 mad_sat(long2 x, long2 y, long2 z);
__attribute__((overloadable)) long3 mad_sat(long3 x, long3 y, long3 z);
__attribute__((overloadable)) long4 mad_sat(long4 x, long4 y, long4 z);
__attribute__((overloadable)) long8 mad_sat(long8 x, long8 y, long8 z);
__attribute__((overloadable)) long16 mad_sat(long16 x, long16 y, long16 z);
__attribute__((overloadable)) ulong mad_sat(ulong x, ulong y, ulong z);
__attribute__((overloadable)) ulong2 mad_sat(ulong2 x, ulong2 y, ulong2 z);
__attribute__((overloadable)) ulong3 mad_sat(ulong3 x, ulong3 y, ulong3 z);
__attribute__((overloadable)) ulong4 mad_sat(ulong4 x, ulong4 y, ulong4 z);
__attribute__((overloadable)) ulong8 mad_sat(ulong8 x, ulong8 y, ulong8 z);
__attribute__((overloadable)) ulong16 mad_sat(ulong16 x, ulong16 y, ulong16 z);
__attribute__((overloadable)) int mul24(int x, int y);
__attribute__((overloadable)) int2 mul24(int2 x, int2 y);
__attribute__((overloadable)) int3 mul24(int3 x, int3 y);
__attribute__((overloadable)) int4 mul24(int4 x, int4 y);
__attribute__((overloadable)) int8 mul24(int8 x, int8 y);
__attribute__((overloadable)) int16 mul24(int16 x, int16 y);
__attribute__((overloadable)) uint mul24(uint x, uint y);
__attribute__((overloadable)) uint2 mul24(uint2 x, uint2 y);
__attribute__((overloadable)) uint3 mul24(uint3 x, uint3 y);
__attribute__((overloadable)) uint4 mul24(uint4 x, uint4 y);
__attribute__((overloadable)) uint8 mul24(uint8 x, uint8 y);
__attribute__((overloadable)) uint16 mul24(uint16 x, uint16 y);
__attribute__((overloadable)) char mul_hi(char x, char y);
__attribute__((overloadable)) char2 mul_hi(char2 x, char2 y);
__attribute__((overloadable)) char3 mul_hi(char3 x, char3 y);
__attribute__((overloadable)) char4 mul_hi(char4 x, char4 y);
__attribute__((overloadable)) char8 mul_hi(char8 x, char8 y);
__attribute__((overloadable)) char16 mul_hi(char16 x, char16 y);
__attribute__((overloadable)) uchar mul_hi(uchar x, uchar y);
__attribute__((overloadable)) uchar2 mul_hi(uchar2 x, uchar2 y);
__attribute__((overloadable)) uchar3 mul_hi(uchar3 x, uchar3 y);
__attribute__((overloadable)) uchar4 mul_hi(uchar4 x, uchar4 y);
__attribute__((overloadable)) uchar8 mul_hi(uchar8 x, uchar8 y);
__attribute__((overloadable)) uchar16 mul_hi(uchar16 x, uchar16 y);
__attribute__((overloadable)) short mul_hi(short x, short y);
__attribute__((overloadable)) short2 mul_hi(short2 x, short2 y);
__attribute__((overloadable)) short3 mul_hi(short3 x, short3 y);
__attribute__((overloadable)) short4 mul_hi(short4 x, short4 y);
__attribute__((overloadable)) short8 mul_hi(short8 x, short8 y);
__attribute__((overloadable)) short16 mul_hi(short16 x, short16 y);
__attribute__((overloadable)) ushort mul_hi(ushort x, ushort y);
__attribute__((overloadable)) ushort2 mul_hi(ushort2 x, ushort2 y);
__attribute__((overloadable)) ushort3 mul_hi(ushort3 x, ushort3 y);
__attribute__((overloadable)) ushort4 mul_hi(ushort4 x, ushort4 y);
__attribute__((overloadable)) ushort8 mul_hi(ushort8 x, ushort8 y);
__attribute__((overloadable)) ushort16 mul_hi(ushort16 x, ushort16 y);
__attribute__((overloadable)) int mul_hi(int x, int y);
__attribute__((overloadable)) int2 mul_hi(int2 x, int2 y);
__attribute__((overloadable)) int3 mul_hi(int3 x, int3 y);
__attribute__((overloadable)) int4 mul_hi(int4 x, int4 y);
// __attribute__((overloadable)) int8 mul_hi(int8 x, int8 y);
// __attribute__((overloadable)) int16 mul_hi(int16 x, int16 y);
// __attribute__((overloadable)) uint mul_hi(uint x, uint y);
// __attribute__((overloadable)) uint2 mul_hi(uint2 x, uint2 y);
// __attribute__((overloadable)) uint3 mul_hi(uint3 x, uint3 y);
// __attribute__((overloadable)) uint4 mul_hi(uint4 x, uint4 y);
// __attribute__((overloadable)) uint8 mul_hi(uint8 x, uint8 y);
// __attribute__((overloadable)) uint16 mul_hi(uint16 x, uint16 y);
// __attribute__((overloadable)) long mul_hi(long x, long y);
// __attribute__((overloadable)) long2 mul_hi(long2 x, long2 y);
// __attribute__((overloadable)) long3 mul_hi(long3 x, long3 y);
// __attribute__((overloadable)) long4 mul_hi(long4 x, long4 y);
// __attribute__((overloadable)) long8 mul_hi(long8 x, long8 y);
// __attribute__((overloadable)) long16 mul_hi(long16 x, long16 y);
// __attribute__((overloadable)) ulong mul_hi(ulong x, ulong y);
// __attribute__((overloadable)) ulong2 mul_hi(ulong2 x, ulong2 y);
// __attribute__((overloadable)) ulong3 mul_hi(ulong3 x, ulong3 y);
// __attribute__((overloadable)) ulong4 mul_hi(ulong4 x, ulong4 y);
// __attribute__((overloadable)) ulong8 mul_hi(ulong8 x, ulong8 y);
// __attribute__((overloadable)) ulong16 mul_hi(ulong16 x, ulong16 y);
// __attribute__((overloadable)) char rhadd(char x, char y);
// __attribute__((overloadable)) char2 rhadd(char2 x, char2 y);
// __attribute__((overloadable)) char3 rhadd(char3 x, char3 y);
// __attribute__((overloadable)) char4 rhadd(char4 x, char4 y);
// __attribute__((overloadable)) char8 rhadd(char8 x, char8 y);
// __attribute__((overloadable)) char16 rhadd(char16 x, char16 y);
// __attribute__((overloadable)) uchar rhadd(uchar x, uchar y);
// __attribute__((overloadable)) uchar2 rhadd(uchar2 x, uchar2 y);
// __attribute__((overloadable)) uchar3 rhadd(uchar3 x, uchar3 y);
// __attribute__((overloadable)) uchar4 rhadd(uchar4 x, uchar4 y);
// __attribute__((overloadable)) uchar8 rhadd(uchar8 x, uchar8 y);
// __attribute__((overloadable)) uchar16 rhadd(uchar16 x, uchar16 y);
// __attribute__((overloadable)) short rhadd(short x, short y);
// __attribute__((overloadable)) short2 rhadd(short2 x, short2 y);
// __attribute__((overloadable)) short3 rhadd(short3 x, short3 y);
// __attribute__((overloadable)) short4 rhadd(short4 x, short4 y);
// __attribute__((overloadable)) short8 rhadd(short8 x, short8 y);
// __attribute__((overloadable)) short16 rhadd(short16 x, short16 y);
// __attribute__((overloadable)) ushort rhadd(ushort x, ushort y);
// __attribute__((overloadable)) ushort2 rhadd(ushort2 x, ushort2 y);
// __attribute__((overloadable)) ushort3 rhadd(ushort3 x, ushort3 y);
// __attribute__((overloadable)) ushort4 rhadd(ushort4 x, ushort4 y);
// __attribute__((overloadable)) ushort8 rhadd(ushort8 x, ushort8 y);
// __attribute__((overloadable)) ushort16 rhadd(ushort16 x, ushort16 y);
// __attribute__((overloadable)) int rhadd(int x, int y);
// __attribute__((overloadable)) int2 rhadd(int2 x, int2 y);
// __attribute__((overloadable)) int3 rhadd(int3 x, int3 y);
// __attribute__((overloadable)) int4 rhadd(int4 x, int4 y);
// __attribute__((overloadable)) int8 rhadd(int8 x, int8 y);
// __attribute__((overloadable)) int16 rhadd(int16 x, int16 y);
// __attribute__((overloadable)) uint rhadd(uint x, uint y);
// __attribute__((overloadable)) uint2 rhadd(uint2 x, uint2 y);
// __attribute__((overloadable)) uint3 rhadd(uint3 x, uint3 y);
// __attribute__((overloadable)) uint4 rhadd(uint4 x, uint4 y);
// __attribute__((overloadable)) uint8 rhadd(uint8 x, uint8 y);
// __attribute__((overloadable)) uint16 rhadd(uint16 x, uint16 y);
// __attribute__((overloadable)) long rhadd(long x, long y);
// __attribute__((overloadable)) long2 rhadd(long2 x, long2 y);
// __attribute__((overloadable)) long3 rhadd(long3 x, long3 y);
// __attribute__((overloadable)) long4 rhadd(long4 x, long4 y);
// __attribute__((overloadable)) long8 rhadd(long8 x, long8 y);
// __attribute__((overloadable)) long16 rhadd(long16 x, long16 y);
// __attribute__((overloadable)) ulong rhadd(ulong x, ulong y);
// __attribute__((overloadable)) ulong2 rhadd(ulong2 x, ulong2 y);
// __attribute__((overloadable)) ulong3 rhadd(ulong3 x, ulong3 y);
// __attribute__((overloadable)) ulong4 rhadd(ulong4 x, ulong4 y);
// __attribute__((overloadable)) ulong8 rhadd(ulong8 x, ulong8 y);
// __attribute__((overloadable)) ulong16 rhadd(ulong16 x, ulong16 y);
// __attribute__((overloadable)) char rotate(char x, char y);
// __attribute__((overloadable)) char2 rotate(char2 x, char2 y);
// __attribute__((overloadable)) char3 rotate(char3 x, char3 y);
// __attribute__((overloadable)) char4 rotate(char4 x, char4 y);
// __attribute__((overloadable)) char8 rotate(char8 x, char8 y);
// __attribute__((overloadable)) char16 rotate(char16 x, char16 y);
// __attribute__((overloadable)) uchar rotate(uchar x, uchar y);
// __attribute__((overloadable)) uchar2 rotate(uchar2 x, uchar2 y);
// __attribute__((overloadable)) uchar3 rotate(uchar3 x, uchar3 y);
// __attribute__((overloadable)) uchar4 rotate(uchar4 x, uchar4 y);
// __attribute__((overloadable)) uchar8 rotate(uchar8 x, uchar8 y);
// __attribute__((overloadable)) uchar16 rotate(uchar16 x, uchar16 y);
// __attribute__((overloadable)) short rotate(short x, short y);
// __attribute__((overloadable)) short2 rotate(short2 x, short2 y);
// __attribute__((overloadable)) short3 rotate(short3 x, short3 y);
// __attribute__((overloadable)) short4 rotate(short4 x, short4 y);
// __attribute__((overloadable)) short8 rotate(short8 x, short8 y);
// __attribute__((overloadable)) short16 rotate(short16 x, short16 y);
// __attribute__((overloadable)) ushort rotate(ushort x, ushort y);
// __attribute__((overloadable)) ushort2 rotate(ushort2 x, ushort2 y);
// __attribute__((overloadable)) ushort3 rotate(ushort3 x, ushort3 y);
// __attribute__((overloadable)) ushort4 rotate(ushort4 x, ushort4 y);
// __attribute__((overloadable)) ushort8 rotate(ushort8 x, ushort8 y);
// __attribute__((overloadable)) ushort16 rotate(ushort16 x, ushort16 y);
// __attribute__((overloadable)) int rotate(int x, int y);
// __attribute__((overloadable)) int2 rotate(int2 x, int2 y);
// __attribute__((overloadable)) int3 rotate(int3 x, int3 y);
// __attribute__((overloadable)) int4 rotate(int4 x, int4 y);
// __attribute__((overloadable)) int8 rotate(int8 x, int8 y);
// __attribute__((overloadable)) int16 rotate(int16 x, int16 y);
// __attribute__((overloadable)) uint rotate(uint x, uint y);
// __attribute__((overloadable)) uint2 rotate(uint2 x, uint2 y);
// __attribute__((overloadable)) uint3 rotate(uint3 x, uint3 y);
// __attribute__((overloadable)) uint4 rotate(uint4 x, uint4 y);
// __attribute__((overloadable)) uint8 rotate(uint8 x, uint8 y);
// __attribute__((overloadable)) uint16 rotate(uint16 x, uint16 y);
// __attribute__((overloadable)) long rotate(long x, long y);
// __attribute__((overloadable)) long2 rotate(long2 x, long2 y);
// __attribute__((overloadable)) long3 rotate(long3 x, long3 y);
// __attribute__((overloadable)) long4 rotate(long4 x, long4 y);
// __attribute__((overloadable)) long8 rotate(long8 x, long8 y);
// __attribute__((overloadable)) long16 rotate(long16 x, long16 y);
// __attribute__((overloadable)) ulong rotate(ulong x, ulong y);
// __attribute__((overloadable)) ulong2 rotate(ulong2 x, ulong2 y);
// __attribute__((overloadable)) ulong3 rotate(ulong3 x, ulong3 y);
// __attribute__((overloadable)) ulong4 rotate(ulong4 x, ulong4 y);
// __attribute__((overloadable)) ulong8 rotate(ulong8 x, ulong8 y);
// __attribute__((overloadable)) ulong16 rotate(ulong16 x, ulong16 y);
// __attribute__((overloadable)) char sub_sat(char x, char y);
// __attribute__((overloadable)) char2 sub_sat(char2 x, char2 y);
// __attribute__((overloadable)) char3 sub_sat(char3 x, char3 y);
// __attribute__((overloadable)) char4 sub_sat(char4 x, char4 y);
// __attribute__((overloadable)) char8 sub_sat(char8 x, char8 y);
// __attribute__((overloadable)) char16 sub_sat(char16 x, char16 y);
// __attribute__((overloadable)) uchar sub_sat(uchar x, uchar y);
// __attribute__((overloadable)) uchar2 sub_sat(uchar2 x, uchar2 y);
// __attribute__((overloadable)) uchar3 sub_sat(uchar3 x, uchar3 y);
// __attribute__((overloadable)) uchar4 sub_sat(uchar4 x, uchar4 y);
// __attribute__((overloadable)) uchar8 sub_sat(uchar8 x, uchar8 y);
// __attribute__((overloadable)) uchar16 sub_sat(uchar16 x, uchar16 y);
// __attribute__((overloadable)) short sub_sat(short x, short y);
// __attribute__((overloadable)) short2 sub_sat(short2 x, short2 y);
// __attribute__((overloadable)) short3 sub_sat(short3 x, short3 y);
// __attribute__((overloadable)) short4 sub_sat(short4 x, short4 y);
// __attribute__((overloadable)) short8 sub_sat(short8 x, short8 y);
// __attribute__((overloadable)) short16 sub_sat(short16 x, short16 y);
// __attribute__((overloadable)) ushort sub_sat(ushort x, ushort y);
// __attribute__((overloadable)) ushort2 sub_sat(ushort2 x, ushort2 y);
// __attribute__((overloadable)) ushort3 sub_sat(ushort3 x, ushort3 y);
// __attribute__((overloadable)) ushort4 sub_sat(ushort4 x, ushort4 y);
// __attribute__((overloadable)) ushort8 sub_sat(ushort8 x, ushort8 y);
// __attribute__((overloadable)) ushort16 sub_sat(ushort16 x, ushort16 y);
// __attribute__((overloadable)) int sub_sat(int x, int y);
// __attribute__((overloadable)) int2 sub_sat(int2 x, int2 y);
// __attribute__((overloadable)) int3 sub_sat(int3 x, int3 y);
// __attribute__((overloadable)) int4 sub_sat(int4 x, int4 y);
// __attribute__((overloadable)) int8 sub_sat(int8 x, int8 y);
// __attribute__((overloadable)) int16 sub_sat(int16 x, int16 y);
// __attribute__((overloadable)) uint sub_sat(uint x, uint y);
// __attribute__((overloadable)) uint2 sub_sat(uint2 x, uint2 y);
// __attribute__((overloadable)) uint3 sub_sat(uint3 x, uint3 y);
// __attribute__((overloadable)) uint4 sub_sat(uint4 x, uint4 y);
// __attribute__((overloadable)) uint8 sub_sat(uint8 x, uint8 y);
// __attribute__((overloadable)) uint16 sub_sat(uint16 x, uint16 y);
// __attribute__((overloadable)) long sub_sat(long x, long y);
// __attribute__((overloadable)) long2 sub_sat(long2 x, long2 y);
// __attribute__((overloadable)) long3 sub_sat(long3 x, long3 y);
// __attribute__((overloadable)) long4 sub_sat(long4 x, long4 y);
// __attribute__((overloadable)) long8 sub_sat(long8 x, long8 y);
// __attribute__((overloadable)) long16 sub_sat(long16 x, long16 y);
// __attribute__((overloadable)) ulong sub_sat(ulong x, ulong y);
// __attribute__((overloadable)) ulong2 sub_sat(ulong2 x, ulong2 y);
// __attribute__((overloadable)) ulong3 sub_sat(ulong3 x, ulong3 y);
// __attribute__((overloadable)) ulong4 sub_sat(ulong4 x, ulong4 y);
// __attribute__((overloadable)) ulong8 sub_sat(ulong8 x, ulong8 y);
// __attribute__((overloadable)) ulong16 sub_sat(ulong16 x, ulong16 y);
// __attribute__((overloadable)) short upsample(char hi, uchar lo); __attribute__((overloadable)) short2 upsample(char2 hi, uchar2 lo); __attribute__((overloadable)) short3 upsample(char3 hi, uchar3 lo); __attribute__((overloadable)) short4 upsample(char4 hi, uchar4 lo); __attribute__((overloadable)) short8 upsample(char8 hi, uchar8 lo); __attribute__((overloadable)) short16 upsample(char16 hi, uchar16 lo); __attribute__((overloadable)) ushort upsample(uchar hi, uchar lo); __attribute__((overloadable)) ushort2 upsample(uchar2 hi, uchar2 lo); __attribute__((overloadable)) ushort3 upsample(uchar3 hi, uchar3 lo); __attribute__((overloadable)) ushort4 upsample(uchar4 hi, uchar4 lo); __attribute__((overloadable)) ushort8 upsample(uchar8 hi, uchar8 lo); __attribute__((overloadable)) ushort16 upsample(uchar16 hi, uchar16 lo); __attribute__((overloadable)) int upsample(short hi, ushort lo); __attribute__((overloadable)) int2 upsample(short2 hi, ushort2 lo); __attribute__((overloadable)) int3 upsample(short3 hi, ushort3 lo); __attribute__((overloadable)) int4 upsample(short4 hi, ushort4 lo); __attribute__((overloadable)) int8 upsample(short8 hi, ushort8 lo); __attribute__((overloadable)) int16 upsample(short16 hi, ushort16 lo); __attribute__((overloadable)) uint upsample(ushort hi, ushort lo); __attribute__((overloadable)) uint2 upsample(ushort2 hi, ushort2 lo); __attribute__((overloadable)) uint3 upsample(ushort3 hi, ushort3 lo); __attribute__((overloadable)) uint4 upsample(ushort4 hi, ushort4 lo); __attribute__((overloadable)) uint8 upsample(ushort8 hi, ushort8 lo); __attribute__((overloadable)) uint16 upsample(ushort16 hi, ushort16 lo); __attribute__((overloadable)) long upsample(int hi, uint lo); __attribute__((overloadable)) long2 upsample(int2 hi, uint2 lo); __attribute__((overloadable)) long3 upsample(int3 hi, uint3 lo); __attribute__((overloadable)) long4 upsample(int4 hi, uint4 lo); __attribute__((overloadable)) long8 upsample(int8 hi, uint8 lo); __attribute__((overloadable)) long16 upsample(int16 hi, uint16 lo); __attribute__((overloadable)) ulong upsample(uint hi, uint lo); __attribute__((overloadable)) ulong2 upsample(uint2 hi, uint2 lo); __attribute__((overloadable)) ulong3 upsample(uint3 hi, uint3 lo); __attribute__((overloadable)) ulong4 upsample(uint4 hi, uint4 lo); __attribute__((overloadable)) ulong8 upsample(uint8 hi, uint8 lo); __attribute__((overloadable)) ulong16 upsample(uint16 hi, uint16 lo);
// __attribute__((overloadable)) char clamp(char x, char y, char z);
// __attribute__((overloadable)) char2 clamp(char2 x, char2 y, char2 z);
// __attribute__((overloadable)) char2 clamp(char2 x, char y, char z);
// __attribute__((overloadable)) char3 clamp(char3 x, char3 y, char3 z);
// __attribute__((overloadable)) char3 clamp(char3 x, char y, char z);
// __attribute__((overloadable)) char4 clamp(char4 x, char4 y, char4 z);
// __attribute__((overloadable)) char4 clamp(char4 x, char y, char z);
// __attribute__((overloadable)) char8 clamp(char8 x, char8 y, char8 z);
// __attribute__((overloadable)) char8 clamp(char8 x, char y, char z);
// __attribute__((overloadable)) char16 clamp(char16 x, char16 y, char16 z);
// __attribute__((overloadable)) char16 clamp(char16 x, char y, char z);
// __attribute__((overloadable)) uchar clamp(uchar x, uchar y, uchar z);
// __attribute__((overloadable)) uchar2 clamp(uchar2 x, uchar2 y, uchar2 z);
// __attribute__((overloadable)) uchar2 clamp(uchar2 x, uchar y, uchar z);
// __attribute__((overloadable)) uchar3 clamp(uchar3 x, uchar3 y, uchar3 z);
// __attribute__((overloadable)) uchar3 clamp(uchar3 x, uchar y, uchar z);
// __attribute__((overloadable)) uchar4 clamp(uchar4 x, uchar4 y, uchar4 z);
// __attribute__((overloadable)) uchar4 clamp(uchar4 x, uchar y, uchar z);
// __attribute__((overloadable)) uchar8 clamp(uchar8 x, uchar8 y, uchar8 z);
// __attribute__((overloadable)) uchar8 clamp(uchar8 x, uchar y, uchar z);
// __attribute__((overloadable)) uchar16 clamp(uchar16 x, uchar16 y, uchar16 z);
// __attribute__((overloadable)) uchar16 clamp(uchar16 x, uchar y, uchar z);
// __attribute__((overloadable)) short clamp(short x, short y, short z);
// __attribute__((overloadable)) short2 clamp(short2 x, short2 y, short2 z);
// __attribute__((overloadable)) short2 clamp(short2 x, short y, short z);
// __attribute__((overloadable)) short3 clamp(short3 x, short3 y, short3 z);
// __attribute__((overloadable)) short3 clamp(short3 x, short y, short z);
// __attribute__((overloadable)) short4 clamp(short4 x, short4 y, short4 z);
// __attribute__((overloadable)) short4 clamp(short4 x, short y, short z);
// __attribute__((overloadable)) short8 clamp(short8 x, short8 y, short8 z);
// __attribute__((overloadable)) short8 clamp(short8 x, short y, short z);
// __attribute__((overloadable)) short16 clamp(short16 x, short16 y, short16 z);
// __attribute__((overloadable)) short16 clamp(short16 x, short y, short z);
// __attribute__((overloadable)) ushort clamp(ushort x, ushort y, ushort z);
// __attribute__((overloadable)) ushort2 clamp(ushort2 x, ushort2 y, ushort2 z);
// __attribute__((overloadable)) ushort2 clamp(ushort2 x, ushort y, ushort z);
// __attribute__((overloadable)) ushort3 clamp(ushort3 x, ushort3 y, ushort3 z);
// __attribute__((overloadable)) ushort3 clamp(ushort3 x, ushort y, ushort z);
// __attribute__((overloadable)) ushort4 clamp(ushort4 x, ushort4 y, ushort4 z);
// __attribute__((overloadable)) ushort4 clamp(ushort4 x, ushort y, ushort z);
// __attribute__((overloadable)) ushort8 clamp(ushort8 x, ushort8 y, ushort8 z);
// __attribute__((overloadable)) ushort8 clamp(ushort8 x, ushort y, ushort z);
// __attribute__((overloadable)) ushort16 clamp(ushort16 x, ushort16 y, ushort16 z);
// __attribute__((overloadable)) ushort16 clamp(ushort16 x, ushort y, ushort z);
// __attribute__((overloadable)) int clamp(int x, int y, int z);
// __attribute__((overloadable)) int2 clamp(int2 x, int2 y, int2 z);
// __attribute__((overloadable)) int2 clamp(int2 x, int y, int z);
// __attribute__((overloadable)) int3 clamp(int3 x, int3 y, int3 z);
// __attribute__((overloadable)) int3 clamp(int3 x, int y, int z);
// __attribute__((overloadable)) int4 clamp(int4 x, int4 y, int4 z);
// __attribute__((overloadable)) int4 clamp(int4 x, int y, int z);
// __attribute__((overloadable)) int8 clamp(int8 x, int8 y, int8 z);
// __attribute__((overloadable)) int8 clamp(int8 x, int y, int z);
// __attribute__((overloadable)) int16 clamp(int16 x, int16 y, int16 z);
// __attribute__((overloadable)) int16 clamp(int16 x, int y, int z);
// __attribute__((overloadable)) uint clamp(uint x, uint y, uint z);
// __attribute__((overloadable)) uint2 clamp(uint2 x, uint2 y, uint2 z);
// __attribute__((overloadable)) uint2 clamp(uint2 x, uint y, uint z);
// __attribute__((overloadable)) uint3 clamp(uint3 x, uint3 y, uint3 z);
// __attribute__((overloadable)) uint3 clamp(uint3 x, uint y, uint z);
// __attribute__((overloadable)) uint4 clamp(uint4 x, uint4 y, uint4 z);
// __attribute__((overloadable)) uint4 clamp(uint4 x, uint y, uint z);
// __attribute__((overloadable)) uint8 clamp(uint8 x, uint8 y, uint8 z);
// __attribute__((overloadable)) uint8 clamp(uint8 x, uint y, uint z);
// __attribute__((overloadable)) uint16 clamp(uint16 x, uint16 y, uint16 z);
// __attribute__((overloadable)) uint16 clamp(uint16 x, uint y, uint z);
// __attribute__((overloadable)) long clamp(long x, long y, long z);
// __attribute__((overloadable)) long2 clamp(long2 x, long2 y, long2 z);
// __attribute__((overloadable)) long2 clamp(long2 x, long y, long z);
// __attribute__((overloadable)) long3 clamp(long3 x, long3 y, long3 z);
// __attribute__((overloadable)) long3 clamp(long3 x, long y, long z);
// __attribute__((overloadable)) long4 clamp(long4 x, long4 y, long4 z);
// __attribute__((overloadable)) long4 clamp(long4 x, long y, long z);
// __attribute__((overloadable)) long8 clamp(long8 x, long8 y, long8 z);
// __attribute__((overloadable)) long8 clamp(long8 x, long y, long z);
// __attribute__((overloadable)) long16 clamp(long16 x, long16 y, long16 z);
// __attribute__((overloadable)) long16 clamp(long16 x, long y, long z);
// __attribute__((overloadable)) ulong clamp(ulong x, ulong y, ulong z);
// __attribute__((overloadable)) ulong2 clamp(ulong2 x, ulong2 y, ulong2 z);
// __attribute__((overloadable)) ulong2 clamp(ulong2 x, ulong y, ulong z);
// __attribute__((overloadable)) ulong3 clamp(ulong3 x, ulong3 y, ulong3 z);
// __attribute__((overloadable)) ulong3 clamp(ulong3 x, ulong y, ulong z);
// __attribute__((overloadable)) ulong4 clamp(ulong4 x, ulong4 y, ulong4 z);
// __attribute__((overloadable)) ulong4 clamp(ulong4 x, ulong y, ulong z);
// __attribute__((overloadable)) ulong8 clamp(ulong8 x, ulong8 y, ulong8 z);
// __attribute__((overloadable)) ulong8 clamp(ulong8 x, ulong y, ulong z);
// __attribute__((overloadable)) ulong16 clamp(ulong16 x, ulong16 y, ulong16 z);
// __attribute__((overloadable)) ulong16 clamp(ulong16 x, ulong y, ulong z);
// __attribute__((overloadable)) float clamp(float x, float y, float z);
// __attribute__((overloadable)) float2 clamp(float2 x, float2 y, float2 z);
// __attribute__((overloadable)) float2 clamp(float2 x, float y, float z);
// __attribute__((overloadable)) float3 clamp(float3 x, float3 y, float3 z);
// __attribute__((overloadable)) float3 clamp(float3 x, float y, float z);
// __attribute__((overloadable)) float4 clamp(float4 x, float4 y, float4 z);
// __attribute__((overloadable)) float4 clamp(float4 x, float y, float z);
// __attribute__((overloadable)) float8 clamp(float8 x, float8 y, float8 z);
// __attribute__((overloadable)) float8 clamp(float8 x, float y, float z);
// __attribute__((overloadable)) float16 clamp(float16 x, float16 y, float16 z);
// __attribute__((overloadable)) float16 clamp(float16 x, float y, float z);
// __attribute__((overloadable)) double clamp(double x, double y, double z);
// __attribute__((overloadable)) double2 clamp(double2 x, double2 y, double2 z);
// __attribute__((overloadable)) double2 clamp(double2 x, double y, double z);
// __attribute__((overloadable)) double3 clamp(double3 x, double3 y, double3 z);
// __attribute__((overloadable)) double3 clamp(double3 x, double y, double z);
// __attribute__((overloadable)) double4 clamp(double4 x, double4 y, double4 z);
// __attribute__((overloadable)) double4 clamp(double4 x, double y, double z);
// __attribute__((overloadable)) double8 clamp(double8 x, double8 y, double8 z);
// __attribute__((overloadable)) double8 clamp(double8 x, double y, double z);
// __attribute__((overloadable)) double16 clamp(double16 x, double16 y, double16 z);
// __attribute__((overloadable)) double16 clamp(double16 x, double y, double z);
// __attribute__((overloadable)) char max(char a, char b);
// __attribute__((overloadable)) char2 max(char2 a, char2 b);
// __attribute__((overloadable)) char2 max(char2 a, char b);
// __attribute__((overloadable)) char3 max(char3 a, char3 b);
// __attribute__((overloadable)) char3 max(char3 a, char b);
// __attribute__((overloadable)) char4 max(char4 a, char4 b);
// __attribute__((overloadable)) char4 max(char4 a, char b);
// __attribute__((overloadable)) char8 max(char8 a, char8 b);
// __attribute__((overloadable)) char8 max(char8 a, char b);
// __attribute__((overloadable)) char16 max(char16 a, char16 b);
// __attribute__((overloadable)) char16 max(char16 a, char b);
// __attribute__((overloadable)) uchar max(uchar a, uchar b);
// __attribute__((overloadable)) uchar2 max(uchar2 a, uchar2 b);
// __attribute__((overloadable)) uchar2 max(uchar2 a, uchar b);
// __attribute__((overloadable)) uchar3 max(uchar3 a, uchar3 b);
// __attribute__((overloadable)) uchar3 max(uchar3 a, uchar b);
// __attribute__((overloadable)) uchar4 max(uchar4 a, uchar4 b);
// __attribute__((overloadable)) uchar4 max(uchar4 a, uchar b);
// __attribute__((overloadable)) uchar8 max(uchar8 a, uchar8 b);
// __attribute__((overloadable)) uchar8 max(uchar8 a, uchar b);
// __attribute__((overloadable)) uchar16 max(uchar16 a, uchar16 b);
// __attribute__((overloadable)) uchar16 max(uchar16 a, uchar b);
// __attribute__((overloadable)) short max(short a, short b);
// __attribute__((overloadable)) short2 max(short2 a, short2 b);
// __attribute__((overloadable)) short2 max(short2 a, short b);
// __attribute__((overloadable)) short3 max(short3 a, short3 b);
// __attribute__((overloadable)) short3 max(short3 a, short b);
// __attribute__((overloadable)) short4 max(short4 a, short4 b);
// __attribute__((overloadable)) short4 max(short4 a, short b);
// __attribute__((overloadable)) short8 max(short8 a, short8 b);
// __attribute__((overloadable)) short8 max(short8 a, short b);
// __attribute__((overloadable)) short16 max(short16 a, short16 b);
// __attribute__((overloadable)) short16 max(short16 a, short b);
// __attribute__((overloadable)) ushort max(ushort a, ushort b);
// __attribute__((overloadable)) ushort2 max(ushort2 a, ushort2 b);
// __attribute__((overloadable)) ushort2 max(ushort2 a, ushort b);
// __attribute__((overloadable)) ushort3 max(ushort3 a, ushort3 b);
// __attribute__((overloadable)) ushort3 max(ushort3 a, ushort b);
// __attribute__((overloadable)) ushort4 max(ushort4 a, ushort4 b);
// __attribute__((overloadable)) ushort4 max(ushort4 a, ushort b);
// __attribute__((overloadable)) ushort8 max(ushort8 a, ushort8 b);
// __attribute__((overloadable)) ushort8 max(ushort8 a, ushort b);
// __attribute__((overloadable)) ushort16 max(ushort16 a, ushort16 b);
// __attribute__((overloadable)) ushort16 max(ushort16 a, ushort b);
// __attribute__((overloadable)) int max(int a, int b);
// __attribute__((overloadable)) int2 max(int2 a, int2 b);
// __attribute__((overloadable)) int2 max(int2 a, int b);
// __attribute__((overloadable)) int3 max(int3 a, int3 b);
// __attribute__((overloadable)) int3 max(int3 a, int b);
// __attribute__((overloadable)) int4 max(int4 a, int4 b);
// __attribute__((overloadable)) int4 max(int4 a, int b);
// __attribute__((overloadable)) int8 max(int8 a, int8 b);
// __attribute__((overloadable)) int8 max(int8 a, int b);
// __attribute__((overloadable)) int16 max(int16 a, int16 b);
// __attribute__((overloadable)) int16 max(int16 a, int b);
// __attribute__((overloadable)) uint max(uint a, uint b);
// __attribute__((overloadable)) uint2 max(uint2 a, uint2 b);
// __attribute__((overloadable)) uint2 max(uint2 a, uint b);
// __attribute__((overloadable)) uint3 max(uint3 a, uint3 b);
// __attribute__((overloadable)) uint3 max(uint3 a, uint b);
// __attribute__((overloadable)) uint4 max(uint4 a, uint4 b);
// __attribute__((overloadable)) uint4 max(uint4 a, uint b);
// __attribute__((overloadable)) uint8 max(uint8 a, uint8 b);
// __attribute__((overloadable)) uint8 max(uint8 a, uint b);
// __attribute__((overloadable)) uint16 max(uint16 a, uint16 b);
// __attribute__((overloadable)) uint16 max(uint16 a, uint b);
// __attribute__((overloadable)) long max(long a, long b);
// __attribute__((overloadable)) long2 max(long2 a, long2 b);
// __attribute__((overloadable)) long2 max(long2 a, long b);
// __attribute__((overloadable)) long3 max(long3 a, long3 b);
// __attribute__((overloadable)) long3 max(long3 a, long b);
// __attribute__((overloadable)) long4 max(long4 a, long4 b);
// __attribute__((overloadable)) long4 max(long4 a, long b);
// __attribute__((overloadable)) long8 max(long8 a, long8 b);
// __attribute__((overloadable)) long8 max(long8 a, long b);
// __attribute__((overloadable)) long16 max(long16 a, long16 b);
// __attribute__((overloadable)) long16 max(long16 a, long b);
// __attribute__((overloadable)) ulong max(ulong a, ulong b);
// __attribute__((overloadable)) ulong2 max(ulong2 a, ulong2 b);
// __attribute__((overloadable)) ulong2 max(ulong2 a, ulong b);
// __attribute__((overloadable)) ulong3 max(ulong3 a, ulong3 b);
// __attribute__((overloadable)) ulong3 max(ulong3 a, ulong b);
// __attribute__((overloadable)) ulong4 max(ulong4 a, ulong4 b);
// __attribute__((overloadable)) ulong4 max(ulong4 a, ulong b);
// __attribute__((overloadable)) ulong8 max(ulong8 a, ulong8 b);
// __attribute__((overloadable)) ulong8 max(ulong8 a, ulong b);
// __attribute__((overloadable)) ulong16 max(ulong16 a, ulong16 b);
// __attribute__((overloadable)) ulong16 max(ulong16 a, ulong b);
// __attribute__((overloadable)) float max(float a, float b);
// __attribute__((overloadable)) float2 max(float2 a, float2 b);
// __attribute__((overloadable)) float2 max(float2 a, float b);
// __attribute__((overloadable)) float3 max(float3 a, float3 b);
// __attribute__((overloadable)) float3 max(float3 a, float b);
// __attribute__((overloadable)) float4 max(float4 a, float4 b);
// __attribute__((overloadable)) float4 max(float4 a, float b);
// __attribute__((overloadable)) float8 max(float8 a, float8 b);
// __attribute__((overloadable)) float8 max(float8 a, float b);
// __attribute__((overloadable)) float16 max(float16 a, float16 b);
// __attribute__((overloadable)) float16 max(float16 a, float b);
// __attribute__((overloadable)) double max(double a, double b);
// __attribute__((overloadable)) double2 max(double2 a, double2 b);
// __attribute__((overloadable)) double2 max(double2 a, double b);
// __attribute__((overloadable)) double3 max(double3 a, double3 b);
// __attribute__((overloadable)) double3 max(double3 a, double b);
// __attribute__((overloadable)) double4 max(double4 a, double4 b);
// __attribute__((overloadable)) double4 max(double4 a, double b);
// __attribute__((overloadable)) double8 max(double8 a, double8 b);
// __attribute__((overloadable)) double8 max(double8 a, double b);
// __attribute__((overloadable)) double16 max(double16 a, double16 b);
// __attribute__((overloadable)) double16 max(double16 a, double b);
// __attribute__((overloadable)) char min(char a, char b);
// __attribute__((overloadable)) char2 min(char2 a, char2 b);
// __attribute__((overloadable)) char2 min(char2 a, char b);
// __attribute__((overloadable)) char3 min(char3 a, char3 b);
// __attribute__((overloadable)) char3 min(char3 a, char b);
// __attribute__((overloadable)) char4 min(char4 a, char4 b);
// __attribute__((overloadable)) char4 min(char4 a, char b);
// __attribute__((overloadable)) char8 min(char8 a, char8 b);
// __attribute__((overloadable)) char8 min(char8 a, char b);
// __attribute__((overloadable)) char16 min(char16 a, char16 b);
// __attribute__((overloadable)) char16 min(char16 a, char b);
// __attribute__((overloadable)) uchar min(uchar a, uchar b);
// __attribute__((overloadable)) uchar2 min(uchar2 a, uchar2 b);
// __attribute__((overloadable)) uchar2 min(uchar2 a, uchar b);
// __attribute__((overloadable)) uchar3 min(uchar3 a, uchar3 b);
// __attribute__((overloadable)) uchar3 min(uchar3 a, uchar b);
// __attribute__((overloadable)) uchar4 min(uchar4 a, uchar4 b);
// __attribute__((overloadable)) uchar4 min(uchar4 a, uchar b);
// __attribute__((overloadable)) uchar8 min(uchar8 a, uchar8 b);
// __attribute__((overloadable)) uchar8 min(uchar8 a, uchar b);
// __attribute__((overloadable)) uchar16 min(uchar16 a, uchar16 b);
// __attribute__((overloadable)) uchar16 min(uchar16 a, uchar b);
// __attribute__((overloadable)) short min(short a, short b);
// __attribute__((overloadable)) short2 min(short2 a, short2 b);
// __attribute__((overloadable)) short2 min(short2 a, short b);
// __attribute__((overloadable)) short3 min(short3 a, short3 b);
// __attribute__((overloadable)) short3 min(short3 a, short b);
// __attribute__((overloadable)) short4 min(short4 a, short4 b);
// __attribute__((overloadable)) short4 min(short4 a, short b);
// __attribute__((overloadable)) short8 min(short8 a, short8 b);
// __attribute__((overloadable)) short8 min(short8 a, short b);
// __attribute__((overloadable)) short16 min(short16 a, short16 b);
// __attribute__((overloadable)) short16 min(short16 a, short b);
// __attribute__((overloadable)) ushort min(ushort a, ushort b);
// __attribute__((overloadable)) ushort2 min(ushort2 a, ushort2 b);
// __attribute__((overloadable)) ushort2 min(ushort2 a, ushort b);
// __attribute__((overloadable)) ushort3 min(ushort3 a, ushort3 b);
// __attribute__((overloadable)) ushort3 min(ushort3 a, ushort b);
// __attribute__((overloadable)) ushort4 min(ushort4 a, ushort4 b);
// __attribute__((overloadable)) ushort4 min(ushort4 a, ushort b);
// __attribute__((overloadable)) ushort8 min(ushort8 a, ushort8 b);
// __attribute__((overloadable)) ushort8 min(ushort8 a, ushort b);
// __attribute__((overloadable)) ushort16 min(ushort16 a, ushort16 b);
// __attribute__((overloadable)) ushort16 min(ushort16 a, ushort b);
// __attribute__((overloadable)) int min(int a, int b);
// __attribute__((overloadable)) int2 min(int2 a, int2 b);
// __attribute__((overloadable)) int2 min(int2 a, int b);
// __attribute__((overloadable)) int3 min(int3 a, int3 b);
// __attribute__((overloadable)) int3 min(int3 a, int b);
// __attribute__((overloadable)) int4 min(int4 a, int4 b);
// __attribute__((overloadable)) int4 min(int4 a, int b);
// __attribute__((overloadable)) int8 min(int8 a, int8 b);
// __attribute__((overloadable)) int8 min(int8 a, int b);
// __attribute__((overloadable)) int16 min(int16 a, int16 b);
// __attribute__((overloadable)) int16 min(int16 a, int b);
// __attribute__((overloadable)) uint min(uint a, uint b);
// __attribute__((overloadable)) uint2 min(uint2 a, uint2 b);
// __attribute__((overloadable)) uint2 min(uint2 a, uint b);
// __attribute__((overloadable)) uint3 min(uint3 a, uint3 b);
// __attribute__((overloadable)) uint3 min(uint3 a, uint b);
// __attribute__((overloadable)) uint4 min(uint4 a, uint4 b);
// __attribute__((overloadable)) uint4 min(uint4 a, uint b);
// __attribute__((overloadable)) uint8 min(uint8 a, uint8 b);
// __attribute__((overloadable)) uint8 min(uint8 a, uint b);
// __attribute__((overloadable)) uint16 min(uint16 a, uint16 b);
// __attribute__((overloadable)) uint16 min(uint16 a, uint b);
// __attribute__((overloadable)) long min(long a, long b);
// __attribute__((overloadable)) long2 min(long2 a, long2 b);
// __attribute__((overloadable)) long2 min(long2 a, long b);
// __attribute__((overloadable)) long3 min(long3 a, long3 b);
// __attribute__((overloadable)) long3 min(long3 a, long b);
// __attribute__((overloadable)) long4 min(long4 a, long4 b);
// __attribute__((overloadable)) long4 min(long4 a, long b);
// __attribute__((overloadable)) long8 min(long8 a, long8 b);
// __attribute__((overloadable)) long8 min(long8 a, long b);
// __attribute__((overloadable)) long16 min(long16 a, long16 b);
// __attribute__((overloadable)) long16 min(long16 a, long b);
// __attribute__((overloadable)) ulong min(ulong a, ulong b);
// __attribute__((overloadable)) ulong2 min(ulong2 a, ulong2 b);
// __attribute__((overloadable)) ulong2 min(ulong2 a, ulong b);
// __attribute__((overloadable)) ulong3 min(ulong3 a, ulong3 b);
// __attribute__((overloadable)) ulong3 min(ulong3 a, ulong b);
// __attribute__((overloadable)) ulong4 min(ulong4 a, ulong4 b);
// __attribute__((overloadable)) ulong4 min(ulong4 a, ulong b);
// __attribute__((overloadable)) ulong8 min(ulong8 a, ulong8 b);
// __attribute__((overloadable)) ulong8 min(ulong8 a, ulong b);
// __attribute__((overloadable)) ulong16 min(ulong16 a, ulong16 b);
// __attribute__((overloadable)) ulong16 min(ulong16 a, ulong b);
// __attribute__((overloadable)) float min(float a, float b);
// __attribute__((overloadable)) float2 min(float2 a, float2 b);
// __attribute__((overloadable)) float2 min(float2 a, float b);
// __attribute__((overloadable)) float3 min(float3 a, float3 b);
// __attribute__((overloadable)) float3 min(float3 a, float b);
// __attribute__((overloadable)) float4 min(float4 a, float4 b);
// __attribute__((overloadable)) float4 min(float4 a, float b);
// __attribute__((overloadable)) float8 min(float8 a, float8 b);
// __attribute__((overloadable)) float8 min(float8 a, float b);
// __attribute__((overloadable)) float16 min(float16 a, float16 b);
// __attribute__((overloadable)) float16 min(float16 a, float b);
// __attribute__((overloadable)) double min(double a, double b);
// __attribute__((overloadable)) double2 min(double2 a, double2 b);
// __attribute__((overloadable)) double2 min(double2 a, double b);
// __attribute__((overloadable)) double3 min(double3 a, double3 b);
// __attribute__((overloadable)) double3 min(double3 a, double b);
// __attribute__((overloadable)) double4 min(double4 a, double4 b);
// __attribute__((overloadable)) double4 min(double4 a, double b);
// __attribute__((overloadable)) double8 min(double8 a, double8 b);
// __attribute__((overloadable)) double8 min(double8 a, double b);
// __attribute__((overloadable)) double16 min(double16 a, double16 b);
// __attribute__((overloadable)) double16 min(double16 a, double b);
// __attribute__((overloadable)) double2 vload2(size_t offset, const __private double *x); __attribute__((overloadable)) double3 vload3(size_t offset, const __private double *x); __attribute__((overloadable)) double4 vload4(size_t offset, const __private double *x); __attribute__((overloadable)) double8 vload8(size_t offset, const __private double *x); __attribute__((overloadable)) double16 vload16(size_t offset, const __private double *x); __attribute__((overloadable)) double2 vload2(size_t offset, const __local double *x); __attribute__((overloadable)) double3 vload3(size_t offset, const __local double *x); __attribute__((overloadable)) double4 vload4(size_t offset, const __local double *x); __attribute__((overloadable)) double8 vload8(size_t offset, const __local double *x); __attribute__((overloadable)) double16 vload16(size_t offset, const __local double *x); __attribute__((overloadable)) double2 vload2(size_t offset, const __constant double *x); __attribute__((overloadable)) double3 vload3(size_t offset, const __constant double *x); __attribute__((overloadable)) double4 vload4(size_t offset, const __constant double *x); __attribute__((overloadable)) double8 vload8(size_t offset, const __constant double *x); __attribute__((overloadable)) double16 vload16(size_t offset, const __constant double *x); __attribute__((overloadable)) double2 vload2(size_t offset, const __global double *x); __attribute__((overloadable)) double3 vload3(size_t offset, const __global double *x); __attribute__((overloadable)) double4 vload4(size_t offset, const __global double *x); __attribute__((overloadable)) double8 vload8(size_t offset, const __global double *x); __attribute__((overloadable)) double16 vload16(size_t offset, const __global double *x); __attribute__((overloadable)) char2 vload2(size_t offset, const __private char *x); __attribute__((overloadable)) char3 vload3(size_t offset, const __private char *x); __attribute__((overloadable)) char4 vload4(size_t offset, const __private char *x); __attribute__((overloadable)) char8 vload8(size_t offset, const __private char *x); __attribute__((overloadable)) char16 vload16(size_t offset, const __private char *x); __attribute__((overloadable)) char2 vload2(size_t offset, const __local char *x); __attribute__((overloadable)) char3 vload3(size_t offset, const __local char *x); __attribute__((overloadable)) char4 vload4(size_t offset, const __local char *x); __attribute__((overloadable)) char8 vload8(size_t offset, const __local char *x); __attribute__((overloadable)) char16 vload16(size_t offset, const __local char *x); __attribute__((overloadable)) char2 vload2(size_t offset, const __constant char *x); __attribute__((overloadable)) char3 vload3(size_t offset, const __constant char *x); __attribute__((overloadable)) char4 vload4(size_t offset, const __constant char *x); __attribute__((overloadable)) char8 vload8(size_t offset, const __constant char *x); __attribute__((overloadable)) char16 vload16(size_t offset, const __constant char *x); __attribute__((overloadable)) char2 vload2(size_t offset, const __global char *x); __attribute__((overloadable)) char3 vload3(size_t offset, const __global char *x); __attribute__((overloadable)) char4 vload4(size_t offset, const __global char *x); __attribute__((overloadable)) char8 vload8(size_t offset, const __global char *x); __attribute__((overloadable)) char16 vload16(size_t offset, const __global char *x); __attribute__((overloadable)) uchar2 vload2(size_t offset, const __private uchar *x); __attribute__((overloadable)) uchar3 vload3(size_t offset, const __private uchar *x); __attribute__((overloadable)) uchar4 vload4(size_t offset, const __private uchar *x); __attribute__((overloadable)) uchar8 vload8(size_t offset, const __private uchar *x); __attribute__((overloadable)) uchar16 vload16(size_t offset, const __private uchar *x); __attribute__((overloadable)) uchar2 vload2(size_t offset, const __local uchar *x); __attribute__((overloadable)) uchar3 vload3(size_t offset, const __local uchar *x); __attribute__((overloadable)) uchar4 vload4(size_t offset, const __local uchar *x); __attribute__((overloadable)) uchar8 vload8(size_t offset, const __local uchar *x); __attribute__((overloadable)) uchar16 vload16(size_t offset, const __local uchar *x); __attribute__((overloadable)) uchar2 vload2(size_t offset, const __constant uchar *x); __attribute__((overloadable)) uchar3 vload3(size_t offset, const __constant uchar *x); __attribute__((overloadable)) uchar4 vload4(size_t offset, const __constant uchar *x); __attribute__((overloadable)) uchar8 vload8(size_t offset, const __constant uchar *x); __attribute__((overloadable)) uchar16 vload16(size_t offset, const __constant uchar *x); __attribute__((overloadable)) uchar2 vload2(size_t offset, const __global uchar *x); __attribute__((overloadable)) uchar3 vload3(size_t offset, const __global uchar *x); __attribute__((overloadable)) uchar4 vload4(size_t offset, const __global uchar *x); __attribute__((overloadable)) uchar8 vload8(size_t offset, const __global uchar *x); __attribute__((overloadable)) uchar16 vload16(size_t offset, const __global uchar *x); __attribute__((overloadable)) short2 vload2(size_t offset, const __private short *x); __attribute__((overloadable)) short3 vload3(size_t offset, const __private short *x); __attribute__((overloadable)) short4 vload4(size_t offset, const __private short *x); __attribute__((overloadable)) short8 vload8(size_t offset, const __private short *x); __attribute__((overloadable)) short16 vload16(size_t offset, const __private short *x); __attribute__((overloadable)) short2 vload2(size_t offset, const __local short *x); __attribute__((overloadable)) short3 vload3(size_t offset, const __local short *x); __attribute__((overloadable)) short4 vload4(size_t offset, const __local short *x); __attribute__((overloadable)) short8 vload8(size_t offset, const __local short *x); __attribute__((overloadable)) short16 vload16(size_t offset, const __local short *x); __attribute__((overloadable)) short2 vload2(size_t offset, const __constant short *x); __attribute__((overloadable)) short3 vload3(size_t offset, const __constant short *x); __attribute__((overloadable)) short4 vload4(size_t offset, const __constant short *x); __attribute__((overloadable)) short8 vload8(size_t offset, const __constant short *x); __attribute__((overloadable)) short16 vload16(size_t offset, const __constant short *x); __attribute__((overloadable)) short2 vload2(size_t offset, const __global short *x); __attribute__((overloadable)) short3 vload3(size_t offset, const __global short *x); __attribute__((overloadable)) short4 vload4(size_t offset, const __global short *x); __attribute__((overloadable)) short8 vload8(size_t offset, const __global short *x); __attribute__((overloadable)) short16 vload16(size_t offset, const __global short *x); __attribute__((overloadable)) ushort2 vload2(size_t offset, const __private ushort *x); __attribute__((overloadable)) ushort3 vload3(size_t offset, const __private ushort *x); __attribute__((overloadable)) ushort4 vload4(size_t offset, const __private ushort *x); __attribute__((overloadable)) ushort8 vload8(size_t offset, const __private ushort *x); __attribute__((overloadable)) ushort16 vload16(size_t offset, const __private ushort *x); __attribute__((overloadable)) ushort2 vload2(size_t offset, const __local ushort *x); __attribute__((overloadable)) ushort3 vload3(size_t offset, const __local ushort *x); __attribute__((overloadable)) ushort4 vload4(size_t offset, const __local ushort *x); __attribute__((overloadable)) ushort8 vload8(size_t offset, const __local ushort *x); __attribute__((overloadable)) ushort16 vload16(size_t offset, const __local ushort *x); __attribute__((overloadable)) ushort2 vload2(size_t offset, const __constant ushort *x); __attribute__((overloadable)) ushort3 vload3(size_t offset, const __constant ushort *x); __attribute__((overloadable)) ushort4 vload4(size_t offset, const __constant ushort *x); __attribute__((overloadable)) ushort8 vload8(size_t offset, const __constant ushort *x); __attribute__((overloadable)) ushort16 vload16(size_t offset, const __constant ushort *x); __attribute__((overloadable)) ushort2 vload2(size_t offset, const __global ushort *x); __attribute__((overloadable)) ushort3 vload3(size_t offset, const __global ushort *x); __attribute__((overloadable)) ushort4 vload4(size_t offset, const __global ushort *x); __attribute__((overloadable)) ushort8 vload8(size_t offset, const __global ushort *x); __attribute__((overloadable)) ushort16 vload16(size_t offset, const __global ushort *x); __attribute__((overloadable)) int2 vload2(size_t offset, const __private int *x); __attribute__((overloadable)) int3 vload3(size_t offset, const __private int *x); __attribute__((overloadable)) int4 vload4(size_t offset, const __private int *x); __attribute__((overloadable)) int8 vload8(size_t offset, const __private int *x); __attribute__((overloadable)) int16 vload16(size_t offset, const __private int *x); __attribute__((overloadable)) int2 vload2(size_t offset, const __local int *x); __attribute__((overloadable)) int3 vload3(size_t offset, const __local int *x); __attribute__((overloadable)) int4 vload4(size_t offset, const __local int *x); __attribute__((overloadable)) int8 vload8(size_t offset, const __local int *x); __attribute__((overloadable)) int16 vload16(size_t offset, const __local int *x); __attribute__((overloadable)) int2 vload2(size_t offset, const __constant int *x); __attribute__((overloadable)) int3 vload3(size_t offset, const __constant int *x); __attribute__((overloadable)) int4 vload4(size_t offset, const __constant int *x); __attribute__((overloadable)) int8 vload8(size_t offset, const __constant int *x); __attribute__((overloadable)) int16 vload16(size_t offset, const __constant int *x); __attribute__((overloadable)) int2 vload2(size_t offset, const __global int *x); __attribute__((overloadable)) int3 vload3(size_t offset, const __global int *x); __attribute__((overloadable)) int4 vload4(size_t offset, const __global int *x); __attribute__((overloadable)) int8 vload8(size_t offset, const __global int *x); __attribute__((overloadable)) int16 vload16(size_t offset, const __global int *x); __attribute__((overloadable)) uint2 vload2(size_t offset, const __private uint *x); __attribute__((overloadable)) uint3 vload3(size_t offset, const __private uint *x); __attribute__((overloadable)) uint4 vload4(size_t offset, const __private uint *x); __attribute__((overloadable)) uint8 vload8(size_t offset, const __private uint *x); __attribute__((overloadable)) uint16 vload16(size_t offset, const __private uint *x); __attribute__((overloadable)) uint2 vload2(size_t offset, const __local uint *x); __attribute__((overloadable)) uint3 vload3(size_t offset, const __local uint *x); __attribute__((overloadable)) uint4 vload4(size_t offset, const __local uint *x); __attribute__((overloadable)) uint8 vload8(size_t offset, const __local uint *x); __attribute__((overloadable)) uint16 vload16(size_t offset, const __local uint *x); __attribute__((overloadable)) uint2 vload2(size_t offset, const __constant uint *x); __attribute__((overloadable)) uint3 vload3(size_t offset, const __constant uint *x); __attribute__((overloadable)) uint4 vload4(size_t offset, const __constant uint *x); __attribute__((overloadable)) uint8 vload8(size_t offset, const __constant uint *x); __attribute__((overloadable)) uint16 vload16(size_t offset, const __constant uint *x); __attribute__((overloadable)) uint2 vload2(size_t offset, const __global uint *x); __attribute__((overloadable)) uint3 vload3(size_t offset, const __global uint *x); __attribute__((overloadable)) uint4 vload4(size_t offset, const __global uint *x); __attribute__((overloadable)) uint8 vload8(size_t offset, const __global uint *x); __attribute__((overloadable)) uint16 vload16(size_t offset, const __global uint *x); __attribute__((overloadable)) long2 vload2(size_t offset, const __private long *x); __attribute__((overloadable)) long3 vload3(size_t offset, const __private long *x); __attribute__((overloadable)) long4 vload4(size_t offset, const __private long *x); __attribute__((overloadable)) long8 vload8(size_t offset, const __private long *x); __attribute__((overloadable)) long16 vload16(size_t offset, const __private long *x); __attribute__((overloadable)) long2 vload2(size_t offset, const __local long *x); __attribute__((overloadable)) long3 vload3(size_t offset, const __local long *x); __attribute__((overloadable)) long4 vload4(size_t offset, const __local long *x); __attribute__((overloadable)) long8 vload8(size_t offset, const __local long *x); __attribute__((overloadable)) long16 vload16(size_t offset, const __local long *x); __attribute__((overloadable)) long2 vload2(size_t offset, const __constant long *x); __attribute__((overloadable)) long3 vload3(size_t offset, const __constant long *x); __attribute__((overloadable)) long4 vload4(size_t offset, const __constant long *x); __attribute__((overloadable)) long8 vload8(size_t offset, const __constant long *x); __attribute__((overloadable)) long16 vload16(size_t offset, const __constant long *x); __attribute__((overloadable)) long2 vload2(size_t offset, const __global long *x); __attribute__((overloadable)) long3 vload3(size_t offset, const __global long *x); __attribute__((overloadable)) long4 vload4(size_t offset, const __global long *x); __attribute__((overloadable)) long8 vload8(size_t offset, const __global long *x); __attribute__((overloadable)) long16 vload16(size_t offset, const __global long *x); __attribute__((overloadable)) ulong2 vload2(size_t offset, const __private ulong *x); __attribute__((overloadable)) ulong3 vload3(size_t offset, const __private ulong *x); __attribute__((overloadable)) ulong4 vload4(size_t offset, const __private ulong *x); __attribute__((overloadable)) ulong8 vload8(size_t offset, const __private ulong *x); __attribute__((overloadable)) ulong16 vload16(size_t offset, const __private ulong *x); __attribute__((overloadable)) ulong2 vload2(size_t offset, const __local ulong *x); __attribute__((overloadable)) ulong3 vload3(size_t offset, const __local ulong *x); __attribute__((overloadable)) ulong4 vload4(size_t offset, const __local ulong *x); __attribute__((overloadable)) ulong8 vload8(size_t offset, const __local ulong *x); __attribute__((overloadable)) ulong16 vload16(size_t offset, const __local ulong *x); __attribute__((overloadable)) ulong2 vload2(size_t offset, const __constant ulong *x); __attribute__((overloadable)) ulong3 vload3(size_t offset, const __constant ulong *x); __attribute__((overloadable)) ulong4 vload4(size_t offset, const __constant ulong *x); __attribute__((overloadable)) ulong8 vload8(size_t offset, const __constant ulong *x); __attribute__((overloadable)) ulong16 vload16(size_t offset, const __constant ulong *x); __attribute__((overloadable)) ulong2 vload2(size_t offset, const __global ulong *x); __attribute__((overloadable)) ulong3 vload3(size_t offset, const __global ulong *x); __attribute__((overloadable)) ulong4 vload4(size_t offset, const __global ulong *x); __attribute__((overloadable)) ulong8 vload8(size_t offset, const __global ulong *x); __attribute__((overloadable)) ulong16 vload16(size_t offset, const __global ulong *x); __attribute__((overloadable)) float2 vload2(size_t offset, const __private float *x); __attribute__((overloadable)) float3 vload3(size_t offset, const __private float *x); __attribute__((overloadable)) float4 vload4(size_t offset, const __private float *x); __attribute__((overloadable)) float8 vload8(size_t offset, const __private float *x); __attribute__((overloadable)) float16 vload16(size_t offset, const __private float *x); __attribute__((overloadable)) float2 vload2(size_t offset, const __local float *x); __attribute__((overloadable)) float3 vload3(size_t offset, const __local float *x); __attribute__((overloadable)) float4 vload4(size_t offset, const __local float *x); __attribute__((overloadable)) float8 vload8(size_t offset, const __local float *x); __attribute__((overloadable)) float16 vload16(size_t offset, const __local float *x); __attribute__((overloadable)) float2 vload2(size_t offset, const __constant float *x); __attribute__((overloadable)) float3 vload3(size_t offset, const __constant float *x); __attribute__((overloadable)) float4 vload4(size_t offset, const __constant float *x); __attribute__((overloadable)) float8 vload8(size_t offset, const __constant float *x); __attribute__((overloadable)) float16 vload16(size_t offset, const __constant float *x); __attribute__((overloadable)) float2 vload2(size_t offset, const __global float *x); __attribute__((overloadable)) float3 vload3(size_t offset, const __global float *x); __attribute__((overloadable)) float4 vload4(size_t offset, const __global float *x); __attribute__((overloadable)) float8 vload8(size_t offset, const __global float *x); __attribute__((overloadable)) float16 vload16(size_t offset, const __global float *x);
// __attribute__((overloadable)) void vstore2(double2 vec, size_t offset, __private double *out); __attribute__((overloadable)) void vstore3(double3 vec, size_t offset, __private double *out); __attribute__((overloadable)) void vstore4(double4 vec, size_t offset, __private double *out); __attribute__((overloadable)) void vstore8(double8 vec, size_t offset, __private double *out); __attribute__((overloadable)) void vstore16(double16 vec, size_t offset, __private double *out); __attribute__((overloadable)) void vstore2(double2 vec, size_t offset, __local double *out); __attribute__((overloadable)) void vstore3(double3 vec, size_t offset, __local double *out); __attribute__((overloadable)) void vstore4(double4 vec, size_t offset, __local double *out); __attribute__((overloadable)) void vstore8(double8 vec, size_t offset, __local double *out); __attribute__((overloadable)) void vstore16(double16 vec, size_t offset, __local double *out); __attribute__((overloadable)) void vstore2(double2 vec, size_t offset, __global double *out); __attribute__((overloadable)) void vstore3(double3 vec, size_t offset, __global double *out); __attribute__((overloadable)) void vstore4(double4 vec, size_t offset, __global double *out); __attribute__((overloadable)) void vstore8(double8 vec, size_t offset, __global double *out); __attribute__((overloadable)) void vstore16(double16 vec, size_t offset, __global double *out); __attribute__((overloadable)) void vstore2(char2 vec, size_t offset, __private char *out); __attribute__((overloadable)) void vstore3(char3 vec, size_t offset, __private char *out); __attribute__((overloadable)) void vstore4(char4 vec, size_t offset, __private char *out); __attribute__((overloadable)) void vstore8(char8 vec, size_t offset, __private char *out); __attribute__((overloadable)) void vstore16(char16 vec, size_t offset, __private char *out); __attribute__((overloadable)) void vstore2(char2 vec, size_t offset, __local char *out); __attribute__((overloadable)) void vstore3(char3 vec, size_t offset, __local char *out); __attribute__((overloadable)) void vstore4(char4 vec, size_t offset, __local char *out); __attribute__((overloadable)) void vstore8(char8 vec, size_t offset, __local char *out); __attribute__((overloadable)) void vstore16(char16 vec, size_t offset, __local char *out); __attribute__((overloadable)) void vstore2(char2 vec, size_t offset, __global char *out); __attribute__((overloadable)) void vstore3(char3 vec, size_t offset, __global char *out); __attribute__((overloadable)) void vstore4(char4 vec, size_t offset, __global char *out); __attribute__((overloadable)) void vstore8(char8 vec, size_t offset, __global char *out); __attribute__((overloadable)) void vstore16(char16 vec, size_t offset, __global char *out); __attribute__((overloadable)) void vstore2(uchar2 vec, size_t offset, __private uchar *out); __attribute__((overloadable)) void vstore3(uchar3 vec, size_t offset, __private uchar *out); __attribute__((overloadable)) void vstore4(uchar4 vec, size_t offset, __private uchar *out); __attribute__((overloadable)) void vstore8(uchar8 vec, size_t offset, __private uchar *out); __attribute__((overloadable)) void vstore16(uchar16 vec, size_t offset, __private uchar *out); __attribute__((overloadable)) void vstore2(uchar2 vec, size_t offset, __local uchar *out); __attribute__((overloadable)) void vstore3(uchar3 vec, size_t offset, __local uchar *out); __attribute__((overloadable)) void vstore4(uchar4 vec, size_t offset, __local uchar *out); __attribute__((overloadable)) void vstore8(uchar8 vec, size_t offset, __local uchar *out); __attribute__((overloadable)) void vstore16(uchar16 vec, size_t offset, __local uchar *out); __attribute__((overloadable)) void vstore2(uchar2 vec, size_t offset, __global uchar *out); __attribute__((overloadable)) void vstore3(uchar3 vec, size_t offset, __global uchar *out); __attribute__((overloadable)) void vstore4(uchar4 vec, size_t offset, __global uchar *out); __attribute__((overloadable)) void vstore8(uchar8 vec, size_t offset, __global uchar *out); __attribute__((overloadable)) void vstore16(uchar16 vec, size_t offset, __global uchar *out); __attribute__((overloadable)) void vstore2(short2 vec, size_t offset, __private short *out); __attribute__((overloadable)) void vstore3(short3 vec, size_t offset, __private short *out); __attribute__((overloadable)) void vstore4(short4 vec, size_t offset, __private short *out); __attribute__((overloadable)) void vstore8(short8 vec, size_t offset, __private short *out); __attribute__((overloadable)) void vstore16(short16 vec, size_t offset, __private short *out); __attribute__((overloadable)) void vstore2(short2 vec, size_t offset, __local short *out); __attribute__((overloadable)) void vstore3(short3 vec, size_t offset, __local short *out); __attribute__((overloadable)) void vstore4(short4 vec, size_t offset, __local short *out); __attribute__((overloadable)) void vstore8(short8 vec, size_t offset, __local short *out); __attribute__((overloadable)) void vstore16(short16 vec, size_t offset, __local short *out); __attribute__((overloadable)) void vstore2(short2 vec, size_t offset, __global short *out); __attribute__((overloadable)) void vstore3(short3 vec, size_t offset, __global short *out); __attribute__((overloadable)) void vstore4(short4 vec, size_t offset, __global short *out); __attribute__((overloadable)) void vstore8(short8 vec, size_t offset, __global short *out); __attribute__((overloadable)) void vstore16(short16 vec, size_t offset, __global short *out); __attribute__((overloadable)) void vstore2(ushort2 vec, size_t offset, __private ushort *out); __attribute__((overloadable)) void vstore3(ushort3 vec, size_t offset, __private ushort *out); __attribute__((overloadable)) void vstore4(ushort4 vec, size_t offset, __private ushort *out); __attribute__((overloadable)) void vstore8(ushort8 vec, size_t offset, __private ushort *out); __attribute__((overloadable)) void vstore16(ushort16 vec, size_t offset, __private ushort *out); __attribute__((overloadable)) void vstore2(ushort2 vec, size_t offset, __local ushort *out); __attribute__((overloadable)) void vstore3(ushort3 vec, size_t offset, __local ushort *out); __attribute__((overloadable)) void vstore4(ushort4 vec, size_t offset, __local ushort *out); __attribute__((overloadable)) void vstore8(ushort8 vec, size_t offset, __local ushort *out); __attribute__((overloadable)) void vstore16(ushort16 vec, size_t offset, __local ushort *out); __attribute__((overloadable)) void vstore2(ushort2 vec, size_t offset, __global ushort *out); __attribute__((overloadable)) void vstore3(ushort3 vec, size_t offset, __global ushort *out); __attribute__((overloadable)) void vstore4(ushort4 vec, size_t offset, __global ushort *out); __attribute__((overloadable)) void vstore8(ushort8 vec, size_t offset, __global ushort *out); __attribute__((overloadable)) void vstore16(ushort16 vec, size_t offset, __global ushort *out); __attribute__((overloadable)) void vstore2(int2 vec, size_t offset, __private int *out); __attribute__((overloadable)) void vstore3(int3 vec, size_t offset, __private int *out); __attribute__((overloadable)) void vstore4(int4 vec, size_t offset, __private int *out); __attribute__((overloadable)) void vstore8(int8 vec, size_t offset, __private int *out); __attribute__((overloadable)) void vstore16(int16 vec, size_t offset, __private int *out); __attribute__((overloadable)) void vstore2(int2 vec, size_t offset, __local int *out); __attribute__((overloadable)) void vstore3(int3 vec, size_t offset, __local int *out); __attribute__((overloadable)) void vstore4(int4 vec, size_t offset, __local int *out); __attribute__((overloadable)) void vstore8(int8 vec, size_t offset, __local int *out); __attribute__((overloadable)) void vstore16(int16 vec, size_t offset, __local int *out); __attribute__((overloadable)) void vstore2(int2 vec, size_t offset, __global int *out); __attribute__((overloadable)) void vstore3(int3 vec, size_t offset, __global int *out); __attribute__((overloadable)) void vstore4(int4 vec, size_t offset, __global int *out); __attribute__((overloadable)) void vstore8(int8 vec, size_t offset, __global int *out); __attribute__((overloadable)) void vstore16(int16 vec, size_t offset, __global int *out); __attribute__((overloadable)) void vstore2(uint2 vec, size_t offset, __private uint *out); __attribute__((overloadable)) void vstore3(uint3 vec, size_t offset, __private uint *out); __attribute__((overloadable)) void vstore4(uint4 vec, size_t offset, __private uint *out); __attribute__((overloadable)) void vstore8(uint8 vec, size_t offset, __private uint *out); __attribute__((overloadable)) void vstore16(uint16 vec, size_t offset, __private uint *out); __attribute__((overloadable)) void vstore2(uint2 vec, size_t offset, __local uint *out); __attribute__((overloadable)) void vstore3(uint3 vec, size_t offset, __local uint *out); __attribute__((overloadable)) void vstore4(uint4 vec, size_t offset, __local uint *out); __attribute__((overloadable)) void vstore8(uint8 vec, size_t offset, __local uint *out); __attribute__((overloadable)) void vstore16(uint16 vec, size_t offset, __local uint *out); __attribute__((overloadable)) void vstore2(uint2 vec, size_t offset, __global uint *out); __attribute__((overloadable)) void vstore3(uint3 vec, size_t offset, __global uint *out); __attribute__((overloadable)) void vstore4(uint4 vec, size_t offset, __global uint *out); __attribute__((overloadable)) void vstore8(uint8 vec, size_t offset, __global uint *out); __attribute__((overloadable)) void vstore16(uint16 vec, size_t offset, __global uint *out); __attribute__((overloadable)) void vstore2(long2 vec, size_t offset, __private long *out); __attribute__((overloadable)) void vstore3(long3 vec, size_t offset, __private long *out); __attribute__((overloadable)) void vstore4(long4 vec, size_t offset, __private long *out); __attribute__((overloadable)) void vstore8(long8 vec, size_t offset, __private long *out); __attribute__((overloadable)) void vstore16(long16 vec, size_t offset, __private long *out); __attribute__((overloadable)) void vstore2(long2 vec, size_t offset, __local long *out); __attribute__((overloadable)) void vstore3(long3 vec, size_t offset, __local long *out); __attribute__((overloadable)) void vstore4(long4 vec, size_t offset, __local long *out); __attribute__((overloadable)) void vstore8(long8 vec, size_t offset, __local long *out); __attribute__((overloadable)) void vstore16(long16 vec, size_t offset, __local long *out); __attribute__((overloadable)) void vstore2(long2 vec, size_t offset, __global long *out); __attribute__((overloadable)) void vstore3(long3 vec, size_t offset, __global long *out); __attribute__((overloadable)) void vstore4(long4 vec, size_t offset, __global long *out); __attribute__((overloadable)) void vstore8(long8 vec, size_t offset, __global long *out); __attribute__((overloadable)) void vstore16(long16 vec, size_t offset, __global long *out); __attribute__((overloadable)) void vstore2(ulong2 vec, size_t offset, __private ulong *out); __attribute__((overloadable)) void vstore3(ulong3 vec, size_t offset, __private ulong *out); __attribute__((overloadable)) void vstore4(ulong4 vec, size_t offset, __private ulong *out); __attribute__((overloadable)) void vstore8(ulong8 vec, size_t offset, __private ulong *out); __attribute__((overloadable)) void vstore16(ulong16 vec, size_t offset, __private ulong *out); __attribute__((overloadable)) void vstore2(ulong2 vec, size_t offset, __local ulong *out); __attribute__((overloadable)) void vstore3(ulong3 vec, size_t offset, __local ulong *out); __attribute__((overloadable)) void vstore4(ulong4 vec, size_t offset, __local ulong *out); __attribute__((overloadable)) void vstore8(ulong8 vec, size_t offset, __local ulong *out); __attribute__((overloadable)) void vstore16(ulong16 vec, size_t offset, __local ulong *out); __attribute__((overloadable)) void vstore2(ulong2 vec, size_t offset, __global ulong *out); __attribute__((overloadable)) void vstore3(ulong3 vec, size_t offset, __global ulong *out); __attribute__((overloadable)) void vstore4(ulong4 vec, size_t offset, __global ulong *out); __attribute__((overloadable)) void vstore8(ulong8 vec, size_t offset, __global ulong *out); __attribute__((overloadable)) void vstore16(ulong16 vec, size_t offset, __global ulong *out); __attribute__((overloadable)) void vstore2(float2 vec, size_t offset, __private float *out); __attribute__((overloadable)) void vstore3(float3 vec, size_t offset, __private float *out); __attribute__((overloadable)) void vstore4(float4 vec, size_t offset, __private float *out); __attribute__((overloadable)) void vstore8(float8 vec, size_t offset, __private float *out); __attribute__((overloadable)) void vstore16(float16 vec, size_t offset, __private float *out); __attribute__((overloadable)) void vstore2(float2 vec, size_t offset, __local float *out); __attribute__((overloadable)) void vstore3(float3 vec, size_t offset, __local float *out); __attribute__((overloadable)) void vstore4(float4 vec, size_t offset, __local float *out); __attribute__((overloadable)) void vstore8(float8 vec, size_t offset, __local float *out); __attribute__((overloadable)) void vstore16(float16 vec, size_t offset, __local float *out); __attribute__((overloadable)) void vstore2(float2 vec, size_t offset, __global float *out); __attribute__((overloadable)) void vstore3(float3 vec, size_t offset, __global float *out); __attribute__((overloadable)) void vstore4(float4 vec, size_t offset, __global float *out); __attribute__((overloadable)) void vstore8(float8 vec, size_t offset, __global float *out); __attribute__((overloadable)) void vstore16(float16 vec, size_t offset, __global float *out);
// __attribute__((overloadable)) float degrees(float x);
// __attribute__((overloadable)) float2 degrees(float2 x);
// __attribute__((overloadable)) float3 degrees(float3 x);
// __attribute__((overloadable)) float4 degrees(float4 x);
// __attribute__((overloadable)) float8 degrees(float8 x);
// __attribute__((overloadable)) float16 degrees(float16 x);
// __attribute__((overloadable)) double degrees(double x);
// __attribute__((overloadable)) double2 degrees(double2 x);
// __attribute__((overloadable)) double3 degrees(double3 x);
// __attribute__((overloadable)) double4 degrees(double4 x);
// __attribute__((overloadable)) double8 degrees(double8 x);
// __attribute__((overloadable)) double16 degrees(double16 x);
// __attribute__((overloadable)) float radians(float x);
// __attribute__((overloadable)) float2 radians(float2 x);
// __attribute__((overloadable)) float3 radians(float3 x);
// __attribute__((overloadable)) float4 radians(float4 x);
// __attribute__((overloadable)) float8 radians(float8 x);
// __attribute__((overloadable)) float16 radians(float16 x);
// __attribute__((overloadable)) double radians(double x);
// __attribute__((overloadable)) double2 radians(double2 x);
// __attribute__((overloadable)) double3 radians(double3 x);
// __attribute__((overloadable)) double4 radians(double4 x);
// __attribute__((overloadable)) double8 radians(double8 x);
// __attribute__((overloadable)) double16 radians(double16 x);
// __attribute__((overloadable)) float mix(float a, float b, float c);
// __attribute__((overloadable)) float2 mix(float2 a, float2 b, float2 c);
// __attribute__((overloadable)) float2 mix(float2 a, float2 b, float c);
// __attribute__((overloadable)) float3 mix(float3 a, float3 b, float3 c);
// __attribute__((overloadable)) float3 mix(float3 a, float3 b, float c);
// __attribute__((overloadable)) float4 mix(float4 a, float4 b, float4 c);
// __attribute__((overloadable)) float4 mix(float4 a, float4 b, float c);
// __attribute__((overloadable)) float8 mix(float8 a, float8 b, float8 c);
// __attribute__((overloadable)) float8 mix(float8 a, float8 b, float c);
// __attribute__((overloadable)) float16 mix(float16 a, float16 b, float16 c);
// __attribute__((overloadable)) float16 mix(float16 a, float16 b, float c);
// __attribute__((overloadable)) double mix(double a, double b, double c);
// __attribute__((overloadable)) double2 mix(double2 a, double2 b, double2 c);
// __attribute__((overloadable)) double2 mix(double2 a, double2 b, double c);
// __attribute__((overloadable)) double3 mix(double3 a, double3 b, double3 c);
// __attribute__((overloadable)) double3 mix(double3 a, double3 b, double c);
// __attribute__((overloadable)) double4 mix(double4 a, double4 b, double4 c);
// __attribute__((overloadable)) double4 mix(double4 a, double4 b, double c);
// __attribute__((overloadable)) double8 mix(double8 a, double8 b, double8 c);
// __attribute__((overloadable)) double8 mix(double8 a, double8 b, double c);
// __attribute__((overloadable)) double16 mix(double16 a, double16 b, double16 c);
// __attribute__((overloadable)) double16 mix(double16 a, double16 b, double c);
// __attribute__((overloadable)) float sign(float x);
// __attribute__((overloadable)) float2 sign(float2 x);
// __attribute__((overloadable)) float3 sign(float3 x);
// __attribute__((overloadable)) float4 sign(float4 x);
// __attribute__((overloadable)) float8 sign(float8 x);
// __attribute__((overloadable)) float16 sign(float16 x);
// __attribute__((overloadable)) double sign(double x);
// __attribute__((overloadable)) double2 sign(double2 x);
// __attribute__((overloadable)) double3 sign(double3 x);
// __attribute__((overloadable)) double4 sign(double4 x);
// __attribute__((overloadable)) double8 sign(double8 x);
// __attribute__((overloadable)) double16 sign(double16 x);
// __attribute__((overloadable)) float smoothstep(float edge0, float edge1, float x);
// __attribute__((overloadable)) float smoothstep(float edge0, float edge1, float x);
// __attribute__((overloadable)) float smoothstep(double edge0, double edge1, float x);
// __attribute__((overloadable)) float2 smoothstep(float2 edge0, float2 edge1, float2 x);
// __attribute__((overloadable)) float2 smoothstep(float edge0, float edge1, float2 x);
// __attribute__((overloadable)) float2 smoothstep(double edge0, double edge1, float2 x);
// __attribute__((overloadable)) float3 smoothstep(float3 edge0, float3 edge1, float3 x);
// __attribute__((overloadable)) float3 smoothstep(float edge0, float edge1, float3 x);
// __attribute__((overloadable)) float3 smoothstep(double edge0, double edge1, float3 x);
// __attribute__((overloadable)) float4 smoothstep(float4 edge0, float4 edge1, float4 x);
// __attribute__((overloadable)) float4 smoothstep(float edge0, float edge1, float4 x);
// __attribute__((overloadable)) float4 smoothstep(double edge0, double edge1, float4 x);
// __attribute__((overloadable)) float8 smoothstep(float8 edge0, float8 edge1, float8 x);
// __attribute__((overloadable)) float8 smoothstep(float edge0, float edge1, float8 x);
// __attribute__((overloadable)) float8 smoothstep(double edge0, double edge1, float8 x);
// __attribute__((overloadable)) float16 smoothstep(float16 edge0, float16 edge1, float16 x);
// __attribute__((overloadable)) float16 smoothstep(float edge0, float edge1, float16 x);
// __attribute__((overloadable)) float16 smoothstep(double edge0, double edge1, float16 x);
// __attribute__((overloadable)) double smoothstep(double edge0, double edge1, double x);
// __attribute__((overloadable)) double smoothstep(float edge0, float edge1, double x);
// __attribute__((overloadable)) double smoothstep(double edge0, double edge1, double x);
// __attribute__((overloadable)) double2 smoothstep(double2 edge0, double2 edge1, double2 x);
// __attribute__((overloadable)) double2 smoothstep(float edge0, float edge1, double2 x);
// __attribute__((overloadable)) double2 smoothstep(double edge0, double edge1, double2 x);
// __attribute__((overloadable)) double3 smoothstep(double3 edge0, double3 edge1, double3 x);
// __attribute__((overloadable)) double3 smoothstep(float edge0, float edge1, double3 x);
// __attribute__((overloadable)) double3 smoothstep(double edge0, double edge1, double3 x);
// __attribute__((overloadable)) double4 smoothstep(double4 edge0, double4 edge1, double4 x);
// __attribute__((overloadable)) double4 smoothstep(float edge0, float edge1, double4 x);
// __attribute__((overloadable)) double4 smoothstep(double edge0, double edge1, double4 x);
// __attribute__((overloadable)) double8 smoothstep(double8 edge0, double8 edge1, double8 x);
// __attribute__((overloadable)) double8 smoothstep(float edge0, float edge1, double8 x);
// __attribute__((overloadable)) double8 smoothstep(double edge0, double edge1, double8 x);
// __attribute__((overloadable)) double16 smoothstep(double16 edge0, double16 edge1, double16 x);
// __attribute__((overloadable)) double16 smoothstep(float edge0, float edge1, double16 x);
// __attribute__((overloadable)) double16 smoothstep(double edge0, double edge1, double16 x);
// __attribute__((overloadable)) float step(float edge, float x);
// __attribute__((overloadable)) float step(float edge, float x);
// __attribute__((overloadable)) float step(double edge, float x);
// __attribute__((overloadable)) float2 step(float2 edge, float2 x);
// __attribute__((overloadable)) float2 step(float edge, float2 x);
// __attribute__((overloadable)) float2 step(double edge, float2 x);
// __attribute__((overloadable)) float3 step(float3 edge, float3 x);
// __attribute__((overloadable)) float3 step(float edge, float3 x);
// __attribute__((overloadable)) float3 step(double edge, float3 x);
// __attribute__((overloadable)) float4 step(float4 edge, float4 x);
// __attribute__((overloadable)) float4 step(float edge, float4 x);
// __attribute__((overloadable)) float4 step(double edge, float4 x);
// __attribute__((overloadable)) float8 step(float8 edge, float8 x);
// __attribute__((overloadable)) float8 step(float edge, float8 x);
// __attribute__((overloadable)) float8 step(double edge, float8 x);
// __attribute__((overloadable)) float16 step(float16 edge, float16 x);
// __attribute__((overloadable)) float16 step(float edge, float16 x);
// __attribute__((overloadable)) float16 step(double edge, float16 x);
// __attribute__((overloadable)) double step(double edge, double x);
// __attribute__((overloadable)) double step(float edge, double x);
// __attribute__((overloadable)) double step(double edge, double x);
// __attribute__((overloadable)) double2 step(double2 edge, double2 x);
// __attribute__((overloadable)) double2 step(float edge, double2 x);
// __attribute__((overloadable)) double2 step(double edge, double2 x);
// __attribute__((overloadable)) double3 step(double3 edge, double3 x);
// __attribute__((overloadable)) double3 step(float edge, double3 x);
// __attribute__((overloadable)) double3 step(double edge, double3 x);
// __attribute__((overloadable)) double4 step(double4 edge, double4 x);
// __attribute__((overloadable)) double4 step(float edge, double4 x);
// __attribute__((overloadable)) double4 step(double edge, double4 x);
// __attribute__((overloadable)) double8 step(double8 edge, double8 x);
// __attribute__((overloadable)) double8 step(float edge, double8 x);
// __attribute__((overloadable)) double8 step(double edge, double8 x);
// __attribute__((overloadable)) double16 step(double16 edge, double16 x);
// __attribute__((overloadable)) double16 step(float edge, double16 x);
// __attribute__((overloadable)) double16 step(double edge, double16 x);
// __attribute__((overloadable)) float3 cross(float3 p0, float3 p1);
// __attribute__((overloadable)) float4 cross(float4 p0, float4 p1);
// __attribute__((overloadable)) double3 cross(double3 p0, double3 p1);
// __attribute__((overloadable)) double4 cross(double4 p0, double4 p1);
// __attribute__((overloadable)) float distance(float p0, float p1);
// __attribute__((overloadable)) float distance(float2 p0, float2 p1);
// __attribute__((overloadable)) float distance(float3 p0, float3 p1);
// __attribute__((overloadable)) float distance(float4 p0, float4 p1);
// __attribute__((overloadable)) double distance(double p0, double p1);
// __attribute__((overloadable)) double distance(double2 p0, double2 p1);
// __attribute__((overloadable)) double distance(double3 p0, double3 p1);
// __attribute__((overloadable)) double distance(double4 p0, double4 p1);
// __attribute__((overloadable)) float dot(float p0, float p1);
// __attribute__((overloadable)) float dot(float2 p0, float2 p1);
// __attribute__((overloadable)) float dot(float3 p0, float3 p1);
// __attribute__((overloadable)) float dot(float4 p0, float4 p1);
// __attribute__((overloadable)) double dot(double p0, double p1);
// __attribute__((overloadable)) double dot(double2 p0, double2 p1);
// __attribute__((overloadable)) double dot(double3 p0, double3 p1);
// __attribute__((overloadable)) double dot(double4 p0, double4 p1);
// __attribute__((overloadable)) float fast_distance(float p0, float p1);
// __attribute__((overloadable)) float fast_distance(float2 p0, float2 p1);
// __attribute__((overloadable)) float fast_distance(float3 p0, float3 p1);
// __attribute__((overloadable)) float fast_distance(float4 p0, float4 p1);
// __attribute__((overloadable)) float fast_length(float p0);
// __attribute__((overloadable)) float fast_length(float2 p0);
// __attribute__((overloadable)) float fast_length(float3 p0);
// __attribute__((overloadable)) float fast_length(float4 p0);
// __attribute__((overloadable)) float fast_normalize(float p);
// __attribute__((overloadable)) float2 fast_normalize(float2 p);
// __attribute__((overloadable)) float3 fast_normalize(float3 p);
// __attribute__((overloadable)) float4 fast_normalize(float4 p);
// __attribute__((overloadable)) float length(float p0);
// __attribute__((overloadable)) float length(float2 p0);
// __attribute__((overloadable)) float length(float3 p0);
// __attribute__((overloadable)) float length(float4 p0);
// __attribute__((overloadable)) double length(double p0);
// __attribute__((overloadable)) double length(double2 p0);
// __attribute__((overloadable)) double length(double3 p0);
// __attribute__((overloadable)) double length(double4 p0);
// __attribute__((overloadable)) float normalize(float p);
// __attribute__((overloadable)) float2 normalize(float2 p);
// __attribute__((overloadable)) float3 normalize(float3 p);
// __attribute__((overloadable)) float4 normalize(float4 p);
// __attribute__((overloadable)) double normalize(double p);
// __attribute__((overloadable)) double2 normalize(double2 p);
// __attribute__((overloadable)) double3 normalize(double3 p);
// __attribute__((overloadable)) double4 normalize(double4 p);
// __attribute__((overloadable)) int all(char v); __attribute__((overloadable)) int all(char2 v); __attribute__((overloadable)) int all(char3 v); __attribute__((overloadable)) int all(char4 v); __attribute__((overloadable)) int all(char8 v); __attribute__((overloadable)) int all(char16 v);
// __attribute__((overloadable)) int all(short v); __attribute__((overloadable)) int all(short2 v); __attribute__((overloadable)) int all(short3 v); __attribute__((overloadable)) int all(short4 v); __attribute__((overloadable)) int all(short8 v); __attribute__((overloadable)) int all(short16 v);
// __attribute__((overloadable)) int all(int v); __attribute__((overloadable)) int all(int2 v); __attribute__((overloadable)) int all(int3 v); __attribute__((overloadable)) int all(int4 v); __attribute__((overloadable)) int all(int8 v); __attribute__((overloadable)) int all(int16 v);
// __attribute__((overloadable)) int all(long v); __attribute__((overloadable)) int all(long2 v); __attribute__((overloadable)) int all(long3 v); __attribute__((overloadable)) int all(long4 v); __attribute__((overloadable)) int all(long8 v); __attribute__((overloadable)) int all(long16 v);
// __attribute__((overloadable)) int any(char v); __attribute__((overloadable)) int any(char2 v); __attribute__((overloadable)) int any(char3 v); __attribute__((overloadable)) int any(char4 v); __attribute__((overloadable)) int any(char8 v); __attribute__((overloadable)) int any(char16 v);
// __attribute__((overloadable)) int any(short v); __attribute__((overloadable)) int any(short2 v); __attribute__((overloadable)) int any(short3 v); __attribute__((overloadable)) int any(short4 v); __attribute__((overloadable)) int any(short8 v); __attribute__((overloadable)) int any(short16 v);
// __attribute__((overloadable)) int any(int v); __attribute__((overloadable)) int any(int2 v); __attribute__((overloadable)) int any(int3 v); __attribute__((overloadable)) int any(int4 v); __attribute__((overloadable)) int any(int8 v); __attribute__((overloadable)) int any(int16 v);
// __attribute__((overloadable)) int any(long v); __attribute__((overloadable)) int any(long2 v); __attribute__((overloadable)) int any(long3 v); __attribute__((overloadable)) int any(long4 v); __attribute__((overloadable)) int any(long8 v); __attribute__((overloadable)) int any(long16 v);
// __attribute__((overloadable)) float bitselect(float x, float y, float z);
// __attribute__((overloadable)) float2 bitselect(float2 x, float2 y, float2 z);
// __attribute__((overloadable)) float3 bitselect(float3 x, float3 y, float3 z);
// __attribute__((overloadable)) float4 bitselect(float4 x, float4 y, float4 z);
// __attribute__((overloadable)) float8 bitselect(float8 x, float8 y, float8 z);
// __attribute__((overloadable)) float16 bitselect(float16 x, float16 y, float16 z);
// __attribute__((overloadable)) double bitselect(double x, double y, double z);
// __attribute__((overloadable)) double2 bitselect(double2 x, double2 y, double2 z);
// __attribute__((overloadable)) double3 bitselect(double3 x, double3 y, double3 z);
// __attribute__((overloadable)) double4 bitselect(double4 x, double4 y, double4 z);
// __attribute__((overloadable)) double8 bitselect(double8 x, double8 y, double8 z);
// __attribute__((overloadable)) double16 bitselect(double16 x, double16 y, double16 z);
// __attribute__((overloadable)) char bitselect(char x, char y, char z);
// __attribute__((overloadable)) char2 bitselect(char2 x, char2 y, char2 z);
// __attribute__((overloadable)) char3 bitselect(char3 x, char3 y, char3 z);
// __attribute__((overloadable)) char4 bitselect(char4 x, char4 y, char4 z);
// __attribute__((overloadable)) char8 bitselect(char8 x, char8 y, char8 z);
// __attribute__((overloadable)) char16 bitselect(char16 x, char16 y, char16 z);
// __attribute__((overloadable)) uchar bitselect(uchar x, uchar y, uchar z);
// __attribute__((overloadable)) uchar2 bitselect(uchar2 x, uchar2 y, uchar2 z);
// __attribute__((overloadable)) uchar3 bitselect(uchar3 x, uchar3 y, uchar3 z);
// __attribute__((overloadable)) uchar4 bitselect(uchar4 x, uchar4 y, uchar4 z);
// __attribute__((overloadable)) uchar8 bitselect(uchar8 x, uchar8 y, uchar8 z);
// __attribute__((overloadable)) uchar16 bitselect(uchar16 x, uchar16 y, uchar16 z);
// __attribute__((overloadable)) short bitselect(short x, short y, short z);
// __attribute__((overloadable)) short2 bitselect(short2 x, short2 y, short2 z);
// __attribute__((overloadable)) short3 bitselect(short3 x, short3 y, short3 z);
// __attribute__((overloadable)) short4 bitselect(short4 x, short4 y, short4 z);
// __attribute__((overloadable)) short8 bitselect(short8 x, short8 y, short8 z);
// __attribute__((overloadable)) short16 bitselect(short16 x, short16 y, short16 z);
// __attribute__((overloadable)) ushort bitselect(ushort x, ushort y, ushort z);
// __attribute__((overloadable)) ushort2 bitselect(ushort2 x, ushort2 y, ushort2 z);
// __attribute__((overloadable)) ushort3 bitselect(ushort3 x, ushort3 y, ushort3 z);
// __attribute__((overloadable)) ushort4 bitselect(ushort4 x, ushort4 y, ushort4 z);
// __attribute__((overloadable)) ushort8 bitselect(ushort8 x, ushort8 y, ushort8 z);
// __attribute__((overloadable)) ushort16 bitselect(ushort16 x, ushort16 y, ushort16 z);
// __attribute__((overloadable)) int bitselect(int x, int y, int z);
// __attribute__((overloadable)) int2 bitselect(int2 x, int2 y, int2 z);
// __attribute__((overloadable)) int3 bitselect(int3 x, int3 y, int3 z);
// __attribute__((overloadable)) int4 bitselect(int4 x, int4 y, int4 z);
// __attribute__((overloadable)) int8 bitselect(int8 x, int8 y, int8 z);
// __attribute__((overloadable)) int16 bitselect(int16 x, int16 y, int16 z);
// __attribute__((overloadable)) uint bitselect(uint x, uint y, uint z);
// __attribute__((overloadable)) uint2 bitselect(uint2 x, uint2 y, uint2 z);
// __attribute__((overloadable)) uint3 bitselect(uint3 x, uint3 y, uint3 z);
// __attribute__((overloadable)) uint4 bitselect(uint4 x, uint4 y, uint4 z);
// __attribute__((overloadable)) uint8 bitselect(uint8 x, uint8 y, uint8 z);
// __attribute__((overloadable)) uint16 bitselect(uint16 x, uint16 y, uint16 z);
// __attribute__((overloadable)) long bitselect(long x, long y, long z);
// __attribute__((overloadable)) long2 bitselect(long2 x, long2 y, long2 z);
// __attribute__((overloadable)) long3 bitselect(long3 x, long3 y, long3 z);
// __attribute__((overloadable)) long4 bitselect(long4 x, long4 y, long4 z);
// __attribute__((overloadable)) long8 bitselect(long8 x, long8 y, long8 z);
// __attribute__((overloadable)) long16 bitselect(long16 x, long16 y, long16 z);
// __attribute__((overloadable)) ulong bitselect(ulong x, ulong y, ulong z);
// __attribute__((overloadable)) ulong2 bitselect(ulong2 x, ulong2 y, ulong2 z);
// __attribute__((overloadable)) ulong3 bitselect(ulong3 x, ulong3 y, ulong3 z);
// __attribute__((overloadable)) ulong4 bitselect(ulong4 x, ulong4 y, ulong4 z);
// __attribute__((overloadable)) ulong8 bitselect(ulong8 x, ulong8 y, ulong8 z);
// __attribute__((overloadable)) ulong16 bitselect(ulong16 x, ulong16 y, ulong16 z);
// __attribute__((overloadable)) int isequal(float x, float y);
// __attribute__((overloadable)) int2 isequal(float2 x, float2 y); __attribute__((overloadable)) int3 isequal(float3 x, float3 y); __attribute__((overloadable)) int4 isequal(float4 x, float4 y); __attribute__((overloadable)) int8 isequal(float8 x, float8 y); __attribute__((overloadable)) int16 isequal(float16 x, float16 y);
// __attribute__((overloadable)) int isequal(double x, double y);
// __attribute__((overloadable)) long2 isequal(double2 x, double2 y); __attribute__((overloadable)) long3 isequal(double3 x, double3 y); __attribute__((overloadable)) long4 isequal(double4 x, double4 y); __attribute__((overloadable)) long8 isequal(double8 x, double8 y); __attribute__((overloadable)) long16 isequal(double16 x, double16 y);
// __attribute__((overloadable)) int isfinite(float x);
// __attribute__((overloadable)) int2 isfinite(float2 x);
// __attribute__((overloadable)) int3 isfinite(float3 x);
// __attribute__((overloadable)) int4 isfinite(float4 x);
// __attribute__((overloadable)) int8 isfinite(float8 x);
// __attribute__((overloadable)) int16 isfinite(float16 x);
// __attribute__((overloadable)) int isfinite(double x);
// __attribute__((overloadable)) long2 isfinite(double2 x);
// __attribute__((overloadable)) long3 isfinite(double3 x);
// __attribute__((overloadable)) long4 isfinite(double4 x);
// __attribute__((overloadable)) long8 isfinite(double8 x);
// __attribute__((overloadable)) long16 isfinite(double16 x);
// __attribute__((overloadable)) int isgreater(float a, float b);
// __attribute__((overloadable)) int2 isgreater(float2 a, float2 b);
// __attribute__((overloadable)) int3 isgreater(float3 a, float3 b);
// __attribute__((overloadable)) int4 isgreater(float4 a, float4 b);
// __attribute__((overloadable)) int8 isgreater(float8 a, float8 b);
// __attribute__((overloadable)) int16 isgreater(float16 a, float16 b);
// __attribute__((overloadable)) int isgreater(double a, double b);
// __attribute__((overloadable)) long2 isgreater(double2 a, double2 b);
// __attribute__((overloadable)) long3 isgreater(double3 a, double3 b);
// __attribute__((overloadable)) long4 isgreater(double4 a, double4 b);
// __attribute__((overloadable)) long8 isgreater(double8 a, double8 b);
// __attribute__((overloadable)) long16 isgreater(double16 a, double16 b);
// __attribute__((overloadable)) int isgreaterequal(float a, float b);
// __attribute__((overloadable)) int2 isgreaterequal(float2 a, float2 b);
// __attribute__((overloadable)) int3 isgreaterequal(float3 a, float3 b);
// __attribute__((overloadable)) int4 isgreaterequal(float4 a, float4 b);
// __attribute__((overloadable)) int8 isgreaterequal(float8 a, float8 b);
// __attribute__((overloadable)) int16 isgreaterequal(float16 a, float16 b);
// __attribute__((overloadable)) int isgreaterequal(double a, double b);
// __attribute__((overloadable)) long2 isgreaterequal(double2 a, double2 b);
// __attribute__((overloadable)) long3 isgreaterequal(double3 a, double3 b);
// __attribute__((overloadable)) long4 isgreaterequal(double4 a, double4 b);
// __attribute__((overloadable)) long8 isgreaterequal(double8 a, double8 b);
// __attribute__((overloadable)) long16 isgreaterequal(double16 a, double16 b);
// __attribute__((overloadable)) int isinf(float);
// __attribute__((overloadable)) int2 isinf(float2); __attribute__((overloadable)) int3 isinf(float3); __attribute__((overloadable)) int4 isinf(float4); __attribute__((overloadable)) int8 isinf(float8); __attribute__((overloadable)) int16 isinf(float16);
// __attribute__((overloadable)) int isinf(double);
// __attribute__((overloadable)) long2 isinf(double2); __attribute__((overloadable)) long3 isinf(double3); __attribute__((overloadable)) long4 isinf(double4); __attribute__((overloadable)) long8 isinf(double8); __attribute__((overloadable)) long16 isinf(double16);
// __attribute__((overloadable)) int isless(float a, float b);
// __attribute__((overloadable)) int2 isless(float2 a, float2 b);
// __attribute__((overloadable)) int3 isless(float3 a, float3 b);
// __attribute__((overloadable)) int4 isless(float4 a, float4 b);
// __attribute__((overloadable)) int8 isless(float8 a, float8 b);
// __attribute__((overloadable)) int16 isless(float16 a, float16 b);
// __attribute__((overloadable)) int isless(double a, double b);
// __attribute__((overloadable)) long2 isless(double2 a, double2 b);
// __attribute__((overloadable)) long3 isless(double3 a, double3 b);
// __attribute__((overloadable)) long4 isless(double4 a, double4 b);
// __attribute__((overloadable)) long8 isless(double8 a, double8 b);
// __attribute__((overloadable)) long16 isless(double16 a, double16 b);
// __attribute__((overloadable)) int islessequal(float a, float b);
// __attribute__((overloadable)) int2 islessequal(float2 a, float2 b);
// __attribute__((overloadable)) int3 islessequal(float3 a, float3 b);
// __attribute__((overloadable)) int4 islessequal(float4 a, float4 b);
// __attribute__((overloadable)) int8 islessequal(float8 a, float8 b);
// __attribute__((overloadable)) int16 islessequal(float16 a, float16 b);
// __attribute__((overloadable)) int islessequal(double a, double b);
// __attribute__((overloadable)) long2 islessequal(double2 a, double2 b);
// __attribute__((overloadable)) long3 islessequal(double3 a, double3 b);
// __attribute__((overloadable)) long4 islessequal(double4 a, double4 b);
// __attribute__((overloadable)) long8 islessequal(double8 a, double8 b);
// __attribute__((overloadable)) long16 islessequal(double16 a, double16 b);
// __attribute__((overloadable)) int islessgreater(float a, float b);
// __attribute__((overloadable)) int2 islessgreater(float2 a, float2 b);
// __attribute__((overloadable)) int3 islessgreater(float3 a, float3 b);
// __attribute__((overloadable)) int4 islessgreater(float4 a, float4 b);
// __attribute__((overloadable)) int8 islessgreater(float8 a, float8 b);
// __attribute__((overloadable)) int16 islessgreater(float16 a, float16 b);
// __attribute__((overloadable)) int islessgreater(double a, double b);
// __attribute__((overloadable)) long2 islessgreater(double2 a, double2 b);
// __attribute__((overloadable)) long3 islessgreater(double3 a, double3 b);
// __attribute__((overloadable)) long4 islessgreater(double4 a, double4 b);
// __attribute__((overloadable)) long8 islessgreater(double8 a, double8 b);
// __attribute__((overloadable)) long16 islessgreater(double16 a, double16 b);
// __attribute__((overloadable)) int isnan(float);
// __attribute__((overloadable)) int2 isnan(float2); __attribute__((overloadable)) int3 isnan(float3); __attribute__((overloadable)) int4 isnan(float4); __attribute__((overloadable)) int8 isnan(float8); __attribute__((overloadable)) int16 isnan(float16);
// __attribute__((overloadable)) int isnan(double);
// __attribute__((overloadable)) long2 isnan(double2); __attribute__((overloadable)) long3 isnan(double3); __attribute__((overloadable)) long4 isnan(double4); __attribute__((overloadable)) long8 isnan(double8); __attribute__((overloadable)) long16 isnan(double16);
// __attribute__((overloadable)) int isnormal(float x);
// __attribute__((overloadable)) int2 isnormal(float2 x);
// __attribute__((overloadable)) int3 isnormal(float3 x);
// __attribute__((overloadable)) int4 isnormal(float4 x);
// __attribute__((overloadable)) int8 isnormal(float8 x);
// __attribute__((overloadable)) int16 isnormal(float16 x);
// __attribute__((overloadable)) int isnormal(double x);
// __attribute__((overloadable)) long2 isnormal(double2 x);
// __attribute__((overloadable)) long3 isnormal(double3 x);
// __attribute__((overloadable)) long4 isnormal(double4 x);
// __attribute__((overloadable)) long8 isnormal(double8 x);
// __attribute__((overloadable)) long16 isnormal(double16 x);
__attribute__((overloadable)) int isnotequal(float a, float b);
__attribute__((overloadable)) int2 isnotequal(float2 a, float2 b);
__attribute__((overloadable)) int3 isnotequal(float3 a, float3 b);
__attribute__((overloadable)) int4 isnotequal(float4 a, float4 b);
__attribute__((overloadable)) int8 isnotequal(float8 a, float8 b);
__attribute__((overloadable)) int16 isnotequal(float16 a, float16 b);
__attribute__((overloadable)) int isnotequal(double a, double b);
__attribute__((overloadable)) long2 isnotequal(double2 a, double2 b);
__attribute__((overloadable)) long3 isnotequal(double3 a, double3 b);
__attribute__((overloadable)) long4 isnotequal(double4 a, double4 b);
__attribute__((overloadable)) long8 isnotequal(double8 a, double8 b);
__attribute__((overloadable)) long16 isnotequal(double16 a, double16 b);
__attribute__((overloadable)) int isordered(float a, float b);
__attribute__((overloadable)) int2 isordered(float2 a, float2 b);
__attribute__((overloadable)) int3 isordered(float3 a, float3 b);
__attribute__((overloadable)) int4 isordered(float4 a, float4 b);
__attribute__((overloadable)) int8 isordered(float8 a, float8 b);
__attribute__((overloadable)) int16 isordered(float16 a, float16 b);
__attribute__((overloadable)) int isordered(double a, double b);
__attribute__((overloadable)) long2 isordered(double2 a, double2 b);
__attribute__((overloadable)) long3 isordered(double3 a, double3 b);
__attribute__((overloadable)) long4 isordered(double4 a, double4 b);
__attribute__((overloadable)) long8 isordered(double8 a, double8 b);
__attribute__((overloadable)) long16 isordered(double16 a, double16 b);
__attribute__((overloadable)) int isunordered(float a, float b);
__attribute__((overloadable)) int2 isunordered(float2 a, float2 b);
__attribute__((overloadable)) int3 isunordered(float3 a, float3 b);
__attribute__((overloadable)) int4 isunordered(float4 a, float4 b);
__attribute__((overloadable)) int8 isunordered(float8 a, float8 b);
__attribute__((overloadable)) int16 isunordered(float16 a, float16 b);
__attribute__((overloadable)) int isunordered(double a, double b);
__attribute__((overloadable)) long2 isunordered(double2 a, double2 b);
__attribute__((overloadable)) long3 isunordered(double3 a, double3 b);
__attribute__((overloadable)) long4 isunordered(double4 a, double4 b);
__attribute__((overloadable)) long8 isunordered(double8 a, double8 b);
__attribute__((overloadable)) long16 isunordered(double16 a, double16 b);
__attribute__((overloadable)) int signbit(float x);
__attribute__((overloadable)) int2 signbit(float2 x);
__attribute__((overloadable)) int3 signbit(float3 x);
__attribute__((overloadable)) int4 signbit(float4 x);
__attribute__((overloadable)) int8 signbit(float8 x);
__attribute__((overloadable)) int16 signbit(float16 x);
__attribute__((overloadable)) int signbit(double x);
__attribute__((overloadable)) long2 signbit(double2 x);
__attribute__((overloadable)) long3 signbit(double3 x);
__attribute__((overloadable)) long4 signbit(double4 x);
__attribute__((overloadable)) long8 signbit(double8 x);
__attribute__((overloadable)) long16 signbit(double16 x);
typedef uint cl_mem_fence_flags;
          void barrier(cl_mem_fence_flags flags);
__attribute__((overloadable)) event_t async_work_group_copy(
  local char *dst,
  const global char *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local char2 *dst,
  const global char2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local char4 *dst,
  const global char4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local char8 *dst,
  const global char8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local char16 *dst,
  const global char16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uchar *dst,
  const global uchar *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uchar2 *dst,
  const global uchar2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uchar4 *dst,
  const global uchar4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uchar8 *dst,
  const global uchar8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uchar16 *dst,
  const global uchar16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local short *dst,
  const global short *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local short2 *dst,
  const global short2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local short4 *dst,
  const global short4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local short8 *dst,
  const global short8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local short16 *dst,
  const global short16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ushort *dst,
  const global ushort *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ushort2 *dst,
  const global ushort2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ushort4 *dst,
  const global ushort4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ushort8 *dst,
  const global ushort8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ushort16 *dst,
  const global ushort16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local int *dst,
  const global int *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local int2 *dst,
  const global int2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local int4 *dst,
  const global int4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local int8 *dst,
  const global int8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local int16 *dst,
  const global int16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uint *dst,
  const global uint *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uint2 *dst,
  const global uint2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uint4 *dst,
  const global uint4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uint8 *dst,
  const global uint8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local uint16 *dst,
  const global uint16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local float *dst,
  const global float *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local float2 *dst,
  const global float2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local float4 *dst,
  const global float4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local float8 *dst,
  const global float8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local float16 *dst,
  const global float16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local long *dst,
  const global long *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local long2 *dst,
  const global long2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local long4 *dst,
  const global long4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local long8 *dst,
  const global long8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local long16 *dst,
  const global long16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ulong *dst,
  const global ulong *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ulong2 *dst,
  const global ulong2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ulong4 *dst,
  const global ulong4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ulong8 *dst,
  const global ulong8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local ulong16 *dst,
  const global ulong16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local double *dst,
  const global double *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local double2 *dst,
  const global double2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local double4 *dst,
  const global double4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local double8 *dst,
  const global double8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  local double16 *dst,
  const global double16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global char *dst,
  const local char *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global char2 *dst,
  const local char2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global char4 *dst,
  const local char4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global char8 *dst,
  const local char8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global char16 *dst,
  const local char16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uchar *dst,
  const local uchar *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uchar2 *dst,
  const local uchar2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uchar4 *dst,
  const local uchar4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uchar8 *dst,
  const local uchar8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uchar16 *dst,
  const local uchar16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global short *dst,
  const local short *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global short2 *dst,
  const local short2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global short4 *dst,
  const local short4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global short8 *dst,
  const local short8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global short16 *dst,
  const local short16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ushort *dst,
  const local ushort *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ushort2 *dst,
  const local ushort2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ushort4 *dst,
  const local ushort4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ushort8 *dst,
  const local ushort8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ushort16 *dst,
  const local ushort16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global int *dst,
  const local int *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global int2 *dst,
  const local int2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global int4 *dst,
  const local int4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global int8 *dst,
  const local int8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global int16 *dst,
  const local int16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uint *dst,
  const local uint *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uint2 *dst,
  const local uint2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uint4 *dst,
  const local uint4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uint8 *dst,
  const local uint8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global uint16 *dst,
  const local uint16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global float *dst,
  const local float *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global float2 *dst,
  const local float2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global float4 *dst,
  const local float4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global float8 *dst,
  const local float8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global float16 *dst,
  const local float16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global long *dst,
  const local long *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global long2 *dst,
  const local long2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global long4 *dst,
  const local long4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global long8 *dst,
  const local long8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global long16 *dst,
  const local long16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ulong *dst,
  const local ulong *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ulong2 *dst,
  const local ulong2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ulong4 *dst,
  const local ulong4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ulong8 *dst,
  const local ulong8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global ulong16 *dst,
  const local ulong16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global double *dst,
  const local double *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global double2 *dst,
  const local double2 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global double4 *dst,
  const local double4 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global double8 *dst,
  const local double8 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_copy(
  global double16 *dst,
  const local double16 *src,
  size_t num_gentypes,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local char *dst,
  const global char *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local char2 *dst,
  const global char2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local char4 *dst,
  const global char4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local char8 *dst,
  const global char8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local char16 *dst,
  const global char16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uchar *dst,
  const global uchar *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uchar2 *dst,
  const global uchar2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uchar4 *dst,
  const global uchar4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uchar8 *dst,
  const global uchar8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uchar16 *dst,
  const global uchar16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local short *dst,
  const global short *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local short2 *dst,
  const global short2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local short4 *dst,
  const global short4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local short8 *dst,
  const global short8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local short16 *dst,
  const global short16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ushort *dst,
  const global ushort *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ushort2 *dst,
  const global ushort2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ushort4 *dst,
  const global ushort4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ushort8 *dst,
  const global ushort8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ushort16 *dst,
  const global ushort16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local int *dst,
  const global int *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local int2 *dst,
  const global int2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local int4 *dst,
  const global int4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local int8 *dst,
  const global int8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local int16 *dst,
  const global int16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uint *dst,
  const global uint *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uint2 *dst,
  const global uint2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uint4 *dst,
  const global uint4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uint8 *dst,
  const global uint8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local uint16 *dst,
  const global uint16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local float *dst,
  const global float *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local float2 *dst,
  const global float2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local float4 *dst,
  const global float4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local float8 *dst,
  const global float8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local float16 *dst,
  const global float16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local long *dst,
  const global long *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local long2 *dst,
  const global long2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local long4 *dst,
  const global long4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local long8 *dst,
  const global long8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local long16 *dst,
  const global long16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ulong *dst,
  const global ulong *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ulong2 *dst,
  const global ulong2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ulong4 *dst,
  const global ulong4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ulong8 *dst,
  const global ulong8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local ulong16 *dst,
  const global ulong16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local double *dst,
  const global double *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local double2 *dst,
  const global double2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local double4 *dst,
  const global double4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local double8 *dst,
  const global double8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  local double16 *dst,
  const global double16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global char *dst,
  const local char *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global char2 *dst,
  const local char2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global char4 *dst,
  const local char4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global char8 *dst,
  const local char8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global char16 *dst,
  const local char16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uchar *dst,
  const local uchar *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uchar2 *dst,
  const local uchar2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uchar4 *dst,
  const local uchar4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uchar8 *dst,
  const local uchar8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uchar16 *dst,
  const local uchar16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global short *dst,
  const local short *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global short2 *dst,
  const local short2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global short4 *dst,
  const local short4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global short8 *dst,
  const local short8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global short16 *dst,
  const local short16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ushort *dst,
  const local ushort *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ushort2 *dst,
  const local ushort2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ushort4 *dst,
  const local ushort4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ushort8 *dst,
  const local ushort8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ushort16 *dst,
  const local ushort16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global int *dst,
  const local int *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global int2 *dst,
  const local int2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global int4 *dst,
  const local int4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global int8 *dst,
  const local int8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global int16 *dst,
  const local int16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uint *dst,
  const local uint *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uint2 *dst,
  const local uint2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uint4 *dst,
  const local uint4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uint8 *dst,
  const local uint8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global uint16 *dst,
  const local uint16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global float *dst,
  const local float *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global float2 *dst,
  const local float2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global float4 *dst,
  const local float4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global float8 *dst,
  const local float8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global float16 *dst,
  const local float16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global long *dst,
  const local long *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global long2 *dst,
  const local long2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global long4 *dst,
  const local long4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global long8 *dst,
  const local long8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global long16 *dst,
  const local long16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ulong *dst,
  const local ulong *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ulong2 *dst,
  const local ulong2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ulong4 *dst,
  const local ulong4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ulong8 *dst,
  const local ulong8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global ulong16 *dst,
  const local ulong16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global double *dst,
  const local double *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global double2 *dst,
  const local double2 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global double4 *dst,
  const local double4 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global double8 *dst,
  const local double8 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) event_t async_work_group_strided_copy(
  global double16 *dst,
  const local double16 *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
__attribute__((overloadable)) void prefetch(const global char *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global char2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global char4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global char8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global char16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uchar *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uchar2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uchar4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uchar8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uchar16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global short *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global short2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global short4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global short8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global short16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ushort *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ushort2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ushort4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ushort8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ushort16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global int *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global int2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global int4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global int8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global int16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uint *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uint2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uint4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uint8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global uint16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global float *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global float2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global float4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global float8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global float16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global long *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global long2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global long4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global long8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global long16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ulong *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ulong2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ulong4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ulong8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global ulong16 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global double *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global double2 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global double4 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global double8 *p, size_t num_gentypes);
__attribute__((overloadable)) void prefetch(const global double16 *p, size_t num_gentypes);
void wait_group_events(int num_events, event_t *event_list);
// __attribute__((overloadable)) int atomic_add (volatile global int *, int); __attribute__((overloadable)) int atomic_add (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_add (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_add (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_and (volatile global int *, int); __attribute__((overloadable)) int atomic_and (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_and (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_and (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_cmpxchg (volatile global int *, int, int); __attribute__((overloadable)) int atomic_cmpxchg (volatile local int *, int, int);
// __attribute__((overloadable)) uint atomic_cmpxchg (volatile global uint *, uint, uint); __attribute__((overloadable)) uint atomic_cmpxchg (volatile local uint *, uint, uint);
// __attribute__((overloadable)) int atomic_max (volatile global int *, int); __attribute__((overloadable)) int atomic_max (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_max (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_max (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_min (volatile global int *, int); __attribute__((overloadable)) int atomic_min (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_min (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_min (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_or (volatile global int *, int); __attribute__((overloadable)) int atomic_or (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_or (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_or (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_sub (volatile global int *, int); __attribute__((overloadable)) int atomic_sub (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_sub (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_sub (volatile local uint *, uint);
// __attribute__((overloadable)) int atomic_xchg (volatile global int *, int); __attribute__((overloadable)) int atomic_xchg (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_xchg (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_xchg (volatile local uint *, uint);
// __attribute__((overloadable)) float atomic_xchg (volatile global float *, float); __attribute__((overloadable)) float atomic_xchg (volatile local float *, float);;
// __attribute__((overloadable)) int atomic_xor (volatile global int *, int); __attribute__((overloadable)) int atomic_xor (volatile local int *, int);
// __attribute__((overloadable)) uint atomic_xor (volatile global uint *, uint); __attribute__((overloadable)) uint atomic_xor (volatile local uint *, uint);
__attribute__((overloadable)) int atom_add(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_add(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_cmpxchg(global int *p, int cmp, int val);
__attribute__((overloadable)) unsigned int atom_cmpxchg(global unsigned int *p, unsigned int cmp, unsigned int val);
__attribute__((overloadable)) int atom_dec(global int *p);
__attribute__((overloadable)) unsigned int atom_dec(global unsigned int *p);
__attribute__((overloadable)) int atom_inc(global int *p);
__attribute__((overloadable)) unsigned int atom_inc(global unsigned int *p);
__attribute__((overloadable)) int atom_sub(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_sub(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_xchg(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_xchg(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_and(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_and(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_max(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_max(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_min(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_min(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_or(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_or(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_xor(global int *p, int val);
__attribute__((overloadable)) unsigned int atom_xor(global unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_add(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_add(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_cmpxchg(local int *p, int cmp, int val);
__attribute__((overloadable)) unsigned int atom_cmpxchg(local unsigned int *p, unsigned int cmp, unsigned int val);
__attribute__((overloadable)) int atom_dec(local int *p);
__attribute__((overloadable)) unsigned int atom_dec(local unsigned int *p);
__attribute__((overloadable)) int atom_inc(local int *p);
__attribute__((overloadable)) unsigned int atom_inc(local unsigned int *p);
__attribute__((overloadable)) int atom_sub(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_sub(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_xchg(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_xchg(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_and(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_and(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_max(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_max(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_min(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_min(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_or(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_or(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int atom_xor(local int *p, int val);
__attribute__((overloadable)) unsigned int atom_xor(local unsigned int *p, unsigned int val);
__attribute__((overloadable)) int get_image_width (image2d_t image);
__attribute__((overloadable)) int get_image_width (image3d_t image);
__attribute__((overloadable)) int get_image_height (image2d_t image);
__attribute__((overloadable)) int get_image_height (image3d_t image);
__attribute__((overloadable)) int get_image_depth (image3d_t image);
__attribute__((overloadable)) int get_image_channel_data_type (image2d_t image);
__attribute__((overloadable)) int get_image_channel_data_type (image3d_t image);
__attribute__((overloadable)) int get_image_channel_order (image2d_t image);
__attribute__((overloadable)) int get_image_channel_order (image3d_t image);
__attribute__((overloadable)) int2 get_image_dim (image2d_t image);
__attribute__((overloadable)) int4 get_image_dim (image3d_t image);
__attribute__((overloadable)) void
write_imagef(image2d_t image, int2 coord, float4 color);
__attribute__((overloadable)) void
write_imagei(image2d_t image, int2 coord, int4 color);
__attribute__((overloadable)) void
write_imageui(image2d_t image, int2 coord, uint4 color);
__attribute__((overloadable)) float4
read_imagef(image2d_t image, sampler_t sampler, int2 coord);
__attribute__((overloadable)) float4
read_imagef(image2d_t image, sampler_t sampler, float2 coord);
__attribute__((overloadable)) int4
read_imagei(image2d_t image, sampler_t sampler, int2 coord);
__attribute__((overloadable)) int4
read_imagei(image2d_t image, sampler_t sampler, float2 coord);
__attribute__((overloadable)) uint4
read_imageui(image2d_t image, sampler_t sampler, int2 coord);
__attribute__((overloadable)) uint4
read_imageui(image2d_t image, sampler_t sampler, float2 coord);
#pragma OPENCL EXTENSION all : disable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// typedef float CONVT;
// typedef float DATA_TYPE;
// typedef float DATATYPE;
// typedef float FLOAT_T;
// typedef float FLOAT_TYPE;
// typedef float FPTYPE;
// typedef float hmc_float;
// typedef float inType;
// typedef float outType;
// typedef float real;
// typedef float REAL;
// typedef float Ty;
// typedef float TyOut;
// typedef float TYPE;
// typedef float VALTYPE;
// typedef float VALUE_TYPE;
// typedef float VECTYPE;
// typedef float WORKTYPE;
// typedef float2 hmc_complex;
// typedef float2 mixed2;
// typedef float2 real2;
// typedef float2 REAL2;
// typedef float3 mixed3;
// typedef float3 real3;
// typedef float3 REAL3;
// typedef float4 FPVECTYPE;
// typedef float4 mixed4;
// typedef float4 real4;
// typedef float4 REAL4;
// typedef float4 T4;
// typedef int BITMAP_INDEX_TYPE;
// typedef int INDEX_TYPE;
// typedef int Ix;
// typedef int KParam;
// typedef int Tp;
// typedef int3 Pixel;
// typedef unsigned int uint32_t;
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2

#define CHAR_BIT  8
#define SCHAR_MAX 127
#define SCHAR_MIN (-128)
#define UCHAR_MAX 255
#define CHAR_MAX  UCHAR_MAX
#define CHAR_MIN  0
#define INT_MAX   2147483647
#define INT_MIN   (-INT_MAX-1)
#define UINT_MAX  (2U*INT_MAX+1)
#define LONG_MAX  2147483647
#define LONG_MIN  (-LONG_MAX-1)
#define ULONG_MAX (2UL*LONG_MAX+1)
#define SHRT_MAX  32767
#define SHRT_MIN  (-SHRT_MAX-1)
#define USHRT_MAX  65535

#define FLT_RADIX 2
#define DECIMAL_DIG 37
#define FLT_MIN 1.175494e-38
#define FLT_MAX 3.402823e+38
#define FLT_EPSILON 1.192093e-07
#define FLT_DIG 6
#define FLT_MANT_DIG 24
#define FLT_MIN_EXP -125
#define FLT_MIN_10_EXP -37
#define FLT_MAX_EXP 128
#define FLT_MAX_10_EXP 38
#define FLT_ROUNDS 1
#define FLT_EVAL_METHOD 1
#define FLT_HAS_SUBNORM 1
