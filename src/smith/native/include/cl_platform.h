/**********************************************************************************
 * Copyright (c) 2008-2013 The Khronos Group Inc.
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
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 **********************************************************************************/

/* $Revision: 11803 $ on $Date: 2010-06-25 10:02:12 -0700 (Fri, 25 Jun 2010) $ */

#ifndef CL_PLATFORM_H
#define CL_PLATFORM_H

#define CL_HAS_ANON_STRUCT 1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define CLK_GLOBAL_MEM_FENCE 1
#define CLK_LOCAL_MEM_FENCE 1

#ifdef cplusplus
extern "C" {
#endif

float exp(float x);
float fabs(float a);
float log(float x);
float pow(float a, float b);
float sqrt(float x);
int atomic_add(__global int *p, int val);
int atomic_and(__global int *p, int val);
int atomic_cmpxchg(__global int *p, int cmp, int val);
int atomic_dec(__global int *p);
int atomic_inc(__global int *p);
int atomic_inc(__global int *p);
int atomic_ma(__global int *p, int val);
int atomic_or(__global int *p, int val);
int atomic_sub(__global int *p, int val);
int atomic_xchg(volatile __global int *p, int val);
int atomic_xor(__global int *p, int val);
int get_global_id(int i);
int get_global_size(int i);
int get_group_id(int i);
int get_local_id(int i);
int get_local_size(int i);
int get_num_groups(int i);
void barrier();

/* scalar types  */
typedef char         uchar;
typedef short        ushort;
typedef unsigned     uint;
typedef unsigned long  ulong;

/* Mirror types to GL types. Mirror types allow us to avoid deciding which 87s to load based on whether we are using GL or GLES here. */
typedef unsigned int GLuint;
typedef int          GLint;
typedef unsigned int GLenum;

/* ---- charn ---- */
typedef union
{
    char   s[2];
#if CL_HAS_ANON_STRUCT
    struct{ char  x, y; };
    struct{ char  s0, s1; };
    struct{ char  lo, hi; };
#endif
#if defined( CL_CHAR2)
    char2     v2;
#endif
}char2;

typedef union
{
    char   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ char  x, y, z, w; };
    struct{ char  s0, s1, s2, s3; };
    struct{ char2 lo, hi; };
#endif
#if defined( CL_CHAR2)
    char2     v2[2];
#endif
#if defined( CL_CHAR4)
    char4     v4;
#endif
}char4;

/* char3 is identical in size, alignment and behavior to char4. See section 6.1.5. */
typedef  char4  char3;

typedef union
{
    char    s[8];
#if CL_HAS_ANON_STRUCT
    struct{ char  x, y, z, w; };
    struct{ char  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ char4 lo, hi; };
#endif
#if defined( CL_CHAR2)
    char2     v2[4];
#endif
#if defined( CL_CHAR4)
    char4     v4[2];
#endif
#if defined( CL_CHAR8 )
    char8     v8;
#endif
}char8;

typedef union
{
    char   s[16];
#if CL_HAS_ANON_STRUCT
    struct{ char  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ char  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ char8 lo, hi; };
#endif
#if defined( CL_CHAR2)
    char2     v2[8];
#endif
#if defined( CL_CHAR4)
    char4     v4[4];
#endif
#if defined( CL_CHAR8 )
    char8     v8[2];
#endif
#if defined( CL_CHAR16 )
    char16    v16;
#endif
}char16;


/* ---- ucharn ---- */
typedef union
{
    uchar   s[2];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uchar  x, y; };
    struct{ uchar  s0, s1; };
    struct{ uchar  lo, hi; };
#endif
#if defined( uchar2)
    uchar2     v2;
#endif
}uchar2;

typedef union
{
    uchar   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ uchar  x, y, z, w; };
    struct{ uchar  s0, s1, s2, s3; };
    struct{ uchar2 lo, hi; };
    int _x;
#endif
#if defined( CL_UCHAR2)
    uchar2     v2[2];
#endif
#if defined( CL_UCHAR4)
    uchar4     v4;
#endif
}uchar4;

/* uchar3 is identical in size, alignment and behavior to uchar4. See section 6.1.5. */
typedef  uchar4  uchar3;

typedef union
{
    uchar    s[8];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uchar  x, y, z, w; };
    struct{ uchar  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ uchar4 lo, hi; };
#endif
#if defined( CL_UCHAR2)
    uchar2     v2[4];
#endif
#if defined( CL_UCHAR4)
    uchar4     v4[2];
#endif
#if defined( CL_UCHAR8 )
    uchar8     v8;
#endif
}uchar8;

typedef union
{
    uchar   s[16];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uchar  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ uchar  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ uchar8 lo, hi; };
#endif
#if defined( CL_UCHAR2)
    uchar2     v2[8];
#endif
#if defined( CL_UCHAR4)
    uchar4     v4[4];
#endif
#if defined( CL_UCHAR8 )
    uchar8     v8[2];
#endif
#if defined( CL_UCHAR16 )
    uchar16    v16;
#endif
}uchar16;


/* ---- shortn ---- */
typedef union
{
    short   s[2];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ short  x, y; };
    struct{ short  s0, s1; };
    struct{ short  lo, hi; };
#endif
#if defined( CL_SHORT2)
    short2     v2;
#endif
}short2;

typedef union
{
    short   s[4];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ short  x, y, z, w; };
    struct{ short  s0, s1, s2, s3; };
    struct{ short2 lo, hi; };
#endif
#if defined( CL_SHORT2)
    short2     v2[2];
#endif
#if defined( CL_SHORT4)
    short4     v4;
#endif
}short4;

/* short3 is identical in size, alignment and behavior to short4. See section 6.1.5. */
typedef  short4  short3;

typedef union
{
    short    s[8];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ short  x, y, z, w; };
    struct{ short  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ short4 lo, hi; };
#endif
#if defined( CL_SHORT2)
    short2     v2[4];
#endif
#if defined( CL_SHORT4)
    short4     v4[2];
#endif
#if defined( CL_SHORT8 )
    short8     v8;
#endif
}short8;

typedef union
{
    short   s[16];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ short  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ short  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ short8 lo, hi; };
#endif
#if defined( CL_SHORT2)
    short2     v2[8];
#endif
#if defined( CL_SHORT4)
    short4     v4[4];
#endif
#if defined( CL_SHORT8 )
    short8     v8[2];
#endif
#if defined( CL_SHORT16 )
    short16    v16;
#endif
}short16;


/* ---- ushortn ---- */
typedef union
{
    ushort   s[2];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ ushort  x, y; };
    struct{ ushort  s0, s1; };
    struct{ ushort  lo, hi; };
#endif
#if defined( CL_USHORT2)
    ushort2     v2;
#endif
}ushort2;

typedef union
{
    ushort   s[4];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ ushort  x, y, z, w; };
    struct{ ushort  s0, s1, s2, s3; };
    struct{ ushort2 lo, hi; };
#endif
#if defined( CL_USHORT2)
    ushort2     v2[2];
#endif
#if defined( CL_USHORT4)
    ushort4     v4;
#endif
}ushort4;

/* ushort3 is identical in size, alignment and behavior to ushort4. See section 6.1.5. */
typedef  ushort4  ushort3;

typedef union
{
    ushort    s[8];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ ushort  x, y, z, w; };
    struct{ ushort  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ ushort4 lo, hi; };
#endif
#if defined( CL_USHORT2)
    ushort2     v2[4];
#endif
#if defined( CL_USHORT4)
    ushort4     v4[2];
#endif
#if defined( CL_USHORT8 )
    ushort8     v8;
#endif
}ushort8;

typedef union
{
    ushort   s[16];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ ushort  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ ushort  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ ushort8 lo, hi; };
#endif
#if defined( CL_USHORT2)
    ushort2     v2[8];
#endif
#if defined( CL_USHORT4)
    ushort4     v4[4];
#endif
#if defined( CL_USHORT8 )
    ushort8     v8[2];
#endif
#if defined( CL_USHORT16 )
    ushort16    v16;
#endif
}ushort16;

/* ---- intn ---- */
typedef union
{
    int   s[2];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ int  x, y; };
    struct{ int  s0, s1; };
    struct{ int  lo, hi; };
#endif
#if defined( CL_INT2)
    int2     v2;
#endif
}int2;

typedef union
{
    int   s[4];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ int  x, y, z, w; };
    struct{ int  s0, s1, s2, s3; };
    struct{ int2 lo, hi; };
#endif
#if defined( CL_INT2)
    int2     v2[2];
#endif
#if defined( CL_INT4)
    int4     v4;
#endif
}int4;

/* int3 is identical in size, alignment and behavior to int4. See section 6.1.5. */
typedef  int4  int3;

typedef union
{
    int    s[8];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ int  x, y, z, w; };
    struct{ int  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ int4 lo, hi; };
#endif
#if defined( CL_INT2)
    int2     v2[4];
#endif
#if defined( CL_INT4)
    int4     v4[2];
#endif
#if defined( CL_INT8 )
    int8     v8;
#endif
}int8;

typedef union
{
    int   s[16];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ int  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ int  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ int8 lo, hi; };
#endif
#if defined( CL_INT2)
    int2     v2[8];
#endif
#if defined( CL_INT4)
    int4     v4[4];
#endif
#if defined( CL_INT8 )
    int8     v8[2];
#endif
#if defined( CL_INT16 )
    int16    v16;
#endif
}int16;


/* ---- uintn ---- */
typedef union
{
    uint   s[2];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uint  x, y; };
    struct{ uint  s0, s1; };
    struct{ uint  lo, hi; };
#endif
#if defined( CL_UINT2)
    uint2     v2;
#endif
}uint2;

typedef union
{
    uint   s[4];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uint  x, y, z, w; };
    struct{ uint  s0, s1, s2, s3; };
    struct{ uint2 lo, hi; };
#endif
#if defined( CL_UINT2)
    uint2     v2[2];
#endif
#if defined( CL_UINT4)
    uint4     v4;
#endif
}uint4;

/* uint3 is identical in size, alignment and behavior to uint4. See section 6.1.5. */
typedef  uint4  uint3;

typedef union
{
    uint    s[8];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uint  x, y, z, w; };
    struct{ uint  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ uint4 lo, hi; };
#endif
#if defined( CL_UINT2)
    uint2     v2[4];
#endif
#if defined( CL_UINT4)
    uint4     v4[2];
#endif
#if defined( CL_UINT8 )
    uint8     v8;
#endif
}uint8;

typedef union
{
    uint   s[16];
    int _x;
#if CL_HAS_ANON_STRUCT
    struct{ uint  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ uint  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ uint8 lo, hi; };
#endif
#if defined( CL_UINT2)
    uint2     v2[8];
#endif
#if defined( CL_UINT4)
    uint4     v4[4];
#endif
#if defined( CL_UINT8 )
    uint8     v8[2];
#endif
#if defined( CL_UINT16 )
    uint16    v16;
#endif
}uint16;

/* ---- longn ---- */
typedef union
{
    long   s[2];
#if CL_HAS_ANON_STRUCT
    struct{ long  x, y; };
    struct{ long  s0, s1; };
    struct{ long  lo, hi; };
#endif
#if defined( CL_LONG2)
    long2     v2;
#endif
}long2;

typedef union
{
    long   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ long  x, y, z, w; };
    struct{ long  s0, s1, s2, s3; };
    struct{ long2 lo, hi; };
#endif
#if defined( CL_LONG2)
    long2     v2[2];
#endif
#if defined( CL_LONG4)
    long4     v4;
#endif
}long4;

/* long3 is identical in size, alignment and behavior to long4. See section 6.1.5. */
typedef  long4  long3;

typedef union
{
    long    s[8];
#if CL_HAS_ANON_STRUCT
    struct{ long  x, y, z, w; };
    struct{ long  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ long4 lo, hi; };
#endif
#if defined( CL_LONG2)
    long2     v2[4];
#endif
#if defined( CL_LONG4)
    long4     v4[2];
#endif
#if defined( CL_LONG8 )
    long8     v8;
#endif
}long8;

typedef union
{
    long   s[16];
#if CL_HAS_ANON_STRUCT
    struct{ long  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ long  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ long8 lo, hi; };
#endif
#if defined( CL_LONG2)
    long2     v2[8];
#endif
#if defined( CL_LONG4)
    long4     v4[4];
#endif
#if defined( CL_LONG8 )
    long8     v8[2];
#endif
#if defined( CL_LONG16 )
    long16    v16;
#endif
}long16;


/* ---- ulongn ---- */
typedef union
{
    ulong   s[2];
#if CL_HAS_ANON_STRUCT
    struct{ ulong  x, y; };
    struct{ ulong  s0, s1; };
    struct{ ulong  lo, hi; };
#endif
#if defined( CL_ULONG2)
    ulong2     v2;
#endif
}ulong2;

typedef union
{
    ulong   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ ulong  x, y, z, w; };
    struct{ ulong  s0, s1, s2, s3; };
    struct{ ulong2 lo, hi; };
#endif
#if defined( CL_ULONG2)
    ulong2     v2[2];
#endif
#if defined( CL_ULONG4)
    ulong4     v4;
#endif
}ulong4;

/* ulong3 is identical in size, alignment and behavior to ulong4. See section 6.1.5. */
typedef  ulong4  ulong3;

typedef union
{
    ulong    s[8];
#if CL_HAS_ANON_STRUCT
    struct{ ulong  x, y, z, w; };
    struct{ ulong  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ ulong4 lo, hi; };
#endif
#if defined( CL_ULONG2)
    ulong2     v2[4];
#endif
#if defined( CL_ULONG4)
    ulong4     v4[2];
#endif
#if defined( CL_ULONG8 )
    ulong8     v8;
#endif
}ulong8;

typedef union
{
    ulong   s[16];
#if CL_HAS_ANON_STRUCT
    struct{ ulong  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ ulong  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ ulong8 lo, hi; };
#endif
#if defined( CL_ULONG2)
    ulong2     v2[8];
#endif
#if defined( CL_ULONG4)
    ulong4     v4[4];
#endif
#if defined( CL_ULONG8 )
    ulong8     v8[2];
#endif
#if defined( CL_ULONG16 )
    ulong16    v16;
#endif
}ulong16;


/* --- floatn ---- */

typedef union
{
    float   s[2];
#if CL_HAS_ANON_STRUCT
    struct{ float  x, y; };
    struct{ float  s0, s1; };
    struct{ float  lo, hi; };
#endif
#if defined( CL_FLOAT2)
    float2     v2;
#endif
}float2;

typedef union
{
    float   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ float   x, y, z, w; };
    struct{ float   s0, s1, s2, s3; };
    struct{ float2  lo, hi; };
#endif
#if defined( CL_FLOAT2)
    float2     v2[2];
#endif
#if defined( CL_FLOAT4)
    float4     v4;
#endif
}float4;

/* float3 is identical in size, alignment and behavior to float4. See section 6.1.5. */
typedef  float4  float3;

typedef union
{
    float    s[8];
#if CL_HAS_ANON_STRUCT
    struct{ float   x, y, z, w; };
    struct{ float   s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ float4  lo, hi; };
#endif
#if defined( CL_FLOAT2)
    float2     v2[4];
#endif
#if defined( CL_FLOAT4)
    float4     v4[2];
#endif
#if defined( CL_FLOAT8 )
    float8     v8;
#endif
}float8;

typedef union
{
    float   s[16];
#if CL_HAS_ANON_STRUCT
    struct{ float  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ float  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ float8 lo, hi; };
#endif
#if defined( CL_FLOAT2)
    float2     v2[8];
#endif
#if defined( CL_FLOAT4)
    float4     v4[4];
#endif
#if defined( CL_FLOAT8 )
    float8     v8[2];
#endif
#if defined( CL_FLOAT16 )
    float16    v16;
#endif
}float16;

/* --- doublen ---- */

typedef union
{
    double   s[2];
#if CL_HAS_ANON_STRUCT
    struct{ double  x, y; };
    struct{ double s0, s1; };
    struct{ double lo, hi; };
#endif
#if defined( CL_DOUBLE2)
    double2     v2;
#endif
}double2;

typedef union
{
    double   s[4];
#if CL_HAS_ANON_STRUCT
    struct{ double  x, y, z, w; };
    struct{ double  s0, s1, s2, s3; };
    struct{ double2 lo, hi; };
#endif
#if defined( CL_DOUBLE2)
    double2     v2[2];
#endif
#if defined( CL_DOUBLE4)
    double4     v4;
#endif
}double4;

/* double3 is identical in size, alignment and behavior to double4. See section 6.1.5. */
typedef  double4  double3;

typedef union
{
    double    s[8];
#if CL_HAS_ANON_STRUCT
    struct{ double  x, y, z, w; };
    struct{ double  s0, s1, s2, s3, s4, s5, s6, s7; };
    struct{ double4 lo, hi; };
#endif
#if defined( CL_DOUBLE2)
    double2     v2[4];
#endif
#if defined( CL_DOUBLE4)
    double4     v4[2];
#endif
#if defined( CL_DOUBLE8 )
    double8     v8;
#endif
}double8;

typedef union
{
    double   s[16];
#if CL_HAS_ANON_STRUCT
    struct{ double  x, y, z, w, spacer4, spacer5, spacer6, spacer7, spacer8, spacer9, sa, sb, sc, sd, se, sf; };
    struct{ double  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
    struct{ double8 lo, hi; };
#endif
#if defined( CL_DOUBLE2)
    double2     v2[8];
#endif
#if defined( CL_DOUBLE4)
    double4     v4[4];
#endif
#if defined( CL_DOUBLE8 )
    double8     v8[2];
#endif
#if defined( CL_DOUBLE16 )
    double16    v16;
#endif
}double16;


#ifdef cplusplus
}
#endif

#endif  /* CL_PLATFORM_H  */
