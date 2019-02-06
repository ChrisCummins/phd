/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef OCLDCT8x8_COMMON_H
#define OCLDCT8x8_COMMON_H

#include <oclUtils.h>

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
typedef cl_uint uint;
#define BLOCK_SIZE 8
#define DCT_FORWARD 666
#define DCT_INVERSE 777

////////////////////////////////////////////////////////////////////////////////
// Reference CPU 8x8 (i)DCT
////////////////////////////////////////////////////////////////////////////////
extern "C" void DCT8x8CPU(float *dst, float *src, uint stride, uint imageH, uint imageW, int dir);

////////////////////////////////////////////////////////////////////////////////
// OpenCL 8x8 (i)DCT
////////////////////////////////////////////////////////////////////////////////
extern "C" void initDCT8x8(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeDCT8x8(void);
extern "C" void DCT8x8(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_uint stride,
    cl_uint imageH,
    cl_uint imageW,
    cl_int dir
);

#endif
