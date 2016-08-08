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

#ifndef OCLCONVOLUTIONSEPARABLE_COMMON_H
#define OCLCONVOLUTIONSEPARABLE_COMMON_H

//Standard utilities and systems includes
#include <oclUtils.h>

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

////////////////////////////////////////////////////////////////////////////////
// OpenCL separable convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void initConvolutionSeparable(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeConvolutionSeparable(void);

extern "C" void convolutionRows(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
);

extern "C" void convolutionColumns(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
);

#endif
