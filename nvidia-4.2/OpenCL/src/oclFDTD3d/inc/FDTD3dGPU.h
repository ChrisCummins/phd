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

#ifndef _FDTD3DGPU_H_
#define _FDTD3DGPU_H_

#include <cstddef>
#if defined(_WIN32) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

// The values are set to give reasonable performance, they can be changed
// but note that setting an excessively large work group size can result in
// build failure due to insufficient local memory, even though it would be
// clamped before execution. This is because the maximum work group size
// cannot be determined before the build.
#define k_localWorkX    32
#define k_localWorkY    8
#define k_localWorkMin  128
#define k_localWorkMax  1024

// Name of the file with the source code for the computation kernel
extern const char* clSourceFile;

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv);
bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv);

#endif
