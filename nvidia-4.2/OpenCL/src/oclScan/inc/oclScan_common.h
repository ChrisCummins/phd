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

#ifndef OCLCOMMON_H
#define OCLCOMMON_H

#include <oclUtils.h>

////////////////////////////////////////////////////////////////////////////////
// Shortcut typenames
////////////////////////////////////////////////////////////////////////////////
typedef cl_uint uint;

////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
extern "C" const uint MAX_BATCH_ELEMENTS;
extern "C" const uint MIN_SHORT_ARRAY_SIZE;
extern "C" const uint MAX_SHORT_ARRAY_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE;

////////////////////////////////////////////////////////////////////////////////
// OpenCL scan
////////////////////////////////////////////////////////////////////////////////
extern "C" void initScan(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeScan(void);
extern "C" size_t scanExclusiveShort(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint batchSize,
    uint arrayLength
);

extern "C" size_t scanExclusiveLarge(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint batchSize,
    uint arrayLength
);

////////////////////////////////////////////////////////////////////////////////
// Reference CPU batched inclusive scan
////////////////////////////////////////////////////////////////////////////////
extern "C" void scanExclusiveHost(
    uint *dst,
    uint *src,
    uint batchSize,
    uint arrayLength
);

#endif
