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

////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////
// Host-side validation routines
////////////////////////////////////////////////////////////////////////////////
extern "C" int validateSortedKeys(
    uint *result,
    uint *data,
    uint batch,
    uint N,
    uint numValues,
    uint dir,
    uint *srcHist,
    uint *resHist
);

extern "C" void fillValues(
    uint *val,
    uint N
);

extern "C" int validateSortedValues(
    uint *resKey,
    uint *resVal,
    uint *srcKey,
    uint batchSize,
    uint arrayLength
);

extern "C" int validateValues(
    uint *oval,
    uint *okey,
    uint N
);

////////////////////////////////////////////////////////////////////////////////
// OpenCL bitonic sort
////////////////////////////////////////////////////////////////////////////////
extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);

extern "C" void closeBitonicSort(void);

extern"C" size_t bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal,
    uint batch,
    uint arrayLength,
    uint dir
);
