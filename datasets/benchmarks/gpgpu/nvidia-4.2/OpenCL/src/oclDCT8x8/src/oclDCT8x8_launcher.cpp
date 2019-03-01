#include <libcecl.h>
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

#include <oclUtils.h>
#include "oclDCT8x8_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for DCT8x8 / IDCT8x8 kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL DCT8x8 program
static cl_program
    cpDCT8x8;

//OpenCL DCT8x8 kernels
static cl_kernel
    ckDCT8x8, ckIDCT8x8;

//Default command queue for DCT8x8 kernels
static cl_command_queue cqDefaultCommandQue;

extern "C" void initDCT8x8(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("Loading OpenCL DCT8x8...\n");
        char *cPathAndName = shrFindFilePath("DCT8x8.cl", argv[0]);
        shrCheckError(cPathAndName != NULL, shrTRUE);
        char *cDCT8x8 = oclLoadProgSource(cPathAndName, "// My comment\n", &kernelLength);
        shrCheckError(cDCT8x8 != NULL, shrTRUE);

    shrLog("Creating DCT8x8 program...\n");
        cpDCT8x8 = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cDCT8x8, &kernelLength, &ciErrNum);
        shrCheckError (ciErrNum, CL_SUCCESS);

    shrLog("Building DCT8x8 program...\n");
        ciErrNum = CECL_PROGRAM(cpDCT8x8, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
        shrCheckError (ciErrNum, CL_SUCCESS);

    shrLog("Creating DCT8x8 kernels...\n");
        ckDCT8x8 = CECL_KERNEL(cpDCT8x8, "DCT8x8", &ciErrNum);
        shrCheckError (ciErrNum, CL_SUCCESS);
        ckIDCT8x8= CECL_KERNEL(cpDCT8x8, "IDCT8x8", &ciErrNum);
        shrCheckError (ciErrNum, CL_SUCCESS);

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cDCT8x8);
}

extern "C" void closeDCT8x8(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseKernel(ckIDCT8x8);
    ciErrNum |= clReleaseKernel(ckDCT8x8);
    ciErrNum |= clReleaseProgram(cpDCT8x8);
}

inline uint iDivUp(uint dividend, uint divisor){
    return dividend / divisor + (dividend % divisor != 0);
}

extern "C" void DCT8x8(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_uint stride,
    cl_uint imageH,
    cl_uint imageW,
    cl_int dir
){
    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    shrCheckError((dir == DCT_FORWARD) || (dir == DCT_INVERSE), CL_TRUE);
    cl_kernel ckDCT = (dir == DCT_FORWARD) ? ckDCT8x8 : ckIDCT8x8;

    const uint BLOCK_X = 32;
    const uint BLOCK_Y = 16;

    size_t localWorkSize[2], globalWorkSize[2];
    cl_uint ciErrNum;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckDCT, 0, sizeof(cl_mem),  (void*)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckDCT, 1, sizeof(cl_mem),  (void*)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckDCT, 2, sizeof(cl_uint), (void*)&stride);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckDCT, 3, sizeof(cl_uint), (void*)&imageH);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckDCT, 4, sizeof(cl_uint), (void*)&imageW);
    shrCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize[0] = BLOCK_X;
    localWorkSize[1] = BLOCK_Y / BLOCK_SIZE;
    globalWorkSize[0] = iDivUp(imageW, BLOCK_X) * localWorkSize[0];
    globalWorkSize[1] = iDivUp(imageH, BLOCK_Y) * localWorkSize[1];

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckDCT, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    shrCheckError (ciErrNum, CL_SUCCESS);
}
