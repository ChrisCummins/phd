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

#include "oclHistogram_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for histogram256 / mergeHistogram256 kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL histogram256 program
static cl_program cpHistogram256;

//OpenCL histogram256 kernels
static cl_kernel
    ckHistogram256, ckMergeHistogram256;

//Default command queue for histogram256 kernels
static cl_command_queue cqDefaultCommandQue;

//histogram256() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
static cl_mem d_PartialHistograms;

static const uint            WARP_SIZE = 32;
static const uint           WARP_COUNT = 6;
static const uint MERGE_WORKGROUP_SIZE = 256;
static const char      *compileOptions = "-D LOG2_WARP_SIZE=5 -D WARP_COUNT=6 -D MERGE_WORKGROUP_SIZE=256";

extern "C" void initHistogram256(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    size_t kernelLength;
    cl_int ciErrNum;

    shrLog("...loading Histogram256.cl\n");
        char *cHistogram256 = oclLoadProgSource(shrFindFilePath("Histogram256.cl", argv[0]), "// My comment\n", &kernelLength);
        shrCheckError(cHistogram256 != NULL, shrTRUE);

    shrLog("...creating histogram256 program\n");
        cpHistogram256 = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cHistogram256, &kernelLength, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...building histogram256 program\n");
        ciErrNum = clBuildProgram(cpHistogram256, 0, NULL, compileOptions, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...creating histogram256 kernels\n");
        ckHistogram256 = clCreateKernel(cpHistogram256, "histogram256", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckMergeHistogram256 = clCreateKernel(cpHistogram256, "mergeHistogram256", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...allocating internal histogram256 buffer\n");
        d_PartialHistograms = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(cl_uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cHistogram256);

    //Save ptx code to separate file
    oclLogPtx(cpHistogram256, oclGetFirstDev(cxGPUContext), "Histogram256.ptx");
}

extern "C" void closeHistogram256(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseMemObject(d_PartialHistograms);
    ciErrNum |= clReleaseKernel(ckMergeHistogram256);
    ciErrNum |= clReleaseKernel(ckHistogram256);
    ciErrNum |= clReleaseProgram(cpHistogram256);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" size_t histogram256(cl_command_queue cqCommandQueue, cl_mem d_Histogram, cl_mem d_Data, uint byteCount){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        shrCheckError( ((byteCount % 4) == 0), shrTRUE );
        uint dataCount = byteCount / 4;
        ciErrNum  = clSetKernelArg(ckHistogram256, 0, sizeof(cl_mem),  (void *)&d_PartialHistograms);
        ciErrNum |= clSetKernelArg(ckHistogram256, 1, sizeof(cl_mem),  (void *)&d_Data);
        ciErrNum |= clSetKernelArg(ckHistogram256, 2, sizeof(cl_uint), (void *)&dataCount);
        shrCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = WARP_SIZE * WARP_COUNT;
        globalWorkSize = PARTIAL_HISTOGRAM256_COUNT * localWorkSize;

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckHistogram256, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    {
        ciErrNum  = clSetKernelArg(ckMergeHistogram256, 0, sizeof(cl_mem),  (void *)&d_Histogram);
        ciErrNum |= clSetKernelArg(ckMergeHistogram256, 1, sizeof(cl_mem),  (void *)&d_PartialHistograms);
        ciErrNum |= clSetKernelArg(ckMergeHistogram256, 2, sizeof(cl_uint), (void *)&PARTIAL_HISTOGRAM256_COUNT);
        shrCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = MERGE_WORKGROUP_SIZE;
        globalWorkSize = HISTOGRAM256_BIN_COUNT * localWorkSize;

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMergeHistogram256, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        return (WARP_SIZE * WARP_COUNT);
    }
}
