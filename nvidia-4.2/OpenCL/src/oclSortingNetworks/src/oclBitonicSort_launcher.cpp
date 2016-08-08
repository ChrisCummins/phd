#include <cecl.h>
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
#include "oclSortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
//OpenCL bitonic sort program
static cl_program cpBitonicSort;

//OpenCL bitonic sort kernels
static cl_kernel
    ckBitonicSortLocal,
    ckBitonicSortLocal1,
    ckBitonicMergeGlobal,
    ckBitonicMergeLocal;

//Default command queue for bitonic kernels
static cl_command_queue cqDefaultCommandQue;

static const uint LOCAL_SIZE_LIMIT = 512U;
static const char  *compileOptions = "-D LOCAL_SIZE_LIMIT=512";

extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("...loading BitonicSort.cl\n");
        char *cBitonicSort = oclLoadProgSource(shrFindFilePath("BitonicSort.cl", argv[0]), "// My comment\n", &kernelLength);
        oclCheckError(cBitonicSort != NULL, shrTRUE);

    shrLog("...creating bitonic sort program\n");
        cpBitonicSort = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cBitonicSort, &kernelLength, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...building bitonic sort program\n");
        ciErrNum = CECL_PROGRAM(cpBitonicSort, 0, NULL, compileOptions, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog( "...creating bitonic sort kernels\n");
        ckBitonicSortLocal = CECL_KERNEL(cpBitonicSort, "bitonicSortLocal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicSortLocal1 = CECL_KERNEL(cpBitonicSort, "bitonicSortLocal1", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicMergeGlobal = CECL_KERNEL(cpBitonicSort, "bitonicMergeGlobal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicMergeLocal = CECL_KERNEL(cpBitonicSort, "bitonicMergeLocal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog( "...checking minimum supported workgroup size\n");
        //Check for work group size
        cl_device_id device;
        size_t szBitonicSortLocal, szBitonicSortLocal1, szBitonicMergeLocal;

        ciErrNum  = clGetCommandQueueInfo(cqParamCommandQue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
        ciErrNum |= clGetKernelWorkGroupInfo(ckBitonicSortLocal,  device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szBitonicSortLocal, NULL);
        ciErrNum |= clGetKernelWorkGroupInfo(ckBitonicSortLocal1, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szBitonicSortLocal1, NULL);
        ciErrNum |= clGetKernelWorkGroupInfo(ckBitonicMergeLocal, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szBitonicMergeLocal, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        if( (szBitonicSortLocal < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicSortLocal1 < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicMergeLocal < (LOCAL_SIZE_LIMIT / 2)) ){
            shrLog("\nERROR !!! Minimum work-group size %u required by this application is not supported on this device.\n\n", LOCAL_SIZE_LIMIT / 2);
            closeBitonicSort();
            free(cBitonicSort);
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(EXIT_FAILURE);
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cBitonicSort);
}

extern "C" void closeBitonicSort(void)
{
    cl_int ciErrNum;
    ciErrNum  = clReleaseKernel(ckBitonicMergeLocal);
    ciErrNum |= clReleaseKernel(ckBitonicMergeGlobal);
    ciErrNum |= clReleaseKernel(ckBitonicSortLocal1);
    ciErrNum |= clReleaseKernel(ckBitonicSortLocal);
    ciErrNum |= clReleaseProgram(cpBitonicSort);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

static cl_uint factorRadix2(cl_uint& log2L, cl_uint L){
    if(!L){
        log2L = 0;
        return 0;
    }else{
        for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}

extern "C" size_t bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal,
    uint batch,
    uint arrayLength,
    uint dir
){
    if(arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported so far
    cl_uint log2L;
    cl_uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    oclCheckError( factorizationRemainder == 1, shrTRUE );

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    dir = (dir != 0);

    cl_int ciErrNum;
    size_t localWorkSize; 
    size_t globalWorkSize;

    if(arrayLength <= LOCAL_SIZE_LIMIT)
    {
        oclCheckError( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0, shrTRUE );
        //Launch bitonicSortLocal
        ciErrNum  = CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 0,   sizeof(cl_mem), (void *)&d_DstKey);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 1,   sizeof(cl_mem), (void *)&d_DstVal);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 2,   sizeof(cl_mem), (void *)&d_SrcKey);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 3,   sizeof(cl_mem), (void *)&d_SrcVal);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 4,  sizeof(cl_uint), (void *)&arrayLength);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal, 5,  sizeof(cl_uint), (void *)&dir);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckBitonicSortLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    else
    {
        //Launch bitonicSortLocal1
        ciErrNum  = CECL_SET_KERNEL_ARG(ckBitonicSortLocal1, 0,  sizeof(cl_mem), (void *)&d_DstKey);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal1, 1,  sizeof(cl_mem), (void *)&d_DstVal);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal1, 2,  sizeof(cl_mem), (void *)&d_SrcKey);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicSortLocal1, 3,  sizeof(cl_mem), (void *)&d_SrcVal);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckBitonicSortLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        for(uint size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        {
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                if(stride >= LOCAL_SIZE_LIMIT)
                {
                    //Launch bitonicMergeGlobal
                    ciErrNum  = CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 0,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 1,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 2,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 3,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 4, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 5, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 6, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeGlobal, 7, sizeof(cl_uint), (void *)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    globalWorkSize = batch * arrayLength / 2;
                    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckBitonicMergeGlobal, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                }
                else
                {
                    //Launch bitonicMergeLocal
                    ciErrNum  = CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 0,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 1,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 2,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 3,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 4, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 5, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 6, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= CECL_SET_KERNEL_ARG(ckBitonicMergeLocal, 7, sizeof(cl_uint), (void *)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
                    globalWorkSize = batch * arrayLength / 2;

                    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckBitonicMergeLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                    break;
                }
            }
        }
    }
    return localWorkSize;
}
