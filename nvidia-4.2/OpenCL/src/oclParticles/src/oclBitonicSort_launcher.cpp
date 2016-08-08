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

// standard utilities and systems includes
#include <oclUtils.h>

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

extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv)
{
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("...loading BitonicSort_b.cl\n");
        char *cBitonicSort = oclLoadProgSource(shrFindFilePath("BitonicSort_b.cl", argv[0]), "// My comment\n", &kernelLength);
        oclCheckError(cBitonicSort != NULL, shrTRUE);

    shrLog("...creating bitonic sort program\n");
        cpBitonicSort = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cBitonicSort, &kernelLength, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...building bitonic sort program\n");
        ciErrNum = clBuildProgram(cpBitonicSort, 0, NULL, NULL, NULL, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			// write out standard error, Build Log and PTX, then cleanup and exit
			shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
			oclLogBuildInfo(cpBitonicSort, oclGetFirstDev(cxGPUContext));
			oclLogPtx(cpBitonicSort, oclGetFirstDev(cxGPUContext), "oclBitonicSort.ptx");
			oclCheckError(ciErrNum, CL_SUCCESS); 
		}

    shrLog("...creating bitonic sort kernels\n");
        ckBitonicSortLocal = clCreateKernel(cpBitonicSort, "bitonicSortLocal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicSortLocal1 = clCreateKernel(cpBitonicSort, "bitonicSortLocal1", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicMergeGlobal = clCreateKernel(cpBitonicSort, "bitonicMergeGlobal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckBitonicMergeLocal = clCreateKernel(cpBitonicSort, "bitonicMergeLocal", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

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

//Note: logically shared with BitonicSort_b.cl!
static const unsigned int LOCAL_SIZE_LIMIT = 512U;

extern"C" void bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir
){
    if(arrayLength < 2)
        return;

    //Only power-of-two array lengths are supported so far
    cl_uint log2L;
    cl_uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    oclCheckError( factorizationRemainder == 1, shrTRUE );

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    dir = (dir != 0);

    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    if(arrayLength <= LOCAL_SIZE_LIMIT)
    {
        oclCheckError( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0, shrTRUE );
        //Launch bitonicSortLocal
        ciErrNum  = clSetKernelArg(ckBitonicSortLocal, 0,   sizeof(cl_mem), (void *)&d_DstKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 1,   sizeof(cl_mem), (void *)&d_DstVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 2,   sizeof(cl_mem), (void *)&d_SrcKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 3,   sizeof(cl_mem), (void *)&d_SrcVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 4,  sizeof(cl_uint), (void *)&arrayLength);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 5,  sizeof(cl_uint), (void *)&dir);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    else
    {
        //Launch bitonicSortLocal1
        ciErrNum  = clSetKernelArg(ckBitonicSortLocal1, 0,  sizeof(cl_mem), (void *)&d_DstKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 1,  sizeof(cl_mem), (void *)&d_DstVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 2,  sizeof(cl_mem), (void *)&d_SrcKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 3,  sizeof(cl_mem), (void *)&d_SrcVal);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        {
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                if(stride >= LOCAL_SIZE_LIMIT)
                {
                    //Launch bitonicMergeGlobal
                    ciErrNum  = clSetKernelArg(ckBitonicMergeGlobal, 0,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 1,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 2,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 3,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 4, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 5, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 6, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 7, sizeof(cl_uint), (void *)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    localWorkSize  = LOCAL_SIZE_LIMIT / 4;
                    globalWorkSize = batch * arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicMergeGlobal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                }
                else
                {
                    //Launch bitonicMergeLocal
                    ciErrNum  = clSetKernelArg(ckBitonicMergeLocal, 0,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 1,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 2,  sizeof(cl_mem), (void *)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 3,  sizeof(cl_mem), (void *)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 4, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 5, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 6, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 7, sizeof(cl_uint), (void *)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
                    globalWorkSize = batch * arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicMergeLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                    break;
                }
            }
        }
    }
}
