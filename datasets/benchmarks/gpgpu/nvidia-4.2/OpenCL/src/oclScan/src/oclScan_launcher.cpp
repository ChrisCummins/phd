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

#include "oclScan_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL scan kernel launchers
////////////////////////////////////////////////////////////////////////////////
//OpenCL scan program handle
static cl_program
    cpProgram;

//OpenCL scan kernel handles
static cl_kernel
    ckScanExclusiveLocal1, ckScanExclusiveLocal2, ckUniformUpdate;

static cl_mem
    d_Buffer;

//All three kernels run 512 threads per workgroup
//Must be a power of two
static const uint  WORKGROUP_SIZE = 256;
static const char *compileOptions = "-D WORKGROUP_SIZE=256";

extern "C" void initScan(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog(" ...loading Scan.cl\n");
        char *cScan = oclLoadProgSource(shrFindFilePath("Scan.cl", argv[0]), "// My comment\n", &kernelLength);
        oclCheckError(cScan != NULL, shrTRUE);

    shrLog(" ...creating scan program\n");
        cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cScan, &kernelLength, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog(" ...building scan program\n");
        ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, compileOptions, NULL, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			// write out standard error, Build Log and PTX, then cleanup and exit
			shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
			oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
			oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclScan.ptx");
			oclCheckError(ciErrNum, CL_SUCCESS); 
		}

    shrLog(" ...creating scan kernels\n");
        ckScanExclusiveLocal1 = CECL_KERNEL(cpProgram, "scanExclusiveLocal1", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckScanExclusiveLocal2 = CECL_KERNEL(cpProgram, "scanExclusiveLocal2", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckUniformUpdate = CECL_KERNEL(cpProgram, "uniformUpdate", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog( " ...checking minimum supported workgroup size\n");
        //Check for work group size
        cl_device_id device;
        size_t szScanExclusiveLocal1, szScanExclusiveLocal2, szUniformUpdate;

        ciErrNum  = clGetCommandQueueInfo(cqParamCommandQue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
        ciErrNum |= CECL_GET_KERNEL_WORK_GROUP_INFO(ckScanExclusiveLocal1,  device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szScanExclusiveLocal1, NULL);
        ciErrNum |= CECL_GET_KERNEL_WORK_GROUP_INFO(ckScanExclusiveLocal2, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szScanExclusiveLocal2, NULL);
        ciErrNum |= CECL_GET_KERNEL_WORK_GROUP_INFO(ckUniformUpdate, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szUniformUpdate, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        if( (szScanExclusiveLocal1 < WORKGROUP_SIZE) || (szScanExclusiveLocal2 < WORKGROUP_SIZE) || (szUniformUpdate < WORKGROUP_SIZE) ){
            shrLog("\nERROR !!! Minimum work-group size %u required by this application is not supported on this device.\n\n", WORKGROUP_SIZE);
            closeScan();
            free(cScan);
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(EXIT_FAILURE);
        }

    shrLog(" ...allocating internal buffers\n");
        d_Buffer = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, (MAX_BATCH_ELEMENTS / (4 * WORKGROUP_SIZE)) * sizeof(uint), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    //Discard temp storage
    free(cScan);
}

extern "C" void closeScan(void){
    cl_int ciErrNum;
    ciErrNum  = clReleaseMemObject(d_Buffer);
    ciErrNum |= clReleaseKernel(ckUniformUpdate);
    ciErrNum |= clReleaseKernel(ckScanExclusiveLocal2);
    ciErrNum |= clReleaseKernel(ckScanExclusiveLocal1);
    ciErrNum |= clReleaseProgram(cpProgram);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

static uint iSnapUp(uint dividend, uint divisor){
    return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
}

static uint factorRadix2(uint& log2L, uint L){
    if(!L){
        log2L = 0;
        return 0;
    }else{
        for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Short scan launcher
////////////////////////////////////////////////////////////////////////////////
static size_t scanExclusiveLocal1(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint n,
    uint size
){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 0, sizeof(cl_mem), (void *)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 1, sizeof(cl_mem), (void *)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 2, 2 * WORKGROUP_SIZE * sizeof(uint), NULL);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 3, sizeof(uint), (void *)&size);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = (n * size) / 4;

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckScanExclusiveLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    return localWorkSize;
}

extern "C" size_t scanExclusiveShort(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint batchSize,
    uint arrayLength
){
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    oclCheckError( factorizationRemainder == 1, shrTRUE);

    //Check supported size range
    oclCheckError( (arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE), shrTRUE );

    //Check total batch size limit
    oclCheckError( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS, shrTRUE );

    //Check all work-groups to be fully packed with data
    oclCheckError( (batchSize * arrayLength) % (4 * WORKGROUP_SIZE) == 0, shrTRUE);

    return scanExclusiveLocal1(
        cqCommandQueue,
        d_Dst,
        d_Src,
        batchSize,
        arrayLength
    );
}

////////////////////////////////////////////////////////////////////////////////
// Large scan launcher
////////////////////////////////////////////////////////////////////////////////
static void scanExclusiveLocal2(
    cl_command_queue cqCommandQueue,
    cl_mem d_Buffer,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint n,
    uint size
){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    uint elements = n * size;
    ciErrNum  = CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 0, sizeof(cl_mem), (void *)&d_Buffer);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 1, sizeof(cl_mem), (void *)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 2, sizeof(cl_mem), (void *)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 3, 2 * WORKGROUP_SIZE * sizeof(uint), NULL);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 4, sizeof(uint), (void *)&elements);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 5, sizeof(uint), (void *)&size);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckScanExclusiveLocal2, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

static size_t uniformUpdate(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Buffer,
    uint n
){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckUniformUpdate, 0, sizeof(cl_mem), (void *)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckUniformUpdate, 1, sizeof(cl_mem), (void *)&d_Buffer);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = n * WORKGROUP_SIZE;

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckUniformUpdate, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    return localWorkSize;
}

extern "C" size_t scanExclusiveLarge(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    uint batchSize,
    uint arrayLength
){
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    oclCheckError( factorizationRemainder == 1, shrTRUE);

    //Check supported size range
    oclCheckError( (arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE), shrTRUE );

    //Check total batch size limit
    oclCheckError( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS, shrTRUE );

    scanExclusiveLocal1(
        cqCommandQueue,
        d_Dst,
        d_Src,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
        4 * WORKGROUP_SIZE
    );

    scanExclusiveLocal2(
        cqCommandQueue,
        d_Buffer,
        d_Dst,
        d_Src,
        batchSize,
        arrayLength / (4 * WORKGROUP_SIZE)
    );

    return uniformUpdate(
        cqCommandQueue,
        d_Dst,
        d_Buffer,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
    );
}
