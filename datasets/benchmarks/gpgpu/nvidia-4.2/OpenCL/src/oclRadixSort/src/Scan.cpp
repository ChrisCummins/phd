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
#include "Scan.h"

Scan::Scan(cl_context GPUContext,
		   cl_command_queue CommandQue,
		   unsigned int numElements, 
		   const char* path) :
		   cxGPUContext(GPUContext), 
		   cqCommandQueue(CommandQue),
		   mNumElements(numElements)
{
	cl_int ciErrNum;
	if (numElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE) 
	{
		d_Buffer = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, numElements / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * sizeof(cl_uint), NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}
	
	//shrLog("Create and build Scan program\n");
	size_t szKernelLength; // Byte size of kernel code

	char *SourceFile = "Scan_b.cl";
    char *cSourcePath = shrFindFilePath(SourceFile, path);
    shrCheckError(cSourcePath != NULL, shrTRUE);
    char *cScan = oclLoadProgSource(cSourcePath, "// My comment\n", &szKernelLength);
    oclCheckError(cScan != NULL, shrTRUE);
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cScan, &szKernelLength, &ciErrNum);
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "Scan.ptx");
        oclCheckError(ciErrNum, CL_SUCCESS); 
    }

	ckScanExclusiveLocal1 = CECL_KERNEL(cpProgram, "scanExclusiveLocal1", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ckScanExclusiveLocal2 = CECL_KERNEL(cpProgram, "scanExclusiveLocal2", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ckUniformUpdate = CECL_KERNEL(cpProgram, "uniformUpdate", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

	free(cScan);
    free(cSourcePath);
}

Scan::~Scan()
{
	cl_int ciErrNum;

	ciErrNum  = clReleaseKernel(ckScanExclusiveLocal1);
	ciErrNum |= clReleaseKernel(ckScanExclusiveLocal2);
	ciErrNum |= clReleaseKernel(ckUniformUpdate);
	if (mNumElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)
    {
		ciErrNum |= clReleaseMemObject(d_Buffer);
    }
    ciErrNum |= clReleaseProgram(cpProgram);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

// main exclusive scan routine
void Scan::scanExclusiveLarge(
    cl_mem d_Dst,
    cl_mem d_Src,
    unsigned int batchSize,
    unsigned int arrayLength
){
    //Check power-of-two factorization
    unsigned int log2L;
    unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
    oclCheckError(factorizationRemainder == 1, shrTRUE);

    //Check supported size range
    oclCheckError( (arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE), shrTRUE );

    //Check total batch size limit
    oclCheckError( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS, shrTRUE );

    scanExclusiveLocal1(
        d_Dst,
        d_Src,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
        4 * WORKGROUP_SIZE
    );

    scanExclusiveLocal2(
        d_Buffer,
        d_Dst,
        d_Src,
        batchSize,
        arrayLength / (4 * WORKGROUP_SIZE)
    );

    uniformUpdate(
        d_Dst,
        d_Buffer,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
    );
}

void Scan::scanExclusiveLocal1(
    cl_mem d_Dst,
    cl_mem d_Src,
    unsigned int n,
    unsigned int size
){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 0, sizeof(cl_mem), (void *)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 1, sizeof(cl_mem), (void *)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 2, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal1, 3, sizeof(unsigned int), (void *)&size);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = (n * size) / 4;

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckScanExclusiveLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void Scan::scanExclusiveLocal2(
    cl_mem d_Buffer,
    cl_mem d_Dst,
    cl_mem d_Src,
    unsigned int n,
    unsigned int size
){
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    unsigned int elements = n * size;
    ciErrNum  = CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 0, sizeof(cl_mem), (void *)&d_Buffer);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 1, sizeof(cl_mem), (void *)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 2, sizeof(cl_mem), (void *)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 3, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 4, sizeof(unsigned int), (void *)&elements);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckScanExclusiveLocal2, 5, sizeof(unsigned int), (void *)&size);
    oclCheckError(ciErrNum, CL_SUCCESS);

     localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckScanExclusiveLocal2, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void Scan::uniformUpdate(
    cl_mem d_Dst,
    cl_mem d_Buffer,
    unsigned int n
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
}
