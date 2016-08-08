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
#include "RadixSort.h"

extern double time1, time2, time3, time4;

RadixSort::RadixSort(cl_context GPUContext,
					 cl_command_queue CommandQue,
					 unsigned int maxElements, 
					 const char* path, 
					 const int ctaSize,
					 bool keysOnly = true) :
					 mNumElements(0),
					 mTempValues(0),
					 mCounters(0),
					 mCountersSum(0),
					 mBlockOffsets(0),
					 cxGPUContext(GPUContext),
					 cqCommandQueue(CommandQue),
					 CTA_SIZE(ctaSize),
					 scan(GPUContext, CommandQue, maxElements/2/CTA_SIZE*16, path)
{

	unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ? 
            (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);
    unsigned int numBlocks2 = ((maxElements % (CTA_SIZE * 2)) == 0) ?
            (maxElements / (CTA_SIZE * 2)) : (maxElements / (CTA_SIZE * 2) + 1);

	cl_int ciErrNum;
	d_tempKeys = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxElements, NULL, &ciErrNum);
	mCounters = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum);
	mCountersSum = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum);
	mBlockOffsets = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum); 

	size_t szKernelLength; // Byte size of kernel code
    char *cSourcePath = shrFindFilePath("RadixSort.cl", path);
    shrCheckError(cSourcePath != NULL, shrTRUE);
    char *cRadixSort = oclLoadProgSource(cSourcePath, "// My comment\n", &szKernelLength);
    oclCheckError(cRadixSort != NULL, shrTRUE);
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cRadixSort, &szKernelLength, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
#ifdef MAC
    char *flags = "-DMAC -cl-fast-relaxed-math";
#else
    char *flags = "-cl-fast-relaxed-math";
#endif
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, flags, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard ciErrNumor, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "RadixSort.ptx");
        oclCheckError(ciErrNum, CL_SUCCESS); 
    }

	ckRadixSortBlocksKeysOnly = clCreateKernel(cpProgram, "radixSortBlocksKeysOnly", &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckFindRadixOffsets        = clCreateKernel(cpProgram, "findRadixOffsets",        &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckScanNaive               = clCreateKernel(cpProgram, "scanNaive",               &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckReorderDataKeysOnly     = clCreateKernel(cpProgram, "reorderDataKeysOnly",     &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	free(cRadixSort);
    free(cSourcePath);
}

RadixSort::~RadixSort()
{
	clReleaseKernel(ckRadixSortBlocksKeysOnly);
	clReleaseKernel(ckFindRadixOffsets);
	clReleaseKernel(ckScanNaive);
	clReleaseKernel(ckReorderDataKeysOnly);
	clReleaseProgram(cpProgram);
	clReleaseMemObject(d_tempKeys);
	clReleaseMemObject(mCounters);
	clReleaseMemObject(mCountersSum);
	clReleaseMemObject(mBlockOffsets);
}

//------------------------------------------------------------------------
// Sorts input arrays of unsigned integer keys and (optional) values
// 
// @param d_keys      Array of keys for data to be sorted
// @param values      Array of values to be sorted
// @param numElements Number of elements to be sorted.  Must be <= 
//                    maxElements passed to the constructor
// @param keyBits     The number of bits in each key to use for ordering
//------------------------------------------------------------------------
void RadixSort::sort(cl_mem d_keys, 
		  unsigned int *values, 
		  unsigned int  numElements,
		  unsigned int  keyBits)
{
	if (values == 0) 
    {
		radixSortKeysOnly(d_keys, numElements, keyBits);
    }
}

//----------------------------------------------------------------------------
// Main key-only radix sort function.  Sorts in place in the keys and values 
// arrays, but uses the other device arrays as temporary storage.  All pointer 
// parameters are device pointers.  Uses cudppScan() for the prefix sum of
// radix counters.
//----------------------------------------------------------------------------
void RadixSort::radixSortKeysOnly(cl_mem d_keys, unsigned int numElements, unsigned int keyBits)
{
	int i = 0;
    while (keyBits > i*bitStep) 
	{
		radixSortStepKeysOnly(d_keys, bitStep, i*bitStep, numElements);
		i++;
	}
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by nbits key bits per step, 
// starting at startbit.
//----------------------------------------------------------------------------
void RadixSort::radixSortStepKeysOnly(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
	// Four step algorithms from Satish, Harris & Garland
	radixSortBlocksKeysOnlyOCL(d_keys, nbits, startbit, numElements);

	findRadixOffsetsOCL(startbit, numElements);

	scan.scanExclusiveLarge(mCountersSum, mCounters, 1, numElements/2/CTA_SIZE*16);

	reorderDataKeysOnlyOCL(d_keys, startbit, numElements);
}

//----------------------------------------------------------------------------
// Wrapper for the kernels of the four steps
//----------------------------------------------------------------------------
void RadixSort::radixSortBlocksKeysOnlyOCL(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
	unsigned int totalBlocks = numElements/4/CTA_SIZE;
	size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
	size_t localWorkSize[1] = {CTA_SIZE};
	cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(ckRadixSortBlocksKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 2, sizeof(unsigned int), (void*)&nbits);
	ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 3, sizeof(unsigned int), (void*)&startbit);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 4, sizeof(unsigned int), (void*)&numElements);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 5, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckRadixSortBlocksKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
{
	unsigned int totalBlocks = numElements/2/CTA_SIZE;
	size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
	size_t localWorkSize[1] = {CTA_SIZE};
	cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(ckFindRadixOffsets, 0, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 1, sizeof(cl_mem), (void*)&mCounters);
    ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
	ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 3, sizeof(unsigned int), (void*)&startbit);
	ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 4, sizeof(unsigned int), (void*)&numElements);
	ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 5, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= clSetKernelArg(ckFindRadixOffsets, 6, 2 * CTA_SIZE *sizeof(unsigned int), NULL);
	ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckFindRadixOffsets, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

#define NUM_BANKS 16
void RadixSort::scanNaiveOCL(unsigned int numElements)
{
	unsigned int nHist = numElements/2/CTA_SIZE*16;
	size_t globalWorkSize[1] = {nHist};
	size_t localWorkSize[1] = {nHist};
	unsigned int extra_space = nHist / NUM_BANKS;
	unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);
	cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(ckScanNaive, 0, sizeof(cl_mem), (void*)&mCountersSum);
	ciErrNum |= clSetKernelArg(ckScanNaive, 1, sizeof(cl_mem), (void*)&mCounters);
	ciErrNum |= clSetKernelArg(ckScanNaive, 2, sizeof(unsigned int), (void*)&nHist);
	ciErrNum |= clSetKernelArg(ckScanNaive, 3, 2 * shared_mem_size, NULL);
	ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckScanNaive, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

void RadixSort::reorderDataKeysOnlyOCL(cl_mem d_keys, unsigned int startbit, unsigned int numElements)
{
	unsigned int totalBlocks = numElements/2/CTA_SIZE;
	size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
	size_t localWorkSize[1] = {CTA_SIZE};
	cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(ckReorderDataKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 3, sizeof(cl_mem), (void*)&mCountersSum);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 4, sizeof(cl_mem), (void*)&mCounters);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 5, sizeof(unsigned int), (void*)&startbit);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 6, sizeof(unsigned int), (void*)&numElements);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 7, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= clSetKernelArg(ckReorderDataKeysOnly, 8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);
	ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckReorderDataKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}
