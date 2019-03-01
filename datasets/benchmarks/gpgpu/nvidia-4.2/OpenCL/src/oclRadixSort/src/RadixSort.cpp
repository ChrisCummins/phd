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
	d_tempKeys = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxElements, NULL, &ciErrNum);
	mCounters = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum);
	mCountersSum = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum);
	mBlockOffsets = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &ciErrNum); 

	size_t szKernelLength; // Byte size of kernel code
    char *cSourcePath = shrFindFilePath("RadixSort.cl", path);
    shrCheckError(cSourcePath != NULL, shrTRUE);
    char *cRadixSort = oclLoadProgSource(cSourcePath, "// My comment\n", &szKernelLength);
    oclCheckError(cRadixSort != NULL, shrTRUE);
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cRadixSort, &szKernelLength, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
#ifdef MAC
    char *flags = "-DMAC -cl-fast-relaxed-math";
#else
    char *flags = "-cl-fast-relaxed-math";
#endif
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, flags, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard ciErrNumor, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "RadixSort.ptx");
        oclCheckError(ciErrNum, CL_SUCCESS); 
    }

	ckRadixSortBlocksKeysOnly = CECL_KERNEL(cpProgram, "radixSortBlocksKeysOnly", &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckFindRadixOffsets        = CECL_KERNEL(cpProgram, "findRadixOffsets",        &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckScanNaive               = CECL_KERNEL(cpProgram, "scanNaive",               &ciErrNum);
	oclCheckError(ciErrNum, CL_SUCCESS);
	ckReorderDataKeysOnly     = CECL_KERNEL(cpProgram, "reorderDataKeysOnly",     &ciErrNum);
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
	ciErrNum  = CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 2, sizeof(unsigned int), (void*)&nbits);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 3, sizeof(unsigned int), (void*)&startbit);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 4, sizeof(unsigned int), (void*)&numElements);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 5, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckRadixSortBlocksKeysOnly, 6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckRadixSortBlocksKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
{
	unsigned int totalBlocks = numElements/2/CTA_SIZE;
	size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
	size_t localWorkSize[1] = {CTA_SIZE};
	cl_int ciErrNum;
	ciErrNum  = CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 0, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 1, sizeof(cl_mem), (void*)&mCounters);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 3, sizeof(unsigned int), (void*)&startbit);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 4, sizeof(unsigned int), (void*)&numElements);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 5, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckFindRadixOffsets, 6, 2 * CTA_SIZE *sizeof(unsigned int), NULL);
	ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckFindRadixOffsets, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
	ciErrNum  = CECL_SET_KERNEL_ARG(ckScanNaive, 0, sizeof(cl_mem), (void*)&mCountersSum);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckScanNaive, 1, sizeof(cl_mem), (void*)&mCounters);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckScanNaive, 2, sizeof(unsigned int), (void*)&nHist);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckScanNaive, 3, 2 * shared_mem_size, NULL);
	ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckScanNaive, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

void RadixSort::reorderDataKeysOnlyOCL(cl_mem d_keys, unsigned int startbit, unsigned int numElements)
{
	unsigned int totalBlocks = numElements/2/CTA_SIZE;
	size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
	size_t localWorkSize[1] = {CTA_SIZE};
	cl_int ciErrNum;
	ciErrNum  = CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 3, sizeof(cl_mem), (void*)&mCountersSum);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 4, sizeof(cl_mem), (void*)&mCounters);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 5, sizeof(unsigned int), (void*)&startbit);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 6, sizeof(unsigned int), (void*)&numElements);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 7, sizeof(unsigned int), (void*)&totalBlocks);
	ciErrNum |= CECL_SET_KERNEL_ARG(ckReorderDataKeysOnly, 8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);
	ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckReorderDataKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}
