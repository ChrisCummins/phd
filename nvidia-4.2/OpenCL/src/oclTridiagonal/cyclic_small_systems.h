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
 
 /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Host code for cyclic reduction (CR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _CYCLIC_SMALL_SYSTEMS_
#define _CYCLIC_SMALL_SYSTEMS_

#include "common.h"

const int cyclicNumKernels = 2;

const char *cyclicKernelNames[cyclicNumKernels] = { 
	"cyclic_small_systems_kernel",			// original version
	"cyclic_branch_free_kernel"				// branch-free modification
};  

cl_kernel crKernel[MAX_GPU_COUNT];

float runCyclicKernel(int devCount, cl_mem *dev_a, cl_mem *dev_b, cl_mem *dev_c, cl_mem *dev_d, cl_mem *dev_x, int system_size, int *workSize)
{
	size_t szGlobalWorkSize[MAX_GPU_COUNT];
    size_t szLocalWorkSize[MAX_GPU_COUNT];
	cl_event GPUExecution[MAX_GPU_COUNT];
	cl_int errcode;

	int iterations = log2(system_size/2);

	for (int i = 0; i < devCount; i++)
	{
		int num_systems = workSize[i];

		// set kernel arguments
		errcode  = clSetKernelArg(crKernel[i], 0, sizeof(cl_mem), (void *) &dev_a[i]);
		errcode |= clSetKernelArg(crKernel[i], 1, sizeof(cl_mem), (void *) &dev_b[i]);
		errcode |= clSetKernelArg(crKernel[i], 2, sizeof(cl_mem), (void *) &dev_c[i]);
		errcode |= clSetKernelArg(crKernel[i], 3, sizeof(cl_mem), (void *) &dev_d[i]);
		errcode |= clSetKernelArg(crKernel[i], 4, sizeof(cl_mem), (void *) &dev_x[i]);
		errcode |= clSetKernelArg(crKernel[i], 5, (system_size+1)*5*sizeof(float), NULL);
		errcode |= clSetKernelArg(crKernel[i], 6, sizeof(int), &system_size);
		errcode |= clSetKernelArg(crKernel[i], 7, sizeof(int), &num_systems);
		errcode |= clSetKernelArg(crKernel[i], 8, sizeof(int), &iterations);
		oclCheckError(errcode, CL_SUCCESS);

		// set execution parameters
		szLocalWorkSize[i] = system_size / 2;
		szGlobalWorkSize[i] = num_systems * szLocalWorkSize[i];
	    
		// warm up
		errcode = clEnqueueNDRangeKernel(cqCommandQue[i], crKernel[i], 1, NULL, &szGlobalWorkSize[i], &szLocalWorkSize[i], 0, NULL, &GPUExecution[i]);
		clFlush(cqCommandQue[i]);
		oclCheckError(errcode, CL_SUCCESS);
	}
	clWaitForEvents(devCount, GPUExecution);

	shrLog("  looping %i times..\n", BENCH_ITERATIONS);	

	// run computations on GPUs in parallel
	double sum_time = 0.0;
	for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
	{
		shrDeltaT(0);
		for (int i = 0; i < devCount; i++)
		{
			errcode = clEnqueueNDRangeKernel(cqCommandQue[i], crKernel[i], 1, NULL, &szGlobalWorkSize[i], &szLocalWorkSize[i], 0, NULL, &GPUExecution[i]);
			clFlush(cqCommandQue[i]);
			oclCheckError(errcode, CL_SUCCESS);
		}
		clWaitForEvents(devCount, GPUExecution);
		sum_time += shrDeltaT(0);
	}
	double time = sum_time / BENCH_ITERATIONS;

#ifdef GPU_PROFILING
	// output detailed timing for each GPU
	for (int i = 0; i < devCount; i++)
    {
		shrLog("  GPU %d time =  %.5f s\n", i, executionTime(GPUExecution[i]));
		clReleaseEvent(GPUExecution[i]);
    }
    shrLog("\n"); 
#endif

	return (float)time;
}

double cyclic_small_systems(int devCount, const char** argv, float *a, float *b, float *c, float *d, float *x, int system_size, int num_systems, int id = 0)
{
    oclCheckError((id < cyclicNumKernels), true);
    shrLog(" %s\n", cyclicKernelNames[id]);

    const unsigned int mem_size = sizeof(float) * num_systems * system_size;

	cl_int errcode;

    cl_mem device_a[MAX_GPU_COUNT];
	cl_mem device_b[MAX_GPU_COUNT];
	cl_mem device_c[MAX_GPU_COUNT];
	cl_mem device_d[MAX_GPU_COUNT];
	cl_mem device_x[MAX_GPU_COUNT];

	int workSize[MAX_GPU_COUNT];		// number of systems assigned to each device
	int workOffset[MAX_GPU_COUNT];		// offset to work set for each device 

	cl_event GPUDone[MAX_GPU_COUNT];

	// create host pointers (TEMPORARY here)
	cl_mem host_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size, a, NULL);
	cl_mem host_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size, b, NULL);
	cl_mem host_c = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size, c, NULL);
	cl_mem host_d = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size, d, NULL);

    // allocate device memory and copy host to device arrays
	workOffset[0] = 0;
	for (int i = 0; i < devCount; i++)
	{
		workSize[i] = num_systems / devCount;	// sign equal work for each GPU
		
		// size and offset in bytes
		int workSizeB = workSize[i] * system_size * sizeof(float);
		int workOffsetB = workOffset[i] * system_size * sizeof(float);

		device_a[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSizeB, NULL, &errcode);
		oclCheckError(errcode, CL_SUCCESS);
		clEnqueueCopyBuffer(cqCommandQue[i], host_a, device_a[i], workOffsetB, 0, workSizeB, 0, NULL, NULL); 

		device_b[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSizeB, NULL, &errcode);
		oclCheckError(errcode, CL_SUCCESS);
		clEnqueueCopyBuffer(cqCommandQue[i], host_b, device_b[i], workOffsetB, 0, workSizeB, 0, NULL, NULL);  

		device_c[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSizeB, NULL, &errcode);
		oclCheckError(errcode, CL_SUCCESS);
		clEnqueueCopyBuffer(cqCommandQue[i], host_c, device_c[i], workOffsetB, 0, workSizeB, 0, NULL, NULL);  

		device_d[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSizeB, NULL, &errcode);
		oclCheckError(errcode, CL_SUCCESS);
		clEnqueueCopyBuffer(cqCommandQue[i], host_d, device_d[i], workOffsetB, 0, workSizeB, 0, NULL, NULL);  

		device_x[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, workSizeB, NULL, &errcode);
		oclCheckError(errcode, CL_SUCCESS);

		if (i < devCount-1) workOffset[i+1] = workOffset[i] + workSize[i];
	}

	// free host pointers
	clReleaseMemObject(host_a);
	clReleaseMemObject(host_b);
	clReleaseMemObject(host_c);
	clReleaseMemObject(host_d);

    // load program from file
	size_t program_length;
	char *source = oclLoadProgSource(shrFindFilePath("cyclic_kernels.cl", argv[0]), "", &program_length);
    oclCheckError(source != NULL, shrTRUE);

    // create program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, &program_length, &errcode);
    oclCheckError(errcode, CL_SUCCESS);

	// build program
    errcode = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if (errcode != CL_SUCCESS)
    {
        // write out standard error, output build log
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "cyclic_kernels.ptx");
        shrLog("\nFAILED\n\n");
        oclCheckError(errcode, CL_SUCCESS);
    }

	// output PTX
	//oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext));

	// create kernels
	for (int i = 0; i < devCount; i++)
	{
		crKernel[i] = clCreateKernel(cpProgram, cyclicKernelNames[id], &errcode);
        oclCheckError(errcode, CL_SUCCESS);
	}

	double time = runCyclicKernel(devCount, device_a, device_b, device_c, device_d, device_x, system_size, workSize);
	
	// copy result from device to host
	for (int i = 0; i < devCount; i++) 
	{
		int workSizeB = workSize[i] * system_size * sizeof(float);
		errcode = clEnqueueReadBuffer(cqCommandQue[i], device_x[i], CL_FALSE, 0, workSizeB, x + workOffset[i] * system_size, 0, NULL, &GPUDone[i]);
		oclCheckError(errcode, CL_SUCCESS);
	}
	clWaitForEvents(devCount, GPUDone);

    // cleanup memory
	for (int i = 0; i < devCount; i++)
	{
		clReleaseMemObject(device_a[i]);
		clReleaseMemObject(device_b[i]);
		clReleaseMemObject(device_c[i]);
		clReleaseMemObject(device_d[i]);
		clReleaseMemObject(device_x[i]);
		clReleaseEvent(GPUDone[i]);
	}
    
    return time;
}

#endif
