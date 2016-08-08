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
 * OpenCL Tridiagonal solvers.
 * Main host code.
 *
 * This sample implements several methods to solve a bunch of small tridiagonal matrices:
 *	PCR		- parallel cyclic reduction O(N log N)
 *  CR		- original cyclic reduction O(N)
 *	Sweep	- serial one-thread-per-system gauss elimination O(N)
 *
 * Original testrig code: UC Davis, Yao Zhang & John Owens
 * Reference paper for the cyclic reduction methods on the GPU:  
 *   Yao Zhang, Jonathan Cohen, and John D. Owens. Fast Tridiagonal Solvers on the GPU. 
 *   In Proceedings of the 15th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP 2010), January 2010.
 * 
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#define OUTPUT_RESULTS
#define MAX_GPU_COUNT		8
#define BENCH_ITERATIONS	5
//#define GPU_PROFILING

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <oclUtils.h>
#include <shrQATest.h>

#include "file_read_write.h"
#include "test_gen_result_check.h"
#include "cpu_solvers.h"

// global OpenCL variables
cl_context       cxGPUContext;
cl_command_queue cqCommandQue[MAX_GPU_COUNT];
cl_device_id     cdDevices[MAX_GPU_COUNT];
cl_uint          selectedDevNums[MAX_GPU_COUNT];

bool             useLmem = false;
bool             useVec4 = false;
int              SWEEP_BLOCK_SIZE = 256;

// available solvers
#include "pcr_small_systems.h"
#include "cyclic_small_systems.h"
#include "sweep_small_systems.h"

////////////////////////////////////////////////////////////////////////////////
// Solve <num_systems> of <system_size> using <devCount> devices
////////////////////////////////////////////////////////////////////////////////
int run(const char** argv, int system_size, int num_systems, int devCount) 
{
	double time_spent_gpu[3];
	double time_spent_cpu[1];
    cl_int errcode;

	// create command-queues
	for (int i = 0; i < devCount; i++)
	{       
        shrLog("Device %d: ", selectedDevNums[i]);
        oclPrintDevName(LOGBOTH, cdDevices[i]);
		shrLog("\n");

		cqCommandQue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
		oclCheckError(errcode, CL_SUCCESS);
	}
	shrLog("\n");

	const unsigned int mem_size = sizeof(float) * num_systems * system_size;

	// allocate host arrays
    float *a = (float*)malloc(mem_size);
    float *b = (float*)malloc(mem_size);
    float *c = (float*)malloc(mem_size);
    float *d = (float*)malloc(mem_size);
    float *x1 = (float*)malloc(mem_size);
    float *x2 = (float*)malloc(mem_size);

	// fill host arrays with data
    for (int i = 0; i < num_systems; i++)
        test_gen_cyclic(&a[i * system_size], &b[i * system_size], &c[i * system_size], &d[i * system_size], &x1[i * system_size], system_size, 0);

    shrLog("  Num_systems = %d, system_size = %d\n", num_systems, system_size);

	// run CPU serial solver
	time_spent_cpu[0] = serial_small_systems(a, b, c, d, x2, system_size, num_systems);

    // Log info
	shrLog("\n----- CPU  solvers -----\n");
	shrLog("  CPU Time =    %.5f s\n", time_spent_cpu[0]);
    shrLog("  Throughput =  %.4f systems/sec\n", (float)num_systems /(time_spent_cpu[0]*1000.0));

	// run GPU solvers
	shrLog("\n----- optimized GPU solvers -----\n\n");
	
	// pcr
    time_spent_gpu[0] = pcr_small_systems(devCount, argv, a, b, c, d, x1, system_size, num_systems, 1);
    shrLogEx(LOGBOTH | MASTER, 0, "oclTridiagonal-pcrsmall, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems, NumDevsUsed = %u\n", 
          (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems, devCount);
    compare_small_systems(x1, x2, system_size, num_systems);

	// cr
    time_spent_gpu[1] = cyclic_small_systems(devCount, argv, a, b, c, d, x1, system_size, num_systems, 1);
    shrLogEx(LOGBOTH | MASTER, 0, "oclTridiagonal-cyclicsmall, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems, NumDevsUsed = %u\n", 
          (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems, devCount);
    compare_small_systems(x1, x2, system_size, num_systems);

	// sweep
    time_spent_gpu[2] = sweep_small_systems(devCount, argv, a, b, c, d, x1, system_size, num_systems, true);
    shrLogEx(LOGBOTH | MASTER, 0, "oclTridiagonal-sweepsmall, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems, NumDevsUsed = %u\n", 
          (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems, devCount);
    compare_small_systems(x1, x2, system_size, num_systems);

#ifdef OUTPUT_RESULTS
	file_write_small_systems(x1, 10, system_size, "oclTriDiagonal_GPU.dat");
	file_write_small_systems(x2, 10, system_size, "oclTriDiagonal_CPU.dat");
	write_timing_results_1d(time_spent_gpu, 1, "oclTriDiagonal_Time_GPU.dat");
	write_timing_results_1d(time_spent_cpu, 1, "oclTriDiagonal_Time_CPU.dat");
#endif 

	// cleanup OpenCL
	for (int i = 0; i < devCount; i++)
		clReleaseCommandQueue(cqCommandQue[i]);

	// free host arrays
    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv) 
{   
    shrQAStart(argc, (char **)argv);

    // set logfile name and start logs
    shrSetLogFileName ("oclTridiagonal.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

	int num_systems = 128 * 128;
	int system_size = 128;
	
	cl_platform_id cpPlatform;
    cl_uint allDevCount = 0, devCount = 0;		
    cl_device_id* cdAllDevices;
	cl_int errcode;

	// get the NVIDIA platform
    errcode = oclGetPlatformID(&cpPlatform);
    oclCheckError(errcode, CL_SUCCESS);

    // get all the devices
    errcode = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &allDevCount);
    oclCheckError(errcode, CL_SUCCESS);
    cdAllDevices = (cl_device_id *)malloc(allDevCount * sizeof(cl_device_id) );
    errcode = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, allDevCount, cdAllDevices, NULL);
    oclCheckError(errcode, CL_SUCCESS);
	
	// create the context
    cxGPUContext = clCreateContext(0, allDevCount, cdAllDevices, NULL, NULL, &errcode);
    oclCheckError(errcode, CL_SUCCESS);

	if(shrCheckCmdLineFlag(argc, (const char**)argv, "num_systems"))
    {
		char* ctaList;
        char* ctaStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "num_systems", &ctaList);

        #ifdef WIN32
            ctaStr = strtok_s (ctaList," ,.-", &next_token);
        #else
            ctaStr = strtok (ctaList," ,.-");
        #endif

		num_systems = atoi(ctaStr);
	}

	if(shrCheckCmdLineFlag(argc, (const char**)argv, "system_size"))
	{
		char* ctaList;
        char* ctaStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "system_size", &ctaList);

        #ifdef WIN32
            ctaStr = strtok_s (ctaList," ,.-", &next_token);
        #else
            ctaStr = strtok (ctaList," ,.-");
        #endif

		system_size = atoi(ctaStr);
	}

	if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        // user specified GPUs
        char* deviceList;
        char* deviceStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

        #ifdef WIN32
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif   
        
		while(deviceStr != NULL) 
        {
            // get the device
            cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
			selectedDevNums[devCount] = atoi(deviceStr);
            cdDevices[devCount] = device;
			devCount++;

			#ifdef WIN32
				deviceStr = strtok_s (NULL," ,.-", &next_token);
			#else            
				deviceStr = strtok (NULL," ,.-");
	        #endif
		}
	}
	else
	{
		// use all available devices
		devCount = allDevCount;
		for (cl_uint i = 0; i < devCount; i++)
		{
			cdDevices[i] = cdAllDevices[i];
			selectedDevNums[i] = i;
		}
	}

	// check lmem flag
	if (shrCheckCmdLineFlag(argc, (const char**)argv, "lmem"))
		useLmem = true;

	// check vectorization flag
	if (shrCheckCmdLineFlag(argc, (const char**)argv, "vec4"))
		useVec4 = true;

	// CTA size for the sweep
	if (shrCheckCmdLineFlag(argc, (const char**)argv, "sweep-cta"))
	{
        char* ctaList;
        char* ctaStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "sweep-cta", &ctaList);

        #ifdef WIN32
            ctaStr = strtok_s (ctaList," ,.-", &next_token);
        #else
            ctaStr = strtok (ctaList," ,.-");
        #endif

		SWEEP_BLOCK_SIZE = atoi(ctaStr);
	}
	if (useVec4) shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE / 4);
		else shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE);
	
	// run the main test
    int result = run(argv, system_size, num_systems, devCount);

	// free OCL context & devices
	clReleaseContext(cxGPUContext);
	free(cdAllDevices);

    // pass or fail (cumulative... all tests in the loop)
    shrQAFinishExit(argc, (const char **)argv, (result == 0) ? QA_PASSED : QA_FAILED);
}
