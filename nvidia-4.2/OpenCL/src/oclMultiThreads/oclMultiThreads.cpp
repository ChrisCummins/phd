/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* 
 * This sample shows the implementation of multi-threaded heterogeneous computing workloads with tight cooperation between CPU and GPU.
 * With OpenCL 1.1 the API introduces three new concepts that are utilized:
 * 1) User Events
 * 2) Thread-Safe API calls
 * 3) Event Callbacks
 *
 * The workloads in the sample follow the form CPU preprocess -> GPU process -> CPU postprocess.
 * Each CPU processing step is handled by its own dedicated thread. GPU workloads are sent to all available GPUs in the system.
 *
 * A user event is used to stall enqueued GPU work until the CPU has finished the preprocessing. Preprocessing is
 * handled by a dedicated CPU thread and relies on thread-safe API calls to signal the GPU that the main processing 
 * can start. The new event callback mechanism of OpenCL is used to launch a new CPU thread on event completion of
 * downloading data from GPU.
 */

#include <stdio.h>
#include <shrQATest.h>
#include <oclUtils.h>
#include <vector>

#ifdef _WIN32
	#include <windows.h>
	const char *getOSName(OSVERSIONINFO *pOSVI)
	{
		pOSVI->dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
		GetVersionEx (pOSVI);

		if ((pOSVI->dwMajorVersion ==6)&&(pOSVI->dwMinorVersion==1))
			return (const char *)"Windows 7";
		else if ((pOSVI->dwMajorVersion ==6)&&(pOSVI->dwMinorVersion==0))
			return (const char *)"Windows Vista";
		else if ((pOSVI->dwMajorVersion ==5)&&(pOSVI->dwMinorVersion==1))
			return (const char *)"Windows XP";
		else if ((pOSVI->dwMajorVersion ==5)&&(pOSVI->dwMinorVersion==0))
			return (const char *)"Windows 2000";
		else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==0))
			return (const char *)"Windows NT 4.0";
		else if ((pOSVI->dwMajorVersion ==3)&&(pOSVI->dwMinorVersion==51))
			return (const char *)"Windows NT 3.51";
		else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==90))
			return (const char *)"Windows ME";
		else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==10))
			return (const char *)"Windows 98";
		else if((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==0))
			return (const char *)"Windows 95";
		else 
			return (const char *)"Windows OS Unknown";
	}
#else
	const char *getOSName()
	{
		return (const char *)"UNIX Operating System";
	}
#endif

#ifdef CL_VERSION_1_1 

#include "multithreading.h"

const int N = 8;
const int buffer_size    = 1 << 23;
const int BLOCK_SIZE     = 16;
const int MAX_GPU_COUNT  = 16;

// Basic Matrix dimensions (can be amplified by command line switch)
// (chosen as multiples of the thread block size for simplicity)
#define WA (5 * BLOCK_SIZE) // Matrix A width
#define HA (10 * BLOCK_SIZE) // Matrix A height
#define WB (5 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

// Globals for size of matrices
unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
int iSizeMultiple = 1;

#ifdef WIN32
#define CALLBACK_FUNC void __stdcall 
#else
#define CALLBACK_FUNC void
#endif

CUTBarrier barrier;

struct cpu_worker_arg_t{
	int*   data_n;
    float* data_fp;
	cl_event user_event;
	int id;
    bool bEnableProfile;
};

bool bOK = true;

std::vector<cl_mem> vDeferredReleaseMem;
std::vector<cl_command_queue> vDeferredReleaseQueue;

// First part of the heterogeneous workload, prcoessing done by CPU
CUT_THREADPROC cpu_preprocess(void* void_arg) 
{
	cpu_worker_arg_t* arg = (cpu_worker_arg_t*) void_arg;

	for( int i=0; i < buffer_size / sizeof(int); ++i ) {		
		arg->data_fp[i] = (float)arg->id;
	}

	// Signal GPU that CPU is done preprocessing via OpenCL user event
	clSetUserEventStatus(arg->user_event, CL_COMPLETE);
	CUT_THREADEND;
}

// last part of the heterogeneous workload, processing done by CPU
CUT_THREADPROC cpu_postprocess(void* void_arg) 
{
    cpu_worker_arg_t* arg = (cpu_worker_arg_t*) void_arg;

	for( int i=0; i < buffer_size / sizeof(int) && bOK; ++i ) {		
		if(arg->data_fp[i] != (float)arg->id + 1.0f) {
			bOK = false;
			shrLog("Results don't match in workload %d!\n", arg->id);
		}
	}

	// Cleanup
	free( arg->data_fp );
	free( void_arg );
	
	// Signal that this job has finished
	cutIncrementBarrier(&barrier);
	CUT_THREADEND;
}

CALLBACK_FUNC event_callback(cl_event event, cl_int event_command_exec_status, void* user_data) 
{
	if( event_command_exec_status != CL_COMPLETE ) {
		shrLog("clEnqueueWriteBuffer() Error: Failed to write buffer!\n");
		cutIncrementBarrier(&barrier);
		return;		
	}
		
    // Profile the OpenCL kernel event information
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time   = (cl_ulong)0;
    size_t return_bytes;

    if ( ((cpu_worker_arg_t *)user_data)->bEnableProfile == true ) 
    {
        int err;
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                sizeof(cl_ulong), &ev_start_time, &return_bytes);

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &ev_end_time, &return_bytes);

        double run_time = (double)(ev_end_time - ev_start_time);
        printf("\t> event_callback() event_id=%d, kernel runtime %f (ms)\n", ((cpu_worker_arg_t *)user_data)->id, run_time*1.0e-6);
    }

    cutStartThread(&cpu_postprocess, user_data);
}

// returns the time it took to run the test
void launch_hybrid_workload(cl_context context, cl_device_id device, 
							char *device_name, cl_kernel kernel, 
							int id, bool bEnableProfile) 
{
	cl_int ciErrNum;

    // We'll used the OpenCL profiling of events to time runs

    shrLog("%s: simpleIncrement(), cl_device_id: %d, event_id: %d\n", device_name, device, id);

    // Setup GPU command queue with device profiling enabled
    cl_command_queue queue = clCreateCommandQueue(context, device, (bEnableProfile ? CL_QUEUE_PROFILING_ENABLE : 0), &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
	    shrLog("clCreateCommandQueue() Error %d: Failed to create OpenCL command queue!\n", ciErrNum);
	    cutIncrementBarrier(&barrier);
		return;
    }

    cpu_worker_arg_t* arg = (cpu_worker_arg_t*) malloc(sizeof(cpu_worker_arg_t)); 
	arg->id = id;
	arg->data_fp = (float*) malloc(buffer_size);
    arg->bEnableProfile = bEnableProfile;
	cl_mem buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, buffer_size, 0,0);

	// Create OpenCL user event and make the first command dependent on its completion.
	// This means that none of the commands will start until user event has been signaled.
	arg->user_event = clCreateUserEvent(context, &ciErrNum);	
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clCreateUserEvent() Error %d: Failed to create user event!\n", ciErrNum);
		cutIncrementBarrier(&barrier);
		return;
	}

	// Launch CPU thread to start the heterogneous workload
	cutStartThread(&cpu_preprocess, (void*) arg);

	// Upload data to GPU.
	ciErrNum = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, buffer_size, arg->data_fp, 1, &arg->user_event, 0);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clEnqueueWriteBuffer() Error %d: Failed to write buffer!\n", ciErrNum);
		cutIncrementBarrier(&barrier);
		return;
	}

	// Do computations on the GPU.
	size_t offset[] = {0};
	size_t globalSize[] = {buffer_size/sizeof(int)};
	size_t localSize[] = {256};

	ciErrNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
	ciErrNum = clEnqueueNDRangeKernel(queue, kernel, 1, offset, globalSize, localSize,0,0,0);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clEnqueueNDRangeKernel() Error %d: Failed to launch kernel!\n", ciErrNum);
		cutIncrementBarrier(&barrier);
		return;
	}

	// Download result from GPU.
	cl_event gpudone_event;
	ciErrNum = clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, buffer_size, arg->data_fp, 0, 0, &gpudone_event);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clEnqueueWriteBuffer() Error %d: Failed to write buffer!\n", ciErrNum);
		cutIncrementBarrier(&barrier);
		return;
	}
	// Set callback that will launch another CPU thread to finish the work when the download from GPU has finished
	ciErrNum = clSetEventCallback(gpudone_event, CL_COMPLETE, &event_callback, arg);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clSetEventCallback() Error %d: Failed to set callback!\n", ciErrNum);
		cutIncrementBarrier(&barrier);
		return;
	}

	// Cleanup
	// NOTE: OpenCL requires that all calls to clRelease* API calls (except clReleaseEvent) are deferred until clSetUserEventStatus is called (see OpenCL Spec 1.1 rev 44 p.143)
	clReleaseEvent(arg->user_event);
	clReleaseEvent(gpudone_event);

	vDeferredReleaseQueue.push_back(queue);
	vDeferredReleaseMem.push_back(buffer);
}

cl_int launch_hybrid_matrixMultiply(cl_context context, 
                                    cl_device_id device, char *device_name, 
                                    cl_kernel kernel, int id, bool bEnableProfile) 
{
	cl_int ciErrNum;

    cl_mem d_A, d_B, d_C;

    iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);
    uiWA = WA * iSizeMultiple;
    uiHA = HA * iSizeMultiple;
    uiWB = WB * iSizeMultiple;
    uiHB = HB * iSizeMultiple;
    uiWC = WC * iSizeMultiple;
    uiHC = HC * iSizeMultiple;
    shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A_data = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B_data = (float*)malloc(mem_size_B);

    // initialize host memory
    srand(2006);
    shrFillArray(h_A_data, size_A);
    shrFillArray(h_B_data, size_B);

    // allocate host memory for result
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // create OpenCL buffer pointing to the host memory
    cl_mem h_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				                mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: clCreateBuffer\n");
        return ciErrNum;
    }

    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = uiHA;
    int workOffset[MAX_GPU_COUNT], workSize[MAX_GPU_COUNT];

    workSize[0]   = sizePerGPU;
    workOffset[0] = 0;

    cl_command_queue commandQueue = 0;

    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, workSize[0] * sizeof(float) * uiWA, NULL,NULL);

    // create OpenCL buffer on device that will be initiatlize from the host memory on first use
    // on device
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         mem_size_B, h_B_data, NULL);

    // Output buffer
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  workSize[0] * uiWC * sizeof(float), NULL,NULL);

    // setup the argument values
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_C);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_A);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_B);
    clSetKernelArg(kernel, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, 0 );
    clSetKernelArg(kernel, 4, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, 0 );
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *) &uiWA);
    clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &uiWB);

    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, uiWC), shrRoundUp(BLOCK_SIZE, workSize[0])};

    shrLog("launching - matrixMultiply() on (%s), cl_device_id: %d\n", device_name, device);

    cpu_worker_arg_t* arg = (cpu_worker_arg_t*) malloc(sizeof(cpu_worker_arg_t)); 

	arg->id = id;
	arg->data_fp = (float*)h_A_data;
    arg->bEnableProfile = bEnableProfile;

	// Create OpenCL user event and make the first command dependent on its completion.
	// This means that none of the commands will start until user event has been signaled.
	arg->user_event = clCreateUserEvent(context, &ciErrNum);	

	// Launch CPU thread to start the heterogneous workload
	cutStartThread(&cpu_preprocess, (void*) arg);

    // Copy only assigned rows (h_A) from CPU host to GPU device    
    ciErrNum = clEnqueueCopyBuffer(commandQueue, 
                                   h_A, // src_buffer
                                   d_A, // dst_buffer
                                  (workOffset[0] * sizeof(float) * uiWA), // src_offset
                                   0, // dst_offset
                                  (workSize[0]   * sizeof(float) * uiWA),  // size_of_bytes to copy
                                   1,  // number_events_in_waitlist
                                   &arg->user_event, /// event_wait_list
                                   0); // event
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clEnqueueCopyBuffer() Error: Failed to copy buffer!\n");
		return -1;
	}

    // Launch Multiplication - non-blocking execution:  launch and push to device(s)
	cl_event GPUExecution;
	globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[0]);
	clEnqueueNDRangeKernel(commandQueue, kernel, 2, 0, globalWorkSize, localWorkSize,
		                   0, NULL, &GPUExecution);
    clFlush(commandQueue);

	// Download result from GPU.
	cl_event gpudone_event;
	ciErrNum = clEnqueueReadBuffer(commandQueue, d_C, CL_FALSE, 0, mem_size_C, arg->data_fp, 0, 0, &gpudone_event);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("clEnqueueReadBuffer() Error: Failed to write buffer!\n");
		return -1;
	}
	// Set callback that will launch another CPU thread to finish the work when the download from GPU has finished
	ciErrNum = clSetEventCallback(gpudone_event, CL_COMPLETE, &event_callback, arg);
	
	// Cleanup
	// NOTE: OpenCL requires that all calls to clRelease* API calls (except clReleaseEvent) are deferred until clSetUserEventStatus is called (see OpenCL Spec 1.1 rev 44 p.143)
	clReleaseEvent(arg->user_event);
	clReleaseEvent(GPUExecution);
	clReleaseEvent(gpudone_event);

	vDeferredReleaseQueue.push_back(commandQueue);
	vDeferredReleaseMem.push_back(d_A);
	vDeferredReleaseMem.push_back(d_B);
	vDeferredReleaseMem.push_back(d_C);

	return 0;
}

// This is a helper function that will load the OpenCL source program, build and return a handle to that OpenCL kernel
int compileOCLKernel(cl_context cxGPUContext, cl_device_id cdDevices, 
                     const char *ocl_source_filename, cl_program *cpProgram, 
                     char **argv)
{
	cl_int ciErrNum;

    size_t program_length;
	const char* source_path = shrFindFilePath(ocl_source_filename, argv[0]);
	oclCheckError(source_path != NULL, shrTRUE);
	char *source = oclLoadProgSource(source_path, "", &program_length);
	if(!source) {
		shrLog("Error: Failed to load compute program %s!\n", source_path);
		return -2000;
    }

	// create the simple increment OpenCL program
	*cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &program_length, &ciErrNum);
	if (ciErrNum != CL_SUCCESS) {
		shrLog("Error: Failed to create program\n");
		return ciErrNum;
    } else {
        shrLog("clCreateProgramWithSource <%s> succeeded, program_length=%d\n", ocl_source_filename, program_length);
    }
	free(source);

	// build the program
    cl_build_status build_status;

	ciErrNum = clBuildProgram(*cpProgram, 0, NULL, "-cl-fast-relaxed-math -cl-nv-verbose", NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then return error
		shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
		oclLogBuildInfo(*cpProgram, oclGetFirstDev(cxGPUContext));
		oclLogPtx(*cpProgram, oclGetFirstDev(cxGPUContext), "oclMultiThreads.ptx");
		return ciErrNum;
    } else {
        shrLog("clBuildProgram <%s> succeeded\n", ocl_source_filename);
        ciErrNum = clGetProgramBuildInfo(*cpProgram, cdDevices, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
        shrLog("clGetProgramBuildInfo returned: ");
        if (build_status == CL_SUCCESS) {
            shrLog("CL_SUCCESS\n");
        } else {
            shrLog("CLErrorNumber = %d\n", ciErrNum);
        }
    }

    // print out the build log, note in the case where there is nothing shown, some OpenCL PTX->SASS caching has happened
    {
        char *build_log;
        size_t ret_val_size;
        ciErrNum = clGetProgramBuildInfo(*cpProgram, cdDevices, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        if (ciErrNum != CL_SUCCESS) {
            shrLog("clGetProgramBuildInfo device %d, failed to get the log size at line %d\n", cdDevices, __LINE__);
        }
        build_log = (char *)malloc(ret_val_size+1);
        ciErrNum = clGetProgramBuildInfo(*cpProgram, cdDevices, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        if (ciErrNum != CL_SUCCESS) {
            shrLog("clGetProgramBuildInfo device %d, failed to get the build log at line %d\n", cdDevices, __LINE__);
        }
        // to be carefully, terminate with \0
        // there's no information in the reference whether the string is 0 terminated or not
        build_log[ret_val_size] = '\0';
        shrLog("%s\n", build_log );
    }
    return 0;
}


int main(int argc, char** argv) 
{
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "help")) {
        printf("[oclMultiThreads] - help\n");
        printf("\t-profile   - Enable OpenCL profiling counters.\n");
        printf("\t-device=n  - Specify one specific GPU device to enable OpenCL kernels.\n");
        return 0;
    }


    shrQAStart(argc, argv);

	// start the logs
    shrSetLogFileName ("oclMultiThreads.txt");

#ifdef _WIN32
	// we are detecting what Windows OS is being used, as the Windows threading types requires Windows Vista/7
	OSVERSIONINFO OSversion;
	printf("Operating System: %s\n", getOSName(&OSversion));
#else
	printf("Operating System: %s\n", getOSName() );
#endif

////////////////////////////////////////////////////////////////////////////////
// OpenCL Setup (initialize the OpenCL context)
////////////////////////////////////////////////////////////////////////////////
    cl_context       cxGPUContext;
    cl_kernel        kernel[MAX_GPU_COUNT];
    cl_program       program[MAX_GPU_COUNT];

	cl_platform_id  cpPlatform    = NULL;
	cl_device_id   *cdDevices     = NULL;
	cl_int          ciErrNum      = CL_SUCCESS;
	cl_uint         ciDeviceCount = 0;

    bool bEnableProfile = false; // This is to enable/disable OpenCL based profiling

	//Get the NVIDIA platform
	ciErrNum = oclGetPlatformID(&cpPlatform);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("Error: Failed to create OpenCL context!\n");
		return ciErrNum;
	}

	//Retrieve of the available GPU type OpenCL devices
	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &ciDeviceCount);
	cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id) );
	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, ciDeviceCount, cdDevices, NULL);

    // Allocate a buffer array to store the names GPU device(s)
    char (*cDevicesName)[256] = new char[ciDeviceCount][256];

    if (ciErrNum != CL_SUCCESS) {
		shrLog("Error: Failed to create OpenCL context!\n");
		return ciErrNum;
	} else {
		shrLog("Detected %d OpenCL devices of type CL_DEVICE_TYPE_CPU\n", ciDeviceCount);
		for (int i=0; i<(int)ciDeviceCount; i++) {
			clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, sizeof(cDevicesName[i]), &cDevicesName[i], NULL);
            shrLog("> OpenCL Device #%d (%s), cl_device_id: %d\n", i, cDevicesName[i], cdDevices[i]);
		}
	}

	//Create the OpenCL context
	cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		shrLog("Error: Failed to create OpenCL context!\n");
		return ciErrNum;
	}

    if(shrCheckCmdLineFlag(argc, (const char**)argv, "profile"))
    {
        bEnableProfile = true;
    }

	if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
	{
		// User specified GPUs
		char* deviceList;
		char* deviceStr;
		char* next_token;
		shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

#ifdef WIN32
		deviceStr = strtok_s (deviceList," ,.-", &next_token);
#else
		deviceStr = strtok (deviceList," ,.-");
#endif   
		ciDeviceCount = 0;

		while(deviceStr != NULL) 
		{
			// get and print the device for this queue
			cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
			if( device == (cl_device_id) -1  ) {
				shrLog(" Device %s does not exist!\n", deviceStr);
				return -1;
			}

			shrLog("Device %s: ", deviceStr);
			oclPrintDevName(LOGBOTH, device);            
			shrLog("\n");

			++ciDeviceCount;

#ifdef WIN32
			deviceStr = strtok_s (NULL," ,.-", &next_token);
#else            
			deviceStr = strtok (NULL," ,.-");
#endif
		}

		free(deviceList);
	} 


////////////////////////////////////////////////////////////////////////////////
// Launch Heterogeneous Workloads
////////////////////////////////////////////////////////////////////////////////
	barrier = cutCreateBarrier(N);

#if 0
    compileOCLKernel(cxGPUContext, cdDevices[0], "matrixMul.cl", &program[0], argv);

	for( int i=0; i < N; ++i ) {
		kernel[i] = clCreateKernel(program[0], "matrixMul", &ciErrNum);
		if (ciErrNum != CL_SUCCESS) {
			shrLog("Error: Failed to create Matrix Multiply kernel\n");
			return ciErrNum;
		}
		launch_hybrid_matrixMultiply(cxGPUContext, cdDevices[i%ciDeviceCount], cDevicesName[i%ciDeviceCount], kernel[i] , i, bEnableProfile);
	}
#else
    compileOCLKernel(cxGPUContext, cdDevices[0], "kernel.cl",    &program[0], argv);

	for( int i=0; i < N; ++i ) {
		kernel[i]= clCreateKernel(program[0], "simpleIncrement", &ciErrNum);
	
		if (ciErrNum != CL_SUCCESS) {
			shrLog("Error: Failed to create Simple Increment kernel\n");
			return ciErrNum;
		}
		launch_hybrid_workload(cxGPUContext, cdDevices[i%ciDeviceCount], cDevicesName[i%ciDeviceCount], kernel[i], i, bEnableProfile);
	}
#endif
	
	clReleaseContext(cxGPUContext);

	// Wait until all work is done by both CPU & GPU(s)
	cutWaitForBarrier(&barrier);

	// Cleanup

	for( std::vector<cl_mem>::iterator it = vDeferredReleaseMem.begin(), it_end = vDeferredReleaseMem.end(); it != it_end; it++ ) clReleaseMemObject(*it);
	for( std::vector<cl_command_queue>::iterator it= vDeferredReleaseQueue.begin(), it_end = vDeferredReleaseQueue.end(); it != it_end; it++ ) clReleaseCommandQueue(*it);
	for( int i=0; i<N; ++i ) clReleaseKernel(kernel[i]);

	clReleaseProgram(program[0]);

    delete [] cDevicesName;

	shrQAFinishExit(argc, (const char **)argv, (bOK ? QA_PASSED : QA_FAILED));

	return 0;
}

#else
// If OpenCL 1.1 is not available (i.e. Mac OSX 10.6.8), then we report this
int main(int argc, char** argv) 
{
    shrQAStart(argc, argv);

	// start the logs
    shrSetLogFileName ("oclMultiThreads.txt");

	shrLog("The OS system does not support OpenCL 1.1\n");

	shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}

#endif
