#include <libcecl.h>
/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

/* Problem size */
#define TMAX 500
#define NX 2048
#define NY 2048

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

DATA_TYPE alpha = 23;
DATA_TYPE beta = 15;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem fict_mem_obj;
cl_mem ex_mem_obj;
cl_mem ey_mem_obj;
cl_mem hz_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("fdtd.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < TMAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = CECL_CREATE_CONTEXT( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = CECL_CREATE_COMMAND_QUEUE(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	fict_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * TMAX, NULL, &errcode);
	ex_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * (NY + 1), NULL, &errcode);
	ey_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (NX + 1) * NY, NULL, &errcode);
	hz_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = CECL_WRITE_BUFFER(clCommandQue, fict_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * TMAX, _fict_, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, ex_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * (NY + 1), ex, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, ey_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (NX + 1) * NY, ey, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, hz_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, hz, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}

 
void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = CECL_PROGRAM_WITH_SOURCE(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = CECL_PROGRAM(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = CECL_KERNEL(clProgram, "fdtd_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	
	// Create the OpenCL kernel
	clKernel2 = CECL_KERNEL(clProgram, "fdtd_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel3 = CECL_KERNEL(clProgram, "fdtd_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int nx = NX;
	int ny = NY;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
	int t;
	for(t=0;t<TMAX;t++)
	{
		// Set the arguments of the kernel
		errcode =  CECL_SET_KERNEL_ARG(clKernel1, 0, sizeof(cl_mem), (void *)&fict_mem_obj);
		errcode =  CECL_SET_KERNEL_ARG(clKernel1, 1, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 2, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 3, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 4, sizeof(int), (void *)&t);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 5, sizeof(int), (void *)&nx);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 6, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clEnqueueBarrier(clCommandQue);

		// Set the arguments of the kernel
		errcode =  CECL_SET_KERNEL_ARG(clKernel2, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 3, sizeof(int), (void *)&nx);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 4, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clEnqueueBarrier(clCommandQue);

		// Set the arguments of the kernel
		errcode =  CECL_SET_KERNEL_ARG(clKernel3, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 3, sizeof(int), (void *)&nx);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 4, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clFinish(clCommandQue);
	}

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(fict_mem_obj);
	errcode = clReleaseMemObject(ex_mem_obj);
	errcode = clReleaseMemObject(ey_mem_obj);
	errcode = clReleaseMemObject(hz_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for(t=0; t < TMAX; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       			for (j = 0; j < NY; j++)
				{
       				ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


int main(void) 
{
	double t_start, t_end;
	
	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;

	_fict_ = (DATA_TYPE*)malloc(TMAX*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	
	int i;
	init_arrays(_fict_, ex, ey, hz);
	read_cl_file();
	cl_initialization();
	cl_mem_init(_fict_, ex, ey, hz);
	cl_load_prog();

	cl_launch_kernel();

	errcode = CECL_READ_BUFFER(clCommandQue, hz_mem_obj, CL_TRUE, 0, NX * NY * sizeof(DATA_TYPE), hz_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");	

	t_start = rtclock();
	runFdtd(_fict_, ex, ey, hz);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(hz, hz_outputFromGpu);
	cl_clean_up();
	
	free(_fict_);
	free(ex);
	free(ey);
	free(hz);
	free(hz_outputFromGpu);
	
    	return 0;
}

