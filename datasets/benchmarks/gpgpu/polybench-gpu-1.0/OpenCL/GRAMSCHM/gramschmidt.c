#include <libcecl.h>
/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define M 2048
#define N 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef double DATA_TYPE;


char str_temp[1024];
 

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

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem q_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("gramschmidt.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
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


void cl_mem_init(DATA_TYPE* A)
{
	a_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	r_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	q_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = CECL_WRITE_BUFFER(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, A, 0, NULL, NULL);
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
	clKernel1 = CECL_KERNEL(clProgram, "gramschmidt_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel2 = CECL_KERNEL(clProgram, "gramschmidt_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel3 = CECL_KERNEL(clProgram, "gramschmidt_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int m = M;
	int n = N;

	size_t localWorkSize[2], globalWorkSizeKernel1[2], globalWorkSizeKernel2[2], globalWorkSizeKernel3[2];

	localWorkSize[0] = DIM_THREAD_BLOCK_X;
	localWorkSize[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel1[0] = DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel1[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel2[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel2[1] = 1;
	globalWorkSizeKernel3[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel3[1] = 1;

	t_start = rtclock();
	
	int k;
	for (k = 0; k < N; k++)
	{
		// Set the arguments of the kernel
		errcode =  CECL_SET_KERNEL_ARG(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  CECL_SET_KERNEL_ARG(clKernel1, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 3, sizeof(int), (void *)&k);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 4, sizeof(int), (void *)&m);
		errcode |= CECL_SET_KERNEL_ARG(clKernel1, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel1, 1, NULL, globalWorkSizeKernel1, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clEnqueueBarrier(clCommandQue);


		errcode =  CECL_SET_KERNEL_ARG(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  CECL_SET_KERNEL_ARG(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 3, sizeof(int), (void *)&k);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 4, sizeof(int), (void *)&m);
		errcode |= CECL_SET_KERNEL_ARG(clKernel2, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel2, 1, NULL, globalWorkSizeKernel2, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
		clEnqueueBarrier(clCommandQue);


		errcode =  CECL_SET_KERNEL_ARG(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  CECL_SET_KERNEL_ARG(clKernel3, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 3, sizeof(int), (void *)&k);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 4, sizeof(int), (void *)&m);
		errcode |= CECL_SET_KERNEL_ARG(clKernel3, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel3, 1, NULL, globalWorkSizeKernel3, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
		clEnqueueBarrier(clCommandQue);

	}
	clFinish(clCommandQue);

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
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


int main(void) 
{
	double t_start, t_end;
	int i;

	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	
	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

	init_array(A);
	read_cl_file();
	cl_initialization();
	cl_mem_init(A);
	cl_load_prog();

	cl_launch_kernel();

	errcode = CECL_READ_BUFFER(clCommandQue, a_mem_obj, CL_TRUE, 0, M*N*sizeof(DATA_TYPE), A_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");   

	t_start = rtclock();
	gramschmidt(A, R, Q);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	
	compareResults(A, A_outputFromGpu);
	cl_clean_up();

	free(A);
	free(A_outputFromGpu);
	free(R);
	free(Q);  

	return 0;
}

