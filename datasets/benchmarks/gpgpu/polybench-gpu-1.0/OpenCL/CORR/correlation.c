#include <libcecl.h>
/**
 * correlation.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define M 2048
#define N 2048

/* Thread block dimensions for kernel 1*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_X 32
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_Y 1


#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

#define FLOAT_N 3214212.01
#define EPS 0.005

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_corr;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i<=M; i++)
	{
		for (j=0; j<=N; j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
				printf("I: %d J: %d \n 1: %f\n 2: %f\n", i, j, symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]);		
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("correlation.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* data)
{
	int i, j;
	
	for (i=0; i<=M; i++) 
	{
    	for (j=0; j<=N; j++) 
		{
       		data[i*N + j] = ((DATA_TYPE) i*j)/ (M+1);	
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


void cl_mem_init(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat)
{
	data_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	symmat_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	stddev_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1), NULL, &errcode);
	mean_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = CECL_WRITE_BUFFER(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), data, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), symmat, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, stddev_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), stddev, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), mean, 0, NULL, NULL);
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
	clKernel_mean = CECL_KERNEL(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel_std = CECL_KERNEL(clProgram, "std_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel_reduce = CECL_KERNEL(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");

	clKernel_corr = CECL_KERNEL(clProgram, "corr_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel4\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int m = M;
	int n = N;

	DATA_TYPE float_n = FLOAT_N;
	DATA_TYPE eps = EPS;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];
	size_t localWorkSize_Kernel4[2], globalWorkSize_Kernel4[2];

	localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = 1;

	localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	localWorkSize_Kernel4[0] = DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	localWorkSize_Kernel4[1] = DIM_LOCAL_WORK_GROUP_KERNEL_4_Y;
	globalWorkSize_Kernel4[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_4_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	globalWorkSize_Kernel4[1] = 1;


	t_start = rtclock();	
	
	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_mean, 3, sizeof(int), (void *)&m);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel_std, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  CECL_SET_KERNEL_ARG(clKernel_std, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_std, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_std, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_std, 4, sizeof(DATA_TYPE), (void *)&eps);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_std, 5, sizeof(int), (void *)&m);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_std, 6, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");
 
	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel_std, 1, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  CECL_SET_KERNEL_ARG(clKernel_reduce, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_reduce, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_reduce, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_reduce, 4, sizeof(int), (void *)&m);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_reduce, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");
 
	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel	
	errcode =  CECL_SET_KERNEL_ARG(clKernel_corr, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_corr, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_corr, 2, sizeof(int), (void *)&m);
	errcode |= CECL_SET_KERNEL_ARG(clKernel_corr, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments4\n");

	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel_corr, 1, NULL, globalWorkSize_Kernel4, localWorkSize_Kernel4, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel4\n");
	clEnqueueBarrier(clCommandQue);

	DATA_TYPE val = 1.0;
	CECL_WRITE_BUFFER(clCommandQue, symmat_mem_obj, CL_TRUE, ((M)*(M+1) + (M))*sizeof(DATA_TYPE), sizeof(DATA_TYPE), &val, 0, NULL, NULL);

	clFinish(clCommandQue);

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel_reduce);
	errcode = clReleaseKernel(clKernel_mean);
	errcode = clReleaseKernel(clKernel_std);
	errcode = clReleaseKernel(clKernel_corr);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(symmat_mem_obj);
	errcode = clReleaseMemObject(data_mem_obj);
	errcode = clReleaseMemObject(mean_mem_obj);
	errcode = clReleaseMemObject(stddev_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat)
{
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 1; j <= M; j++)
   	{
  		mean[j] = 0.0;

   		for (i = 1; i <= N; i++)
		{
			mean[j] += data[i*(M+1) + j];
   		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 1; j <= M; j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 1; i <= N; i++)
		{
			stddev[j] += (data[i*(M+1) + j] - mean[j]) * (data[i*(M+1) + j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
    	stddev[j] = sqrt_of_array_cell(stddev, j);
    	stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }

 	// Center and reduce the column vectors. 
  	for (i = 1; i <= N; i++)
	{
    	for (j = 1; j <= M; j++)
    	{
			data[i*(M+1) + j] -= mean[j];
			data[i*(M+1) + j] /= sqrt(FLOAT_N) ;
			data[i*(M+1) + j] /= stddev[j];
    	}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 1; j1 <= M-1; j1++)
    {	
    	symmat[j1*(M+1) + j1] = 1.0;
    
		for (j2 = j1+1; j2 <= M; j2++)
		{
	  		symmat[j1*(M+1) + j2] = 0.0;

	  		for (i = 1; i <= N; i++)
			{
	   			symmat[j1*(M+1) + j2] += (data[i*(M+1) + j1] * data[i*(M+1) + j2]);
			}

	  		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
		}
    }
 
	symmat[M*(M+1) + M] = 1.0;
}


int main(void) 
{
	double t_start, t_end;
	
	DATA_TYPE* data;
	DATA_TYPE* mean;
	DATA_TYPE* stddev;
	DATA_TYPE* symmat;
	DATA_TYPE* symmat_outputFromGpu;

	data = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	mean = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
	stddev = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
	symmat = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	symmat_outputFromGpu = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	
	init_arrays(data);
	read_cl_file();
	cl_initialization();
	cl_mem_init(data, mean, stddev, symmat);
	cl_load_prog();

	cl_launch_kernel();

	errcode = CECL_READ_BUFFER(clCommandQue, symmat_mem_obj, CL_TRUE, 0, (M+1) * (N+1) * sizeof(DATA_TYPE), symmat_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	correlation(data, mean, stddev, symmat);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(symmat, symmat_outputFromGpu);
	cl_clean_up();
	
	free(data);
	free(mean);
	free(stddev);
	free(symmat);
	free(symmat_outputFromGpu);
	
    return 0;
}

