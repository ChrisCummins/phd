#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1
#define EXPECTED_FAILURE 2

#define  GlobalThreadSize 256
#define  GroupSize 64

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size+1];
		if(!str)
		{
			f.close();
			return SUCCESS;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return SUCCESS;
	}
	std::cout<<"Error: failed to open file\n:"<<filename<<std::endl;
	return FAILURE;
}

int main(int argc, char* argv[])
{
	
	cl_int status = 0;//store the return status

	cl_uint numPlatforms;//store the number of platforms query by clGetPlatformIDs()

	cl_platform_id platform = NULL;//store the chosen platform

	//get platform 
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Getting platforms!"<<std::endl;
		return FAILURE;
	}

	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status=clGetPlatformIDs(numPlatforms,platforms,NULL);
		platform=platforms[0];
		free(platforms);
	}

	if (NULL == platform)
	{
		std::cout<<"Error: No available platform found!"<<std::endl;
		return FAILURE;
	}


	/* Query the context and get the available devices */
	cl_uint numDevice=0;
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&numDevice);
	cl_device_id *devices=(cl_device_id*)malloc(numDevice*sizeof(cl_device_id));
	if (devices == 0) 
	{
        std::cout << "No device available\n";
        return FAILURE;
    }
	clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,numDevice,devices,NULL);

	/* Create Context using the platform selected above */   
	cl_context context=CECL_CREATE_CONTEXT(NULL,numDevice,devices,NULL,NULL,NULL);
	
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating context failed!"<<std::endl;
		return FAILURE;
	}

	/*
	*The API CECL_CREATE_COMMAND_QUEUE creates a command-queue on a specific device.
	*/
	cl_command_queue commandQueue = CECL_CREATE_COMMAND_QUEUE(context, devices[0], 0, &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating command queue failed!"<<std::endl;
		return FAILURE;
	}

	//set input data
	cl_uint inputSizeBytes = GlobalThreadSize *  sizeof(cl_uint);
	cl_float *input = (cl_float *) malloc(inputSizeBytes);

	for(int i=0;i< GlobalThreadSize;i++)
	{
		input[i] = (float)i;
	}
	//create input buffer
	cl_mem inputBuffer = CECL_BUFFER(
                      context, 
                      CL_MEM_READ_ONLY|
					  CL_MEM_USE_HOST_PTR,
                      sizeof(cl_uint) * GlobalThreadSize,
                      (void *)input, 
                      &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating input buffer failed!"<<std::endl;
		return FAILURE;
	}

	//create output buffer
	cl_mem outputBuffer = CECL_BUFFER(
                      context, 
                      CL_MEM_WRITE_ONLY,
                      sizeof(cl_uint) * GlobalThreadSize,
                      NULL, 
                      &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating output buffer failed!"<<std::endl;
		return FAILURE;
	}


	//get the printf kernel
	const char* filename = "./BasicDebug_Kernel.cl";
	std::string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] ={strlen(source)};
	//
	const char* filename2 = "./BasicDebug_Kernel2.cl";
	std::string sourceStr2;
	status = convertToString(filename2, sourceStr2);
	const char *source2 = sourceStr2.c_str();
	size_t sourceSize2[] ={strlen(source2)};

	//create program
	cl_program program = CECL_PROGRAM_WITH_SOURCE(context, 1, &source, sourceSize, &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating program object failed!"<<std::endl;
		return FAILURE;
	}
	cl_program program2 = CECL_PROGRAM_WITH_SOURCE(context, 1, &source2, sourceSize2, &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating program object failed!"<<std::endl;
		return FAILURE;
	}

	char buildOption[128];
    sprintf(buildOption, "-g -D WGSIZE=%d", GroupSize);

	//build program with the command line option '-g' so we can debug kernel
	status = CECL_PROGRAM(program,1, devices, buildOption, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Building program failed!"<<std::endl;
		return FAILURE;
	}
	
	status = CECL_PROGRAM(program2,1, devices, "-g", NULL, NULL);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Building program failed!"<<std::endl;
		return FAILURE;
	}

	//create printf kernel	
	cl_kernel kernel1 = CECL_KERNEL(program, "printfKernel", &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating kernel failed!"<<std::endl;
		return FAILURE;
	}
	//set kernel args.
	status = CECL_SET_KERNEL_ARG(kernel1, 0, sizeof(cl_mem), (void *)&inputBuffer);

	//create debug kernel	
	cl_kernel kernel2 = CECL_KERNEL(program2, "debugKernel", &status);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating kernel failed!"<<std::endl;
		return FAILURE;
	}
	//set kernel args.
	status = CECL_SET_KERNEL_ARG(kernel2, 0, sizeof(cl_mem), (void *)&inputBuffer);
	status = CECL_SET_KERNEL_ARG(kernel2, 1, sizeof(cl_mem), (void *)&outputBuffer);
	

	size_t global_threads[1];
	size_t local_threads[1];
	global_threads[0] = GlobalThreadSize;
    local_threads[0] = GroupSize;

	//execute the kernel
	status = CECL_ND_RANGE_KERNEL(commandQueue, kernel1, 1, NULL, global_threads, local_threads, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Enqueue kernel onto command queue failed!"<<std::endl;
		return FAILURE;
	}
	status = clFinish(commandQueue);

	status = CECL_ND_RANGE_KERNEL(commandQueue, kernel2, 1, NULL, global_threads, local_threads, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		std::cout<<"Error: Enqueue kernel onto command queue failed!"<<std::endl;
		return FAILURE;
	}
	status = clFinish(commandQueue);

	// Clean the resources.
	status = clReleaseKernel(kernel1);//Release kernel.
	status = clReleaseKernel(kernel2);
	status = clReleaseMemObject(inputBuffer);//Release mem object.
	status = clReleaseMemObject(outputBuffer);
	status = clReleaseProgram(program);//Release program.
	status = clReleaseProgram(program2);//Release program2.
	status = clReleaseCommandQueue(commandQueue);//Release command queue.
	status = clReleaseContext(context);//Release context.

	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}

	free(input);
	std::cout<<"Passed!\n";
	return SUCCESS;
}
