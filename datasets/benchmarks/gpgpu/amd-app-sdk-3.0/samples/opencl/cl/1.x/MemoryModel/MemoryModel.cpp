#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
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
#include <fstream>
#include <cstring>
using namespace std;
const int  width=256;
const int  sizespace=4;
const int  localsize=64;

/* read the kernel into a string */
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
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';

        s = str;
        delete[] str;
        return 0;
    }
    std::cout<<"Error: faileded to open file \n"<<filename;
    return -1;
}

int main(int argc,char *argv[])
{

    int m_test[width];
    int *outBuf=new int[width];
    int *verificationBuf=new int[width];
    const int mask[sizespace] = {1, -1, 2, -2}; //used to compute result on cpu

    /* initialization array */
    std::cout << "Input Array: " << std::endl;
    for(int i=0; i<width; i++)
    {
        m_test[i]=i+1;
        if(i%16==0)
        {
            std::cout<< std::endl;
        }
        std::cout<< m_test[i] << " ";
    }
    std::cout<< std::endl << std::endl;
    std::cout<<"GPU computing......." << std::endl;

    /* Get Platforms and choose an available one */
    cl_int status;
    cl_platform_id platform=NULL;
    cl_uint numPlatforms = 0;
    status=clGetPlatformIDs(0,NULL,&numPlatforms);
    if(status!=CL_SUCCESS)
    {
        std::cout<<"Get paltform failed\n";
        return -1;
    }
    if(numPlatforms>0)
    {
        cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(
                                        cl_platform_id));
        status=clGetPlatformIDs(numPlatforms,platforms,NULL);
        platform=platforms[0];
        free(platforms);
    }

    /* Query the context and get the available devices */
    cl_uint numDevice=0;
    status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&numDevice);
    cl_device_id *device=(cl_device_id*)malloc(numDevice*sizeof(cl_device_id));
    if (device == 0)
    {
        std::cout << "No device available\n";
        return -1;
    }
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,numDevice,device,NULL);

    /* Create Context using the platform selected above */
    cl_context context=CECL_CREATE_CONTEXT(NULL,numDevice,device,NULL,NULL,NULL);

    /* create command queue */
    cl_command_queue queue0=CECL_CREATE_COMMAND_QUEUE(context,device[0],
                            CL_QUEUE_PROFILING_ENABLE,NULL);

    /* create cl_buffer objects */
    cl_mem clsrc=CECL_BUFFER(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                width * sizeof(int),m_test,NULL);
    cl_mem clout=CECL_BUFFER(context,CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                width * sizeof(int),NULL,NULL);

    const char *filename="MemoryModel.cl";
    string sourceStr;
    status=convertToString(filename, sourceStr);
    const char *source=sourceStr.c_str();
    size_t sourceSize[]= {strlen(source)};

    /* Create program object */
    cl_program program=CECL_PROGRAM_WITH_SOURCE(context,1,&source,sourceSize,NULL);

    /* Build program object */
    status=CECL_PROGRAM(program,1,device,NULL,NULL,NULL);
    if(status!=CL_SUCCESS)
    {
        std::cout<<"Building program failed\n";
        return -1;
    }

    /* Create kernel object */
    cl_kernel kernel=CECL_KERNEL(program,"MemoryModel", NULL);

    /* Set Kernel arguments */
    CECL_SET_KERNEL_ARG(kernel,0,sizeof(cl_mem),(void *)&clout);
    CECL_SET_KERNEL_ARG(kernel,1,sizeof(cl_mem),(void *)&clsrc);

    size_t globalws[1]= {width};
    size_t localws[1]= {localsize};

    /* Enqueue kernel into command queue and run it */
    CECL_ND_RANGE_KERNEL(queue0,kernel,1,0,globalws,localws,0,NULL,NULL);

    clFinish(queue0);

    /* Read the output back to host memory */
    status = CECL_READ_BUFFER(
                 queue0, clout,
                 CL_TRUE,        /* Blocking Read Back */
                 0, width*sizeof(int),(void*)outBuf, 0, NULL, NULL);

    /* Display the computed result of GPU */
    std::cout<< std::endl;
    std::cout << "Result Array: " << std::endl;
    for(int i=0; i<width; i++)
    {
        if(i%16==0)
        {
            std::cout<< std::endl;
        }
        std::cout<<outBuf[i]<<" ";
    }
    std::cout<< std::endl;

    /*compute the results on CPU,in order to verify*/
    for(int i=0; i<width; i++)
    {
        int interation=i/localsize;
        int t=interation*localsize;
        int result = 0;
        for (int j = 0; j < 4; j++)
        {
            result += m_test[(i+j)%localsize+t];
        }
        result *= mask[interation];
        verificationBuf[i]= result;
    }
    // compare the results and see if they match
    if(memcmp(outBuf, verificationBuf,width*sizeof(int)) == 0)
    {
        std::cout<<"Passed!\n" << std::endl;
    }
    else
    {
        std::cout<<"Failed\n" << std::endl;
    }

    /*clean up*/
    if(outBuf)
    {
        free(outBuf);
    }
    if(verificationBuf)
    {
        free(verificationBuf);
    }
    /* Release OpenCL resource object */
    status =clReleaseKernel(kernel);
    status =clReleaseMemObject(clsrc);
    status =clReleaseMemObject(clout);
    status =clReleaseProgram(program);
    status =clReleaseCommandQueue(queue0);
    status =clReleaseContext(context);
    free(device);
    //system("pause");
    return 0;
}
