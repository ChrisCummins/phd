#include <cecl.h>
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
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <shrQATest.h>
#include <oclUtils.h>


using namespace std;

bool bNoPrompt = false;
bool bQATest   = false;

int *pArgc     = NULL;
char **pArgv   = NULL;

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "inlinePTX.cl";

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_uint        uiNumDevices;    // OpenCL total number of devices
cl_device_id*  cdDevices;       // OpenCL device(s)
cl_program     cpProgram;       // OpenCL program
cl_kernel      ckKernel;        // OpenCL kernel
cl_mem         cmDevBuffer;     // OpenCL device buffer
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;	        // 1D var for # of work items in the work group	
size_t szParmDataBytes;	        // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;

#ifdef STRCASECMP
#undef STRCASECMP
#endif
#ifdef STRNCASECMP
#undef STRNCASECMP
#endif

#ifdef _WIN32
   #define STRCASECMP  _stricmp
   #define STRNCASECMP _strnicmp
#else
   #define STRCASECMP  strcasecmp
   #define STRNCASECMP strncasecmp
#endif


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID=0; elemID<length; elemID++)
    {
        h_ptr[elemID] = elemID % 32;
    }
}


void Cleanup (int iExitCode)
{
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED); 

    if (bNoPrompt)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutableName);
        getchar();
    }
    exit (iExitCode);
}


int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    shrQAStart(argc, argv);

    cout << "OpenCL inline PTX assembler sample" << endl;
    cout << "================================" << endl;
    cout << "Self-test started" << endl;


    const int N = 1000;

    bNoPrompt = shrCheckCmdLineFlag(argc, (const char **)argv, "noprompt");
    bQATest   = shrCheckCmdLineFlag(argc, (const char **)argv, "qatest");

#if defined (__APPLE__) || defined(MACOSX)
    cout << "oclInlinePTX is not currently supported on Mac OSX" << endl;
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, QA_PASSED);
    exit(EXIT_SUCCESS);
#endif

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }


    //Get the devices
//    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiNumDevices, cdDevices, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, cdDevices, NULL, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[0], 0, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_CREATE_COMMAND_QUEUE, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevBuffer = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int) * N, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_BUFFER, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_PROGRAM_WITH_SOURCE, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    ciErr1 = CECL_PROGRAM(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_PROGRAM, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	oclLogBuildInfo(cpProgram, cdDevices[0]);
        Cleanup(EXIT_FAILURE);
    }

    // Create the kernel
    ckKernel = CECL_KERNEL(cpProgram, "sequence_gpu", &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_KERNEL, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Set the Argument values
    ciErr1 = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevBuffer);
    ciErr1 |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_int), (void*)&N);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_SET_KERNEL_ARG, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // --------------------------------------------------------

    int *h_ptr = (int*) malloc(N * sizeof(int));

    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp(szLocalWorkSize, N);

    // Launch kernel
    ciErr1 = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_ND_RANGE_KERNEL, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }


    sequence_cpu(h_ptr, N);

    cout << "OpenCL and CPU algorithm implementations finished" << endl;

    int *h_d_ptr;    
	h_d_ptr = (int*) malloc(N * sizeof(int));

    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = CECL_READ_BUFFER(cqCommandQueue, cmDevBuffer, CL_TRUE, 0, sizeof(int) * N, h_d_ptr, 0, NULL, NULL);
    shrLog("CECL_READ_BUFFER (Dst)...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_READ_BUFFER, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    bool bValid = true;
    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
        }
    }

	if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cmDevBuffer)clReleaseMemObject(cmDevBuffer);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    cout << "TEST Results " << endl;
    shrQAFinishExit(argc, (const char **)argv, (bValid ? QA_PASSED : QA_FAILED));
}
