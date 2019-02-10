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

// *********************************************************************
// oclVectorAdd Notes:  
//
// A simple OpenCL API demo application that implements 
// element by element vector addition between 2 float arrays. 
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
//
// Uses some 'shr' and 'ocl' functions from oclUtils and shrUtils libraries for 
// compactness, but these are NOT required libs for OpenCL developement in general.
// *********************************************************************

// common SDK header for standard utilities and system libs 
#include <oclUtils.h>
#include <shrQATest.h>

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "VectorAdd.cl";

// Host buffers for demo
// *********************************************************************
void *srcA, *srcB, *dst;        // Host buffers for OpenCL test
void* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;

// demo config vars
int iNumElements = 11444777;	// Length of float arrays to process (odd # for illustration)
shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int argc, char **argv, int iExitCode);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    shrQAStart(argc, argv);

    // get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    
    // start logs 
	cExecutableName = argv[0];
    shrSetLogFileName ("oclVectorAdd.txt");
    shrLog("%s Starting...\n\n# of float elements per Array \t= %i\n", argv[0], iNumElements); 

    // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

    // Allocate and initialize host arrays 
    shrLog( "Allocate and Init Host Mem...\n"); 
    srcA = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
    srcB = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
    dst = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
    Golden = (void *)malloc(sizeof(cl_float) * iNumElements);
    shrFillArray((float*)srcA, iNumElements);
    shrFillArray((float*)srcB, iNumElements);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

    shrLog("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    shrLog("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = CECL_CREATE_CONTEXT(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    shrLog("CECL_CREATE_CONTEXT...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_CREATE_CONTEXT, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, 0, &ciErr1);
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_CREATE_COMMAND_QUEUE, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevSrcA = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr1);
    cmDevSrcB = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevDst = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    shrLog("CECL_BUFFER...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_BUFFER, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    
    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    shrLog("CECL_PROGRAM_WITH_SOURCE...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_PROGRAM_WITH_SOURCE, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    ciErr1 = CECL_PROGRAM(cpProgram, 0, NULL, NULL, NULL, NULL);
    shrLog("CECL_PROGRAM...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_PROGRAM, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create the kernel
    ckKernel = CECL_KERNEL(cpProgram, "VectorAdd", &ciErr1);
    shrLog("CECL_KERNEL (VectorAdd)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_KERNEL, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values
    ciErr1 = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErr1 |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErr1 |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
    ciErr1 |= CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
    shrLog("CECL_SET_KERNEL_ARG 0 - 3...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_SET_KERNEL_ARG, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = CECL_WRITE_BUFFER(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcA, 0, NULL, NULL);
    ciErr1 |= CECL_WRITE_BUFFER(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcB, 0, NULL, NULL);
    shrLog("CECL_WRITE_BUFFER (SrcA and SrcB)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_WRITE_BUFFER, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Launch kernel
    ciErr1 = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    shrLog("CECL_ND_RANGE_KERNEL (VectorAdd)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_ND_RANGE_KERNEL, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = CECL_READ_BUFFER(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
    shrLog("CECL_READ_BUFFER (Dst)...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in CECL_READ_BUFFER, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------

    // Compute and compare results for golden-host and report errors and pass/fail
    shrLog("Comparing against Host/C++ computation...\n\n"); 
    VectorAddHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
    shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);

    // Cleanup and leave
    Cleanup (argc, argv, (bMatch == shrTRUE) ? EXIT_SUCCESS : EXIT_FAILURE);
}

void Cleanup (int argc, char **argv, int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevDst)clReleaseMemObject(cmDevDst);

    // Free host memory
    free(srcA); 
    free(srcB);
    free (dst);
    free(Golden);

    // finalize logs and leave
    shrQAFinishExit(argc, (const char **)argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}

// "Golden" Host processing vector addition function for comparison purposes
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
    int i;
    for (i = 0; i < iNumElements; i++) 
    {
        pfResult[i] = pfData1[i] + pfData2[i]; 
    }
}
