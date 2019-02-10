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
// oclDotProduct Notes:  
//
// A simple OpenCL API demo application that implements a
// vector dot product computation between 2 float arrays. 
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
//
// Uses 'shr' and 'ocl' functions from oclUtils and shrUtils libraries for compactness. 
// But these are NOT required libs for OpenCL developement in general.
// *********************************************************************

// standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "DotProduct.cl";

// Host buffers for demo
// *********************************************************************
void *srcA, *srcB, *dst;        // Host buffers for OpenCL test
void* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id   *cdDevices;      // OpenCL device
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
size_t szGlobalWorkSize;        // Total # of work items in the 1D range
size_t szLocalWorkSize;		    // # of work items in the 1D work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErrNum;			    // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

// demo config vars
int iNumElements= 1277944;	    // Length of float arrays to process (odd # for illustration)
shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int *gp_argc = NULL;
char ***gp_argv = NULL;

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    gp_argc = &argc;
    gp_argv = &argv;

    shrQAStart(argc, argv);

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("clGetPlatformID...\n"); 

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("clGetPlatformID...\n"); 

    //Get all the devices
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    // Get command line device options and config accordingly
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("\n  # of Compute Units = %u\n", uiNumComputeUnits); 

    // get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");

    // start logs
	cExecutableName = argv[0];
    shrSetLogFileName ("oclDotProduct.txt");
    shrLog("%s Starting...\n\n# of float elements per Array \t= %u\n", argv[0], iNumElements); 

    // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

    // Allocate and initialize host arrays
    shrLog( "Allocate and Init Host Mem...\n"); 
    srcA = (void *)malloc(sizeof(cl_float4) * szGlobalWorkSize);
    srcB = (void *)malloc(sizeof(cl_float4) * szGlobalWorkSize);
    dst = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
    Golden = (void *)malloc(sizeof(cl_float) * iNumElements);
    shrFillArray((float*)srcA, 4 * iNumElements);
    shrFillArray((float*)srcB, 4 * iNumElements);

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get a GPU device
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevices[uiTargetDevice], NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the context
    cxGPUContext = CECL_CREATE_CONTEXT(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n"); 
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    shrLog("CECL_BUFFER (SrcA, SrcB and Dst in Device GMEM)...\n"); 
    cmDevSrcA = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize * 4, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevSrcB = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize * 4, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevDst = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // Create the program
    shrLog("CECL_PROGRAM_WITH_SOURCE...\n"); 
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

        // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    shrLog("CECL_PROGRAM...\n"); 
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclDotProduct.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // Create the kernel
    shrLog("CECL_KERNEL (DotProduct)...\n"); 
    ckKernel = CECL_KERNEL(cpProgram, "DotProduct", &ciErrNum);

    // Set the Argument values
    shrLog("CECL_SET_KERNEL_ARG 0 - 3...\n\n"); 
    ciErrNum = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    shrLog("CECL_WRITE_BUFFER (SrcA and SrcB)...\n"); 
    ciErrNum = CECL_WRITE_BUFFER(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize * 4, srcA, 0, NULL, NULL);
    ciErrNum |= CECL_WRITE_BUFFER(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize * 4, srcB, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch kernel
    shrLog("CECL_ND_RANGE_KERNEL (DotProduct)...\n"); 
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Read back results and check accumulated errors
    shrLog("CECL_READ_BUFFER (Dst)...\n\n"); 
    ciErrNum = CECL_READ_BUFFER(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Compute and compare results for golden-host and report errors and pass/fail
    shrLog("Comparing against Host/C++ computation...\n\n"); 
    DotProductHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
    shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
    int i, j, k;
    for (i = 0, j = 0; i < iNumElements; i++) 
    {
        pfResult[i] = 0.0f;
        for (k = 0; k < 4; k++, j++) 
        {
            pfResult[i] += pfData1[j] * pfData2[j]; 
        } 
    }
}

// Cleanup and exit code
// *********************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if (cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if (cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if (cmDevDst)clReleaseMemObject(cmDevDst);

    // Free host memory
    free(srcA); 
    free(srcB);
    free (dst);
    free(Golden);

    if (cdDevices) free(cdDevices);

    shrQAFinishExit(*gp_argc, (const char **)*gp_argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
