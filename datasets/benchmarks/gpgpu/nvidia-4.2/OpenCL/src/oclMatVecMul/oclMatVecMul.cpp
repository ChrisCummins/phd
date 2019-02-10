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

/* Matrix-vector multiplication: W = M * V.
 * Host code.
 *
 * This sample implements matrix-vector multiplication.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles and optimizatoins, not with the goal of providing
 * the most performant generic kernel for matrix-vector multiplication.
 *
 * CUBLAS provides high-performance matrix-vector multiplication on GPU.
 */

// standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>

#ifndef _WIN32
    typedef uint64_t memsize_t;
#else
    typedef unsigned __int64 memsize_t;
#endif

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "oclMatVecMul.cl";

// Host buffers for demo
// *********************************************************************
float *M, *V, *W;               // Host buffers for M, V, and W
float* Golden;                  // Host buffer for host golden processing cross check

// OpenCL Vars
cl_platform_id cpPlatform;      // OpenCL platform
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_uint uiNumDevices;           // OpenCL device count
cl_device_id* cdDevices;        // OpenCL device list    
cl_uint targetDevice = 0;	    // Default Device to compute on
cl_uint uiNumDevsUsed = 1;      // Number of devices used in this sample      
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_event ceEvent;               // OpenCL event
cl_mem cmM, cmV, cmW;           // OpenCL buffers for M, V, and W
size_t szGlobalWorkSize;        // Total # of work items in the 1D range
size_t szLocalWorkSize;         // # of work items in the 1D work group    
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErrNum;                // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

// demo config vars
int width = 1100;               // Matrix width
int height;                     // Matrix height
shrBOOL bNoPrompt = shrFALSE;
#define MAX_HEIGHT 100000
bool bPassFlag = true;          // accumulator for test status

// Forward Declarations
// *********************************************************************
void MatVecMulHost(const float* M, const float* V, int width, int height, float* W);
bool getTargetDeviceGlobalMemSize(memsize_t* result, const int argc, const char **argv);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main function 
// *********************************************************************
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);
    // get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char **)argv, "noprompt");

    // start logs
	cExecutableName = argv[0];
    shrSetLogFileName ("oclMatVecMul.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // calculate matrix height given GPU memory
    shrLog("Determining Matrix height from available GPU mem...\n");
    memsize_t memsize;
    getTargetDeviceGlobalMemSize(&memsize, argc, (const char **)argv);
    height = memsize/width/16;
    if (height > MAX_HEIGHT)
        height = MAX_HEIGHT;
    shrLog(" Matrix width\t= %u\n Matrix height\t= %u\n\n", width, height); 

    // Allocate and initialize host arrays
    shrLog("Allocate and Init Host Mem...\n\n");
    unsigned int size = width * height;
    unsigned int mem_size_M = size * sizeof(float);
    M = (float*)malloc(mem_size_M);
    unsigned int mem_size_V = width * sizeof(float);
    V = (float*)malloc(mem_size_V);
    unsigned int mem_size_W = height * sizeof(float);
    W = (float*)malloc(mem_size_W);
    shrFillArray(M, size);
    shrFillArray(V, width);
    Golden = (float*)malloc(mem_size_W);
    MatVecMulHost(M, V, width, height, Golden);

    //Get the NVIDIA platform
    shrLog("Get the Platform ID...\n\n");
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    //Get all the devices
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on targetDevice
    shrLog(" # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char **)argv, "device", &targetDevice)== shrTRUE) 
    {
        targetDevice = CLAMP(targetDevice, 0, (uiNumDevices - 1));
    }
    shrLog(" Using Device %u: ", targetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[targetDevice]);  
    cl_uint num_compute_units;
    clGetDeviceInfo(cdDevices[targetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_compute_units), &num_compute_units, NULL);
    shrLog("\n # of Compute Units = %u\n\n", num_compute_units); 

    //Create the context
    shrLog("CECL_CREATE_CONTEXT...\n"); 
    cxGPUContext = CECL_CREATE_CONTEXT(0, uiNumDevsUsed, &cdDevices[targetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n"); 
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[targetDevice], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    shrLog("CECL_BUFFER (M, V and W in device global memory, mem_size_m = %u)...\n", mem_size_M); 
    cmM = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, mem_size_M, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmV = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, mem_size_V, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmW = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, mem_size_W, NULL, &ciErrNum);
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

    // Build the program
    shrLog("CECL_PROGRAM...\n"); 
    ciErrNum = CECL_PROGRAM(cpProgram, uiNumDevsUsed, &cdDevices[targetDevice], "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatVecMul.ptx");
        shrQAFinish(argc, (const char **)argv, QA_FAILED);
        Cleanup(EXIT_FAILURE); 
    }

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    shrLog("CECL_WRITE_BUFFER (M and V)...\n\n"); 
    ciErrNum = CECL_WRITE_BUFFER(cqCommandQueue, cmM, CL_FALSE, 0, mem_size_M, M, 0, NULL, NULL);
    ciErrNum |= CECL_WRITE_BUFFER(cqCommandQueue, cmV, CL_FALSE, 0, mem_size_V, V, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Kernels
    const char* kernels[] = {
        "MatVecMulUncoalesced0",
        "MatVecMulUncoalesced1",
        "MatVecMulCoalesced0",
        "MatVecMulCoalesced1",
        "MatVecMulCoalesced2",
        "MatVecMulCoalesced3" };

    for (int k = 0; k < (int)(sizeof(kernels)/sizeof(char*)); ++k) {
        shrLog("Running with Kernel %s...\n\n", kernels[k]); 

        // Clear result
        shrLog("  Clear result with CECL_WRITE_BUFFER (W)...\n"); 
        memset(W, 0, mem_size_W);
        ciErrNum = CECL_WRITE_BUFFER(cqCommandQueue, cmW, CL_FALSE, 0, mem_size_W, W, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Create the kernel
        shrLog("  CECL_KERNEL...\n"); 
        if (ckKernel) {
            clReleaseKernel(ckKernel);
            ckKernel = 0;
        }
        ckKernel = CECL_KERNEL(cpProgram, kernels[k], &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Set and log Global and Local work size dimensions
        szLocalWorkSize = 256;
        if (k == 0)
            szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, height);  // rounded up to the nearest multiple of the LocalWorkSize
        else
            // Some experiments should be done here for determining the best global work size for a given device
            // We will assume here that we can run 2 work-groups per compute unit
            szGlobalWorkSize = 2 * num_compute_units * szLocalWorkSize;
        shrLog("  Global Work Size \t\t= %u\n  Local Work Size \t\t= %u\n  # of Work Groups \t\t= %u\n", 
               szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

        // Set the Argument values
        shrLog("  CECL_SET_KERNEL_ARG...\n\n");
        int n = 0;
        ciErrNum = CECL_SET_KERNEL_ARG(ckKernel,  n++, sizeof(cl_mem), (void*)&cmM);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, n++, sizeof(cl_mem), (void*)&cmV);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, n++, sizeof(cl_int), (void*)&width);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, n++, sizeof(cl_int), (void*)&height);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, n++, sizeof(cl_mem), (void*)&cmW);
        if (k > 1)
            ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, n++, szLocalWorkSize * sizeof(float), 0);    
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch kernel
        shrLog("  CECL_ND_RANGE_KERNEL (%s)...\n", kernels[k]); 
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, &ceEvent);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Read back results and check accumulated errors
        shrLog("  CECL_READ_BUFFER (W)...\n"); 
        ciErrNum = CECL_READ_BUFFER(cqCommandQueue, cmW, CL_TRUE, 0, mem_size_W, W, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    #ifdef GPU_PROFILING
        // Execution time
        ciErrNum = clWaitForEvents(1, &ceEvent);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        cl_ulong start, end;
        ciErrNum = clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        ciErrNum |= clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        double dSeconds = 1.0e-9 * (double)(end - start);
        shrLog("  Kernel execution time: %.5f s\n\n", dSeconds);
    #endif

        // Compare results for golden-host and report errors and pass/fail
        shrLog("  Comparing against Host/C++ computation...\n\n"); 
        shrBOOL res = shrCompareL2fe(Golden, W, height, 1e-6f);
        shrLog("    GPU Result %s CPU Result within allowable tolerance\n\n", (res == shrTRUE) ? "MATCHES" : "DOESN'T MATCH");
        bPassFlag &= (res == shrTRUE); 

        // Release event
        ciErrNum = clReleaseEvent(ceEvent);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        ceEvent = 0;
    }

    // Master status Pass/Fail (all tests)
    shrQAFinish(argc, (const char **)argv, (bPassFlag ? QA_PASSED : QA_FAILED) );

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

// "Golden" Host processing matrix vector multiplication function for comparison purposes
// *********************************************************************
void MatVecMulHost(const float* M, const float* V, int width, int height, float* W)
{
    for (int i = 0; i < height; ++i) {
        double sum = 0;
        for (int j = 0; j < width; ++j) {
            double a = M[i * width + j];
            double b = V[j];
            sum += a * b;
        }
        W[i] = (float)sum;
    }
}

// Cleanup and exit code
// *********************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");
    if(cdDevices)free(cdDevices);
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(ceEvent)clReleaseEvent(ceEvent);  
    if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if (cmM)clReleaseMemObject(cmM);
    if (cmV)clReleaseMemObject(cmV);
    if (cmW)clReleaseMemObject(cmW);

    // Free host memory
    free(M); 
    free(V);
    free(W);
    free(Golden);

    shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    exit (iExitCode);
}

bool getTargetDeviceGlobalMemSize(memsize_t* result, const int argc, const char **argv)
{
    bool ok = true;
    cl_platform_id    platform     = 0;
    cl_context        context      = 0;
    cl_device_id     *devices      = 0;
    cl_uint           deviceCount  = 0;
    cl_ulong          memsize      = 0;
    cl_int            errnum       = 0;

    //Get the NVIDIA platform
    if (ok)
    {
        shrLog(" oclGetPlatformID...\n");
        errnum = oclGetPlatformID(&cpPlatform);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, "oclGetPlatformID (returned %d).\n", errnum);
            ok = false;
        }
    }

    //Get the devices
    if (ok)
    {
        shrLog(" clGetDeviceIDs");
        errnum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id) );
        errnum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, "clGetDeviceIDs (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Create the OpenCL context
    if (ok)
    {
        shrLog(" CECL_CREATE_CONTEXT...\n");
        context = CECL_CREATE_CONTEXT(0, deviceCount, devices, NULL, NULL, &errnum);
        if (context == (cl_context)0) 
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, "CECL_CREATE_CONTEXT (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Select target device (device 0 by default)
    if (ok)
    {
        char *device = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "device", &device))
        {
            targetDevice = (cl_uint)atoi(device);
            if (targetDevice >= deviceCount)
            {
                shrLogEx(LOGBOTH | ERRORMSG, 0, "invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
            free(device);
    }

    // Query target device for maximum memory allocation
    if (ok)
    {
        shrLog(" clGetDeviceInfo...\n"); 
        errnum = clGetDeviceInfo(devices[targetDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, "clGetDeviceInfo (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Save the result
    if (ok)
    {
        *result = (memsize_t)memsize;
    }

    // Cleanup
    if (devices)
        free(devices);
    if (context)
        clReleaseContext(context);
    return ok;
}
