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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication with multi GPU support.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 *
 */

// standard utilities and system includes
#include <oclUtils.h>
#include <shrQATest.h>

// project include
#include "matrixMul.h"

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 8;

// Globals for size of matrices
unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
int iSizeMultiple = 1;

// global variables
cl_context cxGPUContext;
cl_kernel multiplicationKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, const char** argv);
void printDiff(float*, float*, int, int, int, float);
void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C );

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);

    // start the logs
    shrSetLogFileName ("oclMatrixMul.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // run the code
    bool bOK = (runTest(argc, (const char **)argv) == CL_SUCCESS);
    shrLog("%s\n\n", (bOK ? "PASSED" : "FAILED"));

    // finish
    shrQAFinishExit(argc, (const char **)argv, (bOK ? QA_PASSED : QA_FAILED));
}

void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C )
{
    cl_mem d_A[MAX_GPU_COUNT];
    cl_mem d_C[MAX_GPU_COUNT];
    cl_mem d_B[MAX_GPU_COUNT];

    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];

    // Start the computation on each available GPU
    
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = uiHA / ciDeviceCount;

    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];

    workOffset[0] = 0;
    for(unsigned int i=0; i < ciDeviceCount; ++i) 
    {
        // Input buffer
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (uiHA - workOffset[i]);        

        d_A[i] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float) * uiWA, NULL,NULL);

        // Copy only assigned rows from host to device
        clEnqueueCopyBuffer(commandQueue[i], h_A, d_A[i], workOffset[i] * sizeof(float) * uiWA, 
                            0, workSize[i] * sizeof(float) * uiWA, 0, NULL, NULL);        
        
        // create OpenCL buffer on device that will be initiatlize from the host memory on first use
        // on device
        d_B[i] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                mem_size_B, h_B_data, NULL);

        // Output buffer
        d_C[i] = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY,  workSize[i] * uiWC * sizeof(float), NULL,NULL);
              
        // set the args values
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 0, sizeof(cl_mem), (void *) &d_C[i]);
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 1, sizeof(cl_mem), (void *) &d_A[i]);
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 2, sizeof(cl_mem), (void *) &d_B[i]);
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 3, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 4, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 5, sizeof(cl_int), (void *) &uiWA);
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 6, sizeof(cl_int), (void *) &uiWB);
        CECL_SET_KERNEL_ARG(multiplicationKernel[i], 7, sizeof(cl_int), (void *) &workSize[i]);

        if(i+1 < ciDeviceCount)
            workOffset[i + 1] = workOffset[i] + workSize[i];
    }
    
    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, uiWC), shrRoundUp(BLOCK_SIZE, workSize[0])};
    
    // Launch kernels on devices
#ifdef GPU_PROFILING
	
	int nIter = 30;

    for (int j = -1; j < nIter; j++) 
    {
        // Sync all queues to host and start timer first time through loop
        if(j == 0){
            for(unsigned int i = 0; i < ciDeviceCount; i++) 
            {
                clFinish(commandQueue[i]);
            }

            shrDeltaT(0);
        }
#endif
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {
			// Multiplication - non-blocking execution:  launch and push to device(s)
			globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[i]);
			CECL_ND_RANGE_KERNEL(commandQueue[i], multiplicationKernel[i], 2, 0, globalWorkSize, localWorkSize,
				                   0, NULL, &GPUExecution[i]);
            clFlush(commandQueue[i]);
		}

#ifdef GPU_PROFILING
    }
#endif

    // sync all queues to host
	for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
		clFinish(commandQueue[i]);
	}

#ifdef GPU_PROFILING

    // stop and log timer 
    double dSeconds = shrDeltaT(0)/(double)nIter;
    double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    double gflops = 1.0e-9 * dNumOps/dSeconds;
    shrLogEx(LOGBOTH | MASTER, 0, "oclMatrixMul, Throughput = %.4f GFlops/s, Time = %.5f s, Size = %.0f, NumDevsUsed = %d, Workgroup = %u\n", 
            gflops, dSeconds, dNumOps, ciDeviceCount, localWorkSize[0] * localWorkSize[1]);

    // Print kernel timing per GPU
    shrLog("\n");
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        shrLog("  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution[i]));
    }
    shrLog("\n");
#endif

    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        // Non-blocking copy of result from device to host
        CECL_READ_BUFFER(commandQueue[i], d_C[i], CL_FALSE, 0, uiWC * sizeof(float) * workSize[i], 
                            h_C + workOffset[i] * uiWC, 0, NULL, &GPUDone[i]);
    }

	// CPU sync with GPU
    clWaitForEvents(ciDeviceCount, GPUDone);


    // Release mem and event objects    
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        clReleaseMemObject(d_A[i]);
        clReleaseMemObject(d_C[i]);
        clReleaseMemObject(d_B[i]);
	    clReleaseEvent(GPUExecution[i]);
	    clReleaseEvent(GPUDone[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for 
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, const char** argv)
{
    cl_platform_id cpPlatform = NULL;
    cl_uint ciDeviceCount = 0;
    cl_device_id *cdDevices = NULL;
    cl_int ciErrNum = CL_SUCCESS;

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    //Create the context
    cxGPUContext = CECL_CREATE_CONTEXT(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: Failed to create OpenCL context!\n");
        return ciErrNum;
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
           
            // create command queue
            commandQueue[ciDeviceCount] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in CECL_CREATE_COMMAND_QUEUE call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
                
            ++ciDeviceCount;

            #ifdef WIN32
                deviceStr = strtok_s (NULL," ,.-", &next_token);
            #else            
                deviceStr = strtok (NULL," ,.-");
            #endif
        }

        free(deviceList);
    } 
    else 
    {
        // Find out how many GPU's to compute on all available GPUs
	    size_t nDeviceBytes;
	    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            shrLog(" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        } 

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            shrLog("Device %d: ", i);
            oclPrintDevName(LOGBOTH, device);            
            shrLog("\n");

            // create command queue
            commandQueue[i] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in CECL_CREATE_COMMAND_QUEUE call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
        }
    }

    // Optional Command-line multiplier for matrix sizes
    shrGetCmdLineArgumenti(argc, (const char**)argv, "sizemult", &iSizeMultiple); 
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
    cl_mem h_A = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				    mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: CECL_BUFFER\n");
        return ciErrNum;
    }

    // Program Setup
    size_t program_length;
    const char* header_path = shrFindFilePath("matrixMul.h", argv[0]);
    oclCheckError(header_path != NULL, shrTRUE);
    char* header = oclLoadProgSource(header_path, "", &program_length);
    if(!header)
    {
        shrLog("Error: Failed to load the header %s!\n", header_path);
        return -1000;
    }
    const char* source_path = shrFindFilePath("matrixMul.cl", argv[0]);
    oclCheckError(source_path != NULL, shrTRUE);
    char *source = oclLoadProgSource(source_path, header, &program_length);
    if(!source)
    {
        shrLog("Error: Failed to load compute program %s!\n", source_path);
        return -2000;
    }

    // create the program
    cl_program cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&source, 
                                                    &program_length, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: Failed to create program\n");
        return ciErrNum;
    }
    free(header);
    free(source);
    
    // build the program
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
        return ciErrNum;
    }

    // write out PTX if requested on the command line
    if(shrCheckCmdLineFlag(argc, argv, "dump-ptx") )
    {
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
    }

    // Create Kernel
    for(unsigned int i = 0; i < ciDeviceCount; ++i) {
        multiplicationKernel[i] = CECL_KERNEL(cpProgram, "matrixMul", &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            shrLog("Error: Failed to create kernel\n");
            return ciErrNum;
        }
    }
        
    // Run multiplication on 1..deviceCount GPUs to compare improvement
    shrLog("\nRunning Computations on 1 - %d GPU's...\n\n", ciDeviceCount);
    for(unsigned int k = 1; k <= ciDeviceCount; ++k) 
    {
        matrixMulGPU(k, h_A, h_B_data, mem_size_B, h_C);
    }

    // compute reference solution
    shrLog("Comparing results with CPU computation... \n\n");
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A_data, h_B_data, uiHA, uiWA, uiWB);

    // check result
    shrBOOL res = shrCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    if (res != shrTRUE) 
    {
        printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
    }

    // clean up OCL resources
    ciErrNum = clReleaseMemObject(h_A);
    for(unsigned int k = 0; k < ciDeviceCount; ++k) 
    {
        ciErrNum |= clReleaseKernel( multiplicationKernel[k] );
        ciErrNum |= clReleaseCommandQueue( commandQueue[k] );
    }
    ciErrNum |= clReleaseProgram(cpProgram);
    ciErrNum |= clReleaseContext(cxGPUContext);
    if(ciErrNum != CL_SUCCESS)
    {
        shrLog("Error: Failure releasing OpenCL resources: %d\n", ciErrNum);
        return ciErrNum;
    }

    // clean up memory
    free(h_A_data);
    free(h_B_data);
    free(h_C);
    free(reference);
    
    return ((shrTRUE == res) ? CL_SUCCESS : -3000);
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}
