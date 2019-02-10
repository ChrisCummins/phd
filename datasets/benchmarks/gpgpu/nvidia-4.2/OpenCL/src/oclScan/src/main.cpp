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

#include <oclUtils.h>
#include <shrQATest.h>

#include "oclScan_common.h"

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    shrQAStart(argc, (char **)argv);

    // Start logs
    shrSetLogFileName ("oclScan.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    cl_platform_id cpPlatform;       //OpenCL platform
    cl_device_id cdDevice;           //OpenCL device
    cl_context      cxGPUContext;    //OpenCL context
    cl_command_queue cqCommandQueue; //OpenCL command que
    cl_mem d_Input, d_Output;        //OpenCL memory buffer objects

    cl_int ciErrNum;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    const uint N = 13 * 1048576 / 2;

    shrLog("Allocating and initializing host arrays...\n");
        h_Input     = (uint *)malloc(N * sizeof(uint));
        h_OutputCPU = (uint *)malloc(N * sizeof(uint));
        h_OutputGPU = (uint *)malloc(N * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < N; i++)
            h_Input[i] = rand();

    shrLog("Initializing OpenCL...\n");
        //Get the NVIDIA platform
        ciErrNum = oclGetPlatformID(&cpPlatform);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Get a GPU device
        ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Create the context
        cxGPUContext = CECL_CREATE_CONTEXT(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Create a command-queue
        cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, 0, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Initializing OpenCL scan...\n");
        initScan(cxGPUContext, cqCommandQueue, argv);

    shrLog("Creating OpenCL memory objects...\n\n");
        d_Input = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(uint), h_Input, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Output = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(uint), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    int globalFlag = 1; // init pass/fail flag to pass
    size_t szWorkgroup;
    const int iCycles = 100;
    shrLog("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);
    for(uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength *= 2)
    {
        shrLog("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
            clFinish(cqCommandQueue);
            shrDeltaT(0);
            for (int i = 0; i<iCycles; i++)
            {
                szWorkgroup = scanExclusiveShort(
                    cqCommandQueue,
                    d_Output,
                    d_Input,
                    N / arrayLength,
                    arrayLength
                );
            }
            clFinish(cqCommandQueue);
            double timerValue = shrDeltaT(0)/(double)iCycles;

        shrLog("Validating the results...\n"); 
            shrLog(" ...reading back OpenCL memory\n");
                ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_Output, CL_TRUE, 0, N * sizeof(uint), h_OutputGPU, 0, NULL, NULL);
                oclCheckError(ciErrNum, CL_SUCCESS);

            shrLog(" ...scanExclusiveHost()\n");
                scanExclusiveHost(
                    h_OutputCPU,
                    h_Input,
                    N / arrayLength,
                    arrayLength
                );

            // Compare GPU results with CPU results and accumulate error for this test
            shrLog(" ...comparing the results\n");
                int localFlag = 1;
                for(uint i = 0; i < N; i++)
                {
                    if(h_OutputCPU[i] != h_OutputGPU[i])
                    {
                        localFlag = 0;
                        break;
                    }
                }

            // Log message on individual test result, then accumulate to global flag
            shrLog(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
            globalFlag = globalFlag && localFlag;

            #ifdef GPU_PROFILING
                if (arrayLength == MAX_SHORT_ARRAY_SIZE)
                {
                    shrLog("\n");
                    shrLogEx(LOGBOTH | MASTER, 0, "oclScan-Short, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n", 
                           (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
                    shrLog("\n");
                }
            #endif
    }

    shrLog("*** Running GPU scan for large arrays (%d identical iterations)...\n\n", iCycles);
    for(uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength *= 2)
    {
        shrLog("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
            clFinish(cqCommandQueue);
            shrDeltaT(0);
            for (int i = 0; i<iCycles; i++)
            {
                szWorkgroup = scanExclusiveLarge(
                    cqCommandQueue,
                    d_Output,
                    d_Input,
                    N / arrayLength,
                    arrayLength
                );
            }
            clFinish(cqCommandQueue);
            double timerValue = shrDeltaT(0)/(double)iCycles;

        shrLog("Validating the results...\n"); 
            shrLog(" ...reading back OpenCL memory\n");
                ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_Output, CL_TRUE, 0, N * sizeof(uint), h_OutputGPU, 0, NULL, NULL);
                oclCheckError(ciErrNum, CL_SUCCESS);

            shrLog(" ...scanExclusiveHost()\n");
                scanExclusiveHost(
                    h_OutputCPU,
                    h_Input,
                    N / arrayLength,
                    arrayLength
                );

            // Compare GPU results with CPU results and accumulate error for this test
            shrLog(" ...comparing the results\n");
                int localFlag = 1;
                for(uint i = 0; i < N; i++)
                {
                    if(h_OutputCPU[i] != h_OutputGPU[i])
                    {
                        localFlag = 0;
                        break;
                    }
                }

            // Log message on individual test result, then accumulate to global flag
            shrLog(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
            globalFlag = globalFlag && localFlag;

            #ifdef GPU_PROFILING
                if (arrayLength == MAX_LARGE_ARRAY_SIZE)
                {
                    shrLog("\n");
                    shrLogEx(LOGBOTH | MASTER, 0, "oclScan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n", 
                           (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
                    shrLog("\n");
                }
            #endif
    }

    shrLog("Shutting down...\n");
        //Release kernels and program
        closeScan();

        //Release other OpenCL Objects
        ciErrNum  = clReleaseMemObject(d_Output);
        ciErrNum |= clReleaseMemObject(d_Input);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(h_OutputGPU);
        free(h_OutputCPU);
        free(h_Input);


    // finish
    // pass or fail (cumulative... all tests in the loop)
    shrQAFinishExit(argc, (const char **)argv, globalFlag ? QA_PASSED : QA_FAILED);

        //Finish
        shrEXIT(argc, argv);
}
