#include <cecl.h>
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

//Standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>

#include "oclSortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
//Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_platform_id cpPlatform;
    cl_device_id cdDevice;
    cl_context cxGPUContext;
    cl_command_queue cqCommandQueue;
    cl_mem d_InputKey, d_InputVal, d_OutputKey, d_OutputVal;

    cl_int ciErrNum;
    uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;

    const uint dir = 1;
    const uint N = 1048576;
    const uint numValues = 65536;

    shrQAStart(argc, (char **)argv);

    // set logfile name and start logs
    shrSetLogFileName ("oclSortingNetworks.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    shrLog("Initializing data...\n");
        h_InputKey      = (uint *)malloc(N * sizeof(uint));
        h_InputVal      = (uint *)malloc(N * sizeof(uint));
        h_OutputKeyGPU  = (uint *)malloc(N * sizeof(uint));
        h_OutputValGPU  = (uint *)malloc(N * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < N; i++)
            h_InputKey[i] = rand() % numValues;
        fillValues(h_InputVal, N);

    shrLog("Initializing OpenCL...\n");
        //Get the NVIDIA platform
        ciErrNum = oclGetPlatformID(&cpPlatform);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Get the devices
        ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Create the context
        cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Create a command-queue
        cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Initializing OpenCL bitonic sorter...\n");
        initBitonicSort(cxGPUContext, cqCommandQueue, argv);

    shrLog("Creating OpenCL memory objects...\n\n");
        d_InputKey = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_uint), h_InputKey, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_InputVal = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_uint), h_InputVal, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_OutputKey = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(cl_uint), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_OutputVal = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(cl_uint), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    //Temp storage for key array validation routine
    uint *srcHist = (uint *)malloc(numValues * sizeof(uint));
    uint *resHist = (uint *)malloc(numValues * sizeof(uint));

#ifdef GPU_PROFILING
    cl_event startTime, endTime;
#endif

    int globalFlag = 1;// init pass/fail flag to pass
    for(uint arrayLength = 64; arrayLength <= N; arrayLength *= 2){
        shrLog("Test array length %u (%u arrays in the batch)...\n", arrayLength, N / arrayLength);

#ifdef GPU_PROFILING
            clFinish(cqCommandQueue);
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startTime);
            oclCheckError(ciErrNum, CL_SUCCESS);
            shrDeltaT(0);
#endif

            size_t szWorkgroup = bitonicSort(
                NULL,
                d_OutputKey,
                d_OutputVal,
                d_InputKey,
                d_InputVal,
                N / arrayLength,
                arrayLength,
                dir
            );
            oclCheckError(szWorkgroup > 0, true); 

#ifdef GPU_PROFILING
            if (arrayLength == N)
            {
                ciErrNum = clEnqueueMarker(cqCommandQueue, &endTime);
                oclCheckError(ciErrNum, CL_SUCCESS);
                clFinish(cqCommandQueue);
                double timerValue = shrDeltaT(0);
                shrLogEx(LOGBOTH | MASTER, 0, "oclSortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n", 
                       (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);

                cl_ulong startTimeVal = 0, endTimeVal = 0;
                ciErrNum = clGetEventProfilingInfo(
                    startTime, 
                    CL_PROFILING_COMMAND_END, 
                    sizeof(cl_ulong),
                    &startTimeVal,
                    NULL
                );

                ciErrNum = clGetEventProfilingInfo(
                    endTime, 
                    CL_PROFILING_COMMAND_END, 
                    sizeof(cl_ulong),
                    &endTimeVal,
                    NULL
                );

                shrLog("OpenCL time: %.5f s\n", 1.0e-9 * (double)(endTimeVal - startTimeVal));
            }
#endif

        //Reading back results from device to host
        ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_OutputKey, CL_TRUE, 0, N * sizeof(cl_uint), h_OutputKeyGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_OutputVal, CL_TRUE, 0, N * sizeof(cl_uint), h_OutputValGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Check if keys array is not corrupted and properly ordered
        int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, dir, srcHist, resHist);

        //Check if values array is not corrupted
        int valuesFlag = validateSortedValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);

        // accumulate any error or failure
        globalFlag = globalFlag && keysFlag && valuesFlag;
    }

    // Start Cleanup
    shrLog("Shutting down...\n");
        //Discard temp storage for key validation routine
        free(srcHist);
        free(resHist);

        //Release kernels and program
        closeBitonicSort();

        //Release other OpenCL Objects
        ciErrNum  = clReleaseMemObject(d_OutputVal);
        ciErrNum |= clReleaseMemObject(d_OutputKey);
        ciErrNum |= clReleaseMemObject(d_InputVal);
        ciErrNum |= clReleaseMemObject(d_InputKey);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(h_OutputValGPU);
        free(h_OutputKeyGPU);
        free(h_InputVal);
        free(h_InputKey);

    // finish
    // pass or fail (cumulative... all tests in the loop)
    shrQAFinishExit(argc, (const char **)argv, globalFlag ? QA_PASSED : QA_FAILED);

        //Finish
        shrEXIT(argc, argv);
}
