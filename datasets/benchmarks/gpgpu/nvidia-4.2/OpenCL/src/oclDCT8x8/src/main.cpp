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

//Standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>
#include "oclDCT8x8_common.h"


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cl_platform_id cpPlatform;       //OpenCL platform
    cl_device_id cdDevice;           //OpenCL device
    cl_context       cxGPUContext;   //OpenCL context
    cl_command_queue cqCommandQueue; //OpenCL command que
    cl_mem      d_Input, d_Output;   //OpenCL memory buffer objects

    cl_int ciErrNum;

    float *h_Input, *h_OutputCPU, *h_OutputGPU;

    const uint
        imageW = 2048,
        imageH = 2048,
        stride = 2048;

    const int dir = DCT_FORWARD;

    shrQAStart(argc, argv);

    // set logfile name and start logs
    shrSetLogFileName ("oclDCT8x8.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    shrLog("Allocating and initializing host memory...\n");
        h_Input     = (float *)malloc(imageH * stride * sizeof(float));
        h_OutputCPU = (float *)malloc(imageH * stride * sizeof(float));
        h_OutputGPU = (float *)malloc(imageH * stride * sizeof(float));
        srand(2009);
        for(uint i = 0; i < imageH; i++)
            for(uint j = 0; j < imageW; j++)
                h_Input[i * stride + j] = (float)rand() / (float)RAND_MAX;

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
        cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Initializing OpenCL DCT 8x8...\n");
        initDCT8x8(cxGPUContext, cqCommandQueue, (const char **)argv);

    shrLog("Creating OpenCL memory objects...\n");
        d_Input = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageH * stride *  sizeof(cl_float), h_Input, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Output = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, imageH * stride * sizeof(cl_float), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Performing DCT8x8 of %u x %u image...\n\n", imageH, imageW);
        //Just a single iteration or a warmup iteration
        DCT8x8(
            cqCommandQueue,
            d_Output,
            d_Input,
            stride,
            imageH,
            imageW,
            dir
        );

#ifdef GPU_PROFILING
    const int numIterations = 16;
    cl_event startMark, endMark;
    ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
    ciErrNum |= clFinish(cqCommandQueue);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrDeltaT(0);

    for(int iter = 0; iter < numIterations; iter++)
        DCT8x8(
            NULL,
            d_Output,
            d_Input,
            stride,
            imageH,
            imageW,
            dir
        );

    ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
    ciErrNum |= clFinish(cqCommandQueue);
    shrCheckError(ciErrNum, CL_SUCCESS);

    //Calculate performance metrics by wallclock time
    double gpuTime = shrDeltaT(0) / (double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclDCT8x8, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", 
            (1.0e-6 * (double)(imageW * imageH)/ gpuTime), gpuTime, (imageW * imageH), 1, 0); 

    //Get profiler time
    cl_ulong startTime = 0, endTime = 0;
    ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
    ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog("\nOpenCL time: %.5f s\n\n", 1.0e-9 * ((double)endTime - (double)startTime) / (double)numIterations);
#endif

    shrLog("Reading back OpenCL results...\n");
        ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_Output, CL_TRUE, 0, imageH * stride * sizeof(cl_float), h_OutputGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Comparing against Host/C++ computation...\n"); 
        DCT8x8CPU(h_OutputCPU, h_Input, stride, imageH, imageW, dir);
        double sum = 0, delta = 0;
        double L2norm;
        for(uint i = 0; i < imageH; i++)
            for(uint j = 0; j < imageW; j++){
                sum += h_OutputCPU[i * stride + j] * h_OutputCPU[i * stride + j];
                delta += (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]) * (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]);
            }
        L2norm = sqrt(delta / sum);
        shrLog("Relative L2 norm: %.3e\n\n", L2norm);

    shrLog("Shutting down...\n");
        //Release kernels and program
        closeDCT8x8();

        //Release other OpenCL objects
        ciErrNum  = clReleaseMemObject(d_Output);
        ciErrNum |= clReleaseMemObject(d_Input);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(h_OutputGPU);
        free(h_OutputCPU);
        free(h_Input);

        //Finish
        shrQAFinishExit(argc, (const char **)argv, (L2norm < 1E-6) ? QA_PASSED : QA_FAILED);
}
