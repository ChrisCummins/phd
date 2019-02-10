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

#include "oclConvolutionSeparable_common.h"

#include <shrQATest.h>

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cl_platform_id   cpPlatform;
    cl_device_id*    cdDevices = NULL;
    cl_context       cxGPUContext;                        //OpenCL context
    cl_command_queue cqCommandQueue;                //OpenCL command queue
    cl_mem c_Kernel, d_Input, d_Buffer, d_Output;   //OpenCL memory buffer objects
    cl_float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

    cl_int ciErrNum;

    const unsigned int imageW = 3072;
    const unsigned int imageH = 3072;

    shrQAStart(argc, argv);

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

    // set logfile name and start logs
    shrSetLogFileName ("oclConvolutionSeparable.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    shrLog("Allocating and initializing host memory...\n");
        h_Kernel    = (cl_float *)malloc(KERNEL_LENGTH * sizeof(cl_float));
        h_Input     = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_Buffer    = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_OutputCPU = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_OutputGPU = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));

        srand(2009);
        for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
            h_Kernel[i] = (cl_float)(rand() % 16);

        for(unsigned int i = 0; i < imageW * imageH; i++)
            h_Input[i] = (cl_float)(rand() % 16);

    shrLog("Initializing OpenCL...\n");
        //Get the NVIDIA platform
        ciErrNum = oclGetPlatformID(&cpPlatform);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Get the devices
        ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevices[uiTargetDevice], NULL);

        //Create the context
        cxGPUContext = CECL_CREATE_CONTEXT(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //Create a command-queue
        cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiTargetDevice], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Initializing OpenCL separable convolution...\n");
        initConvolutionSeparable(cxGPUContext, cqCommandQueue, (const char **)argv);

    shrLog("Creating OpenCL memory objects...\n");
        c_Kernel = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KERNEL_LENGTH * sizeof(cl_float), h_Kernel, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Input = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageW * imageH * sizeof(cl_float), h_Input, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Buffer = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, imageW * imageH * sizeof(cl_float), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Output = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, imageW * imageH * sizeof(cl_float), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Applying separable convolution to %u x %u image...\n\n", imageW, imageH);
        //Just a single run or a warmup iteration
        convolutionRows(
            NULL,
            d_Buffer,
            d_Input,
            c_Kernel,
            imageW,
            imageH
        );

        convolutionColumns(
            NULL,
            d_Output,
            d_Buffer,
            c_Kernel,
            imageW,
            imageH
        );

#ifdef GPU_PROFILING
    const int numIterations = 16;
    cl_event startMark, endMark;
    ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
    ciErrNum |= clFinish(cqCommandQueue);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrDeltaT(0);

    for(int iter = 0; iter < numIterations; iter++){
        convolutionRows(
            cqCommandQueue,
            d_Buffer,
            d_Input,
            c_Kernel,
            imageW,
            imageH
        );

        convolutionColumns(
            cqCommandQueue,
            d_Output,
            d_Buffer,
            c_Kernel,
            imageW,
            imageH
        );
    }
    ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
    ciErrNum |= clFinish(cqCommandQueue);
    shrCheckError(ciErrNum, CL_SUCCESS);

    //Calculate performance metrics by wallclock time
    double gpuTime = shrDeltaT(0) / (double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclConvolutionSeparable, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
            (1.0e-6 * (double)(imageW * imageH)/ gpuTime), gpuTime, (imageW * imageH), 1, 0);

    //Get OpenCL profiler  info
    cl_ulong startTime = 0, endTime = 0;
    ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
    ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog("\nOpenCL time: %.5f s\n\n", 1.0e-9 * ((double)endTime - (double)startTime)/ (double)numIterations);
#endif

    shrLog("Reading back OpenCL results...\n\n");
        ciErrNum = CECL_READ_BUFFER(cqCommandQueue, d_Output, CL_TRUE, 0, imageW * imageH * sizeof(cl_float), h_OutputGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Comparing against Host/C++ computation...\n"); 
        convolutionRowHost(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
        convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
        double sum = 0, delta = 0;
        double L2norm;
        for(unsigned int i = 0; i < imageW * imageH; i++){
            delta += (h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]);
            sum += h_OutputCPU[i] * h_OutputCPU[i];
        }
        L2norm = sqrt(delta / sum);
        shrLog("Relative L2 norm: %.3e\n\n", L2norm);

    // cleanup
    closeConvolutionSeparable();
    ciErrNum  = clReleaseMemObject(d_Output);
    ciErrNum |= clReleaseMemObject(d_Buffer);
    ciErrNum |= clReleaseMemObject(d_Input);
    ciErrNum |= clReleaseMemObject(c_Kernel);
    ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);

    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

   if(cdDevices)free(cdDevices);

    // finish
    shrQAFinishExit(argc, (const char **)argv, (L2norm < 1e-6) ? QA_PASSED : QA_FAILED);
}
