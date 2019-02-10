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

#include "FDTD3dGPU.h"

#include <oclUtils.h>
#include <iostream>
#include <algorithm>

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    bool ok = true;
    cl_platform_id    platform     = 0;
    cl_context        context      = 0;
    cl_device_id     *devices      = 0;
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_ulong          memsize      = 0;
    cl_int            errnum       = 0;

    // Get the NVIDIA platform
    if (ok)
    {
        shrLog(" oclGetPlatformID\n"); 
        errnum = oclGetPlatformID(&platform);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("oclGetPlatformID (no platforms found).\n");
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the platform
    if (ok)
    {
        shrLog(" clGetDeviceIDs\n"); 
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id) );
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
        if ((deviceCount == 0) || (errnum != CL_SUCCESS))
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetDeviceIDs (returned error or no devices found).\n");
            ok = false;
        }
    }

    // Create the OpenCL context
    if (ok)
    {
        shrLog(" CECL_CREATE_CONTEXT\n");
        context = CECL_CREATE_CONTEXT(0, deviceCount, devices, NULL, NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_CREATE_CONTEXT (returned %d).\n", errnum);
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
                shrLogEx(LOGBOTH | ERRORMSG, -2000, STDERROR);
                shrLog("invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
        {
            free(device);
        }
    }

    // Query target device for maximum memory allocation
    if (ok)
    {
        shrLog(" clGetDeviceInfo\n"); 
        errnum = clGetDeviceInfo(devices[targetDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetDeviceInfo (returned %d).\n", errnum);
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

bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    bool ok = true;
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    cl_context        context      = 0;
    cl_platform_id    platform     = 0;
    cl_device_id     *devices      = 0;
    cl_command_queue  commandQueue = 0;
    cl_mem            bufferOut    = 0;
    cl_mem            bufferIn     = 0;
    cl_mem            bufferCoeff  = 0;
    cl_program        program      = 0;
    cl_kernel         kernel       = 0;
    cl_event         *kernelEvents = 0;
#ifdef GPU_PROFILING
    cl_ulong          kernelEventStart;
    cl_ulong          kernelEventEnd;
#endif
    double            hostElapsedTimeS;
    char             *cPathAndName = 0;
    char             *cSourceCL = 0;
    size_t            szKernelLength;
    size_t            globalWorkSize[2];
    size_t            localWorkSize[2];
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_int            errnum       = 0;
    char              buildOptions[128];

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    const int profileTimesteps = timesteps - 1;
    if (ok)
    {
        if (profileTimesteps < 1)
        {
            shrLog(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
        }
    }
#endif

    // Get the NVIDIA platform
    if (ok)
    {
        shrLog(" oclGetPlatformID...\n");
        errnum = oclGetPlatformID(&platform);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("oclGetPlatformID (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the platform
    if (ok)
    {
        shrLog(" clGetDeviceIDs");
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id) );
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetDeviceIDs (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Create the OpenCL context
    if (ok)
    {
        shrLog(" CECL_CREATE_CONTEXT...\n");
        context = CECL_CREATE_CONTEXT(0, deviceCount, devices, NULL, NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_CREATE_CONTEXT (returned %d).\n", errnum);
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
                shrLogEx(LOGBOTH | ERRORMSG, -2001, STDERROR);
                shrLog("invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
        {
            free(device);
        }
    }

    // Create a command-queue
    if (ok)
    {
        shrLog(" CECL_CREATE_COMMAND_QUEUE\n"); 
        commandQueue = CECL_CREATE_COMMAND_QUEUE(context, devices[targetDevice], CL_QUEUE_PROFILING_ENABLE, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_CREATE_COMMAND_QUEUE (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Create memory buffer objects
    if (ok)
    {
        shrLog(" CECL_BUFFER bufferOut\n"); 
        bufferOut = CECL_BUFFER(context, CL_MEM_READ_WRITE, paddedVolumeSize * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_BUFFER (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" CECL_BUFFER bufferIn\n"); 
        bufferIn = CECL_BUFFER(context, CL_MEM_READ_WRITE, paddedVolumeSize * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_BUFFER (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" CECL_BUFFER bufferCoeff\n"); 
        bufferCoeff = CECL_BUFFER(context, CL_MEM_READ_ONLY, (radius + 1) * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_BUFFER (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Load the kernel from file
    if (ok)
    {
        shrLog(" shrFindFilePath\n"); 
        cPathAndName = shrFindFilePath(clSourceFile, argv[0]);
        if (cPathAndName == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2002, STDERROR);
            shrLog("shrFindFilePath returned null.\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" oclLoadProgSource\n"); 
        cSourceCL = oclLoadProgSource(cPathAndName, "// Preamble\n", &szKernelLength);
        if (cSourceCL == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2003, STDERROR);
            shrLog("oclLoadProgSource returned null.\n");
            ok = false;
        }
    }

    // Create the program
    if (ok)
    {
        shrLog(" CECL_PROGRAM_WITH_SOURCE\n");
        program = CECL_PROGRAM_WITH_SOURCE(context, 1, (const char **)&cSourceCL, &szKernelLength, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_PROGRAM_WITH_SOURCE (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Check for a command-line specified work group size
    size_t userWorkSize;
    int    localWorkMaxY;
	if (ok)
    {
        int userWorkSizeInt;
        if (shrGetCmdLineArgumenti(argc, argv, "work-group-size", &userWorkSizeInt))
        {
            // We can't clamp to CL_KERNEL_WORK_GROUP_SIZE yet since that is
            // dependent on the build.
            if (userWorkSizeInt < k_localWorkMin || userWorkSizeInt > k_localWorkMax)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -2004, STDERROR);
                shrLog("invalid work group size specified on command line (must be between %d and %d).\n", k_localWorkMin, k_localWorkMax);
                ok = false;
            }
            // Constrain to a multiple of k_localWorkX
            userWorkSize = (userWorkSizeInt / k_localWorkX * k_localWorkX);
        }
        else
        {
            userWorkSize = k_localWorkY * k_localWorkX;
        }
        
        // Divide by k_localWorkX (integer division to clamp)
        localWorkMaxY = userWorkSize / k_localWorkX;
    }

    // Build the program
    if (ok)
    {
#ifdef WIN32
        if (sprintf_s(buildOptions, sizeof(buildOptions), "-DRADIUS=%d -DMAXWORKX=%d -DMAXWORKY=%d -cl-fast-relaxed-math", radius, k_localWorkX, localWorkMaxY) < 0)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2005, STDERROR);
            shrLog("sprintf_s (failed).\n");
            ok = false;
        }
#else
        if (snprintf(buildOptions, sizeof(buildOptions), "-DRADIUS=%d -DMAXWORKX=%d -DMAXWORKY=%d -cl-fast-relaxed-math", radius, k_localWorkX, localWorkMaxY) < 0)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2005, STDERROR);
            shrLog("snprintf (failed).\n");
            ok = false;
        }
#endif
    }
    if (ok)
    {
        shrLog(" CECL_PROGRAM (%s)\n", buildOptions);
        errnum = CECL_PROGRAM(program, 0, NULL, buildOptions, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            char buildLog[10240];
            clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_PROGRAM (returned %d).\n", errnum);
            shrLog("Log:\n%s\n", buildLog);
            ok = false;
        }
    }

    // Create the kernel
    if (ok)
    {
        shrLog(" CECL_KERNEL\n");
        kernel = CECL_KERNEL(program, "FiniteDifferences", &errnum);
        if (kernel == (cl_kernel)NULL || errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_KERNEL (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the maximum work group size
    size_t maxWorkSize;
    if (ok)
    {
        shrLog(" CECL_GET_KERNEL_WORK_GROUP_INFO\n");
        errnum = CECL_GET_KERNEL_WORK_GROUP_INFO(kernel, devices[targetDevice], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkSize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_GET_KERNEL_WORK_GROUP_INFO (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Set the work group size
    if (ok)
    {
		userWorkSize = CLAMP(userWorkSize, k_localWorkMin, maxWorkSize);
        localWorkSize[0] = k_localWorkX;
        localWorkSize[1] = userWorkSize / k_localWorkX;
        globalWorkSize[0] = localWorkSize[0] * (unsigned int)ceil((float)dimx / localWorkSize[0]);
        globalWorkSize[1] = localWorkSize[1] * (unsigned int)ceil((float)dimy / localWorkSize[1]);
        shrLog(" set local work group size to %dx%d\n", localWorkSize[0], localWorkSize[1]);
        shrLog(" set total work size to %dx%d\n", globalWorkSize[0], globalWorkSize[1]);
    }

    // Copy the input to the device input buffer
    if (ok)
    {
        shrLog(" CECL_WRITE_BUFFER bufferIn\n");
        errnum = CECL_WRITE_BUFFER(commandQueue, bufferIn, CL_TRUE, padding * sizeof(float), volumeSize * sizeof(float), input, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_WRITE_BUFFER bufferIn (returned %d).\n", errnum);
            ok = false;
        }
    }
    // Copy the input to the device output buffer (actually only need the halo)
    if (ok)
    {
        shrLog(" CECL_WRITE_BUFFER bufferOut\n");
        errnum = CECL_WRITE_BUFFER(commandQueue, bufferOut, CL_TRUE, padding * sizeof(float), volumeSize * sizeof(float), input, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_WRITE_BUFFER bufferOut (returned %d).\n", errnum);
            ok = false;
        }
    }
    // Copy the coefficients to the device coefficient buffer
    if (ok)
    {
        shrLog(" CECL_WRITE_BUFFER bufferCoeff\n");
        errnum = CECL_WRITE_BUFFER(commandQueue, bufferCoeff, CL_TRUE, 0, (radius + 1) * sizeof(float), coeff, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_WRITE_BUFFER bufferCoeff (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Allocate the events
    if (ok)
    {
        shrLog(" calloc events\n");
        if ((kernelEvents = (cl_event *)calloc(timesteps, sizeof(cl_event))) == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2006, STDERROR);
            shrLog("Insufficient memory for events calloc, please try a smaller volume (use --help for syntax).\n");
            ok = false;        
        }
    }

    // Start the clock
    shrDeltaT(0);

    // Set the constant arguments
    if (ok)
    {
        shrLog(" CECL_SET_KERNEL_ARG 2-6\n");
        errnum = CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), (void *)&bufferCoeff);
        errnum |= CECL_SET_KERNEL_ARG(kernel, 3, sizeof(int), &dimx);
        errnum |= CECL_SET_KERNEL_ARG(kernel, 4, sizeof(int), &dimy);
        errnum |= CECL_SET_KERNEL_ARG(kernel, 5, sizeof(int), &dimz);
        errnum |= CECL_SET_KERNEL_ARG(kernel, 6, sizeof(int), &padding);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_SET_KERNEL_ARG 2-6 (returned %d).\n", errnum);
            ok = false;     
        }
    }

    // Execute the FDTD
    cl_mem bufferSrc = bufferIn;
    cl_mem bufferDst = bufferOut;
    if (ok)
    {
        shrLog(" GPU FDTD loop\n");
    }
    for (int it = 0 ; ok && it < timesteps ; it++)
    {
        shrLog("\tt = %d ", it);

        // Set the dynamic arguments
        if (ok)
        {
            shrLog(" CECL_SET_KERNEL_ARG 0-1,");
            errnum = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), (void *)&bufferDst);
            errnum |= CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_mem), (void *)&bufferSrc);
            if (errnum != CL_SUCCESS)
            {
                shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
                shrLog("CECL_SET_KERNEL_ARG 0-1 (returned %d).\n", errnum);
                ok = false;               
            }
        }

        // Launch the kernel
        if (ok)
        {
            shrLog(" CECL_ND_RANGE_KERNEL\n");
            errnum = CECL_ND_RANGE_KERNEL(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvents[it]);
            if (errnum != CL_SUCCESS)
            {
                shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
                shrLog("CECL_ND_RANGE_KERNEL (returned %d).\n", errnum);
                ok = false;   
            }
        }
        // Toggle the buffers
        cl_mem tmp = bufferSrc;
        bufferSrc = bufferDst;
        bufferDst = tmp;
    }
    if (ok)
        shrLog("\n");

    // Wait for the kernel to complete
    if (ok)
    {
        shrLog(" clWaitForEvents\n");
        errnum = clWaitForEvents(1, &kernelEvents[timesteps-1]);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clWaitForEvents (returned %d).\n", errnum);
            ok = false;  
        }
    }

    // Stop the clock
    hostElapsedTimeS = shrDeltaT(0);

    // Read the result back, result is in bufferSrc (after final toggle)
    if (ok)
    {
        shrLog(" CECL_READ_BUFFER\n");
        errnum = CECL_READ_BUFFER(commandQueue, bufferSrc, CL_TRUE, padding * sizeof(float), volumeSize * sizeof(float), output, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("CECL_READ_BUFFER bufferSrc (returned %d).\n", errnum);
            ok = false;  
        }
    }

    // Report time
#ifdef GPU_PROFILING
    double elapsedTime = 0.0;
    if (ok && profileTimesteps > 0)
        shrLog(" Collect profile information\n");
    for (int it = 1 ; ok && it <= profileTimesteps ; it++)
    {
        shrLog("\tt = %d ", it);
        shrLog(" clGetEventProfilingInfo,", it);
        errnum = clGetEventProfilingInfo(kernelEvents[it], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelEventStart, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetEventProfilingInfo (returned %d).\n", errnum);
            ok = false;  
        }
        shrLog(" clGetEventProfilingInfo\n", it);
        errnum = clGetEventProfilingInfo(kernelEvents[it], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelEventEnd, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetEventProfilingInfo (returned %d).\n", errnum);
            ok = false;  
        }
        elapsedTime += (double)kernelEventEnd - (double)kernelEventStart;
    }
    if (ok && profileTimesteps > 0)
    {
        shrLog("\n");
        // Convert nanoseconds to seconds
        elapsedTime *= 1.0e-9;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        shrLogEx(LOGBOTH | MASTER, 0, "oclFDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %i, Workgroup = %u\n", 
            throughputM, avgElapsedTime, pointsComputed, 1, localWorkSize[0] * localWorkSize[1]); 
    }
#endif
    
    // Cleanup
    if (kernelEvents)
    {
        for (int it = 0 ; it < timesteps ; it++)
        {
            if (kernelEvents[it])
                clReleaseEvent(kernelEvents[it]);
        }
        free(kernelEvents);
    }
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (cSourceCL)
        free(cSourceCL);
    if (cPathAndName)
        free(cPathAndName);
    if (bufferCoeff)
        clReleaseMemObject(bufferCoeff);
    if (bufferIn)
        clReleaseMemObject(bufferIn);
    if (bufferOut)
        clReleaseMemObject(bufferOut);
    if (commandQueue)
        clReleaseCommandQueue(commandQueue);
    if (devices)
        free(devices);
    if (context)
        clReleaseContext(context);
    return ok;
}
