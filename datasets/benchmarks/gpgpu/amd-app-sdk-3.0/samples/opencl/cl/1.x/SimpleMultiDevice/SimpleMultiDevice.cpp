#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#include "SimpleMultiDevice.hpp"

int
Device::createContext()
{
    //Create context using current device's ID
    context = CECL_CREATE_CONTEXT(cprops,
                              1,
                              &deviceId,
                              0,
                              0,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT failed.");

    return SDK_SUCCESS;
}

int
Device::createQueue()
{
    //Create Command-Queue
    queue = CECL_CREATE_COMMAND_QUEUE(context,
                                 deviceId,
                                 CL_QUEUE_PROFILING_ENABLE,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

    return SDK_SUCCESS;
}

int
Device::createBuffers()
{
    // Create input buffer
    inputBuffer = CECL_BUFFER(context,
                                 CL_MEM_READ_ONLY,
                                 width * sizeof(cl_float),
                                 0,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(inputBuffer)");

    // Create output buffer
    outputBuffer = CECL_BUFFER(context,
                                  CL_MEM_WRITE_ONLY,
                                  width * sizeof(cl_float),
                                  0,
                                  &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(outputBuffer)");

    return SDK_SUCCESS;
}

int
Device::enqueueWriteBuffer()
{
    // Initialize input buffer
    status = CECL_WRITE_BUFFER(queue,
                                  inputBuffer,
                                  1,
                                  0,
                                  width * sizeof(cl_float),
                                  input,
                                  0,
                                  0,
                                  0);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed.");

    return SDK_SUCCESS;
}

int
Device::createProgram(const char **source, const size_t *sourceSize)
{
    // Create program with source
    program = CECL_PROGRAM_WITH_SOURCE(context,
                                        1,
                                        source,
                                        sourceSize,
                                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_PROGRAM_WITH_SOURCE failed.");

    return SDK_SUCCESS;
}

int
Device::buildProgram()
{
    char buildOptions[50];
    sprintf(buildOptions, "-D KERNEL_ITERATIONS=%d", KERNEL_ITERATIONS);
    // Build program source
    status = CECL_PROGRAM(program,
                            1,
                            &deviceId,
                            buildOptions,
                            0,
                            0);
    // Print build log here if build program failed
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
        {
            cl_int logStatus;
            char *buildLog = NULL;
            size_t buildLogSize = 0;
            logStatus = clGetProgramBuildInfo(program,
                                              deviceId,
                                              CL_PROGRAM_BUILD_LOG,
                                              buildLogSize,
                                              buildLog,
                                              &buildLogSize);
            CHECK_OPENCL_ERROR(status, "clGetProgramBuildInfo failed.");

            buildLog = (char*)malloc(buildLogSize);
            if(buildLog == NULL)
            {
                CHECK_ALLOCATION(buildLog, "Failed to allocate host memory. (buildLog)");
            }

            memset(buildLog, 0, buildLogSize);

            logStatus = clGetProgramBuildInfo(program,
                                              deviceId,
                                              CL_PROGRAM_BUILD_LOG,
                                              buildLogSize,
                                              buildLog,
                                              NULL);
            if(logStatus != CL_SUCCESS)
            {
                std::cout << "clGetProgramBuildInfo failed.";
                free(buildLog);
                return SDK_FAILURE;
            }

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << buildLog << std::endl;
            std::cout << " ************************************************\n";
            free(buildLog);
        }

        CHECK_OPENCL_ERROR(status, "CECL_PROGRAM failed.");
    }

    return SDK_SUCCESS;
}

int
Device::createKernel()
{
    kernel = CECL_KERNEL(program, "multiDeviceKernel", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}

int
Device::setKernelArgs()
{
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), &inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(inputBuffer)");

    status = CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_mem), &outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(outputBuffer)");

    return SDK_SUCCESS;
}

int
Device::enqueueKernel(size_t *globalThreads, size_t *localThreads)
{
    status = CECL_ND_RANGE_KERNEL(queue,
                                    kernel,
                                    1,
                                    NULL,
                                    globalThreads,
                                    localThreads,
                                    0,
                                    NULL,
                                    &eventObject);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(queue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    return SDK_SUCCESS;
}

int
Device::waitForKernel()
{
    status = clFinish(queue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

    return SDK_SUCCESS;
}

int
Device::getProfilingData()
{
    status = clGetEventProfilingInfo(eventObject,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong),
                                     &kernelStartTime,
                                     0);
    CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(start time)");

    status = clGetEventProfilingInfo(eventObject,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong),
                                     &kernelEndTime,
                                     0);
    CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(end time)");

    //Measure time in ms
    elapsedTime = 1e-6 * (kernelEndTime - kernelStartTime);

    return SDK_SUCCESS;
}

int
Device::enqueueReadData()
{
    // Allocate memory
    if(output == NULL)
    {
        output = (cl_float*)malloc(width * sizeof(cl_float));
        CHECK_ALLOCATION(output, "Failed to allocate output buffer!\n");
    }

    status = CECL_READ_BUFFER(queue,
                                 outputBuffer,
                                 1,
                                 0,
                                 width * sizeof(cl_float),
                                 output,
                                 0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");

    return SDK_SUCCESS;
}

int
Device::verifyResults()
{
    float error = 0;
    //compare results between verificationOutput and output host buffers
    for(int i = 0; i < width; i++)
    {
        error += (output[i] - verificationOutput[i]);
    }
    error /= width;

    if(error < 0.001)
    {
        std::cout << "Passed!\n" << std::endl;
        verificationCount++;
    }
    else
    {
        std::cout << "Failed!\n" << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
Device::cleanupResources()
{
    int status = clReleaseCommandQueue(queue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(queue)");

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    cl_uint programRefCount;
    status = clGetProgramInfo(program,
                              CL_PROGRAM_REFERENCE_COUNT,
                              sizeof(cl_uint),
                              &programRefCount,
                              0);
    CHECK_OPENCL_ERROR(status, "clGetProgramInfo failed.");

    if(programRefCount)
    {
        status = clReleaseProgram(program);
        CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");
    }

    cl_uint inputRefCount;
    status = clGetMemObjectInfo(inputBuffer,
                                CL_MEM_REFERENCE_COUNT,
                                sizeof(cl_uint),
                                &inputRefCount,
                                0);
    CHECK_OPENCL_ERROR(status, "clGetMemObjectInfo failed.");

    if(inputRefCount)
    {
        status = clReleaseMemObject(inputBuffer);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (inputBuffer)");
    }

    cl_uint outputRefCount;
    status = clGetMemObjectInfo(outputBuffer,
                                CL_MEM_REFERENCE_COUNT,
                                sizeof(cl_uint),
                                &outputRefCount,
                                0);
    CHECK_OPENCL_ERROR(status, "clGetMemObjectInfo failed.");

    if(outputRefCount)
    {
        status = clReleaseMemObject(outputBuffer);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (outputBuffer)");
    }

    cl_uint contextRefCount;
    status = clGetContextInfo(context,
                              CL_CONTEXT_REFERENCE_COUNT,
                              sizeof(cl_uint),
                              &contextRefCount,
                              0);
    CHECK_OPENCL_ERROR(status, "clGetContextInfo failed.");

    if(contextRefCount)
    {
        status = clReleaseContext(context);
        CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");
    }

    status = clReleaseEvent(eventObject);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent failed.");

    return SDK_SUCCESS;
}



//Thread function for a device
void* threadFunc(void *device)
{
    Device *d = (Device*)device;

    size_t globalThreads = width;
    size_t localThreads = GROUP_SIZE;

    d->enqueueKernel(&globalThreads, &localThreads);
    d->waitForKernel();

    return NULL;
}

Device::~Device()
{
    FREE(output);
}


int runMultiGPU()
{
    int status;

    ///////////////////////////////////////////////////////////////////
    //  Case 1 : Single Context (Single Thread)
    //////////////////////////////////////////////////////////////////
    std::cout << sep << "\nMulti GPU Test 1 : Single context Single Thread\n" <<
              sep << std::endl;

    cl_context context = CECL_CREATE_CONTEXT_FROM_TYPE(cprops,
                         CL_DEVICE_TYPE_GPU,
                         0,
                         0,
                         &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT failed.");

    size_t sourceSize = strlen(source);
    cl_program program  = CECL_PROGRAM_WITH_SOURCE(context,
                          1,
                          &source,
                          (const size_t*)&sourceSize,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_PROGRAM_WITH_SOURCE failed.");

    char buildOptions[50];
    sprintf(buildOptions, "-D KERNEL_ITERATIONS=%d", KERNEL_ITERATIONS);

    //Build program for all the devices in the context
    status = CECL_PROGRAM(program, 0, 0, buildOptions, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_PROGRAM failed.");


    //Buffers
    cl_mem inputBuffer;
    cl_mem outputBuffer;

    //Setup for all GPU devices
    for(int i = 0; i < numGPUDevices; i++)
    {
        gpu[i].context = context;
        gpu[i].program = program;

        status = gpu[i].createQueue();
        CHECK_ERROR(status , SDK_SUCCESS, "Creating Commmand Queue(GPU) failed");

        status = gpu[i].createKernel();
        CHECK_ERROR(status , SDK_SUCCESS, "Creating Kernel (GPU) failed");

        // Create buffers
        // Create input buffer
        inputBuffer = CECL_BUFFER(context,
                                     CL_MEM_READ_ONLY,
                                     width * sizeof(cl_float),
                                     0,
                                     &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(inputBuffer)");

        // Create output buffer
        outputBuffer = CECL_BUFFER(context,
                                      CL_MEM_WRITE_ONLY,
                                      width * sizeof(cl_float),
                                      0,
                                      &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(outputBuffer)");

        //Set Buffers
        gpu[i].inputBuffer = inputBuffer;
        status = gpu[i].enqueueWriteBuffer();
        CHECK_ERROR(status , SDK_SUCCESS,
                    "Submitting Write OpenCL Buffer (GPU) failed");
        gpu[i].outputBuffer = outputBuffer;

        //Set kernel arguments
        status = gpu[i].setKernelArgs();
        CHECK_ERROR(status , SDK_SUCCESS, "Setting Kernel Args(GPU) failed");
    }

    size_t globalThreads = width;
    size_t localThreads = GROUP_SIZE;

    //Start a host timer here
    int timer = sampleTimer.createTimer();
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].enqueueKernel(&globalThreads, &localThreads);
        CHECK_ERROR(status , SDK_SUCCESS, "Submitting Opencl Kernel (GPU) failed");
    }

    //Wait for all kernels to finish execution
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].waitForKernel();
        CHECK_ERROR(status , SDK_SUCCESS, "Waiting for Kernel(GPU) failed");
    }

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    double totalTime = sampleTimer.readTimer(timer);

    //Get individual timers
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].getProfilingData();
        CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (GPU) failed");
    }


    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    for(int i = 0; i < numGPUDevices; i++)
    {
        std::cout << "Time of GPU" << i << " : " << gpu[i].elapsedTime <<
                  std::endl;
    }

    if(verify)
    {
        //Enqueue Read output buffer and verify results
        for(int i = 0; i < numGPUDevices; i++)
        {
            status = gpu[i].enqueueReadData();
            CHECK_ERROR(status , SDK_SUCCESS, "Submitting Read buffer (GPU) failed");

            // Verify results
            std::cout << "Verifying results for GPU" << i << " : ";
            gpu[i].verifyResults();
        }
    }

    //Release the resources on all devices
    //Release context
    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT failed.");

    //Release memory buffers
    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (inputBuffer)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed. (outputBuffer)");

    //Release Program object
    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    //Release Kernel object, command-queue, event object
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = clReleaseKernel(gpu[i].kernel);
        CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

        status = clReleaseCommandQueue(gpu[i].queue);
        CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

        status = clReleaseEvent(gpu[i].eventObject);
        CHECK_OPENCL_ERROR(status, "clReleaseEvent failed.");
    }

    ///////////////////////////////////////////////////////////////////
    //  Case 2 : Multiple Context (Single Thread)
    //////////////////////////////////////////////////////////////////
    std::cout << sep << "\nMulti GPU Test 2 : Multiple context Single Thread\n" <<
              sep << std::endl;

    for(int i = 0; i < numGPUDevices; i++)
    {
        //Create context for each device
        status = gpu[i].createContext();
        CHECK_ERROR(status ,SDK_SUCCESS,"createContext failed");

        //Create command-queue;
        status = gpu[i].createQueue();
        CHECK_ERROR(status, SDK_SUCCESS, "Create CommandQueue failed");

        //Create memory buffers
        status = gpu[i].createBuffers();
        CHECK_ERROR(status, SDK_SUCCESS, "Create Buffers");

        //Initialize input buffer
        status = gpu[i].enqueueWriteBuffer();
        CHECK_ERROR(status, SDK_SUCCESS, "EnqueueWriteBuffer Failed");

        //create program object
        status = gpu[i].createProgram(&source, &sourceSize);
        CHECK_ERROR(status, SDK_SUCCESS, "Create Program Failed");

        //Build program
        status = gpu[i].buildProgram();
        CHECK_ERROR(status, SDK_SUCCESS, "Build Program Failed");

        //Create kernel objects for each device
        status = gpu[i].createKernel();
        CHECK_ERROR(status, SDK_SUCCESS, "Create Kernel");
    }

    //Set kernel arguments
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].setKernelArgs();
        CHECK_ERROR(status, SDK_SUCCESS, "setKernelArgs failed");
    }

    //Start a host timer here
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].enqueueKernel(&globalThreads, &localThreads);
        CHECK_ERROR(status, SDK_SUCCESS, "Enqueue Kernel Failed");
    }

    //Wait for all kernels to finish execution
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].waitForKernel();
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForKernel Failed");
    }

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    totalTime = sampleTimer.readTimer(timer);

    //Get individual timers
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].getProfilingData();
        CHECK_ERROR(status, SDK_SUCCESS, "Get Profiling Data Failed");
    }

    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    for(int i = 0; i < numGPUDevices; i++)
    {
        std::cout << "Time of GPU" << i << " : " << gpu[i].elapsedTime <<
                  std::endl;
    }

    if(verify)
    {
        // Read outputdata and verify results
        for(int i = 0; i < numGPUDevices; i++)
        {
            status = gpu[i].enqueueReadData();
            CHECK_ERROR(status, SDK_SUCCESS, "Enqueue Read Data Filed");

            // Verify results
            std::cout << "Verifying results for GPU" << i << " : ";
            status = gpu[i].verifyResults();
            CHECK_ERROR(status, SDK_SUCCESS, "verifyResults failed");
        }
    }


    //Release the resources on all devices
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].cleanupResources();
        CHECK_ERROR(status, SDK_SUCCESS, "cleanup Resourcefailed");
    }

    ////////////////////////////////////////////////////////////////////
    //  Case 3 : Multiple thread and multiple context for each device
    ////////////////////////////////////////////////////////////////////
    std::cout << sep << "\nMulti GPU Test 3 : Multiple context Multiple Thread\n" <<
              sep << std::endl;

    for(int i = 0; i < numGPUDevices; i++)
    {
        //Create context for each device
        status = gpu[i].createContext();
        CHECK_ERROR(status , SDK_SUCCESS, "Creating CL_Context (GPU) failed");

        //Create command-queue;
        status = gpu[i].createQueue();
        CHECK_ERROR(status , SDK_SUCCESS, "Creating Command Queue(GPU) failed");

        //Create memory buffers
        status = gpu[i].createBuffers();
        CHECK_ERROR(status , SDK_SUCCESS, "Createing Buffers (GPU) failed");

        //Initialize input buffer
        status = gpu[i].enqueueWriteBuffer();
        CHECK_ERROR(status , SDK_SUCCESS,
                    "Submitting Write OpenCL Buffer (GPU) failed");

        //create program object
        status = gpu[i].createProgram(&source, &sourceSize);
        CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Program (GPU) failed");

        //Build program
        status = gpu[i].buildProgram();
        CHECK_ERROR(status , SDK_SUCCESS, "Building OpenCL Program (GPU) failed");

        //Create kernel objects for each device
        status = gpu[i].createKernel();
        CHECK_ERROR(status , SDK_SUCCESS, "Create Kernel(GPU) failed");
    }

    //Set kernel arguments
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].setKernelArgs();
        CHECK_ERROR(status , SDK_SUCCESS, "Setting Kernel Arguments(GPU) failed");
    }

    //Start a host timer here
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    //Create thread objects
    SDKThread *gpuThread = new SDKThread[numGPUDevices];

    //Start threads for each gpu device
    for(int i = 0; i < numGPUDevices; i++)
    {
        gpuThread[i].create(::threadFunc, (void *)(gpu + i));
    }

    //Join all gpu threads
    for(int i = 0; i < numGPUDevices; i++)
    {
        gpuThread[i].join();
    }

    delete []gpuThread;

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    totalTime = sampleTimer.readTimer(timer);

    //Get individual timers
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].getProfilingData();
        CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (GPU) failed");
    }

    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    for(int i = 0; i < numGPUDevices; i++)
    {
        std::cout << "Time of GPU" << i << " : " << gpu[i].elapsedTime << std::endl;
    }

    if(verify)
    {
        // Read outputdata and verify results
        for(int i = 0; i < numGPUDevices; i++)
        {
            status = gpu[i].enqueueReadData();
            CHECK_ERROR(status , SDK_SUCCESS, "Submitting Read OpenCL Buffer (GPU) failed");

            // Verify results
            std::cout << "Verifying results for GPU" << i << " : ";

            gpu[i].verifyResults();
        }
    }

    //Release the resources on all devices
    for(int i = 0; i < numGPUDevices; i++)
    {
        status = gpu[i].cleanupResources();
        CHECK_ERROR(status , SDK_SUCCESS, "Cleaning Up OpenCL Resources(GPU) failed");
    }

    return SDK_SUCCESS;
}

int runMultiDevice()
{
    int status;

    ///////////////////////////////////////////////////////////////////
    //  Case 1 : Single Context (Single Thread)
    //////////////////////////////////////////////////////////////////
    std::cout << sep << "\nCPU + GPU Test 1 : Single context Single Thread\n" <<
              sep << std::endl;

    /* Create a list of device IDs having only CPU0 and GPU0 as device IDs */
    cl_device_id *devices = (cl_device_id*)malloc(2 * sizeof(cl_device_id));
    devices[0] = cpu[0].deviceId;
    devices[1] = gpu[0].deviceId;

    cl_context context = CECL_CREATE_CONTEXT(cprops, 2, devices, 0, 0, &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT failed.");

    size_t sourceSize = strlen(source);
    cl_program program  = CECL_PROGRAM_WITH_SOURCE(context,
                          1,
                          &source,
                          (const size_t*)&sourceSize,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_PROGRAM_WITH_SOURCE failed.");

    char buildOptions[50];
    sprintf(buildOptions, "-D KERNEL_ITERATIONS=%d", KERNEL_ITERATIONS);

    //Build program for all the devices in the context
    status = CECL_PROGRAM(program, 0, 0, buildOptions, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_PROGRAM failed.");

    //Allocate objects for CPU
    cpu[0].context = context;
    gpu[0].context = context;

    cpu[0].program = program;
    gpu[0].program = program;

    // Create command queue
    status = cpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Create Command Queue (CPU) failed");

    // Create kernel
    status = cpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Create Kernel (CPU) failed");

    // Create queue
    status = gpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Create Command Queue (GPU) failed");

    // Create kernel
    status = gpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Create Kernel (GPU) failed");

    // Create buffers - A buffer is created on all devices sharing a context
    // So bufffer creation should should not per device in a single-context
    cl_mem inputBuffer = CECL_BUFFER(context,
                                        CL_MEM_READ_ONLY,
                                        width * sizeof(cl_float),
                                        0,
                                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(inputBuffer)");

    cl_mem outputBuffer = CECL_BUFFER(context,
                                         CL_MEM_WRITE_ONLY,
                                         width * sizeof(cl_float),
                                         0,
                                         &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(outputBuffer)");

    cpu[0].inputBuffer = inputBuffer;
    gpu[0].inputBuffer = inputBuffer;

    cpu[0].outputBuffer = outputBuffer;
    gpu[0].outputBuffer = outputBuffer;

    // Initialize input buffer for both devices
    status = cpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS,
                "Submitting Write OpenCL Buffer (CPU) failed");

    status = gpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS,
                "Submitting Write OpenCL Buffer (GPU) failed");

    //Set kernel arguments
    status = cpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguents (CPU) failed");

    status = gpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguments (GPU) failed");


    size_t globalThreads = width;
    size_t localThreads = GROUP_SIZE;

    //Start a host timer here
    int timer = sampleTimer.createTimer();
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    status = cpu[0].enqueueKernel(&globalThreads, &localThreads);
    CHECK_ERROR(status , SDK_SUCCESS, "Submitting Kernel (CPU) failed");

    status = gpu[0].enqueueKernel(&globalThreads, &localThreads);
    CHECK_ERROR(status , SDK_SUCCESS, "Submitting Kernel (GPU) failed");

    //Wait for all kernels to finish execution
    status = cpu[0].waitForKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Wait for Kernel (CPU) failed");

    status = gpu[0].waitForKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Wait for Kernel (GPU) failed");

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    double totalTime = sampleTimer.readTimer(timer);

    status = cpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (CPU) failed");

    status = gpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (GPU) failed");

    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    std::cout << "Time of CPU : " << cpu[0].elapsedTime << std::endl;
    std::cout << "Time of GPU : " << gpu[0].elapsedTime << std::endl;


    if(verify)
    {
        //Read back output data for verification
        status = cpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Submitting Read OpenCL Buffer (CPU) failed");

        status = gpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Submitting Read OpenCL Buffer (GPU) failed");

        // Verify results
        std::cout << "Verifying results for CPU : ";
        cpu[0].verifyResults();
        std::cout << "Verifying results for GPU : ";
        gpu[0].verifyResults();
    }

    //Release context
    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT failed.(context)");


    //Release memory buffers
    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (inputBuffer)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (outputBuffer)");

    //ReleaseCommand-queue
    status = clReleaseCommandQueue(cpu[0].queue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(cpu[0].queue)");

    status = clReleaseCommandQueue(gpu[0].queue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(gpu[0].queue)");

    //Release Program object
    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    //Release Kernel object
    status = clReleaseKernel(cpu[0].kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(cpu[0].kernel)");

    status = clReleaseKernel(gpu[0].kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(gpu[0].kernel)");

    //Release Event object
    status = clReleaseEvent(cpu[0].eventObject);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent failed.(cpu[0].eventObject)");

    status = clReleaseEvent(gpu[0].eventObject);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent failed.(gpu[0].eventObject)");


    //Release the resources on all devices
    /*status = cpu[0].cleanupResources();
    if(status != SDK_SUCCESS)
        return status;

    status = gpu[0].cleanupResources();
    if(status != SDK_SUCCESS)
        return status;*/


    ///////////////////////////////////////////////////////////////////
    //  Case 2 : Multiple Context (Single Thread)
    //////////////////////////////////////////////////////////////////
    std::cout << sep << "\nCPU + GPU Test 2 : Multiple context Single Thread\n" <<
              sep << std::endl;

    status = cpu[0].createContext();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Context (CPU) failed");

    status = cpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Command Queue (CPU) failed");

    status = cpu[0].createBuffers();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Buffer (CPU) failed");

    status = cpu[0].createProgram(&source, &sourceSize);
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Program (CPU) failed");

    status = cpu[0].buildProgram();
    CHECK_ERROR(status , SDK_SUCCESS, "Building OpenCL Program (CPU) failed");

    status = cpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Kernel (CPU) failed");

    status = gpu[0].createContext();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Context (GPU) failed");

    status = gpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Command Queue (GPU) failed");

    status = gpu[0].createBuffers();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Buffer (GPU) failed");

    status = gpu[0].createProgram(&source, &sourceSize);
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Program (GPU) failed");

    status = gpu[0].buildProgram();
    CHECK_ERROR(status , SDK_SUCCESS, "Building OpenCL Program (GPU) failed");

    status = gpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Kernel (GPU) failed");

    // Initialize input buffer for both devices
    status = cpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS,
                "Submitting Write OpenCL Buffer (CPU) failed");

    status = gpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS,
                "Submitting Write OpenCL Buffer (GPU) failed");

    //Set kernel arguments
    status = cpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguments (CPU) failed");

    status = gpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguments (GPU) failed");

    //Start a host timer here
    //int timer = sampleTimer->createTimer();
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    //size_t globalThreads = width;
    //size_t localThreads = 1;
    status = cpu[0].enqueueKernel(&globalThreads, &localThreads);
    CHECK_ERROR(status , SDK_SUCCESS, "Submitting Kernel Failed (CPU) failed");

    status = gpu[0].enqueueKernel(&globalThreads, &localThreads);
    CHECK_ERROR(status , SDK_SUCCESS, "Submitting Kernel Failed (GPU) failed");

    //Wait for all kernels to finish execution
    status = cpu[0].waitForKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Wait For Kernel Execution (CPU) failed");

    status = gpu[0].waitForKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Wait For Kernel Execution (GPU) failed");

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    //double totalTime = sampleTimer->readTimer(timer);

    status = cpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling data (CPU) failed");

    status = gpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (GPU) failed");

    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    std::cout << "Time of CPU : " << cpu[0].elapsedTime << std::endl;
    std::cout << "Time of GPU : " << gpu[0].elapsedTime << std::endl;

    if(verify)
    {
        //Read back output data for verification
        status = cpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Reading OpenCL Buffer (CPU) failed");

        status = gpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Reading OpenCL Buffer (GPU) failed");

        // Verify results
        std::cout << "Verifying results for CPU : ";
        cpu[0].verifyResults();
        std::cout << "Verifying results for GPU : ";
        gpu[0].verifyResults();
    }

    //Release the resources on all devices
    status = cpu[0].cleanupResources();
    CHECK_ERROR(status , SDK_SUCCESS, "CleanUp Resources (CPU) failed");

    status = gpu[0].cleanupResources();
    CHECK_ERROR(status , SDK_SUCCESS, "Clean Up Resources (GPU) failed");

    /////////////////////////////////////////////////////////////////////
    //  Case 3 : Multiple thread and multiple context for each device
    ////////////////////////////////////////////////////////////////////
    std::cout << sep << "\nCPU + GPU Test 3 : Multiple context Multiple Thread\n" <<
              sep << std::endl;

    status = cpu[0].createContext();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Context (CPU) failed");

    status = cpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Command Queue (CPU) failed");

    status = cpu[0].createBuffers();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Buffers (CPU) failed");

    status = cpu[0].createProgram(&source, &sourceSize);
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Program (CPU) failed");

    status = cpu[0].buildProgram();
    CHECK_ERROR(status , SDK_SUCCESS, "Building OpenCL Program (CPU) failed");

    status = cpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Kernel (CPU) failed");

    status = gpu[0].createContext();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpencL Context (GPU) failed");

    status = gpu[0].createQueue();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating Command Queue (GPU) failed");

    status = gpu[0].createBuffers();
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Buffers (GPU) failed");

    status = gpu[0].createProgram(&source, &sourceSize);
    CHECK_ERROR(status , SDK_SUCCESS, "Creating OpenCL Program (GPU) failed");

    status = gpu[0].buildProgram();
    CHECK_ERROR(status , SDK_SUCCESS, "Build OpenCL Kernel (GPU) failed");

    status = gpu[0].createKernel();
    CHECK_ERROR(status , SDK_SUCCESS, "Create Kernel (GPU) failed");

    // Initialize input buffer for both devices
    status = cpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS, "Writing to OpenCL Buffer (CPU) failed");

    status = gpu[0].enqueueWriteBuffer();
    CHECK_ERROR(status , SDK_SUCCESS, "Writing to OpenCL Buffer (GPU) failed");

    //Set kernel arguments
    status = cpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguments (CPU) failed");

    status = gpu[0].setKernelArgs();
    CHECK_ERROR(status , SDK_SUCCESS, "Set Kernel Arguments (GPU) failed");

    //Start a host timer here
    sampleTimer.resetTimer(timer);
    sampleTimer.startTimer(timer);

    //Create a thread for CPU and GPU device each
    SDKThread cpuThread;
    SDKThread gpuThread;

    cpuThread.create(::threadFunc, (void *)cpu);
    gpuThread.create(::threadFunc, (void *)gpu);

    cpuThread.join();
    gpuThread.join();

    //Stop the host timer here
    sampleTimer.stopTimer(timer);

    //Measure total time
    totalTime = sampleTimer.readTimer(timer);

    status = cpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data (CPU) failed");

    status = gpu[0].getProfilingData();
    CHECK_ERROR(status , SDK_SUCCESS, "Getting Profiling Data(GPU) failed");

    //Print total time and individual times
    std::cout << "Total time : " << totalTime * 1000 << std::endl;
    std::cout << "Time of CPU : " << cpu[0].elapsedTime << std::endl;
    std::cout << "Time of GPU : " << gpu[0].elapsedTime << std::endl;

    if(verify)
    {
        //Read back output data for verification
        status = cpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Reading data from OpenCL Buffer failed");

        status = gpu[0].enqueueReadData();
        CHECK_ERROR(status , SDK_SUCCESS, "Reading data from OpenCL Buffer failed");

        // Verify results
        std::cout << "Verifying results for CPU : ";
        cpu[0].verifyResults();
        std::cout << "Verifying results for GPU : ";
        gpu[0].verifyResults();
    }

    //Release the resources on all devices
    status = cpu[0].cleanupResources();
    CHECK_ERROR(status , SDK_SUCCESS, "Clean up resources (cpu[0]) failed");

    status = gpu[0].cleanupResources();
    CHECK_ERROR(status , SDK_SUCCESS, "Clean up Reources (gpu[0]) failed");

    FREE(devices)

    return SDK_SUCCESS;
}
/*
 * \brief Host Initialization
 *        Allocate and initialize memory
 *        on the host. Print input array.
 */
int
initializeHost(void)
{
    width = NUM_THREADS;
    input = NULL;
    verificationOutput = NULL;

    /////////////////////////////////////////////////////////////////
    // Allocate and initialize memory used by host
    /////////////////////////////////////////////////////////////////
    cl_uint sizeInBytes = width * sizeof(cl_uint);
    input = (cl_float*) malloc(sizeInBytes);
    CHECK_ALLOCATION(input, "Error: Failed to allocate input memory on host\n");

    verificationOutput = (cl_float*) malloc(sizeInBytes);
    CHECK_ALLOCATION(verificationOutput,
                     "Error: Failed to allocate verificationOutput memory on host\n");

    //Initilize input data
    for(int i = 0; i < width; i++)
    {
        input[i] = (cl_float)i;
    }

    return SDK_SUCCESS;
}

/*
 * Converts the contents of a file into a string
 */
std::string
convertToString(const char *filename)
{
    size_t size;
    char*  str;
    std::string s;

    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            return NULL;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';

        s = str;
        delete[] str;
        return s;
    }
    return NULL;
}


/*
 * \brief OpenCL related initialization
 *        Create Context, Device list, Command Queue
 *        Create OpenCL memory buffer objects
 *        Load CL file, compile, link CL source
 *  Build program and kernel objects
 */
int
initializeCL(void)
{
    cl_int status = 0;
    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */

    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_OPENCL_ERROR(status, "clGetPlatformIDs failed.");

    if(numPlatforms > 0)
    {
        cl_platform_id* platforms = (cl_platform_id *)malloc(numPlatforms*sizeof(
                                        cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        CHECK_OPENCL_ERROR(status, "clGetPlatformIDs failed.");

        for(unsigned int i=0; i < numPlatforms; ++i)
        {
            char pbuff[100];
            status = clGetPlatformInfo(
                         platforms[i],
                         CL_PLATFORM_VENDOR,
                         sizeof(pbuff),
                         pbuff,
                         NULL);
            CHECK_OPENCL_ERROR(status, "clGetPlatformInfo failed.");

            platform = platforms[i];
            if(!strcmp(pbuff, "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
        FREE(platforms);
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */
    cps[0] = CL_CONTEXT_PLATFORM;
    cps[1] = (cl_context_properties)platform;
    cps[2] = 0;

    cprops = (NULL == platform) ? NULL : cps;

    // Get Number of CPU devices available
    status = clGetDeviceIDs(platform,
                            CL_DEVICE_TYPE_GPU,
                            0,
                            0,
                            (cl_uint*)&numCPUDevices);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed.(numCPUDevices)");

    // Get Number of CPU devices available
    status = clGetDeviceIDs(platform,
                            CL_DEVICE_TYPE_GPU,
                            0,
                            0,
                            (cl_uint*)&numDevices);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed.(numDevices)");

    // Get number of GPU Devices
    numGPUDevices = numDevices - numCPUDevices;

    // If no GPU is present then exit
    if(numGPUDevices < 1)
    {
        OPENCL_EXPECTED_ERROR("Only CPU device is present. Exiting!");
    }

    // Allocate memory for list of Devices
    cpu = new Device[numCPUDevices];
    //Get CPU Device IDs
    cl_device_id* cpuDeviceIDs = new cl_device_id[numCPUDevices];
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numCPUDevices,
                            cpuDeviceIDs, 0);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed.");

    for(int i = 0; i < numCPUDevices; i++)
    {
        cpu[i].dType = CL_DEVICE_TYPE_GPU;
        cpu[i].deviceId = cpuDeviceIDs[i];
    }

    delete[] cpuDeviceIDs;

    gpu = new Device[numGPUDevices];
    //Get GPU Device IDs
    cl_device_id* gpuDeviceIDs = new cl_device_id[numGPUDevices];
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numGPUDevices,
                            gpuDeviceIDs, 0);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed.");

    for(int i = 0; i < numGPUDevices; i++)
    {
        gpu[i].dType = CL_DEVICE_TYPE_GPU;
        gpu[i].deviceId = gpuDeviceIDs[i];
    }

    delete[] gpuDeviceIDs;

    /////////////////////////////////////////////////////////////////
    // Load CL file
    /////////////////////////////////////////////////////////////////
    const char *filename  = "SimpleMultiDevice_Kernels.cl";
    sourceStr = convertToString(filename);
    source = sourceStr.c_str();

    return SDK_SUCCESS;
}

int
run()
{
    int status;

    // If a GPU is present then run CPU + GPU concurrently
    if(numGPUDevices > 0 && numCPUDevices > 0)
    {
        /* 3 tests :
         a) Single context - Single thread
         b) Multiple context - Single thread
         c) Multiple context - Multple Threads*/

        // 3 Tests * 2 devices
        requiredCount += 3 * 2;
        status = runMultiDevice();
        CHECK_ERROR(status , SDK_SUCCESS,
                    "Running OpenCL Kernel in MultiDevice Failed");
    }

    // If more than 1 GPU is present then run MultiGPU concurrently
    if(numGPUDevices > 1)
    {
        /* 3 tests :
         a) Single context - Single thread
         b) Multiple context - Single thread
         c) Multiple context - Multple Threads*/

        // 3 Tests * numGPUDevices
        requiredCount += 3 * numGPUDevices;
        status = runMultiGPU();
        CHECK_ERROR(status , SDK_SUCCESS, "Running OpenCL Kernel in MultiGPU Failed");
    }
    return SDK_SUCCESS;

}


/*
 * \brief Releases program's resources
 */
void
cleanupHost(void)
{
    if(input != NULL)
    {
        free(input);
        input = NULL;
    }
    if(verificationOutput != NULL)
    {
        free(verificationOutput);
        verificationOutput = NULL;
    }
    if(cpu != NULL)
    {
        delete[] cpu;
        cpu = NULL;
    }
    if(gpu != NULL)
    {
        delete[] gpu;
        gpu = NULL;
    }
}


/*
 * \brief Print no more than 256 elements of the given array.
 *
 *        Print Array name followed by elements.
 */
void print1DArray(
    const std::string arrayName,
    const unsigned int * arrayData,
    const unsigned int length)
{
    cl_uint i;
    cl_uint numElementsToPrint = (256 < length) ? 256 : length;

    std::cout << std::endl;
    std::cout << arrayName << ":" << std::endl;
    for(i = 0; i < numElementsToPrint; ++i)
    {
        std::cout << arrayData[i] << " ";
    }
    std::cout << std::endl;

}

// OpenCL MAD definition for CPU
float mad(float a, float b, float c)
{
    return a * b + c;
}

// OpenCL HYPOT definition for CPU
#if (_MSC_VER != 1800)
float hypot(float a, float b)
{
    return sqrt(a * a + b * b);
}
#endif


int
CPUKernel()
{
    for(int i = 0; i < width; i++)
    {
        float a = mad(input[i], input[i], 1);
        float b = mad(input[i], input[i], 2);

        for(int j = 0; j < KERNEL_ITERATIONS; j++)
        {
            a = hypot(a, b);
            b = hypot(a, b);
        }
        verificationOutput[i] = (a + b);
    }
    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    for(int i = 1; i < argc; i++)
    {
        if(!strcmp(argv[i], "-e") || !strcmp(argv[i], "--verify"))
        {
            verify = true;
        }
        if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            printf("Usage\n");
            printf("-h, --help\tPrint this help.\n");
            printf("-e, --verify\tVerify results against reference implementation.\n");
            exit(0);
        }
		if(!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version"))
        {
            std::cout << "SDK version : " << SAMPLE_VERSION << std::endl;
            exit(0);
        }
    }

    int status;

    // Initialize Host application
    if (initializeHost() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Run host computation if verification is true
    if (CPUKernel() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Initialize OpenCL resources
    status = initializeCL();
    if(status != SDK_SUCCESS)
    {
        if(status == SDK_EXPECTED_FAILURE)
        {
            return SDK_SUCCESS;
        }

        return SDK_FAILURE;
    }

    // Run the CL program
    if (run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Release host resources
    cleanupHost();

    if(verify)
    {
        // If any one test fails then print FAILED
        if(verificationCount != requiredCount)
        {
            std::cout << "\n\nFAILED!\n" << std::endl;
            return SDK_FAILURE;
        }
        else
        {
            std::cout << "\n\nPASSED!\n" << std::endl;
            return SDK_SUCCESS;
        }
    }

    return SDK_SUCCESS;
}
