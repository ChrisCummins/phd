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


#include "Reduction.hpp"


int
Reduction::setupReduction()
{
    // make sure length is multiple of group size * 4
    unsigned int mulFactor = GROUP_SIZE * VECTOR_SIZE * MULTIPLY;
    length = (length < mulFactor) ? mulFactor : length;
    length = (length / mulFactor) * mulFactor;

    length = length / VECTOR_SIZE;

#if defined (_WIN32)
    input = (cl_uint*)_aligned_malloc(length * sizeof(cl_uint4), 16);
#else
    input = (cl_uint*)memalign(16, length * sizeof(cl_uint4));
#endif

    CHECK_ALLOCATION(input, "Failed to allocate host memory. (input)");

    // random initialisation of input
    fillRandom<cl_uint>(input, length * VECTOR_SIZE, 1, 0, 5);

    // Unless quiet mode has been enabled, print the INPUT array
    if(!sampleArgs->quiet)
        printArray<cl_uint>(
            "Input",
            input,
            length * VECTOR_SIZE,
            1);

    return SDK_SUCCESS;
}

int
Reduction::setupCL()
{
    cl_int status = CL_SUCCESS;

    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_GPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_GPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    // Creating Context for the found platform
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = CECL_CREATE_CONTEXT_FROM_TYPE(cps,
                                      dType,
                                      NULL,
                                      NULL,
                                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    // Create command queue

    commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                        devices[sampleArgs->deviceId],
                                        0,
                                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create memory objects for input array
    inputBuffer = CECL_BUFFER(context,
                                 inMemFlags,
                                 length * sizeof(cl_uint4),
                                 NULL,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("Reduction_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    kernel = CECL_KERNEL(program,
                            "reduce",
                            &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    status = setWorkGroupSize();
    CHECK_ERROR(status, SDK_SUCCESS, "setWorkGroupSize failed");

    return SDK_SUCCESS;
}

int Reduction::setWorkGroupSize()
{
    cl_int status = 0;
    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, " setKernelWorkGroupInfo() failed");
    /**
     * If groupSize exceeds the maximum supported on kernel
     * fall back
     */
    if(groupSize > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }
        groupSize = kernelInfo.kernelWorkGroupSize;
    }

    if(groupSize > deviceInfo.maxWorkItemSizes[0] ||
            groupSize > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support"
                  << "requested number of work items.";
        return SDK_FAILURE;
    }

    if(kernelInfo.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    globalThreads[0] = length / MULTIPLY;
    localThreads[0] = groupSize;

    return SDK_SUCCESS;
}

int
Reduction::runCLKernels()
{
    cl_int status;
    cl_event ndrEvent;
    cl_int eventStatus = CL_QUEUED;

    // This algorithm reduces each group of work-items to a single value
    // on OpenCL device and later each reduced items per group is further
    // reduced to a single value on CPU


    // Transfer input to device
    cl_event inMapEvt;
    void* mapPtr = CECL_MAP_BUFFER(commandQueue,
                                      inputBuffer,
                                      CL_FALSE,
                                      CL_MAP_WRITE,
                                      0,
                                      length * sizeof(cl_uint4),
                                      0,
                                      NULL,
                                      &inMapEvt,
                                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed. (inputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inMapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");
    memcpy(mapPtr, input, length * sizeof(cl_uint4));

    cl_event inUnmapEvent;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     inputBuffer,
                                     mapPtr,
                                     0,
                                     NULL,
                                     &inUnmapEvent);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inUnmapEvent);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");

    // Set appropriate arguments to the kernel the input array
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    // temporary output buffer
    status = CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_mem), (void*)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    // local array
    status = CECL_SET_KERNEL_ARG(kernel, 2, groupSize * sizeof(cl_uint4), NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (local memory)");

    // Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(commandQueue,
                                    kernel,
                                    1,
                                    NULL,
                                    globalThreads,
                                    localThreads,
                                    0,
                                    NULL,
                                    &ndrEvent);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvent);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    cl_event outMapEvt;
    cl_uint* outMapPtr = (cl_uint*)CECL_MAP_BUFFER(commandQueue,
                         outputBuffer,
                         CL_FALSE,
                         CL_MAP_READ,
                         0,
                         numBlocks * sizeof(cl_uint4),
                         0,
                         NULL,
                         &outMapEvt,
                         &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(outputBuffer) failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outMapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt) Failed");

    // Add individual sum of blocks
    output = 0;
    for(int i = 0; i < numBlocks * VECTOR_SIZE; ++i)
    {
        output += outMapPtr[i];
    }

    cl_event outUnmapEvt;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     outputBuffer,
                                     (void*)outMapPtr,
                                     0,
                                     NULL,
                                     &outUnmapEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(outputBuffer) failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outUnmapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outUnMapEvt) Failed");;

    return SDK_SUCCESS;
}

/*
 * Reduces the input array (in place)
 * length specifies the length of the array
 */
void
Reduction::reductionCPUReference(cl_uint * input,
                                 const cl_uint length,
                                 cl_uint& output)
{
    for(cl_uint i = 0; i < length; ++i)
    {
        output += input[i];
    }
}

int Reduction::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Now add customized options
    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error.\n");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the Input array";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);
    delete array_length;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory Allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    return SDK_SUCCESS;
}


int
Reduction::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("Reduction_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int
Reduction::setup()
{
    if (setupReduction() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);

    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int Reduction::run()
{
    cl_int status = CL_SUCCESS;

    /* Allocate memory for output buffer as output depends on groupSize
       which also depends on device */
    numBlocks = length / ((cl_uint)groupSize * MULTIPLY);
    outputPtr = (cl_uint*)malloc(numBlocks * VECTOR_SIZE * sizeof(cl_uint));
    CHECK_ALLOCATION(outputPtr, "Failed to allocate host memory. (outputPtr)");

    memset(outputPtr, 0, numBlocks * VECTOR_SIZE * sizeof(cl_uint));

    // Create memory objects for temporary output array
    outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       numBlocks * sizeof(cl_uint4),
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------"
              << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    // Run the kernel for a number of iterations
    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute total time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>("Output", &output, 1, 1);
    }

    return SDK_SUCCESS;
}

int Reduction::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
         * reference implementation
         * it overwrites the input array with the output
         */
        reductionCPUReference(input, length * VECTOR_SIZE, refOutput);

        // compare the results and see if they match
        if(refOutput == output)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void Reduction::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[3] = {"Elements", "Time(sec)", "(DataTransfer + Kernel)Time(sec)"};
        std::string stats[3];

        sampleTimer->totalTime = setupTime + kernelTime;
        stats[0]  = toString(length * VECTOR_SIZE, std::dec);
        stats[1]  = toString(sampleTimer->totalTime, std::dec);
        stats[2]  = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 3);
    }
}

int
Reduction::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    return SDK_SUCCESS;
}

Reduction::~Reduction()
{
    // release program resources (input memory etc.)
#ifdef _WIN32
    ALIGNED_FREE(input);
#else
    FREE(input);
#endif

    FREE(outputPtr);
    FREE(devices);
}

int
main(int argc, char * argv[])
{
    Reduction clReduction;

    if (clReduction.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clReduction.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clReduction.sampleArgs->isDumpBinaryEnabled())
    {
        return clReduction.genBinaryImage();
    }

    if(clReduction.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clReduction.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clReduction.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clReduction.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clReduction.printStats();
    return SDK_SUCCESS;
}
