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


#include "PrefixSum.hpp"

int PrefixSum::setupPrefixSum()
{
    if(length < 2)
    {
        length = 2;
    }
    // allocate and init memory used by host
    cl_uint sizeBytes = length * sizeof(cl_float);

    input = (cl_float *) malloc(sizeBytes);
    CHECK_ALLOCATION(input, "Failed to allocate host memory. (input)");
    // random initialisation of input
    fillRandom<cl_float>(input, length, 1, 0, 10);

    if(sampleArgs->verify)
    {
        verificationOutput = (cl_float *) malloc(sizeBytes);
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");
        memset(verificationOutput, 0, sizeBytes);
    }
    // Unless quiet mode has been enabled, print the INPUT array
    if(!sampleArgs->quiet)
    {
        printArray<cl_float>("Input : ", input, length, 1);
    }

    return SDK_SUCCESS;
}

int
PrefixSum::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("PrefixSum_Kernels.cl");
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
PrefixSum::setupCL(void)
{
    cl_int status = 0;
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

    // Get platform
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    // If we could find our platform, use it. Otherwise use just available platform.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = CECL_CREATE_CONTEXT_FROM_TYPE(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    //Set device info of given cl_device_id
    status = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Create command queue
    commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                        devices[sampleArgs->deviceId],
                                        0,
                                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY,
                      sizeof(cl_float) * length,
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY,
                       sizeof(cl_float) * length,
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("PrefixSum_Kernels.cl");
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
    group_kernel = CECL_KERNEL(program, "group_prefixSum", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL::group_prefixSum failed.");

    global_kernel = CECL_KERNEL(program, "global_prefixSum", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL::global_prefixSum failed.");

    // Move data host to device
    cl_float *ptr;
    status = mapBuffer( inputBuffer, ptr, length, CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(textBuf)");
    memcpy(ptr, input, (length * sizeof(cl_float)));
    status = unmapBuffer(inputBuffer, ptr);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    return SDK_SUCCESS;
}

int
PrefixSum::runGroupKernel(size_t offset)
{
    size_t dataSize = length/offset;
    size_t localThreads = kernelInfo.kernelWorkGroupSize;
    size_t globalThreads = (dataSize+1) / 2;    // Actual threads needed
    // Set global thread size multiple of local thread size.
    globalThreads = ((globalThreads + localThreads - 1) / localThreads) *
                    localThreads;

    // Set appropriate arguments to the kernel
    // 1st argument to the kernel - outputBuffer
    int status = CECL_SET_KERNEL_ARG(
                     group_kernel,
                     0,
                     sizeof(cl_mem),
                     (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(outputBuffer)");
    // 2nd argument to the kernel - inputBuffer
    status = CECL_SET_KERNEL_ARG(
                 group_kernel,
                 1,
                 sizeof(cl_mem),
                 // After the 1st kernel run, we read the input from outputBuffer and update in outputBuffer.
                 (offset>1) ? (void *)&outputBuffer  : (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(inputBuffer)");
    // 3rd argument to the kernel - local memory
    status = CECL_SET_KERNEL_ARG(
                 group_kernel,
                 2,
                 2*localThreads*sizeof(cl_float),
                 NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(kernel)");
    // 4th argument to the kernel - length
    status = CECL_SET_KERNEL_ARG(
                 group_kernel,
                 3,
                 sizeof(cl_int),
                 (void*)&length);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(length)");
    // 5th argument to the kernel - memory offset for each input element
    status = CECL_SET_KERNEL_ARG(
                 group_kernel,
                 4,
                 sizeof(cl_int),
                 (void*)&offset);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(offset)");
    // Enqueue a kernel run call
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 group_kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    return SDK_SUCCESS;
}

int
PrefixSum::runGlobalKernel(size_t offset)
{
    size_t localThreads = kernelInfo.kernelWorkGroupSize;
    size_t localDataSize = localThreads << 1;   // Each thread work on 2 elements

    // Set number of threads needed for global_kernel.
    size_t globalThreads = length - offset;
    globalThreads -= (globalThreads / (offset * localDataSize)) * offset;

    // Set global thread size multiple of local thread size.
    globalThreads = ((globalThreads + localThreads - 1) / localThreads) *
                    localThreads;

    // Set appropriate arguments to the kernel
    // 1st argument to the kernel - Global Buffer
    int status = CECL_SET_KERNEL_ARG(global_kernel,
                                0,
                                sizeof(cl_mem),
                                (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(outputBuffer)");
    // 2nd argument to the kernel - offset
    status = CECL_SET_KERNEL_ARG(global_kernel,
                            1,
                            sizeof(cl_int),
                            (void*)&offset);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(offset)");
    // 3rd argument to the kernel - offset
    status = CECL_SET_KERNEL_ARG(global_kernel,
                            2,
                            sizeof(cl_int),
                            (void*)&length);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(length)");
    // Run the kernel
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 global_kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    return SDK_SUCCESS;
}

int
PrefixSum::runCLKernels(void)
{
    cl_int status;

    status =  kernelInfo.setKernelWorkGroupInfo(group_kernel,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");
    size_t localThreads = kernelInfo.kernelWorkGroupSize;
    size_t localDataSize = localThreads << 1;   // Each thread work on 2 elements

    cl_ulong availableLocalMemory = deviceInfo.localMemSize -
                                    kernelInfo.localMemoryUsed;
    cl_ulong neededLocalMemory = localDataSize * sizeof(cl_float);
    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return SDK_SUCCESS;
    }

    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 inputBuffer,
                 CL_FALSE,
                 0,
                 sizeof(cl_float) * length,
                 input,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed.(inputBuffer)");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");
    status = waitForEventAndRelease(&writeEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(writeEvt) Failed");

    for(size_t offset=1; offset<length; offset *= localDataSize)
    {
        if ((length/offset) > 1)  // Need atlest 2 element for process the kernel
        {
            if(runGroupKernel(offset) != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }

        // Call global_kernel for update all elements
        if(offset > 1)
        {
            if(runGlobalKernel(offset) != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
    }

    return SDK_SUCCESS;
}

void
PrefixSum::prefixSumCPUReference(
    cl_float * output,
    cl_float * input,
    const cl_uint length)
{
    output[0] = input[0];

    for(cl_uint i = 1; i< length; ++i)
    {
        output[i] = input[i] + output[i-1];
    }
}

int PrefixSum::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error. (array_length)");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the input array";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);
    delete array_length;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error. (num_iterations)");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    return SDK_SUCCESS;
}

int PrefixSum::setup()
{
    if(!isPowerOf2(length))
    {
        length = roundToPowerOf2(length);
    }
    if(setupPrefixSum() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if (setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    setupTime = (cl_double)sampleTimer->readTimer(timer);
    return SDK_SUCCESS;
}


int PrefixSum::run()
{
    int status = 0;

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    kernelTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}

int PrefixSum::verifyResults()
{
    int status = SDK_SUCCESS;
    if(sampleArgs->verify)
    {
        // Read the device output buffer
        cl_float *ptrOutBuff;
        int status = mapBuffer (outputBuffer, ptrOutBuff,  length * sizeof(cl_float),
                                CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(resultBuf)");

        // reference implementation
        prefixSumCPUReference(verificationOutput, input, length);

        // compare the results and see if they match
        float epsilon = length * 1e-7f;
        if(compare(ptrOutBuff, verificationOutput, length, epsilon))
        {
            std::cout << "Passed!\n" << std::endl;
            status = SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            status = SDK_FAILURE;
        }

        if(!sampleArgs->quiet)
        {
            printArray<cl_float>("Output : ", ptrOutBuff, length, 1);
        }

        // un-map outputBuffer
        int result = unmapBuffer(outputBuffer, ptrOutBuff);
        CHECK_ERROR(result, SDK_SUCCESS,
                    "Failed to unmap device buffer.(resultCountBuf)");
    }
    return status;
}

void PrefixSum::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Samples",
            "Setup Time(sec)",
            "Avg. kernel time (sec)",
            "Samples used /sec"
        };
        std::string stats[4];
        double avgKernelTime = kernelTime / iterations;

        stats[0] = toString(length, std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString(avgKernelTime, std::dec);
        stats[3] = toString((length/avgKernelTime), std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int PrefixSum::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status = 0;

    status = clReleaseKernel(group_kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(program)");

    status = clReleaseKernel(global_kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(program)");

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

template<typename T>
int PrefixSum::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
                         size_t sizeInBytes, cl_map_flags flags)
{
    cl_int status;
    hostPointer = (T*) CECL_MAP_BUFFER(commandQueue,
                                          deviceBuffer,
                                          CL_TRUE,
                                          flags,
                                          0,
                                          sizeInBytes,
                                          0,
                                          NULL,
                                          NULL,
                                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

    return SDK_SUCCESS;
}

int
PrefixSum::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{

    PrefixSum clPrefixSum;
    // Initialize
    if(clPrefixSum.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clPrefixSum.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clPrefixSum.sampleArgs->isDumpBinaryEnabled())
    {
        //GenBinaryImage
        return clPrefixSum.genBinaryImage();
    }

    // Setup
    if(clPrefixSum.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Run
    if(clPrefixSum.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // VerifyResults
    if(clPrefixSum.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup
    if (clPrefixSum.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clPrefixSum.printStats();
    return SDK_SUCCESS;
}
