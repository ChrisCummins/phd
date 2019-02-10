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


#include "ScanLargeArrays.hpp"

template<typename T>
int ScanLargeArrays::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
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

    return SDK_SUCCESS;
}

int
ScanLargeArrays::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    return SDK_SUCCESS;
}


int
ScanLargeArrays::setupScanLargeArrays()
{
    // input buffer size
    cl_uint sizeBytes = length * sizeof(cl_float);

    /*
     * Map cl_mem inputBuffer to host for writing
     * Note the usage of CL_MAP_WRITE_INVALIDATE_REGION flag
     * This flag indicates the runtime that whole buffer is mapped for writing and
     * there is no need of device->host transfer. Hence map call will be faster
     */
    int status = mapBuffer( inputBuffer, input,
                            sizeBytes,
                            CL_MAP_WRITE_INVALIDATE_REGION );
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    // random initialisation of input
    fillRandom<cl_float>(input, length, 1, 0, 255);

    // Unless quiet mode has been enabled, print the INPUT array
    if(!sampleArgs->quiet)
    {
        printArray<cl_float>("Input", input, length, 1);
    }

    /* Unmaps cl_mem inputBuffer from host
     * host->device transfer happens if device exists in different address-space
     */
    status = unmapBuffer(inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    // if verification is enabled
    if(sampleArgs->verify)
    {
        // allocate memory for verification output array
        verificationOutput = (cl_float*)malloc(sizeBytes);
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verify)");
        memset(verificationOutput, 0, sizeBytes);
    }

    return SDK_SUCCESS;
}

int
ScanLargeArrays::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("ScanLargeArrays_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Generate Binary Image Failed");
    return SDK_SUCCESS;
}


int
ScanLargeArrays::setupCL(void)
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

    /*
     * If we could find our platform, use it. Otherwise use just available platform.
     */
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
    CHECK_OPENCL_ERROR(status,"CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        if(checkVal(status, 0, "CECL_CREATE_COMMAND_QUEUE failed."))
        {
            return SDK_FAILURE;
        }
    }

    // Get Device specific Information

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("ScanLargeArrays_Kernels.cl");
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
    bScanKernel = CECL_KERNEL(program, "ScanLargeArrays", &status);
    CHECK_OPENCL_ERROR(status,"CECL_KERNEL failed.(bScanKernel)");

    bAddKernel = CECL_KERNEL(program, "blockAddition", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(bAddKernel)");

    // get a kernel object handle for a kernel with the given name
    pScanKernel = CECL_KERNEL(program, "prefixSum", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(pScanKernel)");

    status = kernelInfoBScan.setKernelWorkGroupInfo(bScanKernel,devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, " setKernelWorkGroupInfo() failed");

    status = kernelInfoBAdd.setKernelWorkGroupInfo(pScanKernel,devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, " setKernelWorkGroupInfo() failed");

    status = kernelInfoPScan.setKernelWorkGroupInfo(bAddKernel,devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, " setKernelWorkGroupInfo() failed");

    // Find minimum of all kernel's group-sizes
    size_t temp = min(kernelInfoBScan.kernelWorkGroupSize,
                      kernelInfoPScan.kernelWorkGroupSize);
    temp = (temp > kernelInfoBAdd.kernelWorkGroupSize) ?
           kernelInfoBAdd.kernelWorkGroupSize : temp;

    if(blockSize > (cl_uint)temp)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << blockSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << temp << std::endl;
            std::cout << "Falling back to " << temp << std::endl;
        }
        blockSize = (cl_uint)temp;
    }

    blockSize = min(blockSize,length/2);
    // Calculate number of passes required
    float t = log((float)length) / log((float)blockSize);
    pass = (cl_uint)t;

    // If t is equal to pass
    if(fabs(t - (float)pass) < 1e-7)
    {
        pass--;
    }

    // Create input buffer on device
    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY,
                      sizeof(cl_float) * length,
                      0,
                      &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(inputBuffer)");

    // Allocate output buffers
    outputBuffer = (cl_mem*)malloc(pass * sizeof(cl_mem));

    for(int i = 0; i < (int)pass; i++)
    {
        int size = (int)(length / pow((float)blockSize,(float)i));
        outputBuffer[i] = CECL_BUFFER(
                              context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_float) * size,
                              0,
                              &status);
        CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(outputBuffer)");
    }

    // Allocate blockSumBuffers
    blockSumBuffer = (cl_mem*)malloc(pass * sizeof(cl_mem));

    for(int i = 0; i < (int)pass; i++)
    {
        int size = (int)(length / pow((float)blockSize,(float)(i + 1)));
        blockSumBuffer[i] = CECL_BUFFER(
                                context,
                                CL_MEM_READ_WRITE,
                                sizeof(cl_float) * size,
                                0,
                                &status);

        CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(blockSumBuffer)");
    }

    // Create a tempBuffer on device
    int tempLength = (int)(length / pow((float)blockSize, (float)pass));

    tempBuffer = CECL_BUFFER(context,
                                CL_MEM_READ_WRITE,
                                sizeof(cl_float) * tempLength,
                                0,
                                &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(tempBuffer)");

    return SDK_SUCCESS;
}

int
ScanLargeArrays::bScan(cl_uint len,
                       cl_mem *inputBuffer,
                       cl_mem *outputBuffer,
                       cl_mem *blockSumBuffer)
{
    cl_int status;

    // set the block size
    size_t globalThreads[1]= {len / 2};
    size_t localThreads[1] = {blockSize / 2};

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[0] > deviceInfo.maxWorkGroupSize)
    {
        std::cout<<"Unsupported: Device does not"
                 "support requested number of work items.";

        return SDK_FAILURE;

    }

    // Set appropriate arguments to the kernel

    // 1st argument to the kernel - outputBuffer
    status = CECL_SET_KERNEL_ARG(
                 bScanKernel,
                 0,
                 sizeof(cl_mem),
                 (void *)outputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(outputBuffer)");

    // 2nd argument to the kernel - inputBuffer
    status = CECL_SET_KERNEL_ARG(
                 bScanKernel,
                 1,
                 sizeof(cl_mem),
                 (void *)inputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(inputBuffer)");

    // 3rd argument to the kernel - local memory
    status = CECL_SET_KERNEL_ARG(
                 bScanKernel,
                 2,
                 blockSize * sizeof(cl_float),
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(local memory)");

    // 4th argument to the kernel - block_size
    status = CECL_SET_KERNEL_ARG(
                 bScanKernel,
                 3,
                 sizeof(cl_int),
                 &blockSize);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(blockSize)");

    // 5th argument to the kernel - SumBuffer
    status = CECL_SET_KERNEL_ARG(
                 bScanKernel,
                 4,
                 sizeof(cl_mem),
                 blockSumBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(blockSumBuffer)");

    if(kernelInfoBScan.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient"
                  "local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    // Enqueue a kernel run call
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 bScanKernel,
                 1,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.(commandQueue)");

    return SDK_SUCCESS;
}

int
ScanLargeArrays::pScan(cl_uint len,
                       cl_mem *inputBuffer,
                       cl_mem *outputBuffer)
{
    cl_int status;

    size_t globalThreads[1]= {len / 2};
    size_t localThreads[1] = {len / 2};

    if(kernelInfoPScan.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local memory on device."
                  << std::endl;
        return SDK_FAILURE;
    }

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[0] > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support"
                  "requested number of work items." << std::endl;
        return SDK_FAILURE;
    }
    // Set appropriate arguments to the kernel

    // 1st argument to the kernel - outputBuffer
    status = CECL_SET_KERNEL_ARG(
                 pScanKernel,
                 0,
                 sizeof(cl_mem),
                 (void *)outputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(outputBuffer)");

    // 2nd argument to the kernel - inputBuffer
    status = CECL_SET_KERNEL_ARG(
                 pScanKernel,
                 1,
                 sizeof(cl_mem),
                 (void *)inputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(inputBuffer)");

    // 3rd argument to the kernel - local memory
    status = CECL_SET_KERNEL_ARG(
                 pScanKernel,
                 2,
                 (len+1) * sizeof(cl_float),
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(local memory)");

    // 4th argument to the kernel - block_size
    status = CECL_SET_KERNEL_ARG(
                 pScanKernel,
                 3,
                 sizeof(cl_int),
                 &len);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(blockSize)");

    // Enqueue a kernel run call
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 pScanKernel,
                 1,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.(commandQueue)");

    return SDK_SUCCESS;
}

int
ScanLargeArrays::bAddition(cl_uint len,
                           cl_mem *inputBuffer,
                           cl_mem *outputBuffer)
{
    cl_int   status;

    // set the block size
    size_t globalThreads[1]= {len};
    size_t localThreads[1] = {blockSize};

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[0] > deviceInfo.maxWorkGroupSize)
    {
        std::cout<<"Unsupported: Device does not support"
                 "requested number of work items.";
        return SDK_FAILURE;
    }

    // 1st argument to the kernel - inputBuffer
    status = CECL_SET_KERNEL_ARG(
                 bAddKernel,
                 0,
                 sizeof(cl_mem),
                 (void*)inputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(outputBuffer)");

    // 2nd argument to the kernel - outputBuffer
    status = CECL_SET_KERNEL_ARG(
                 bAddKernel,
                 1,
                 sizeof(cl_mem),
                 (void *)outputBuffer);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(inputBuffer)");

    if(kernelInfoBAdd.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local memory on device."
                  << std::endl;
        return SDK_FAILURE;
    }

    // Enqueue a kernel run call
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 bAddKernel,
                 1,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.(commandQueue)");

    return SDK_SUCCESS;
}


int
ScanLargeArrays::runCLKernels(void)
{

    // Do block-wise sum
    if(bScan(length, &inputBuffer, &outputBuffer[0], &blockSumBuffer[0]))
    {
        return SDK_FAILURE;
    }

    for(int i = 1; i < (int)pass; i++)
    {
        if(bScan((cl_uint)(length / pow((float)blockSize, (float)i)),
                 &blockSumBuffer[i - 1],
                 &outputBuffer[i],
                 &blockSumBuffer[i]))
        {
            return SDK_FAILURE;
        }
    }

    int tempLength = (int)(length / pow((float)blockSize, (float)pass));

    // Do scan to tempBuffer
    if(pScan(tempLength, &blockSumBuffer[pass - 1], &tempBuffer))
    {
        return SDK_FAILURE;
    }

    // Do block-addition on outputBuffers
    if(bAddition((cl_uint)(length / pow((float)blockSize, (float)(pass - 1))),
                 &tempBuffer, &outputBuffer[pass - 1]))
    {
        return SDK_FAILURE;
    }

    for(int i = pass - 1; i > 0; i--)
    {
        if(bAddition((cl_uint)(length / pow((float)blockSize, (float)(i - 1))),
                     &outputBuffer[i], &outputBuffer[i - 1]))
        {
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

/*
* Naive implementation of Scan
*/
void
ScanLargeArrays::scanLargeArraysCPUReference(
    cl_float * output,
    cl_float * input,
    const cl_uint length)
{
    output[0] = 0;

    for(cl_uint i = 1; i < length; ++i)
    {
        output[i] = input[i-1] + output[i-1];
    }
}

int ScanLargeArrays::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length,"Memory Allocation error.(array_length)");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the input array";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);
    delete array_length;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option,"Memory Allocation error.(iteration_option)");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    return SDK_SUCCESS;
}

int ScanLargeArrays::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    if(isPowerOf2(length) != SDK_SUCCESS)
    {
        length = roundToPowerOf2(length);
    }

    if((length/blockSize>GROUP_SIZE)&&(((length)&(length-1))!=0))
    {
        std::cout <<"Invalid length."<<std::endl;
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(setupScanLargeArrays() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int ScanLargeArrays::run()
{
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " <<
              iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" <<
              std::endl;

    // create and initialize timers
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
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}

int ScanLargeArrays::verifyResults()
{
    if(sampleArgs->verify)
    {
        /*
         * Map cl_mem inputBuffer to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        int status = mapBuffer( inputBuffer, input,
                                (length * sizeof(cl_float)),
                                CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

        // reference implementation
        scanLargeArraysCPUReference(verificationOutput, input, length);

        /*
         * Unmap cl_mem inputBuffer from host
         * there will be no data-transfers since cl_mem inputBuffer was mapped for reading
         */
        status = unmapBuffer(inputBuffer, input);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

        /*
         * Map cl_mem outputBuffer[0] to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        status = mapBuffer( outputBuffer[0], output,
                            (length * sizeof(cl_float)),
                            CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to map device buffer.(outputBuffer[0])");

        // compare the results and see if they match
        bool pass = compare(output, verificationOutput, length, (float)0.001);

        if(!sampleArgs->quiet)
        {
            printArray<cl_float>("Output", output, length, 1);
        }

        /*
         * Unmap cl_mem outputBuffer[0] from host
         * there will be no data-transfers since cl_mem outputBuffer was mapped for reading
         */
        status = unmapBuffer(outputBuffer[0], output);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(outputBuffer[0])");

        if(pass)
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

void ScanLargeArrays::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Setup time (sec)", "Avg. kernel time (sec)", "Elements/sec"};
        std::string stats[4];

        double avgTime = (kernelTime / iterations);

        stats[0]  = toString(length, std::dec);
        stats[1]  = toString(setupTime, std::dec);
        stats[2]  = toString(avgTime, std::dec);
        stats[3]  = toString((length / avgTime), std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int
ScanLargeArrays::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(pScanKernel);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(pScanKernel))");

    status = clReleaseKernel(bScanKernel);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(bScanKernel))");

    status = clReleaseKernel(bAddKernel);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(bAddKernel))");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(program))");

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(tempBuffer))");

    status = clReleaseMemObject(tempBuffer);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(tempBuffer))");

    for(int i = 0; i < (int)pass; i++)
    {
        status = clReleaseMemObject(outputBuffer[i]);
        CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(outputBuffer))");

        status = clReleaseMemObject(blockSumBuffer[i]);
        CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(blockSumBuffer))");
    }

    FREE(outputBuffer);
    FREE(blockSumBuffer);

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status,"clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status,"clReleaseContext failed.(context)");
    // release program resources (input memory etc.)
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{

    ScanLargeArrays clScanLargeArrays;

    // Initialize
    if(clScanLargeArrays.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clScanLargeArrays.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clScanLargeArrays.sampleArgs->isDumpBinaryEnabled())
    {
        return clScanLargeArrays.genBinaryImage();
    }

    // Setup
    if(clScanLargeArrays.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    //Run
    if(clScanLargeArrays.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // VerifyResults
    if(clScanLargeArrays.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup
    if(clScanLargeArrays.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clScanLargeArrays.printStats();
    return SDK_SUCCESS;
}
