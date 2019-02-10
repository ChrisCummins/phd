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


#include "Histogram.hpp"

#include <math.h>

int
Histogram::calculateHostBin()
{
    int status = mapBuffer( dataBuf, data, sizeof(cl_uint) * width * height,
                            CL_MAP_READ);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(dataBuf in calcHostBin)");

    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            hostBin[data[i * width + j]]++;
        }
    }

    status = unmapBuffer( dataBuf, data );
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(dataBuf in calcHostBin)");

    return SDK_SUCCESS;
}

template<typename T>
int Histogram::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
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
Histogram::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
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
Histogram::setupHistogram()
{
    int i = 0;

    int status = mapBuffer( dataBuf, data, sizeof(cl_uint) * width * height,
                            CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(dataBuf)");

    for(i = 0; i < width * height; i++)
    {
        data[i] = rand() % (cl_uint)(binSize);
    }

    status = unmapBuffer( dataBuf, data );
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(dataBuf)");

    hostBin = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    CHECK_ALLOCATION(hostBin, "Failed to allocate host memory. (hostBin)");

    memset(hostBin, 0, binSize * sizeof(cl_uint));

    deviceBin = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    CHECK_ALLOCATION(deviceBin, "Failed to allocate host memory. (deviceBin)");

    memset(deviceBin, 0, binSize * sizeof(cl_uint));

    return SDK_SUCCESS;
}

int
Histogram::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("Histogram_Kernels.cl");
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
Histogram::setupCL(void)
{
    cl_int status = 0;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_GPU;
    }
    else //sampleArgs->deviceType = "gpu"
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
    if(scalar && vector)//if both options are specified
    {
        std::cout<<"Ignoring --scalar and --vector option and using the default vector width of the device"<<std::endl;
        vectorWidth = deviceInfo.preferredFloatVecWidth;
    }

    else if(scalar)
    {
        vectorWidth = 1;
    }
    else if(vector)
    {
        vectorWidth = 4;
    }
    else //if no option is specified.
    {
        vectorWidth = deviceInfo.preferredFloatVecWidth;
    }

    if(!sampleArgs->quiet)
    {
        if(vectorWidth == 1)
        {
            std::cout<<"Selecting scalar kernel\n"<<std::endl;
        }
        else
        {
            std::cout<<"Selecting vector kernel\n"<<std::endl;
        }
    }

    subHistgCnt = (width * height) / (groupSize * groupIterations);

    // Check if byte-addressable store is supported
    if(!strstr(deviceInfo.extensions, "cl_khr_byte_addressable_store"))
    {
        byteRWSupport = false;
        OPENCL_EXPECTED_ERROR("Device does not support cl_khr_byte_addressable_store extension!");
    }

    dataBuf = CECL_BUFFER(
                  context,
                  CL_MEM_READ_ONLY,
                  sizeof(cl_uint) * width  * height,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (dataBuf)");

    midDeviceBinBuf = CECL_BUFFER(
                          context,
                          CL_MEM_WRITE_ONLY,
                          sizeof(cl_uint) * binSize * subHistgCnt,
                          NULL,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (midDeviceBinBuf)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("Histogram_Kernels.cl");
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
    CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    const char *kernelName = (vectorWidth == 4)? "histogram256_vector":
                             "histogram256_scalar";

    kernel = CECL_KERNEL(program, kernelName, &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}

int Histogram::setWorkGroupSize()
{
    cl_int status = 0;
    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, " setKernelWorkGroupInfo() failed");

    if((size_t)groupSize > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
        }
        groupSize = (cl_int)kernelInfo.kernelWorkGroupSize;
    }

    globalThreads = (width * height) / (GROUP_ITERATIONS);

    localThreads = groupSize;

    if(localThreads > deviceInfo.maxWorkItemSizes[0] ||
            localThreads > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not"
                  << "support requested number of work items." << std::endl;
        return SDK_FAILURE;
    }

    if(kernelInfo.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local "
                  << " memory on device." << std::endl;
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

int
Histogram::runCLKernels(void)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;

    status = this->setWorkGroupSize();
    CHECK_ERROR(status, SDK_SUCCESS, "setKernelWorkGroupSize() failed");

    // whether sort is to be in increasing order. CL_TRUE implies increasing
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), (void*)&dataBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (dataBuf)");

    status = CECL_SET_KERNEL_ARG(kernel, 1, groupSize * binSize * sizeof(cl_uchar),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (local memory)");

    status = CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), (void*)&midDeviceBinBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (deviceBinBuf)");

    // Enqueue a kernel run call.

    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt1) Failed");

    status = mapBuffer( midDeviceBinBuf, midDeviceBin,
                        subHistgCnt * binSize * sizeof(cl_uint), CL_MAP_READ);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(midDeviceBinBuf)");

    // Clear deviceBin array
    memset(deviceBin, 0, binSize * sizeof(cl_uint));

    // Calculate final histogram bin
    for(int i = 0; i < subHistgCnt; ++i)
    {
        for(int j = 0; j < binSize; ++j)
        {
            deviceBin[j] += midDeviceBin[i * binSize + j];
        }
    }

    status = unmapBuffer( midDeviceBinBuf, midDeviceBin);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(midDeviceBinBuf)");

    return SDK_SUCCESS;
}

int
Histogram::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* width_option = new Option;
    CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

    width_option->_sVersion = "x";
    width_option->_lVersion = "width";
    width_option->_description = "Width of the input";
    width_option->_type = CA_ARG_INT;
    width_option->_value = &width;

    sampleArgs->AddOption(width_option);
    delete width_option;

    Option* height_option = new Option;
    CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

    height_option->_sVersion = "y";
    height_option->_lVersion = "height";
    height_option->_description = "Height of the input";
    height_option->_type = CA_ARG_INT;
    height_option->_value = &height;

    sampleArgs->AddOption(height_option);
    delete height_option;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* scalar_option = new Option;
    CHECK_ALLOCATION(scalar_option, "Memory allocation error.\n");

    scalar_option->_sVersion = "";
    scalar_option->_lVersion = "scalar";
    scalar_option->_description =
        "Run scalar version of the kernel (--scalar and --vector options are mutually exclusive)";
    scalar_option->_type = CA_NO_ARGUMENT;
    scalar_option->_value = &scalar;

    sampleArgs->AddOption(scalar_option);
    delete scalar_option;

    Option* vector_option = new Option;
    CHECK_ALLOCATION(vector_option, "Memory allocation error.\n");

    vector_option->_sVersion = "";
    vector_option->_lVersion = "vector";
    vector_option->_description =
        "Run vector version of the kernel (--scalar and --vector options are mutually exclusive)";
    vector_option->_type = CA_NO_ARGUMENT;
    vector_option->_value = &vector;

    sampleArgs->AddOption(vector_option);
    delete vector_option;



    return SDK_SUCCESS;
}

int
Histogram::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int status = 0;

    /* width must be multiples of binSize and
     * height must be multiples of groupSize
     */
    width = (width / binSize ? width / binSize: 1) * binSize;
    height = (height / groupSize ? height / groupSize: 1) * groupSize;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = setupHistogram();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample Resource Setup Failed");

    sampleTimer->stopTimer(timer);

    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
Histogram::run()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

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
    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer));

    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>("deviceBin", deviceBin, binSize, 1);
    }

    return SDK_SUCCESS;
}

int
Histogram::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        /**
         * Reference implementation on host device
         * calculates the histogram bin on host
         */
        calculateHostBin();

        // compare the results and see if they match
        bool result = true;
        for(int i = 0; i < binSize; ++i)
        {
            if(hostBin[i] != deviceBin[i])
            {
                result = false;
                break;
            }
        }

        if(result)
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

void Histogram::printStats()
{
    if(sampleArgs->timing)
    {
        if(!byteRWSupport)
        {
            return;
        }

        // calculate total time
        double avgKernelTime = kernelTime/iterations;

        std::string strArray[5] =
        {
            "Width",
            "Height",
            "Setup Time(sec)",
            "Avg. Kernel Time (sec)",
            "Elements/sec"
        };
        std::string stats[5];

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(setupTime, std::dec);
        stats[3] = toString(avgKernelTime, std::dec);
        stats[4] = toString(((width*height)/avgKernelTime), std::dec);

        printStatistics(strArray, stats, 5);
    }
}

int Histogram::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(dataBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(dataBuf)");

    status = clReleaseMemObject(midDeviceBinBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(midDeviceBinBuf)");

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // Release program resources (input memory etc.)
    FREE(hostBin);
    FREE(deviceBin);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    int status = 0;
    // Create MonteCalroAsian object
    Histogram clHistogram;

    // Initialization
    if(clHistogram.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(clHistogram.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clHistogram.sampleArgs->isDumpBinaryEnabled())
    {
        return clHistogram.genBinaryImage();
    }

    // Setup
    status = clHistogram.setup();
    if(status != SDK_SUCCESS)
    {
        return (status == SDK_EXPECTED_FAILURE)? SDK_SUCCESS : SDK_FAILURE;
    }

    // Run
    if(clHistogram.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Verify
    if(clHistogram.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup resources created
    if(clHistogram.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Print performance statistics
    clHistogram.printStats();

    return SDK_SUCCESS;
}
