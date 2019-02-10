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


#include "BlackScholesDP.hpp"

#include <math.h>
#include <malloc.h>


/**
 *  Constants
 */

#define S_LOWER_LIMIT 10.0
#define S_UPPER_LIMIT 100.0
#define K_LOWER_LIMIT 10.0
#define K_UPPER_LIMIT 100.0
#define T_LOWER_LIMIT 1.0
#define T_UPPER_LIMIT 10.0
#define R_LOWER_LIMIT 0.01
#define R_UPPER_LIMIT 0.05
#define SIGMA_LOWER_LIMIT 0.01
#define SIGMA_UPPER_LIMIT 0.10

double
BlackScholesDP::phi(double X)
{
    double y, absX, t;

    // the coeffs
    const double c1 =  0.319381530;
    const double c2 = -0.356563782;
    const double c3 =  1.781477937;
    const double c4 = -1.821255978;
    const double c5 =  1.330274429;

    const double oneBySqrt2pi = 0.398942280;

    absX = fabs(X);
    t = 1.0 / (1.0 + 0.2316419 * absX);

    y = 1.0 - oneBySqrt2pi * exp(-X * X / 2.0) *
        t * (c1 +
             t * (c2 +
                  t * (c3 +
                       t * (c4 + t * c5))));

    return (X < 0) ? (1.0 - y) : y;
}

void
BlackScholesDP::blackScholesDPCPU()
{
    int y;
    for (y = 0; y < width * height * 4; ++y)
    {
        double d1, d2;
        double sigmaSqrtT;
        double KexpMinusRT;
        double s = S_LOWER_LIMIT * randArray[y] + S_UPPER_LIMIT * (1.0 - randArray[y]);
        double k = K_LOWER_LIMIT * randArray[y] + K_UPPER_LIMIT * (1.0 - randArray[y]);
        double t = T_LOWER_LIMIT * randArray[y] + T_UPPER_LIMIT * (1.0 - randArray[y]);
        double r = R_LOWER_LIMIT * randArray[y] + R_UPPER_LIMIT * (1.0 - randArray[y]);
        double sigma = SIGMA_LOWER_LIMIT * randArray[y] + SIGMA_UPPER_LIMIT *
                       (1.0 - randArray[y]);

        sigmaSqrtT = sigma * sqrt(t);

        d1 = (log(s / k) + (r + sigma * sigma / 2.0) * t) / sigmaSqrtT;
        d2 = d1 - sigmaSqrtT;

        KexpMinusRT = k * exp(-r * t);
        hostCallPrice[y] = s * phi(d1) - KexpMinusRT * phi(d2);
        hostPutPrice[y]  = KexpMinusRT * phi(-d2) - s * phi(-d1);
    }
}

int
BlackScholesDP::setupBlackScholesDP()
{
    int i = 0;

    // Calculate width and height from samples
    samples = samples / 4;
    samples = (samples / GROUP_SIZE)? (samples / GROUP_SIZE) * GROUP_SIZE:
              GROUP_SIZE;

    unsigned int tempVar1 = (unsigned int)sqrt((double)samples);
    tempVar1 = (tempVar1 / GROUP_SIZE)? (tempVar1 / GROUP_SIZE) * GROUP_SIZE:
               GROUP_SIZE;
    samples = tempVar1 * tempVar1;

    width = tempVar1;
    height = width;

#if defined (_WIN32)
    randArray = (cl_double*)_aligned_malloc(width * height * sizeof(cl_double4),
                                            16);
#else
    randArray = (cl_double*)memalign(16, width * height * sizeof(cl_double4));
#endif

    if(randArray == NULL)
    {
        error("Failed to allocate host memory. (randArray)");
        return SDK_FAILURE;
    }
    for(i = 0; i < width * height * 4; i++)
    {
        randArray[i] = (double)rand() / (double)RAND_MAX;
    }

    deviceCallPrice = (cl_double*)malloc(width * height * sizeof(cl_double4));
    CHECK_ALLOCATION(deviceCallPrice,
                     "Failed to allocate host memory. (deviceCallPrice)");
    memset(deviceCallPrice, 0, width * height * sizeof(cl_double4));

    devicePutPrice = (cl_double*)malloc(width * height * sizeof(cl_double4));
    CHECK_ALLOCATION(devicePutPrice,
                     "Failed to allocate host memory. (devicePutPrice)");
    memset(devicePutPrice, 0, width * height * sizeof(cl_double4));

    hostCallPrice = (cl_double*)malloc(width * height * sizeof(cl_double4));
    CHECK_ALLOCATION(hostCallPrice,
                     "Failed to allocate host memory. (hostCallPrice)");
    memset(hostCallPrice, 0, width * height * sizeof(cl_double4));

    hostPutPrice = (cl_double*)malloc(width * height * sizeof(cl_double4));
    CHECK_ALLOCATION(hostPutPrice,
                     "Failed to allocate host memory. (hostPutPrice)");
    memset(hostPutPrice, 0, width * height * sizeof(cl_double4));

    return SDK_SUCCESS;
}

int
BlackScholesDP::genBinaryImage()
{
    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    bifData binaryData;
    binaryData.kernelName = std::string("BlackScholesDP_Kernels.cl");
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
BlackScholesDP::setupCL(void)
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

    context = CECL_CREATE_CONTEXT_FROM_TYPE(cps,
                                      dType,
                                      NULL,
                                      NULL,
                                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
    status = getDevices(context,&devices,sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                            devices[sampleArgs->deviceId],
                                            prop,
                                            &status);
        CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

    }

    // Get Device specific Information
    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    std::string buildOptions = std::string("");
    // Check if cl_khr_fp64 extension is supported
    if(strstr(deviceInfo.extensions, "cl_khr_fp64"))
    {
        buildOptions.append("-D KHR_DP_EXTENSION");
    }
    else
    {
        /* Check if cl_amd_fp64 extension is supported */
        if(!strstr(deviceInfo.extensions, "cl_amd_fp64"))
        {
            OPENCL_EXPECTED_ERROR("Device does not support cl_amd_fp64 extension!");
        }
    }

    // Exit if SDK version is 2.2 or less
    std::string deviceVersionStr = std::string(deviceInfo.deviceVersion);
    size_t vStart = deviceVersionStr.find_last_of("v");
    size_t vEnd = deviceVersionStr.find(" ", vStart);
    std::string vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);
    if(vStrVal.compare("2.2") <= 0 && dType == CL_DEVICE_TYPE_GPU)
    {
        OPENCL_EXPECTED_ERROR("Few double math functions are not supported in SDK2.2 or less!");
    }

    // Exit if OpenCL version is 1.0
    vStart = deviceVersionStr.find(" ", 0);
    vEnd = deviceVersionStr.find(" ", vStart + 1);
    vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);
    if(vStrVal.compare("1.0") <= 0 && dType == CL_DEVICE_TYPE_GPU)
    {
        OPENCL_EXPECTED_ERROR("Unsupported device!");
    }

    //Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    randBuf = CECL_BUFFER(context,
                             inMemFlags,
                             sizeof(cl_double4) * width  * height,
                             NULL,
                             &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(randBuf)");


    callPriceBuf = CECL_BUFFER(context,
                                  CL_MEM_WRITE_ONLY,
                                  sizeof(cl_double4) * width * height,
                                  NULL,
                                  &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(callPriceBuf)");


    putPriceBuf = CECL_BUFFER(context,
                                 CL_MEM_WRITE_ONLY,
                                 sizeof(cl_double4) * width * height,
                                 NULL,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(putPriceBuf)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("BlackScholesDP_Kernels.cl");
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
    kernel = CECL_KERNEL(program, "blackScholes", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(kernel)");

    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "kernelInfo.setKernelWorkGroupInfo failed");

    // Calculte 2D block size according to required work-group size by kernel
    kernelInfo.kernelWorkGroupSize = kernelInfo.kernelWorkGroupSize > GROUP_SIZE ?
                                     GROUP_SIZE : kernelInfo.kernelWorkGroupSize;
    while((blockSizeX * blockSizeY) < kernelInfo.kernelWorkGroupSize)
    {
        bool next = false;
        if(2 * blockSizeX * blockSizeY <= kernelInfo.kernelWorkGroupSize)
        {
            blockSizeX <<= 1;
            next = true;
        }
        if(2 * blockSizeX * blockSizeY <= kernelInfo.kernelWorkGroupSize)
        {
            next = true;
            blockSizeY <<= 1;
        }

        // Break if no if statement is executed
        if(next == false)
        {
            break;
        }
    }

    return SDK_SUCCESS;
}


int
BlackScholesDP::runCLKernels(void)
{
    cl_int   status;
    cl_event ndrEvt;
    size_t globalThreads[2] = {width, height};
    size_t localThreads[2] = {blockSizeX, blockSizeY};

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
            (size_t)blockSizeX * blockSizeY > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support"
                  "requested number of work items.";
        return SDK_FAILURE;
    }

    cl_event inMapEvt;
    void* mapPtr = CECL_MAP_BUFFER(
                       commandQueue,
                       randBuf,
                       CL_FALSE,
                       CL_MAP_WRITE,
                       0,
                       sizeof(cl_double4) * width  * height,
                       0,
                       NULL,
                       &inMapEvt,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed.(randBuf)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");


    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
                     inMapEvt,
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     sizeof(cl_int),
                     &eventStatus,
                     NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventInfo failed.");
    }

    status = clReleaseEvent(inMapEvt);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent failed.(inMapEvt)");
    memcpy(mapPtr, randArray, sizeof(cl_double4) * width  * height);

    cl_event unmapEvent;
    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 randBuf,
                 mapPtr,
                 0,
                 NULL,
                 &unmapEvent);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed.(randBuf)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(randBuf)");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
                     unmapEvent,
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     sizeof(cl_int),
                     &eventStatus,
                     NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventInfo failed.");
    }

    status = clReleaseEvent(unmapEvent);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(unmapEvent)");

    // whether sort is to be in increasing order. CL_TRUE implies increasing
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), (void *)&randBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(randBuf)");

    status = CECL_SET_KERNEL_ARG(kernel, 1, sizeof(width), (const void *)&width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(width)");

    status = CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), (void *)&callPriceBuf);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed.(callPriceBuf)");

    status = CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), (void *)&putPriceBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed.(putPriceBuf)");

    /*
     * Enqueue a kernel run call.
     */
    status = CECL_ND_RANGE_KERNEL(commandQueue,
                                    kernel,
                                    2,
                                    NULL,
                                    globalThreads,
                                    localThreads,
                                    0,
                                    NULL,
                                    &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    // wait for the kernel call to finish execution
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");


    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    cl_event callEvent;
    cl_event putEvent;

    // Enqueue the results to application pointer
    status = CECL_READ_BUFFER(commandQueue,
                                 callPriceBuf,
                                 CL_FALSE,
                                 0,
                                 width * height * sizeof(cl_double4),
                                 deviceCallPrice,
                                 0,
                                 NULL,
                                 &callEvent);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");


    // Enqueue the results to application pointer
    status = CECL_READ_BUFFER(commandQueue,
                                 putPriceBuf,
                                 CL_FALSE,
                                 0,
                                 width * height * sizeof(cl_double4),
                                 devicePutPrice,
                                 0,
                                 NULL,
                                 &putEvent);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commanQueue)");

    status = waitForEventAndRelease(&callEvent);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(callEvent) Failed");

    status = waitForEventAndRelease(&putEvent);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(putEvent) Failed");

    return SDK_SUCCESS;
}

int
BlackScholesDP::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* num_samples = new Option;
    CHECK_ALLOCATION(num_samples, "Failed to allocate memory (num_samples)");

    num_samples->_sVersion = "x";
    num_samples->_lVersion = "samples";
    num_samples->_description = "Number of samples to be calculated";
    num_samples->_type = CA_ARG_INT;
    num_samples->_value = &samples;

    sampleArgs->AddOption(num_samples);

    num_samples->_sVersion = "i";
    num_samples->_lVersion = "iterations";
    num_samples->_description = "Number of iterations";
    num_samples->_type = CA_ARG_INT;
    num_samples->_value = &iterations;

    sampleArgs->AddOption(num_samples);

    delete num_samples;

    return SDK_SUCCESS;
}

int
BlackScholesDP::setup()
{
    if(setupBlackScholesDP() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);


    int returnVal = setupCL();
    if(returnVal != SDK_SUCCESS)
    {
        return returnVal;
    }

    sampleTimer->stopTimer(timer);
    /* Compute setup time */
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
BlackScholesDP::run()
{

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout <<"-------------------------------------------" << std::endl;

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
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<cl_double>("deviceCallPrice",
                              deviceCallPrice,
                              width,
                              1);

        printArray<cl_double>("devicePutPrice",
                              devicePutPrice,
                              width,
                              1);
    }

    return SDK_SUCCESS;
}

int
BlackScholesDP::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        blackScholesDPCPU();

        if(!sampleArgs->quiet)
        {
            printArray<cl_double>("hostCallPrice",
                                  hostCallPrice,
                                  width,
                                  1);

            printArray<cl_double>("hostPutPrice",
                                  hostPutPrice,
                                  width,
                                  1);
        }
        // compare the call/put price results and see if they match
        bool callPriceResult = compare(hostCallPrice, deviceCallPrice,
                                       width * height * 4);
        bool putPriceResult = compare(hostPutPrice, devicePutPrice, width * height * 4,
                                      1e-4);

        if(!(callPriceResult ? (putPriceResult ? true : false) : false))
        {
            std::cout << "Failed\n"  << std::endl;
            return SDK_FAILURE;
        }
        else
        {
            std::cout << "Passed!\n"  << std::endl;
            return SDK_SUCCESS;
        }
    }

    return SDK_SUCCESS;
}

void
BlackScholesDP::printStats()
{
    if(sampleArgs->timing)
    {
        int actualSamples = width * height * 4;
        sampleTimer->totalTime = setupTime + kernelTime;

        std::string strArray[4] =
        {
            "Option Samples",
            "Time(sec)",
            "[Transfer+kernel]Time(sec)",
            "Options/sec"
        };

        std::string stats[4];
        stats[0] = toString(actualSamples, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
        stats[2] = toString(kernelTime, std::dec);
        stats[3] = toString(actualSamples / sampleTimer->totalTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int BlackScholesDP::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;
    status = clReleaseMemObject(randBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(randBuf)");

    status = clReleaseMemObject(callPriceBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(callPriceBuf)");

    status = clReleaseMemObject(putPriceBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(putPriceBuf)");

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // Release program resources (input memory etc.)
    if(randArray)
    {
#ifdef _WIN32
        ALIGNED_FREE(randArray);
#else
        FREE(randArray);
#endif
    }

    FREE(deviceCallPrice);
    FREE(devicePutPrice);
    FREE(hostCallPrice);
    FREE(hostPutPrice);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    // Create MonteCalroAsian object
    BlackScholesDP clBlackScholesDP;

    // Initialization
    if(clBlackScholesDP.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(clBlackScholesDP.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clBlackScholesDP.sampleArgs->isDumpBinaryEnabled())
    {
        return clBlackScholesDP.genBinaryImage();
    }
    else
    {
        // Setup
        int returnVal = clBlackScholesDP.setup();
        if(returnVal != SDK_SUCCESS)
        {
            return (returnVal == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;
        }

        // Run
        if(clBlackScholesDP.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Verifty
        if(clBlackScholesDP.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Cleanup resources created
        if(clBlackScholesDP.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Print performance statistics
        clBlackScholesDP.printStats();
    }
    return SDK_SUCCESS;
}