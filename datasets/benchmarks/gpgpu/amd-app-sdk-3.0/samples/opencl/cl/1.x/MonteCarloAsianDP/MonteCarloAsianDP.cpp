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


#include "MonteCarloAsianDP.hpp"

#include <math.h>
#include <malloc.h>


/*
 *  structure for attributes of Monte carlo
 *  simulation
 */

typedef struct _MonteCalroAttrib
{
    cl_double4 strikePrice;
    cl_double4 c1;
    cl_double4 c2;
    cl_double4 c3;
    cl_double4 initPrice;
    cl_double4 sigma;
    cl_double4 timeStep;
} MonteCarloAttrib;


int
MonteCarloAsianDP::setupMonteCarloAsianDP()
{
    steps = (steps < 4) ? 4 : steps;
    steps = (steps / 2) * 2;

    int i = 0;
    const cl_double finalValue = 0.8;
    const cl_double stepValue = finalValue / (cl_double)steps;

    // Allocate and init memory used by host
    sigma = (cl_double*)malloc(steps * sizeof(cl_double));
    CHECK_ALLOCATION(sigma, "Failed to allocate host memory. (sigma)");


    sigma[0] = 0.01;
    for(i = 1; i < steps; i++)
    {
        sigma[i] = sigma[i - 1] + stepValue;
    }

    price = (cl_double*) malloc(steps * sizeof(cl_double));
    CHECK_ALLOCATION(price, "Failed to allocate host memory. (price)");

    memset((void*)price, 0, steps * sizeof(cl_double));

    vega = (cl_double*) malloc(steps * sizeof(cl_double));
    CHECK_ALLOCATION(vega, "Failed to allocate host memory. (vega)");

    memset((void*)vega, 0, steps * sizeof(cl_double));

    refPrice = (cl_double*) malloc(steps * sizeof(cl_double));
    CHECK_ALLOCATION(refPrice, "Failed to allocate host memory. (refPrice)");
    memset((void*)refPrice, 0, steps * sizeof(cl_double));

    refVega = (cl_double*) malloc(steps * sizeof(cl_double));
    CHECK_ALLOCATION(refVega, "Failed to allocate host memory. (refVega)");
    memset((void*)refVega, 0, steps * sizeof(cl_double));

    // Set samples and exercize points
    noOfSum = 12;
    noOfTraj = 1024;

    width = noOfTraj / 4;
    height = noOfTraj / 2;

#if defined (_WIN32)
    randNum = (cl_uint*)_aligned_malloc(width * height * sizeof(cl_uint4), 16);
#else
    randNum = (cl_uint*)memalign(16, width * height * sizeof(cl_uint4));
#endif

    CHECK_ALLOCATION(randNum, "Failed to allocate host memory. (randNum)");

    priceVals = (cl_double*)malloc(width * height * 2 * sizeof(cl_double4));
    CHECK_ALLOCATION(priceVals, "Failed to allocate host memory. (priceVals)");
    memset((void*)priceVals, 0, width * height * 2 * sizeof(cl_double4));

    priceDeriv = (cl_double*)malloc(width * height * 2 * sizeof(cl_double4));
    CHECK_ALLOCATION(priceDeriv, "Failed to allocate host memory. (priceDeriv)");
    memset((void*)priceDeriv, 0, width * height * 2 * sizeof(cl_double4));

    /*
     * Unless quiet mode has been enabled, print the INPUT array.
     * No more than 256 values are printed because it clutters the screen
     * and it is not practical to manually compare a large set of numbers
     */
    if(!sampleArgs->quiet)
    {
        printArray<cl_double>("sigma values",
                              sigma,
                              steps,
                              1);
    }

    return SDK_SUCCESS;
}

int
MonteCarloAsianDP::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("MonteCarloAsianDP_Kernels.cl");
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
MonteCarloAsianDP::setupCL(void)
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
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    status = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "deviceInfo.setDeviceInfo failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                            devices[sampleArgs->deviceId],
                                            prop,
                                            &status);
        CHECK_OPENCL_ERROR(status,"CECL_CREATE_COMMAND_QUEUE failed.");

    }

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
            expectedError("Device does not support cl_amd_fp64 extension!");
            return SDK_EXPECTED_FAILURE;
        }
    }

    // Exit if SDK version is 2.2 or less
    std::string deviceVersionStr = std::string(deviceInfo.deviceVersion);
    size_t vStart = deviceVersionStr.find_last_of("v");
    size_t vEnd = deviceVersionStr.find(" ", vStart);
    std::string vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);
    if(vStrVal.compare("2.2") <= 0 && dType == CL_DEVICE_TYPE_GPU)
    {
        expectedError("Few double math functions are not supported in SDK2.2 or less!");
        return SDK_EXPECTED_FAILURE;
    }

    // Exit if OpenCL version is 1.0
    vStart = deviceVersionStr.find(" ", 0);
    vEnd = deviceVersionStr.find(" ", vStart + 1);
    vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);
    if(vStrVal.compare("1.0") <= 0 && dType == CL_DEVICE_TYPE_GPU)
    {
        expectedError("Unsupported device!");
        return SDK_EXPECTED_FAILURE;
    }

    // Set Persistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    randBuf = CECL_BUFFER(context,
                             inMemFlags,
                             sizeof(cl_uint4) * width  * height,
                             NULL,
                             &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(randBuf)");

    randBufAsync = CECL_BUFFER(context,
                                  inMemFlags,
                                  sizeof(cl_uint4) * width  * height,
                                  NULL,
                                  &status);

    CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBufAsync) failed.");


    priceBuf = CECL_BUFFER(context,
                              CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(cl_double4) * width * height * 2,
                              NULL,
                              &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(priceBuf)");

    priceDerivBuf = CECL_BUFFER(context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(cl_double4) * width * height * 2,
                                   NULL,
                                   &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(priceDerivBuf)");

    priceBufAsync = CECL_BUFFER(context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(cl_double4) * width * height * 2,
                                   NULL,
                                   &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(priceBufAsync)");

    priceDerivBufAsync = CECL_BUFFER(context,
                                        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                        sizeof(cl_double4) * width * height * 2,
                                        NULL,
                                        &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed.(priceDerivBufAsync)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("MonteCarloAsianDP_Kernels.cl");
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
    kernel = CECL_KERNEL(program, "calPriceVega", &status);
    CHECK_OPENCL_ERROR(status,"CECL_KERNEL failed.");

    // Settinf the kernel Information
    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "kernelInfo.setKernelWorkGroupInfo failed");

    if((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }

        // Three possible cases
        if(blockSizeX > kernelInfo.kernelWorkGroupSize)
        {
            blockSizeX = kernelInfo.kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }
    return SDK_SUCCESS;
}


int
MonteCarloAsianDP::runCLKernels(void)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;

    size_t globalThreads[2] = {width, height};
    size_t localThreads[2] = {blockSizeX, blockSizeY};

    /*
     * Declare attribute structure
     */
    MonteCarloAttrib attributes;

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
            (size_t)blockSizeX * blockSizeY > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support requested"
                  ":number of work items.";
        return SDK_FAILURE;
    }

    status = CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_int), (void*)&noOfSum);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (noOfSum)");

    status = CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_uint), (void*)&width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (width)");


    double timeStep = maturity / (noOfSum - 1);

    // Initialize random number generator
    srand(1);
    void* inMapPtr1 = NULL;
    void* inMapPtr2 = NULL;
    void* outMapPtr11 = NULL;
    void* outMapPtr12 = NULL;
    void* outMapPtr21 = NULL;
    void* outMapPtr22 = NULL;
    cl_double* ptr21 = NULL;
    cl_double* ptr22 = NULL;

    cl_event inMapEvt1;
    cl_event inMapEvt2;
    cl_event inUnmapEvt1;
    cl_event inUnmapEvt2;

    cl_event outMapEvt11;
    cl_event outMapEvt12;
    cl_event outUnmapEvt11;
    cl_event outUnmapEvt12;

    cl_event outMapEvt21;
    cl_event outMapEvt22;
    cl_event outUnmapEvt21;
    cl_event outUnmapEvt22;
    cl_event ndrEvt;

    size_t inSize = width * height * sizeof(cl_uint4);
    size_t outSize = width * height * sizeof(cl_double4);
    for(int k = 0; k < steps / 2; k++)
    {
        // Map input buffer for kernel 1
        inMapPtr1 = CECL_MAP_BUFFER(
                        commandQueue,
                        randBuf,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        inSize,
                        0,
                        NULL,
                        &inMapEvt1,
                        &status);
        CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(randBuf) failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        // Generate data for input for kernel 1
        for(int j = 0; j < (width * height * 4); j++)
        {
            randNum[j] = (cl_uint)rand();
        }

        status = waitForEventAndRelease(&inMapEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");
        memcpy(inMapPtr1, (void*)randNum, inSize);

        // Unmap of input buffer of kernel 1
        status = clEnqueueUnmapMemObject(
                     commandQueue,
                     randBuf,
                     inMapPtr1,
                     0,
                     NULL,
                     &inUnmapEvt1);
        CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(randBuf) failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");


        // Get data from output buffers of kernel 2
        if(k != 0)
        {
            // Wait for kernel 2 to complete
            status = waitForEventAndRelease(&ndrEvt);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

            outMapPtr21 = CECL_MAP_BUFFER(
                              commandQueue,
                              priceBufAsync,
                              CL_FALSE,
                              CL_MAP_READ,
                              0,
                              outSize * 2,
                              0,
                              NULL,
                              &outMapEvt21,
                              &status);
            CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceBufAsync) failed.");
            outMapPtr22 = CECL_MAP_BUFFER(
                              commandQueue,
                              priceDerivBufAsync,
                              CL_FALSE,
                              CL_MAP_READ,
                              0,
                              outSize * 2,
                              0,
                              NULL,
                              &outMapEvt22,
                              &status);
            CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceDerivBufAsync) failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        }

        // Set up arguments required for kernel 1
        double c1 = (interest - 0.5 * sigma[k * 2] * sigma[k * 2]) * timeStep;
        double c2 = sigma[k * 2] * sqrt(timeStep);
        double c3 = (interest + 0.5 * sigma[k * 2] * sigma[k * 2]);

        const cl_double4 c1F4 = {c1, c1, c1, c1};
        attributes.c1 = c1F4;

        const cl_double4 c2F4 = {c2, c2, c2, c2};
        attributes.c2 = c2F4;

        const cl_double4 c3F4 = {c3, c3, c3, c3};
        attributes.c3 = c3F4;

        const cl_double4 initPriceF4 = {initPrice, initPrice, initPrice, initPrice};
        attributes.initPrice = initPriceF4;

        const cl_double4 strikePriceF4 = {strikePrice, strikePrice, strikePrice, strikePrice};
        attributes.strikePrice = strikePriceF4;

        const cl_double4 sigmaF4 = {sigma[k * 2], sigma[k * 2], sigma[k * 2], sigma[k * 2]};
        attributes.sigma = sigmaF4;

        const cl_double4 timeStepF4 = {timeStep, timeStep, timeStep, timeStep};
        attributes.timeStep = timeStepF4;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(attributes), (void*)&attributes);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributes) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), (void*)&randBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 4, sizeof(cl_mem), (void*)&priceBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 5, sizeof(cl_mem), (void*)&priceDerivBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceDerivBuf) failed.");

        // Wait for input of kernel 1 to complete
        status = waitForEventAndRelease(&inUnmapEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inUnmapEvt1) Failed");
        inMapPtr1 = NULL;

        // Enqueue kernel 1
        status = CECL_ND_RANGE_KERNEL(commandQueue,
                                        kernel,
                                        2,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &ndrEvt);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL() failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        // Generate data of input for kernel 2
        // Fill data of input buffer for kernel 2
        if(k <= steps - 1)
        {
            // Map input buffer for kernel 1
            inMapPtr2 = CECL_MAP_BUFFER(
                            commandQueue,
                            randBufAsync,
                            CL_FALSE,
                            CL_MAP_WRITE,
                            0,
                            inSize,
                            0,
                            NULL,
                            &inMapEvt2,
                            &status);
            CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(randBufAsync) failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");

            // Generate data for input for kernel 1
            for(int j = 0; j < (width * height * 4); j++)
            {
                randNum[j] = (cl_uint)rand();
            }

            // Wait for map of input of kernel 1
            status = waitForEventAndRelease(&inMapEvt2);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt2) Failed");

            memcpy(inMapPtr2, (void*)randNum, inSize);

            // Unmap of input buffer of kernel 1
            status = clEnqueueUnmapMemObject(
                         commandQueue,
                         randBufAsync,
                         inMapPtr2,
                         0,
                         NULL,
                         &inUnmapEvt2);
            CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(randBufAsync) failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        }

        // Wait for output buffers of kernel 2 to complete
        // Calculate the results from output of kernel 2
        if(k != 0)
        {
            // Wait for output buffers of kernel 2 to complete
            status = waitForEventAndRelease(&outMapEvt21);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapevt21) Failed");

            status = waitForEventAndRelease(&outMapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapevt22) Failed");

            // Calculate the results from output of kernel 2
            ptr21 = (cl_double*)outMapPtr21;
            ptr22 = (cl_double*)outMapPtr22;
            for(int i = 0; i < noOfTraj * noOfTraj; i++)
            {
                price[k * 2 - 1] += ptr21[i];
                vega[k * 2 - 1] += ptr22[i];
            }

            price[k * 2 - 1] /= (noOfTraj * noOfTraj);
            vega[k * 2 - 1] /= (noOfTraj * noOfTraj);

            price[k * 2 - 1] = exp(-interest * maturity) * price[k * 2 - 1];
            vega[k * 2 - 1] = exp(-interest * maturity) * vega[k * 2 - 1];

            // Unmap of output buffers of kernel 2
            status = clEnqueueUnmapMemObject(
                         commandQueue,
                         priceBufAsync,
                         outMapPtr21,
                         0,
                         NULL,
                         &outUnmapEvt21);
            CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(priceBufAsync) failed.");

            status = clEnqueueUnmapMemObject(
                         commandQueue,
                         priceDerivBufAsync,
                         outMapPtr22,
                         0,
                         NULL,
                         &outUnmapEvt22);
            CHECK_OPENCL_ERROR(status,
                               "clEnqueueUnmapMemObject(priceDerivBufAsync) failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");

            status = waitForEventAndRelease(&outUnmapEvt21);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "WaitForEventAndRelease(outUnmapevt21) Failed");

            status = waitForEventAndRelease(&outUnmapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "WaitForEventAndRelease(outUnmapevt22) Failed");
        }

        // Wait for kernel 1 to complete
        status = waitForEventAndRelease(&ndrEvt);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

        // Get data from output buffers of kernel 1
        outMapPtr11 = CECL_MAP_BUFFER(
                          commandQueue,
                          priceBuf,
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          outSize * 2,
                          0,
                          NULL,
                          &outMapEvt11,
                          &status);
        CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceBuf) failed.");
        outMapPtr12 = CECL_MAP_BUFFER(
                          commandQueue,
                          priceDerivBuf,
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          outSize * 2,
                          0,
                          NULL,
                          &outMapEvt12,
                          &status);
        CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceDerivBuf) failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        // Set up arguments required for kernel 2
        double c21 = (interest - 0.5 * sigma[k * 2 + 1] * sigma[k * 2 + 1]) * timeStep;
        double c22 = sigma[k * 2 + 1] * sqrt(timeStep);
        double c23 = (interest + 0.5 * sigma[k * 2 + 1] * sigma[k * 2 + 1]);

        const cl_double4 c1F42 = {c21, c21, c21, c21};
        attributes.c1 = c1F42;

        const cl_double4 c2F42 = {c22, c22, c22, c22};
        attributes.c2 = c2F42;

        const cl_double4 c3F42 = {c23, c23, c23, c23};
        attributes.c3 = c3F42;

        const cl_double4 initPriceF42 = {initPrice, initPrice, initPrice, initPrice};
        attributes.initPrice = initPriceF42;

        const cl_double4 strikePriceF42 = {strikePrice, strikePrice, strikePrice, strikePrice};
        attributes.strikePrice = strikePriceF42;

        const cl_double4 sigmaF42 = {sigma[k * 2 + 1], sigma[k * 2 + 1], sigma[k * 2 + 1], sigma[k * 2 + 1]};
        attributes.sigma = sigmaF42;

        const cl_double4 timeStepF42 = {timeStep, timeStep, timeStep, timeStep};
        attributes.timeStep = timeStepF42;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(attributes), (void*)&attributes);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributes) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), (void*)&randBufAsync);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 4, sizeof(cl_mem), (void*)&priceBufAsync);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 5, sizeof(cl_mem), (void*)&priceDerivBufAsync);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceDerivBuf) failed.");

        // Wait for input of kernel 2 to complete
        status = waitForEventAndRelease(&inUnmapEvt2);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inUnmapEvt2) Failed");
        inMapPtr2 = NULL;

        // Enqueue kernel 2
        status = CECL_ND_RANGE_KERNEL(commandQueue,
                                        kernel,
                                        2,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &ndrEvt);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL() failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        // Wait for output buffers of kernel 1 to complete
        status = waitForEventAndRelease(&outMapEvt11);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapevt11) Failed");


        status = waitForEventAndRelease(&outMapEvt12);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapevt12 ) Failed");

        // Calculate the results from output of kernel 2
        ptr21 = (cl_double*)outMapPtr11;
        ptr22 = (cl_double*)outMapPtr12;
        for(int i = 0; i < noOfTraj * noOfTraj; i++)
        {
            price[k * 2] += ptr21[i];
            vega[k * 2] += ptr22[i];
        }

        price[k * 2] /= (noOfTraj * noOfTraj);
        vega[k * 2] /= (noOfTraj * noOfTraj);

        price[k * 2] = exp(-interest * maturity) * price[k * 2];
        vega[k * 2] = exp(-interest * maturity) * vega[k * 2];

        // Unmap of output buffers of kernel 2
        status = clEnqueueUnmapMemObject(
                     commandQueue,
                     priceBuf,
                     outMapPtr11,
                     0,
                     NULL,
                     &outUnmapEvt11);
        CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(priceBuf) failed.");

        status = clEnqueueUnmapMemObject(
                     commandQueue,
                     priceDerivBuf,
                     outMapPtr12,
                     0,
                     NULL,
                     &outUnmapEvt12);
        CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(priceDerivBuf) failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        status = waitForEventAndRelease(&outUnmapEvt11);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "WaitForEventAndRelease(outUnmapEvt11) Failed");

        status = waitForEventAndRelease(&outUnmapEvt12);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "WaitForEventAndRelease(outUnmapEvt12) Failed");
    }

    // Wait for kernel 1 to complete
    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    // Gather last kernel 2 execution here
    outMapPtr21 = CECL_MAP_BUFFER(
                      commandQueue,
                      priceBufAsync,
                      CL_FALSE,
                      CL_MAP_READ,
                      0,
                      outSize * 2,
                      0,
                      NULL,
                      &outMapEvt21,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceBuf) failed.");

    outMapPtr22 = CECL_MAP_BUFFER(
                      commandQueue,
                      priceDerivBufAsync,
                      CL_FALSE,
                      CL_MAP_READ,
                      0,
                      outSize * 2,
                      0,
                      NULL,
                      &outMapEvt22,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceDerivBuf) failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = waitForEventAndRelease(&outMapEvt21);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt21) Failed");

    status = waitForEventAndRelease(&outMapEvt22);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt22) Failed");

    // Calculate the results from output of kernel 2
    ptr21 = (cl_double*)outMapPtr21;
    ptr22 = (cl_double*)outMapPtr22;
    for(int i = 0; i < noOfTraj * noOfTraj; i++)
    {
        price[steps - 1] += ptr21[i];
        vega[steps - 1] += ptr22[i];
    }

    price[steps - 1] /= (noOfTraj * noOfTraj);
    vega[steps - 1] /= (noOfTraj * noOfTraj);

    price[steps - 1] = exp(-interest * maturity) * price[steps - 1];
    vega[steps - 1] = exp(-interest * maturity) * vega[steps - 1];

    return SDK_SUCCESS;
}

int
MonteCarloAsianDP::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    const int optionsCount = 5;
    Option *optionList = new Option[optionsCount];
    CHECK_ALLOCATION(optionList,"Failed to allocate host memory. (optionList)");

    optionList[0]._sVersion = "c";
    optionList[0]._lVersion = "steps";
    optionList[0]._description = "Steps of Monte carlo simuation";
    optionList[0]._type = CA_ARG_INT;
    optionList[0]._value = &steps;

    optionList[1]._sVersion = "P";
    optionList[1]._lVersion = "initPrice";
    optionList[1]._description = "Initial price(Default value 50)";
    optionList[1]._type = CA_ARG_DOUBLE;
    optionList[1]._value = &initPrice;

    optionList[2]._sVersion = "s";
    optionList[2]._lVersion = "strikePrice";
    optionList[2]._description = "Strike price (Default value 55)";
    optionList[2]._type = CA_ARG_DOUBLE;
    optionList[2]._value = &strikePrice;

    optionList[3]._sVersion = "r";
    optionList[3]._lVersion = "interest";
    optionList[3]._description = "interest rate (Default value 0.06)";
    optionList[3]._type = CA_ARG_DOUBLE;
    optionList[3]._value = &interest;

    optionList[4]._sVersion = "m";
    optionList[4]._lVersion = "maturity";
    optionList[4]._description = "Maturity (Default value 1)";
    optionList[4]._type = CA_ARG_DOUBLE;
    optionList[4]._value = &maturity;


    for(cl_int i = 0; i < optionsCount; ++i)
    {
        sampleArgs->AddOption(&optionList[i]);
    }

    delete[] optionList;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option,
                     "Failed to allocate host memory. (iteration_option)");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);

    delete iteration_option;

    return SDK_SUCCESS;
}

int MonteCarloAsianDP::setup()
{
    int status=setupMonteCarloAsianDP();
    CHECK_ERROR(status, SDK_SUCCESS, "MonteCarloAsianDP::setup) failed");

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int MonteCarloAsianDP::run()
{
    int status = 0;
    // Warmup
    for(int i = 0; i < 2; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout<<"Executing kernel for " <<
             iterations << " iterations" << std::endl;
    std::cout<<"-------------------------------------------" <<
             std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<cl_double>("price", price, steps, 1);
        printArray<cl_double>("vega", vega, steps, 1);
    }

    return SDK_SUCCESS;
}

void
MonteCarloAsianDP::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Steps",
            "Time(sec)",
            "[Transfer+kernel](sec)",
            "Samples used /sec"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;
        stats[0] = toString(steps, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
        stats[2] = toString(kernelTime, std::dec);
        stats[3] = toString((noOfTraj * (noOfSum - 1) * steps) /
                            kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

void
MonteCarloAsianDP::lshift128(unsigned int* input,
                             unsigned int shift,
                             unsigned int * output)
{
    unsigned int invshift = 32u - shift;

    output[0] = input[0] << shift;
    output[1] = (input[1] << shift) | (input[0] >> invshift);
    output[2] = (input[2] << shift) | (input[1] >> invshift);
    output[3] = (input[3] << shift) | (input[2] >> invshift);
}

void
MonteCarloAsianDP::rshift128(unsigned int* input,
                             unsigned int shift,
                             unsigned int* output)
{
    unsigned int invshift = 32u - shift;
    output[3]= input[3] >> shift;
    output[2] = (input[2] >> shift) | (input[0] >> invshift);
    output[1] = (input[1] >> shift) | (input[1] >> invshift);
    output[0] = (input[0] >> shift) | (input[2] >> invshift);
}

void
MonteCarloAsianDP::generateRand(unsigned int* seed,
                                double *gaussianRand1,
                                double *gaussianRand2,
                                unsigned int* nextRand)
{

    unsigned int mulFactor = 4;
    unsigned int temp[8][4];

    unsigned int state1[4] = {seed[0], seed[1], seed[2], seed[3]};
    unsigned int state2[4] = {0u, 0u, 0u, 0u};
    unsigned int state3[4] = {0u, 0u, 0u, 0u};
    unsigned int state4[4] = {0u, 0u, 0u, 0u};
    unsigned int state5[4] = {0u, 0u, 0u, 0u};

    unsigned int stateMask = 1812433253u;
    unsigned int thirty = 30u;
    unsigned int mask4[4] = {stateMask, stateMask, stateMask, stateMask};
    unsigned int thirty4[4] = {thirty, thirty, thirty, thirty};
    unsigned int one4[4] = {1u, 1u, 1u, 1u};
    unsigned int two4[4] = {2u, 2u, 2u, 2u};
    unsigned int three4[4] = {3u, 3u, 3u, 3u};
    unsigned int four4[4] = {4u, 4u, 4u, 4u};

    unsigned int r1[4] = {0u, 0u, 0u, 0u};
    unsigned int r2[4] = {0u, 0u, 0u, 0u};

    unsigned int a[4] = {0u, 0u, 0u, 0u};
    unsigned int b[4] = {0u, 0u, 0u, 0u};

    unsigned int e[4] = {0u, 0u, 0u, 0u};
    unsigned int f[4] = {0u, 0u, 0u, 0u};

    unsigned int thirteen  = 13u;
    unsigned int fifteen = 15u;
    unsigned int shift = 8u * 3u;

    unsigned int mask11 = 0xfdff37ffu;
    unsigned int mask12 = 0xef7f3f7du;
    unsigned int mask13 = 0xff777b7du;
    unsigned int mask14 = 0x7ff7fb2fu;

    const double one = 1.0;
    const double intMax = 4294967296.0;
    const double PI = 3.14159265358979;
    const double two = 2.0;

    double r[4] = {0.0, 0.0, 0.0, 0.0};
    double phi[4] = {0.0, 0.0, 0.0, 0.0};

    double temp1[4] = {0.0, 0.0, 0.0, 0.0};
    double temp2[4] = {0.0, 0.0, 0.0, 0.0};

    //Initializing states.
    for(int c = 0; c < 4; ++c)
    {
        state2[c] = mask4[c] * (state1[c] ^ (state1[c] >> thirty4[c])) + one4[c];
        state3[c] = mask4[c] * (state2[c] ^ (state2[c] >> thirty4[c])) + two4[c];
        state4[c] = mask4[c] * (state3[c] ^ (state3[c] >> thirty4[c])) + three4[c];
        state5[c] = mask4[c] * (state4[c] ^ (state4[c] >> thirty4[c])) + four4[c];
    }

    unsigned int i = 0;
    for(i = 0; i < mulFactor; ++i)
    {
        switch(i)
        {
        case 0:
            for(int c = 0; c < 4; ++c)
            {
                r1[c] = state4[c];
                r2[c] = state5[c];
                a[c] = state1[c];
                b[c] = state3[c];
            }
            break;
        case 1:
            for(int c = 0; c < 4; ++c)
            {
                r1[c] = r2[c];
                r2[c] = temp[0][c];
                a[c] = state2[c];
                b[c] = state4[c];
            }
            break;
        case 2:
            for(int c = 0; c < 4; ++c)
            {
                r1[c] = r2[c];
                r2[c] = temp[1][c];
                a[c] = state3[c];
                b[c] = state5[c];
            }
            break;
        case 3:
            for(int c = 0; c < 4; ++c)
            {
                r1[c] = r2[c];
                r2[c] = temp[2][c];
                a[c] = state4[c];
                b[c] = state1[c];
            }
            break;
        default:
            break;

        }

        lshift128(a, shift, e);
        rshift128(r1, shift, f);

        temp[i][0] = a[0] ^ e[0] ^ ((b[0] >> thirteen) & mask11) ^ f[0] ^
                     (r2[0] << fifteen);
        temp[i][1] = a[1] ^ e[1] ^ ((b[1] >> thirteen) & mask12) ^ f[1] ^
                     (r2[1] << fifteen);
        temp[i][2] = a[2] ^ e[2] ^ ((b[2] >> thirteen) & mask13) ^ f[2] ^
                     (r2[2] << fifteen);
        temp[i][3] = a[3] ^ e[3] ^ ((b[3] >> thirteen) & mask14) ^ f[3] ^
                     (r2[3] << fifteen);

    }

    for(int c = 0; c < 4; ++c)
    {
        temp1[c] = temp[0][c] * one / intMax;
        temp2[c] = temp[1][c] * one / intMax;
    }

    for(int c = 0; c < 4; ++c)
    {
        // Applying Box Mullar Transformations.
        r[c] = sqrt((-two) * log(temp1[c]));
        phi[c]  = two * PI * temp2[c];
        gaussianRand1[c] = r[c] * cos(phi[c]);
        gaussianRand2[c] = r[c] * sin(phi[c]);

        nextRand[c] = temp[2][c];
    }
}

void
MonteCarloAsianDP::calOutputs(double strikePrice,
                              double* meanDeriv1,
                              double*  meanDeriv2,
                              double* meanPrice1,
                              double* meanPrice2,
                              double* pathDeriv1,
                              double* pathDeriv2,
                              double* priceVec1,
                              double* priceVec2)
{
    double temp1[4] = {0.0, 0.0, 0.0, 0.0};
    double temp2[4] = {0.0, 0.0, 0.0, 0.0};
    double temp3[4] = {0.0, 0.0, 0.0, 0.0};
    double temp4[4] = {0.0, 0.0, 0.0, 0.0};

    double tempDiff1[4] = {0.0, 0.0, 0.0, 0.0};
    double tempDiff2[4] = {0.0, 0.0, 0.0, 0.0};

    for(int c = 0; c < 4; ++c)
    {
        tempDiff1[c] = meanPrice1[c] - strikePrice;
        tempDiff2[c] = meanPrice2[c] - strikePrice;
    }
    if(tempDiff1[0] > 0.0)
    {
        temp1[0] = 1.0;
        temp3[0] = tempDiff1[0];
    }
    if(tempDiff1[1] > 0.0)
    {
        temp1[1] = 1.0;
        temp3[1] = tempDiff1[1];
    }
    if(tempDiff1[2] > 0.0)
    {
        temp1[2] = 1.0;
        temp3[2] = tempDiff1[2];
    }
    if(tempDiff1[3] > 0.0)
    {
        temp1[3] = 1.0;
        temp3[3] = tempDiff1[3];
    }

    if(tempDiff2[0] > 0.0)
    {
        temp2[0] = 1.0;
        temp4[0] = tempDiff2[0];
    }
    if(tempDiff2[1] > 0.0)
    {
        temp2[1] = 1.0;
        temp4[1] = tempDiff2[1];
    }
    if(tempDiff2[2] > 0.0)
    {
        temp2[2] = 1.0;
        temp4[2] = tempDiff2[2];
    }
    if(tempDiff2[3] > 0.0)
    {
        temp2[3] = 1.0;
        temp4[3] = tempDiff2[3];
    }

    for(int c = 0; c < 4; ++c)
    {
        pathDeriv1[c] = meanDeriv1[c] * temp1[c];
        pathDeriv2[c] = meanDeriv2[c] * temp2[c];
        priceVec1[c] = temp3[c];
        priceVec2[c] = temp4[c];
    }
}

void MonteCarloAsianDP::cpuReferenceImpl()
{
    double timeStep = maturity / (noOfSum - 1);

    // Initialize random number generator
    srand(1);

    for(int k = 0; k < steps; k++)
    {
        double c1 = (interest - 0.5 * sigma[k] * sigma[k]) * timeStep;
        double c2 = sigma[k] * sqrt(timeStep);
        double c3 = (interest + 0.5 * sigma[k] * sigma[k]);

        for(int j = 0; j < (width * height); j++)
        {
            unsigned int nextRand[4] = {0u, 0u, 0u, 0u};
            for(int c = 0; c < 4; ++c)
            {
                nextRand[c] = (cl_uint)rand();
            }

            double trajPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
            double sumPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
            double sumDeriv1[4] = {0.0, 0.0, 0.0, 0.0};
            double meanPrice1[4] = {0.0, 0.0, 0.0, 0.0};
            double meanDeriv1[4] = {0.0, 0.0, 0.0, 0.0};
            double price1[4] = {0.0, 0.0, 0.0, 0.0};
            double pathDeriv1[4] = {0.0, 0.0, 0.0, 0.0};

            double trajPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
            double sumPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
            double sumDeriv2[4] = {0.0, 0.0, 0.0, 0.0};
            double meanPrice2[4] = {0.0, 0.0, 0.0, 0.0};
            double meanDeriv2[4] = {0.0, 0.0, 0.0, 0.0};
            double price2[4] = {0.0, 0.0, 0.0, 0.0};
            double pathDeriv2[4] = {0.0, 0.0, 0.0, 0.0};

            //Run the Monte Carlo simulation a total of Num_Sum - 1 times
            for(int i = 1; i < noOfSum; i++)
            {
                unsigned int tempRand[4] =  {0u, 0u, 0u, 0u};
                for(int c = 0; c < 4; ++c)
                {
                    tempRand[c] = nextRand[c];
                }

                double gaussian1[4] = {0.0, 0.0, 0.0, 0.0};
                double gaussian2[4] = {0.0, 0.0, 0.0, 0.0};
                generateRand(tempRand, gaussian1, gaussian2, nextRand);
                //Calculate the trajectory price and sum price for all trajectories
                for(int c = 0; c < 4; ++c)
                {
                    trajPrice1[c] = trajPrice1[c] * exp(c1 + c2 * gaussian1[c]);
                    trajPrice2[c] = trajPrice2[c] * exp(c1 + c2 * gaussian2[c]);

                    sumPrice1[c] = sumPrice1[c] + trajPrice1[c];
                    sumPrice2[c] = sumPrice2[c] + trajPrice2[c];

                    double temp = c3 * timeStep * i;

                    // Calculate the derivative price for all trajectories
                    sumDeriv1[c] = sumDeriv1[c] + trajPrice1[c]
                                   * ((log(trajPrice1[c] / initPrice) - temp) / sigma[k]);
                    sumDeriv2[c] = sumDeriv2[c] + trajPrice2[c]
                                   * ((log(trajPrice2[c] / initPrice) - temp) / sigma[k]);
                }

            }

            //Calculate the average price and “average derivative” of each simulated path
            for(int c = 0; c < 4; ++c)
            {
                meanPrice1[c] = sumPrice1[c] / noOfSum;
                meanPrice2[c] = sumPrice2[c] / noOfSum;
                meanDeriv1[c] = sumDeriv1[c] / noOfSum;
                meanDeriv2[c] = sumDeriv2[c] / noOfSum;
            }

            calOutputs(strikePrice, meanDeriv1, meanDeriv2, meanPrice1, meanPrice2,
                       pathDeriv1, pathDeriv2, price1, price2);

            for(int c = 0; c < 4; ++c)
            {
                priceVals[j * 8 + c] = price1[c];
                priceVals[j * 8 + 1 * 4 + c] = price2[c];
                priceDeriv[j * 8 + c] = pathDeriv1[c];
                priceDeriv[j * 8 + 1 * 4 + c] = pathDeriv2[c];
            }
        }

        // Replace Following "for" loop with reduction kernel
        for(int i = 0; i < noOfTraj * noOfTraj; i++)
        {
            refPrice[k] += priceVals[i];
            refVega[k] += priceDeriv[i];
        }

        refPrice[k] /= (noOfTraj * noOfTraj);
        refVega[k] /= (noOfTraj * noOfTraj);

        refPrice[k] = exp(-interest * maturity) * refPrice[k];
        refVega[k] = exp(-interest * maturity) * refVega[k];
    }
}

int MonteCarloAsianDP::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        cpuReferenceImpl();

        double epsilon = (0.2 * maturity);
        // compare the results and see if they match
        for(int i = 0; i < steps; ++i)
        {
            if(fabs(price[i] - refPrice[i]) > epsilon)
            {
                std::cout << "Failed\n";
                return SDK_FAILURE;
            }
            if(fabs(vega[i] - refVega[i]) > epsilon)
            {
                std::cout << "Failed\n" << std::endl;
                return SDK_FAILURE;
            }
        }
        std::cout << "Passed!\n" << std::endl;
    }

    return SDK_SUCCESS;
}

int MonteCarloAsianDP::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(priceBuf);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(priceBuf)");

    status = clReleaseMemObject(priceDerivBuf);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(priceDerivBuf)");

    status = clReleaseMemObject(randBuf);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(randBuf)");

    status = clReleaseMemObject(priceBufAsync);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(priceBufAsync)");

    status = clReleaseMemObject(priceDerivBufAsync);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(priceDerivBufAsync)");

    status = clReleaseMemObject(randBufAsync);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(randBufAsync)");

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status,"clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status,"clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status,"clReleaseContext failed.(context)");


    // Release program resources (input memory etc.)

    FREE(sigma);
    FREE(price);
    FREE(vega);
    FREE(refPrice);
    FREE(refVega);


    if(randNum)
    {
#if defined (_WIN32)
        ALIGNED_FREE(randNum);
#else
        FREE(randNum);
#endif
        randNum = NULL;
    }

    FREE(priceVals);
    FREE(priceDeriv);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    // Create MonteCalroAsian object
    MonteCarloAsianDP clMonteCarloAsianDP;

    // Initialization
    if(clMonteCarloAsianDP.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(clMonteCarloAsianDP.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clMonteCarloAsianDP.sampleArgs->isDumpBinaryEnabled())
    {
        return clMonteCarloAsianDP.genBinaryImage();
    }
    else
    {
        // Setup
        int returnVal = clMonteCarloAsianDP.setup();

        if(returnVal != SDK_SUCCESS)
        {
            return returnVal;
        }

        // Run
        if(clMonteCarloAsianDP.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clMonteCarloAsianDP.verifyResults()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Cleanup resources created
        if(clMonteCarloAsianDP.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clMonteCarloAsianDP.printStats();
    }
    return SDK_SUCCESS;
}