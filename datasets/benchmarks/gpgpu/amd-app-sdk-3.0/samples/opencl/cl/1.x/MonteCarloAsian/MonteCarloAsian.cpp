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


#include "MonteCarloAsian.hpp"

#include <math.h>
#include <malloc.h>


int
MonteCarloAsian::setupMonteCarloAsian()
{
    if(!disableAsync)
    {
        steps = (steps < 4) ? 4 : steps;
        steps = (steps / 2) * 2;
    }

	if (interest <= 0 || interest > 1)
	{
		std::cout << "Interest rate must be in the range > 0 and <= 1.0!" << std::endl;
		return SDK_EXPECTED_FAILURE;
	}
    // Validate flags
    if(dUseOutAllocHostPtr)
        if(!(disableAsync && disableMapping))
        {
            std::cout << "Note : Neglected --dOutAllocHostPtr flag" << std::endl;
            dUseOutAllocHostPtr = false;
        }

    if(dUseInPersistent)
        if(!(disableAsync && disableMapping))
        {
            std::cout << "Note : Neglected --dInPersistent flag" << std::endl;
            dUseInPersistent = false;
        }

    int i = 0;
    const cl_float finalValue = 0.8f;
    const cl_float stepValue = finalValue / (cl_float)steps;

    // Allocate and init memory used by host
    sigma = (cl_float*)malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(sigma, "Failed to allocate host memory. (sigma)");

    sigma[0] = 0.01f;
    for(i = 1; i < steps; i++)
    {
        sigma[i] = sigma[i - 1] + stepValue;
    }

    price = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(price, "Failed to allocate host memory. (price)");
    memset((void*)price, 0, steps * sizeof(cl_float));

    vega = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(price, "Failed to allocate host memory. (vega)");
    memset((void*)vega, 0, steps * sizeof(cl_float));

    refPrice = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(refPrice, "Failed to allocate host memory. (refPrice)");
    memset((void*)refPrice, 0, steps * sizeof(cl_float));

    refVega = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(refVega, "Failed to allocate host memory. (refVega)");
    memset((void*)refVega, 0, steps * sizeof(cl_float));

    // Set samples and exercise points
    noOfSum = 12;
    noOfTraj = 1024;

    width = noOfTraj / 4;
    height = noOfTraj / 2;

#if defined (_WIN32)
    randNum = (cl_uint*)_aligned_malloc(width * height * sizeof(cl_uint4) * steps,
                                        16);
#else
    randNum = (cl_uint*)memalign(16, width * height * sizeof(cl_uint4) * steps);
#endif
    CHECK_ALLOCATION(randNum, "Failed to allocate host memory. (randNum)");

    // Generate random data
    for(int i = 0; i < (width * height * 4 * steps); i++)
    {
        randNum[i] = (cl_uint)rand();
    }

    priceVals = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceVals, "Failed to allocate host memory. (priceVals)");
    memset((void*)priceVals, 0, width * height * 2 * sizeof(cl_float4));

    priceDeriv = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceDeriv, "Failed to allocate host memory. (priceDeriv)");
    memset((void*)priceDeriv, 0, width * height * 2 * sizeof(cl_float4));

    if(!disableAsync && disableMapping)
    {
        priceValsAsync = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
        CHECK_ALLOCATION(priceValsAsync,
                         "Failed to allocate host memory. (priceValsAsync)");
        memset((void*)priceValsAsync, 0, width * height * 2 * sizeof(cl_float4));

        priceDerivAsync = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
        CHECK_ALLOCATION(priceDerivAsync,
                         "Failed to allocate host memory. (priceDerivAsync)");
        memset((void*)priceDerivAsync, 0, width * height * 2 * sizeof(cl_float4));
    }
    /*
     * Unless quiet mode has been enabled, print the INPUT array.
     * No more than 256 values are printed because it clutters the screen
     * and it is not practical to manually compare a large set of numbers
     */

    if(!sampleArgs->quiet)
        printArray<cl_float>(
            "sigma values",
            sigma,
            steps,
            1);

    return SDK_SUCCESS;
}

int
MonteCarloAsian::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("MonteCarloAsian_Kernels.cl");
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
MonteCarloAsian::setupCL(void)
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
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE() failed.");

    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    //Set device info of given cl_device_id

    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // If both options were ambiguously specified, use default vector-width for the device
    if(useScalarKernel && useVectorKernel)
    {
        std::cout <<
                  "Ignoring \"--scalar\" & \"--vector\" options. Using default vector-width for the device"
                  << std::endl;
        vectorWidth = deviceInfo.preferredFloatVecWidth;
    }
    else if(useScalarKernel)
    {
        vectorWidth = 1;
    }
    else if(useVectorKernel)
    {
        vectorWidth = 4;
    }
    else                            // If the options were not specified at command-line
    {
        vectorWidth = deviceInfo.preferredFloatVecWidth;
    }

    if(vectorWidth == 1)
    {
        std::cout << "Selecting scalar kernel" << std::endl;
    }
    else
    {
        std::cout << "Selecting vector kernel" << std::endl;
    }

    commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                        devices[sampleArgs->deviceId],
                                        0,
                                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE(commandQueue) failed.");

    if(!dUseInPersistent)
    {
        // Set Persistent memory only for AMD platform
        cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
        if(sampleArgs->isAmdPlatform())
        {
            inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
        }

        // Create buffer for randBuf
        randBuf = CECL_BUFFER(context,
                                 inMemFlags,
                                 sizeof(cl_uint4) * width  * height,
                                 NULL,
                                 &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBuf) failed.");

        if(!disableAsync)
        {
            // create Buffer for randBuf in asynchronous mode
            randBufAsync = CECL_BUFFER(context,
                                          inMemFlags,
                                          sizeof(cl_uint4) * width  * height,
                                          NULL,
                                          &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBufAsync) failed.");
        }
    }
    else
    {
        // create Normal Buffer, if persistent memory is not in used
        randBuf = CECL_BUFFER(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 sizeof(cl_uint4) * width  * height,
                                 randNum,
                                 &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBuf) failed.");
    }

    if(!dUseOutAllocHostPtr)
    {
        // create Buffer for PriceBuf
        priceBuf = CECL_BUFFER(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(cl_float4) * width * height * 2,
                                  NULL,
                                  &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBuf) failed.");

        // create Buffer for PriceDeriveBuffer
        priceDerivBuf = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       sizeof(cl_float4) * width * height * 2,
                                       NULL,
                                       &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBuf) failed.");

        if(!disableAsync)
        {
            // create Buffer for priceBufAsync
            priceBufAsync = CECL_BUFFER(context,
                                           CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(cl_float4) * width * height * 2,
                                           NULL,
                                           &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBufAsync) failed.");

            // create Buffer for priceDerivBufAsync
            priceDerivBufAsync = CECL_BUFFER(context,
                                                CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                sizeof(cl_float4) * width * height * 2,
                                                NULL,
                                                &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBufAsync) failed.");
        }
    }
    else
    {
        // create Buffer for priceBuf
        priceBuf = CECL_BUFFER(context,
                                  CL_MEM_WRITE_ONLY,
                                  sizeof(cl_float4) * width * height * 2,
                                  NULL,
                                  &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBuf) failed.");

        // create Buffer for priceDerivBuf
        priceDerivBuf = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(cl_float4) * width * height * 2,
                                       NULL,
                                       &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBuf) failed.");
    }

    // Create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("MonteCarloAsian_Kernels.cl");
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
    CHECK_ERROR(retValue, SDK_SUCCESS, "sampleCommon::buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    if (vectorWidth == 1)
    {
        kernel = CECL_KERNEL(program, "calPriceVega_Scalar", &status);
        CHECK_OPENCL_ERROR(status, "CECL_KERNEL(calPriceVega_Scalar) failed.");
    }
    else
    {
        kernel = CECL_KERNEL(program, "calPriceVega_Vector", &status);
        CHECK_OPENCL_ERROR(status, "CECL_KERNEL(calPriceVega_Vector) failed.");
    }

    // Check group-size against what is returned by kernel
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
MonteCarloAsian::setKernelArgs(int step, cl_mem *rand, cl_mem *price,
                               cl_mem *priceDeriv)
{
    cl_int status;
    float timeStep = maturity / (noOfSum - 1);

    // Set up arguments required for kernel 1
    float c1 = (interest - 0.5f * sigma[step] * sigma[step]) * timeStep;
    float c2 = sigma[step] * sqrt(timeStep);
    float c3 = (interest + 0.5f * sigma[step] * sigma[step]);

    if (vectorWidth == 1)
    {
        attributesScalar.c1 = c1;
        attributesScalar.c2 = c2;
        attributesScalar.c3 = c3;
        attributesScalar.initPrice = initPrice;
        attributesScalar.strikePrice = strikePrice;
        attributesScalar.sigma = sigma[step];
        attributesScalar.timeStep = timeStep;

        status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(attributesScalar),
                                (void*)&attributesScalar);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributesScalar) failed.");
    }
    else
    {
        const cl_float4 c1F4 = {c1, c1, c1, c1};
        attributes.c1 = c1F4;

        const cl_float4 c2F4 = {c2, c2, c2, c2};
        attributes.c2 = c2F4;

        const cl_float4 c3F4 = {c3, c3, c3, c3};
        attributes.c3 = c3F4;

        const cl_float4 initPriceF4 = {initPrice, initPrice, initPrice, initPrice};
        attributes.initPrice = initPriceF4;

        const cl_float4 strikePriceF4 = {strikePrice, strikePrice, strikePrice, strikePrice};
        attributes.strikePrice = strikePriceF4;

        const cl_float4 sigmaF4 = {sigma[step], sigma[step], sigma[step], sigma[step]};
        attributes.sigma = sigmaF4;

        const cl_float4 timeStepF4 = {timeStep, timeStep, timeStep, timeStep};
        attributes.timeStep = timeStepF4;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(attributes), (void*)&attributes);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributes) failed.");
    }

    status = CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), (void*)rand);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

    status = CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), (void*)price);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceBuf) failed.");

    status = CECL_SET_KERNEL_ARG(kernel, 4, sizeof(cl_mem), (void*)priceDeriv);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceDerivBuf) failed.");

    if(vectorWidth == 1)
    {
        status = CECL_SET_KERNEL_ARG(kernel, 5,
                                blockSizeX*blockSizeY*VECTOR_SIZE*sizeof(cl_float2), NULL);
    }
    else
    {
        status = CECL_SET_KERNEL_ARG(kernel, 5, blockSizeX*blockSizeY*sizeof(cl_float8),
                                NULL);
    }
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(sData1) failed.");

    return SDK_SUCCESS;
}

int
MonteCarloAsian::runAsyncWithMappingDisabled()
{
    cl_event inWriteEvt1;
    cl_event inWriteEvt2;
    cl_event outReadEvt11;
    cl_event outReadEvt12;
    cl_event outReadEvt21;
    cl_event outReadEvt22;
    cl_event ndrEvt;
    size_t size = width * height * sizeof(cl_float4);
    size_t count = width * height * 4;
    cl_int status;
    size_t sizeAfterReduction = width * height  / blockSizeX /blockSizeY * 4;

    for(int k = 0; k < steps / 2; k++)
    {
        // Fill data of input buffer for kernel 1
        status = CECL_WRITE_BUFFER(
                     commandQueue,
                     randBuf,
                     CL_FALSE,
                     0,
                     size,
                     (randNum + (2 * k * count)),
                     0,
                     NULL,
                     &inWriteEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(randBuf) failed.");
        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        // Get data from output buffers of kernel 2
        if(k != 0)
        {
            // Wait for kernel 2 to complete
            status = waitForEventAndRelease(&ndrEvt);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(ndrEvt) failed.");

            status = CECL_READ_BUFFER(
                         commandQueue,
                         priceBufAsync,
                         CL_FALSE,
                         0,
                         size * 2,
                         priceValsAsync,
                         0,
                         NULL,
                         &outReadEvt21);
            CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceBufAsync) failed.");
            status = CECL_READ_BUFFER(
                         commandQueue,
                         priceDerivBufAsync,
                         CL_FALSE,
                         0,
                         size * 2,
                         priceDerivAsync,
                         0,
                         NULL,
                         &outReadEvt22);
            CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceDerivBufAsync) failed.");
            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        }
        // Set up arguments required for kernel 1
        status = setKernelArgs(k * 2, &randBuf, &priceBuf, &priceDerivBuf);
        CHECK_OPENCL_ERROR(status, "setKernelArgs failed.");

        // Wait for input of kernel to complete
        status = waitForEventAndRelease(&inWriteEvt1);
        CHECK_OPENCL_ERROR(status, "clWaitForEventsAndRelease(inWriteEvt1) failed.");

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
        /* Generate data of input for kernel 2
         Fill data of input buffer for kernel 2*/
        if(k <= steps - 1)
        {
            // Fill data of input buffer for kernel 2
            status = CECL_WRITE_BUFFER(
                         commandQueue,
                         randBufAsync,
                         CL_FALSE,
                         0,
                         size,
                         (randNum + (((2 * k) +1) * count)),
                         0,
                         NULL,
                         &inWriteEvt2);
            CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(randBufAsync) failed.");
            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        }
        // Wait for output buffers of kernel 2 to complete
        // Calculate the results from output of kernel 2
        if(k != 0)
        {
            // Wait for output buffers of kernel 2 to complete
            status = waitForEventAndRelease(&outReadEvt21);
            CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt21) failed.");
            status = waitForEventAndRelease(&outReadEvt22);
            CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt22) failed.");
            // Calculate the results from output of kernel 2
            for(size_t i = 0; i < sizeAfterReduction; i++)
            {
                price[k * 2 - 1] += priceValsAsync[i];
                vega[k * 2 - 1] += priceDerivAsync[i];
            }
            price[k * 2 - 1] /= (noOfTraj * noOfTraj);
            vega[k * 2 - 1] /= (noOfTraj * noOfTraj);
            price[k * 2 - 1] = exp(-interest * maturity) * price[k * 2 - 1];
            vega[k * 2 - 1] = exp(-interest * maturity) * vega[k * 2 - 1];
        }
        // Wait for kernel 1 to complete
        status = waitForEventAndRelease(&ndrEvt);
        CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(ndrEvt) failed.");
        // Get data from output buffers of kernel 1
        status = CECL_READ_BUFFER(commandQueue,
                                     priceBuf,
                                     CL_TRUE,
                                     0,
                                     size * 2,
                                     priceVals,
                                     0,
                                     NULL,
                                     &outReadEvt11);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceBuf) failed.");
        status = CECL_READ_BUFFER(commandQueue,
                                     priceDerivBuf,
                                     CL_TRUE,
                                     0,
                                     size * 2,
                                     priceDeriv,
                                     0,
                                     NULL,
                                     &outReadEvt12);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceDerivBuf) failed.");
        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        // Set up arguments required for kernel 2
        status = setKernelArgs((k * 2 + 1), &randBufAsync, &priceBufAsync,
                               &priceDerivBufAsync);
        CHECK_OPENCL_ERROR(status, "setKernelArgs failed.");
        // Wait for input of kernel 2 to complete
        status = waitForEventAndRelease(&inWriteEvt2);
        CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(inWriteEvt2) failed.");

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
        status = waitForEventAndRelease(&outReadEvt11);
        CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt11) failed.");
        status = waitForEventAndRelease(&outReadEvt12);
        CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt12) failed.");
        for(size_t i = 0; i < sizeAfterReduction; i++)
        {
            price[k * 2] += priceVals[i];
            vega[k * 2] += priceDeriv[i];
        }
        price[k * 2] /= (noOfTraj * noOfTraj);
        vega[k * 2] /= (noOfTraj * noOfTraj);
        price[k * 2] = exp(-interest * maturity) * price[k * 2];
        vega[k * 2] = exp(-interest * maturity) * vega[k * 2];
    }
    // Wait for kernel 1 to complete
    status = waitForEventAndRelease(&ndrEvt);
    CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(ndrEvt) failed.");
    // Gather last kernel 2 execution here
    status = CECL_READ_BUFFER(
                 commandQueue,
                 priceBufAsync,
                 CL_FALSE,
                 0,
                 size * 2,
                 priceValsAsync,
                 0,
                 NULL,
                 &outReadEvt21);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceBufAsync) failed.");
    status = CECL_READ_BUFFER(
                 commandQueue,
                 priceDerivBufAsync,
                 CL_FALSE,
                 0,
                 size * 2,
                 priceDerivAsync,
                 0,
                 NULL,
                 &outReadEvt22);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceDerivBufAsync) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");
    // Wait for output buffers of kernel 2 to complete
    status = waitForEventAndRelease(&outReadEvt21);
    CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt21) failed.");
    status = waitForEventAndRelease(&outReadEvt22);
    CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(outReadEvt22) failed.");

    // Calculate the results from output of kernel 2
    for(size_t i = 0; i < sizeAfterReduction; i++)
    {
        price[steps - 1] += priceValsAsync[i];
        vega[steps - 1] += priceDerivAsync[i];
    }
    price[steps - 1] /= (noOfTraj * noOfTraj);
    vega[steps - 1] /= (noOfTraj * noOfTraj);
    price[steps - 1] = exp(-interest * maturity) * price[steps - 1];
    vega[steps - 1] = exp(-interest * maturity) * vega[steps - 1];

    return SDK_SUCCESS;
}

int
MonteCarloAsian::runAsyncWithMappingEnabled()
{
    void* inMapPtr1 = NULL;
    void* inMapPtr2 = NULL;
    void* outMapPtr11 = NULL;
    void* outMapPtr12 = NULL;
    void* outMapPtr21 = NULL;
    void* outMapPtr22 = NULL;
    cl_float* ptr21 = NULL;
    cl_float* ptr22 = NULL;
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
    cl_int status;
    cl_int eventStatus = CL_QUEUED;

    size_t count = width * height * 4;
    size_t sizeAfterReduction = width * height  / blockSizeX /blockSizeY * 4;
    size_t size = width * height * sizeof(cl_float4);

    for(int k = 0; k < steps / 2; k++)
    {
        // Map input buffer for kernel 1
        status = asyncMapBuffer( randBuf, inMapPtr1, size,
                                 CL_MAP_WRITE_INVALIDATE_REGION, &inMapEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(randBuf)");

        // Wait for map of input of kernel 1
        status = waitForEventAndRelease(&inMapEvt1);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inMapEvt1) failed.");

        memcpy(inMapPtr1, (void*)(randNum + (2 * k * count)), size);
        // Unmap of input buffer of kernel 1
        status = asyncUnmapBuffer(randBuf, inMapPtr1, &inUnmapEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer(randBuf).");

        // Get data from output buffers of kernel 2
        if(k != 0)
        {
            // Wait for kernel 2 to complete
            status = waitForEventAndRelease(&ndrEvt);
            CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(ndrEvt) failed.");

            status = asyncMapBuffer( priceBufAsync, outMapPtr21, sizeAfterReduction,
                                     CL_MAP_READ, &outMapEvt21);
            CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceBufAsync)");
            status = asyncMapBuffer( priceDerivBufAsync, outMapPtr22, sizeAfterReduction,
                                     CL_MAP_READ, &outMapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "Failed to map device buffer.(priceDerivBufAsync)");
        }

        // Set up arguments required for kernel 1
        status = setKernelArgs(k * 2, &randBuf, &priceBuf, &priceDerivBuf);
        CHECK_OPENCL_ERROR(status, "setKernelArgs failed.");

        // Wait for input of kernel 1 to complete
        status = waitForEventAndRelease(&inUnmapEvt1);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inUnmapEvt1) failed.");

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

        // Fill data of input buffer for kernel 2
        if(k <= steps - 1)
        {
            // Map input buffer for kernel 1
            status = asyncMapBuffer( randBufAsync, inMapPtr2, size,
                                     CL_MAP_WRITE_INVALIDATE_REGION, &inMapEvt2);
            CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(randBufAsync)");

            // Wait for map of input of kernel 1
            status = waitForEventAndRelease(&inMapEvt2);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inMapEvt2) failed.");

            memcpy(inMapPtr2, (void*)(randNum + (((2 * k) +1) * count)), size);

            // Unmap of input buffer of kernel 1
            status = asyncUnmapBuffer(randBufAsync, inMapPtr2, &inUnmapEvt2);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "Failed to unmap device buffer(randBufAsync).");
        }
        // Wait for output buffers of kernel 2 to complete
        // Calculate the results from output of kernel 2
        if(k != 0)
        {
            // Wait for output buffers of kernel 2 to complete
            status = waitForEventAndRelease(&outMapEvt21);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt21) failed.");
            status = waitForEventAndRelease(&outMapEvt22);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt22) failed.");

            // Calculate the results from output of kernel 2
            ptr21 = (cl_float*)outMapPtr21;
            ptr22 = (cl_float*)outMapPtr22;
            for(size_t i = 0; i < sizeAfterReduction; i++)
            {
                price[k * 2 - 1] += ptr21[i];
                vega[k * 2 - 1] += ptr22[i];
            }
            price[k * 2 - 1] /= (noOfTraj * noOfTraj);
            vega[k * 2 - 1] /= (noOfTraj * noOfTraj);
            price[k * 2 - 1] = exp(-interest * maturity) * price[k * 2 - 1];
            vega[k * 2 - 1] = exp(-interest * maturity) * vega[k * 2 - 1];

            // Unmap of output buffers of kernel 2
            status = asyncUnmapBuffer(priceBufAsync, outMapPtr21, &outUnmapEvt21);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "Failed to unmap device buffer(priceBufAsync).");
            status = asyncUnmapBuffer(priceDerivBufAsync, outMapPtr22, &outUnmapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "Failed to unmap device buffer(priceDerivBufAsync).");

            status = waitForEventAndRelease(&outUnmapEvt21);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outUnmapEvt21) failed.");
            status = waitForEventAndRelease(&outUnmapEvt22);
            CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outUnmapEvt22) failed.");
        }
        // Wait for kernel 1 to complete
        status = waitForEventAndRelease(&ndrEvt);
        CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(ndrEvt) failed.");

        // Get data from output buffers of kernel 1
        status = asyncMapBuffer( priceBuf, outMapPtr11, sizeAfterReduction, CL_MAP_READ,
                                 &outMapEvt11);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceBuf)");
        status = asyncMapBuffer( priceDerivBuf, outMapPtr12, sizeAfterReduction,
                                 CL_MAP_READ, &outMapEvt12);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceDerivBuf)");

        // Set up arguments required for kernel 2
        status = setKernelArgs((k * 2 + 1), &randBufAsync, &priceBufAsync,
                               &priceDerivBufAsync);
        CHECK_OPENCL_ERROR(status, "setKernelArgs failed.");
        // Wait for input of kernel 2 to complete
        status = waitForEventAndRelease(&inUnmapEvt2);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inUnmapEvt2) failed.");

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
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt11) failed.");
        status = waitForEventAndRelease(&outMapEvt12);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt12) failed.");

        // Calculate the results from output of kernel 2
        ptr21 = (cl_float*)outMapPtr11;
        ptr22 = (cl_float*)outMapPtr12;
        for(size_t i = 0; i < sizeAfterReduction; i++)
        {
            price[k * 2] += ptr21[i];
            vega[k * 2] += ptr22[i];

        }
        price[k * 2] /= (noOfTraj * noOfTraj);
        vega[k * 2] /= (noOfTraj * noOfTraj);
        price[k * 2] = exp(-interest * maturity) * price[k * 2];
        vega[k * 2] = exp(-interest * maturity) * vega[k * 2];

        // Unmap of output buffers of kernel 2
        status = asyncUnmapBuffer(priceBuf, outMapPtr11, &outUnmapEvt11);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer(priceBuf).");
        status = asyncUnmapBuffer(priceDerivBuf, outMapPtr12, &outUnmapEvt12);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer(priceDerivBuf).");

        status = waitForEventAndRelease(&outUnmapEvt11);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outUnmapEvt11) failed.");
        status = waitForEventAndRelease(&outUnmapEvt12);
        CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outUnmapEvt12) failed.");
    }
    // Wait for kernel 1 to complete
    status = waitForEventAndRelease(&ndrEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(ndrEvt) failed.");

    // Gather last kernel 2 execution here
    status = asyncMapBuffer( priceBufAsync, outMapPtr21, sizeAfterReduction,
                             CL_MAP_READ, &outMapEvt21);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceBufAsync)");

    status = asyncMapBuffer( priceDerivBufAsync, outMapPtr22, sizeAfterReduction,
                             CL_MAP_READ, &outMapEvt22);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(priceDerivBufAsync)");

    status = waitForEventAndRelease(&outMapEvt21);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt21) failed.");
    status = waitForEventAndRelease(&outMapEvt22);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(outMapEvt22) failed.");

    // Calculate the results from output of kernel 2
    ptr21 = (cl_float*)outMapPtr21;
    ptr22 = (cl_float*)outMapPtr22;
    for(size_t i = 0; i < sizeAfterReduction; i++)
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
MonteCarloAsian::runCLKernels(void)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    globalThreads[0] = width, globalThreads[1] = height;
    localThreads[0] = blockSizeX, localThreads[1] = blockSizeY;

    size_t count = width * height * 4;
    size_t sizeAfterReduction = width * height  / blockSizeX /blockSizeY * 4;

    if(vectorWidth == 1)
    {
        globalThreads[0] *= VECTOR_SIZE;
        localThreads[0] *= VECTOR_SIZE;
    }

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
            (localThreads[0] * localThreads[1]) > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support requested"
                  << ":number of work items.";
        return SDK_FAILURE;
    }

    status = CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_int), (void*)&noOfSum);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(noOfSum) failed.");

    if(vectorWidth == 1)
    {
        status = CECL_SET_KERNEL_ARG(kernel, 6, localThreads[0] * sizeof(cl_uint), NULL);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(rand1) failed.");

        status = CECL_SET_KERNEL_ARG(kernel, 7, localThreads[0] * sizeof(cl_uint), NULL);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(rand2) failed.");
    }

    float timeStep = maturity / (noOfSum - 1);

    // Initialize random number generator
    srand(1);
    if(!disableAsync)
    {
        if(disableMapping)
        {
            status = runAsyncWithMappingDisabled();
            if(status != SDK_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            status = runAsyncWithMappingEnabled();
            if(status != SDK_SUCCESS)
            {
                return status;
            }
        }
    }
    else
    {
        cl_event events[1];

        for(int k = 0; k < steps; k++)
        {
            if(!dUseInPersistent)
            {
                if(disableMapping)
                {
                    status = CECL_WRITE_BUFFER(
                                 commandQueue,
                                 randBuf,
                                 CL_TRUE,
                                 0,
                                 width * height * sizeof(cl_float4),
                                 (randNum + (k * count)),
                                 0,
                                 NULL,
                                 NULL);
                    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(randBuf) failed.");
                }
                else
                {
                    cl_event inEvent;
                    cl_event inUnEvent;
                    void* mapPtr;
                    status = asyncMapBuffer( randBuf, mapPtr, (width * height * sizeof(cl_float4)),
                                             CL_MAP_WRITE_INVALIDATE_REGION, &inEvent);
                    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(randBuf)");

                    status = waitForEventAndRelease(&inEvent);
                    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inEvent) failed.");

                    memcpy(mapPtr, (void*)(randNum + (k * count)),
                           width * height * sizeof(cl_float4));

                    status = asyncUnmapBuffer(randBuf, mapPtr, &inUnEvent);
                    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer(randBuf).");

                    status = waitForEventAndRelease(&inUnEvent);
                    CHECK_OPENCL_ERROR(status, "clWaitForEventAndRelease(inUnEvent) failed.");
                }
            }

            status = setKernelArgs(k, &randBuf, &priceBuf, &priceDerivBuf);
            CHECK_OPENCL_ERROR(status, "setKernelArgs failed.");

            // Enqueue a kernel run call.
            status = CECL_ND_RANGE_KERNEL(commandQueue,
                                            kernel,
                                            2,
                                            NULL,
                                            globalThreads,
                                            localThreads,
                                            0,
                                            NULL,
                                            &events[0]);

            CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL() failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");

            // wait for the kernel call to finish execution
            status = waitForEventAndRelease(&events[0]);
            CHECK_ERROR(status,0, "WaitForEventAndRelease(events[0]) Failed");

            if(!disableMapping)
            {
                cl_event inEvent;
                cl_event inUnEvent;
                void* mapPtr;
                status = asyncMapBuffer( priceBuf, mapPtr,
                                         (width * height * 2 * sizeof(cl_float4)), CL_MAP_READ, &inEvent);
                CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceBuf)");

                status = waitForEventAndRelease(&inEvent);
                CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inEvent) failed.");

                memcpy((void*)priceVals, mapPtr, width * height * 2 * sizeof(cl_float4));

                status = asyncUnmapBuffer(priceBuf, mapPtr, &inUnEvent);
                CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer(priceBuf).");

                status = waitForEventAndRelease(&inUnEvent);
                CHECK_ERROR(status,0, "WaitForEventAndRelease(inUnEvent) Failed");

                status = asyncMapBuffer( priceDerivBuf, mapPtr,
                                         (width * height * 2 * sizeof(cl_float4)), CL_MAP_READ, &inEvent);
                CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(priceDerivBuf)");

                status = waitForEventAndRelease(&inEvent);
                CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(inEvent) failed.");

                memcpy((void*)priceDeriv, mapPtr, width * height * 2 * sizeof(cl_float4));

                status = asyncUnmapBuffer(priceDerivBuf, mapPtr, &inUnEvent);
                CHECK_ERROR(status, SDK_SUCCESS,
                            "Failed to unmap device buffer(priceDerivBuf).");

                status = waitForEventAndRelease(&inUnEvent);
                CHECK_OPENCL_ERROR(status, "WaitForEventAndRelease(inUnEvent) Failed");
            }
            else
            {
                // Enqueue the results to application pointer
                status = CECL_READ_BUFFER(commandQueue,
                                             priceBuf,
                                             CL_TRUE,
                                             0,
                                             width * height * 2 * sizeof(cl_float4),
                                             priceVals,
                                             0,
                                             NULL,
                                             &events[0]);
                CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceBuf) failed.");

                // wait for the read buffer to finish execution
                status = waitForEventAndRelease(&events[0]);
                CHECK_OPENCL_ERROR(status, "clWaitForEventsAndRelease(events[0]) failed.");

                // Enqueue the results to application pointer
                status = CECL_READ_BUFFER(commandQueue,
                                             priceDerivBuf,
                                             CL_TRUE,
                                             0,
                                             width * height * 2 * sizeof(cl_float4),
                                             priceDeriv,
                                             0,
                                             NULL,
                                             &events[0]);
                CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(priceDerivBuf) failed.");

                // wait for the read buffer to finish execution
                status = waitForEventAndRelease(&events[0]);
                CHECK_OPENCL_ERROR(status, "clWaitForEventsAndRelease(events[0]) failed.");

            }
            // Replace Following "for" loop with reduction kernel

            for(size_t i = 0; i < sizeAfterReduction; i++)
            {
                price[k] += priceVals[i];
                vega[k] += priceDeriv[i];
            }

            price[k] /= (noOfTraj * noOfTraj);
            vega[k] /= (noOfTraj * noOfTraj);

            price[k] = exp(-interest * maturity) * price[k];
            vega[k] = exp(-interest * maturity) * vega[k];
        }
    }

    return SDK_SUCCESS;
}

int
MonteCarloAsian::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initialization failed");
    const int optionsCount = 7;
    Option *optionList = new Option[optionsCount];
    CHECK_ALLOCATION(optionList, " Allocate memory failed (optionList)\n");

    optionList[0]._sVersion = "c";
    optionList[0]._lVersion = "steps";
    optionList[0]._description = "Steps of Monte carlo simuation";
    optionList[0]._type = CA_ARG_INT;
    optionList[0]._value = &steps;

    optionList[1]._sVersion = "P";
    optionList[1]._lVersion = "initPrice";
    optionList[1]._description = "Initial price(Default value 50)";
    optionList[1]._type = CA_ARG_FLOAT;//STRING;
    optionList[1]._value = &initPrice;

    optionList[2]._sVersion = "s";
    optionList[2]._lVersion = "strikePrice";
    optionList[2]._description = "Strike price (Default value 55)";
    optionList[2]._type = CA_ARG_FLOAT;//STRING;
    optionList[2]._value = &strikePrice;

    optionList[3]._sVersion = "r";
    optionList[3]._lVersion = "interest";
    optionList[3]._description = "interest rate > 0 && <=1.0 (Default value 0.06)";
    optionList[3]._type = CA_ARG_FLOAT;//STRING;
    optionList[3]._value = &interest;

    optionList[4]._sVersion = "m";
    optionList[4]._lVersion = "maturity";
    optionList[4]._description = "Maturity (Default value 1)";
    optionList[4]._type = CA_ARG_FLOAT;//STRING;
    optionList[4]._value = &maturity;

    optionList[5]._sVersion = "";
    optionList[5]._lVersion = "scalar";
    optionList[5]._description =
        "Run scalar version of the kernel(--scalar and --vector options are mutually exclusive)";
    optionList[5]._type = CA_NO_ARGUMENT;
    optionList[5]._value = &useScalarKernel;

    optionList[6]._sVersion = "";
    optionList[6]._lVersion = "vector";
    optionList[6]._description =
        "Run vector version of the kernel(--scalar and --vector options are mutually exclusive)";
    optionList[6]._type = CA_NO_ARGUMENT;
    optionList[6]._value = &useVectorKernel;

    for(cl_int i = 0; i < optionsCount; ++i)
    {
        sampleArgs->AddOption(&optionList[i]);
    }

    delete[] optionList;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option,
                     "Failed to allocate memory (iteration_option)\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);

    delete iteration_option;

    Option* inPersistent_option = new Option;
    CHECK_ALLOCATION(inPersistent_option,
                     "Failed to allocate memory (inPersistent_option)\n");

    inPersistent_option->_sVersion = "";
    inPersistent_option->_lVersion = "dInPersistent";
    inPersistent_option->_description =
        "Disables the Persistent memory for input buffers";
    inPersistent_option->_type = CA_NO_ARGUMENT;
    inPersistent_option->_value = &dUseInPersistent;

    sampleArgs->AddOption(inPersistent_option);

    delete inPersistent_option;

    Option* outAllocHostPtr_option = new Option;
    CHECK_ALLOCATION(outAllocHostPtr_option,
                     "Failed to allocate memory (outAllocHostPtr_option)\n");

    outAllocHostPtr_option->_sVersion = "";
    outAllocHostPtr_option->_lVersion = "dOutAllocHostPtr";
    outAllocHostPtr_option->_description =
        "Disables the Alloc host ptr for output buffers";
    outAllocHostPtr_option->_type = CA_NO_ARGUMENT;
    outAllocHostPtr_option->_value = &dUseOutAllocHostPtr;

    sampleArgs->AddOption(outAllocHostPtr_option);

    delete outAllocHostPtr_option;

    Option* disableMapping_option = new Option;
    CHECK_ALLOCATION(disableMapping_option,
                     "Failed to allocate memory (disableMapping_option)\n");

    disableMapping_option->_sVersion = "";
    disableMapping_option->_lVersion = "dMapping";
    disableMapping_option->_description =
        "Disables mapping/unmapping and uses read/write buffers.";
    disableMapping_option->_type = CA_NO_ARGUMENT;
    disableMapping_option->_value = &disableMapping;

    sampleArgs->AddOption(disableMapping_option);

    delete disableMapping_option;

    Option* disableAsync_option = new Option;
    CHECK_ALLOCATION(disableAsync_option,
                     "Failed to allocate memory (disableAsync_option)\n");

    disableAsync_option->_sVersion = "";
    disableAsync_option->_lVersion = "dAsync";
    disableAsync_option->_description = "Disables Asynchronous.";
    disableAsync_option->_type = CA_NO_ARGUMENT;
    disableAsync_option->_value = &disableAsync;

    sampleArgs->AddOption(disableAsync_option);

    delete disableAsync_option;

    return SDK_SUCCESS;
}

int MonteCarloAsian::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
	int status = setupMonteCarloAsian();
    if(status != SDK_SUCCESS)
    {
		return status;
    }

    // create and initialize timers
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


int MonteCarloAsian::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
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
        printArray<cl_float>("price", price, steps, 1);
        printArray<cl_float>("vega", vega, steps, 1);
    }

    return SDK_SUCCESS;
}

void
MonteCarloAsian::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Steps",
            "Setup Time(sec)",
            "Avg. Kernel Time (sec)",
            "Samples used /sec"
        };
        std::string stats[4];

        double avgKernelTime = (kernelTime / iterations);
        stats[0] = toString(steps, std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString(avgKernelTime, std::dec);
        stats[3] = toString((noOfTraj * (noOfSum - 1) * steps) /
                            avgKernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

void
MonteCarloAsian::lshift128(unsigned int* input,
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
MonteCarloAsian::rshift128(unsigned int* input,
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
MonteCarloAsian::generateRand(unsigned int* seed,
                              float *gaussianRand1,
                              float *gaussianRand2,
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
    const float one = 1.0f;
    const float intMax = 4294967296.0f;
    const float PI = 3.14159265358979f;
    const float two = 2.0f;

    float r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float phi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float temp1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float temp2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
MonteCarloAsian::calOutputs(float strikePrice,
                            float* meanDeriv1,
                            float*  meanDeriv2,
                            float* meanPrice1,
                            float* meanPrice2,
                            float* pathDeriv1,
                            float* pathDeriv2,
                            float* priceVec1,
                            float* priceVec2)
{
    float temp1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float temp2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float temp3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float temp4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float tempDiff1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tempDiff2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for(int c = 0; c < 4; ++c)
    {
        tempDiff1[c] = meanPrice1[c] - strikePrice;
        tempDiff2[c] = meanPrice2[c] - strikePrice;
    }
    if(tempDiff1[0] > 0.0f)
    {
        temp1[0] = 1.0f;
        temp3[0] = tempDiff1[0];
    }
    if(tempDiff1[1] > 0.0f)
    {
        temp1[1] = 1.0f;
        temp3[1] = tempDiff1[1];
    }
    if(tempDiff1[2] > 0.0f)
    {
        temp1[2] = 1.0f;
        temp3[2] = tempDiff1[2];
    }
    if(tempDiff1[3] > 0.0f)
    {
        temp1[3] = 1.0f;
        temp3[3] = tempDiff1[3];
    }

    if(tempDiff2[0] > 0.0f)
    {
        temp2[0] = 1.0f;
        temp4[0] = tempDiff2[0];
    }
    if(tempDiff2[1] > 0.0f)
    {
        temp2[1] = 1.0f;
        temp4[1] = tempDiff2[1];
    }
    if(tempDiff2[2] > 0.0f)
    {
        temp2[2] = 1.0f;
        temp4[2] = tempDiff2[2];
    }
    if(tempDiff2[3] > 0.0f)
    {
        temp2[3] = 1.0f;
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

void MonteCarloAsian::cpuReferenceImpl()
{
    float timeStep = maturity / (noOfSum - 1);
    // Initialize random number generator
    srand(1);

    for(int k = 0; k < steps; k++)
    {
        float c1 = (interest - 0.5f * sigma[k] * sigma[k]) * timeStep;
        float c2 = sigma[k] * sqrt(timeStep);
        float c3 = (interest + 0.5f * sigma[k] * sigma[k]);

        for(int j = 0; j < (width * height); j++)
        {
            unsigned int nextRand[4] = {0u, 0u, 0u, 0u};
            for(int c = 0; c < 4; ++c)
            {
                nextRand[c] = (cl_uint)rand();
            }

            float trajPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
            float sumPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
            float sumDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float meanPrice1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float meanDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float price1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float pathDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            float trajPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
            float sumPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
            float sumDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float meanPrice2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float meanDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float price2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float pathDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            //Run the Monte Carlo simulation a total of Num_Sum - 1 times
            for(int i = 1; i < noOfSum; i++)
            {
                unsigned int tempRand[4] =  {0u, 0u, 0u, 0u};
                for(int c = 0; c < 4; ++c)
                {
                    tempRand[c] = nextRand[c];
                }

                float gaussian1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float gaussian2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                generateRand(tempRand, gaussian1, gaussian2, nextRand);

                //Calculate the trajectory price and sum price for all trajectories
                for(int c = 0; c < 4; ++c)
                {
                    trajPrice1[c] = trajPrice1[c] * exp(c1 + c2 * gaussian1[c]);
                    trajPrice2[c] = trajPrice2[c] * exp(c1 + c2 * gaussian2[c]);

                    sumPrice1[c] = sumPrice1[c] + trajPrice1[c];
                    sumPrice2[c] = sumPrice2[c] + trajPrice2[c];

                    float temp = c3 * timeStep * i;

                    // Calculate the derivative price for all trajectories
                    sumDeriv1[c] = sumDeriv1[c] + trajPrice1[c]
                                   * ((log(trajPrice1[c] / initPrice) - temp) / sigma[k]);

                    sumDeriv2[c] = sumDeriv2[c] + trajPrice2[c]
                                   * ((log(trajPrice2[c] / initPrice) - temp) / sigma[k]);
                }

            }

            //Calculate the average price and average derivative of each simulated path
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

int MonteCarloAsian::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        cpuReferenceImpl();

        // compare the results and see if they match
        for(int i = 0; i < steps; ++i)
        {
            if(fabs(price[i] - refPrice[i]) > 0.2f)
            {
                std::cout << "Failed\n";
                return SDK_FAILURE;
            }
            if(fabs(vega[i] - refVega[i]) > 0.2f)
            {
                std::cout << "Failed\n" << std::endl;
                return SDK_FAILURE;
            }
        }
        std::cout << "Passed!\n" << std::endl;
    }

    return SDK_SUCCESS;
}

int MonteCarloAsian::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(priceBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBuf) failed.");

    status = clReleaseMemObject(priceDerivBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceDerivBuf) failed.");

    status = clReleaseMemObject(randBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBuf) failed.");

    if(!disableAsync)
    {
        status = clReleaseMemObject(priceBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBufAsync) failed.");

        status = clReleaseMemObject(priceDerivBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceDerivBufAsync) failed.");

        status = clReleaseMemObject(randBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBufAsync) failed.");
    }

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel(kernel) failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram(program) failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue(readKernel) failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext(context) failed.");

    // Release program resources (input memory etc.)
    FREE(sigma);
    FREE(price);
    FREE(vega);
    FREE(refPrice);
    FREE(refVega);
    FREE(priceVals);
    FREE(priceDeriv);

    if(!disableAsync && disableMapping)
    {
        FREE(priceValsAsync);
        FREE(priceDerivAsync);
    }
    FREE(devices);
    return SDK_SUCCESS;
}

template<typename T>
int MonteCarloAsian::asyncMapBuffer(cl_mem deviceBuffer, T* &hostPointer,
                                    size_t sizeInBytes, cl_map_flags flags, cl_event *event)
{
    cl_int status;
    hostPointer = (T*) CECL_MAP_BUFFER(commandQueue,
                                          deviceBuffer,
                                          CL_FALSE,
                                          flags,
                                          0,
                                          sizeInBytes,
                                          0,
                                          NULL,
                                          event,
                                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed");

    // Flush the enqueued commands on commandQueue, guarantes commands submitted to device.
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    return SDK_SUCCESS;
}

int
MonteCarloAsian::asyncUnmapBuffer(cl_mem deviceBuffer, void* hostPointer,
                                  cl_event *event)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     event);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    // Flush the enqueued commands on commandQueue, guarantes commands submitted to device.
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    // Create MonteCalroAsian object
    MonteCarloAsian clMonteCarloAsian;

    // Initialization
    if(clMonteCarloAsian.initialize()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(clMonteCarloAsian.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clMonteCarloAsian.sampleArgs->isDumpBinaryEnabled())
    {
        return clMonteCarloAsian.genBinaryImage();
    }
    else
    {
        // Setup
		int status = clMonteCarloAsian.setup();
        if(status != SDK_SUCCESS)
        {
            return status;
        }

        // Run
        if(clMonteCarloAsian.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        //VerifyResults.
        if(clMonteCarloAsian.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Cleanup resources created
        if(clMonteCarloAsian.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Print performance statistics
        clMonteCarloAsian.printStats();
    }

    return SDK_SUCCESS;
}
