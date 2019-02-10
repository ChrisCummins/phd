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


#include "MonteCarloAsianMultiGPU.hpp"

#include <math.h>
#include <malloc.h>


/*
 *  structure for attributes of Monte carlo
 *  simulation
 */

typedef struct _MonteCalroAttrib
{
    cl_float4 strikePrice;
    cl_float4 c1;
    cl_float4 c2;
    cl_float4 c3;
    cl_float4 initPrice;
    cl_float4 sigma;
    cl_float4 timeStep;
} MonteCarloAttrib;


int
MonteCarloAsianMultiGPU::setupMonteCarloAsianMultiGPU()
{
    steps = (steps < 4) ? 4 : steps;
    steps = (steps / 2) * 2;

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
    memset((void*)price,
           0,
           steps * sizeof(cl_float));

    vega = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(price, "Failed to allocate host memory. (vega)");
    memset((void*)vega,
           0,
           steps * sizeof(cl_float));

    refPrice = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(refPrice, "Failed to allocate host memory. (refPrice)");
    memset((void*)refPrice,
           0,
           steps * sizeof(cl_float));

    refVega = (cl_float*) malloc(steps * sizeof(cl_float));
    CHECK_ALLOCATION(refVega, "Failed to allocate host memory. (refVega)");
    memset((void*)refVega,
           0,
           steps * sizeof(cl_float));

    // Set samples and exercize points
    noOfSum = 12;
    noOfTraj = 1024;

    width = noOfTraj / 4;
    height = noOfTraj / 2;

#if defined (_WIN32)
    randNum = (cl_uint*)_aligned_malloc(width * height * sizeof(cl_uint4),
                                        16);
#else
    randNum = (cl_uint*)memalign(16,
                                 width * height * sizeof(cl_uint4));
#endif
    CHECK_ALLOCATION(randNum, "Failed to allocate host memory. (randNum)");

    priceVals = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceVals, "Failed to allocate host memory. (priceVals)");

    memset((void*)priceVals,
           0,
           width * height * 2 * sizeof(cl_float4));

    priceDeriv = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceDeriv, "Failed to allocate host memory. (priceDeriv)");

    memset((void*)priceDeriv,
           0,
           width * height * 2 * sizeof(cl_float4));

    priceValsAsync = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceValsAsync,
                     "Failed to allocate host memory. (priceValsAsync)");

    memset((void*)priceValsAsync,
           0,
           width * height * 2 * sizeof(cl_float4));

    priceDerivAsync = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
    CHECK_ALLOCATION(priceDerivAsync,
                     "Failed to allocate host memory. (priceDerivAsync)");

    memset((void*)priceDerivAsync,
           0,
           width * height * 2 * sizeof(cl_float4));

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
MonteCarloAsianMultiGPU::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("MonteCarloAsianMultiGPU_Kernels.cl");
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
MonteCarloAsianMultiGPU::loadBalancing()
{
    /**
    * Calculating the peak GFlops of each device
    **/

    peakGflopsGPU = new cl_double[numGPUDevices];

    CHECK_ALLOCATION(peakGflopsGPU, "Allocation failed(peakGflopGPU)");
    double toatalPeakGflops = 0;//Sum of peakGflops of all the devices
    for (int i = 0; i < numGPUDevices; i++)
    {
        cl_int numComputeUnits = devicesInfo[i].maxComputeUnits;

        cl_int maxClockFrequency = devicesInfo[i].maxClockFrequency;

        char * deviceName = devicesInfo[i].name;

        /**
        * In cayman device we have four processing elements in a stream processor
        **/
        int numProcessingElts;

        if (!strcmp(deviceName,
                    "Cayman"))
        {
            numProcessingElts = 4;
        }
        else
        {
            numProcessingElts = 5;
        }
        /**
        * We have 16 stream processors per compute unit and numProcessingelts number of
        * processing elements in a stream processor. A processing elt can execute 2 floating point operations
        * per cycle. So the peakGflops formula would be
        *
        *   peakGflops = numComputeUnits * numStreamProcessor per Compute Unit * numProcessingElt per stream processor
        *                       * maxClockFrequency * num Floating Point operations per cycle per processing elt;
        *                       We have to divide the result by 1000 to get the Gflops as freq is in MHz.
        *
        **/
        peakGflopsGPU[i] = (numComputeUnits * 16 * numProcessingElts * maxClockFrequency
                            * 2) / 1000;
        toatalPeakGflops = peakGflopsGPU[i] + toatalPeakGflops;
    }

    /**
    * ratios function will be used in load balancing statically
    **/
    double *ratios = new cl_double[numGPUDevices];
    CHECK_ALLOCATION(ratios, "Allocation failed!!(ratios)");

    numStepsPerGPU = new cl_int[numGPUDevices];
    CHECK_ALLOCATION(numStepsPerGPU, "Allocation failed(numStepsPerGPU)!!");

    /**
    * This array is used while copying the result into output array
    */
    cumulativeStepsPerGPU = new cl_int[numGPUDevices];
    CHECK_ALLOCATION(cumulativeStepsPerGPU,
                     "Allocation failed(cumulativeStepsPerGPU)!!");

    int cumulativeSumSteps = 0;

    for (int i = 0; i < numGPUDevices; i++)
    {
        ratios[i] = peakGflopsGPU[i] / toatalPeakGflops;
        numStepsPerGPU[i] = static_cast<cl_int>(steps * ratios[i]);

        cumulativeSumSteps += numStepsPerGPU[i];
        cumulativeStepsPerGPU[i] = cumulativeSumSteps;

        /**
        * There can be a possibility that some values are missed. To avoid, add the missing values to the last GPU
        **/
        if (i == numGPUDevices - 1 && cumulativeSumSteps < steps)
        {
            numStepsPerGPU[i] = numStepsPerGPU[i] + (steps - cumulativeSumSteps);
            cumulativeStepsPerGPU[i] += (steps - cumulativeSumSteps);
        }
    }

    if (ratios)
    {
        delete []ratios;
        ratios = NULL;
    }

    return SDK_SUCCESS;
}

int
MonteCarloAsianMultiGPU::setupCL(void)
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

    //If -d option is enabled or if device-type is CPU, make noMultiGPUSupport to true
    if(sampleArgs->isDeviceIdEnabled() || (dType == CL_DEVICE_TYPE_GPU))
    {
        noMultiGPUSupport =true;
    }

    // Get platform
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform,
                               sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform,
                              dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    if(dType == CL_DEVICE_TYPE_GPU)
    {
        //Get Number of devices available
        status = clGetDeviceIDs(platform,
                                CL_DEVICE_TYPE_GPU,
                                0,
                                0,
                                (cl_uint*)&numGPUDevices);
        CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed");

        gpuDeviceIDs = new cl_device_id[numGPUDevices];
        CHECK_ALLOCATION(gpuDeviceIDs, "Allocation failed(gpuDeviceIDs)");

        status = clGetDeviceIDs(platform,
                                CL_DEVICE_TYPE_GPU,
                                numGPUDevices,
                                gpuDeviceIDs,
                                0);

        CHECK_OPENCL_ERROR(status, "clGetDeviceIDs failed.");
    }

    if(!noMultiGPUSupport)
    {
        commandQueues = new cl_command_queue[numGPUDevices];
        CHECK_ALLOCATION(commandQueues, "Allocation failed(commandQueues)");

        programs = new cl_program[numGPUDevices];
        CHECK_ALLOCATION(programs, "Allocation failed(programs)");

        kernels = new cl_kernel[numGPUDevices];
        CHECK_ALLOCATION(kernels, "Allocation failed(kernels)");

        devicesInfo = new SDKDeviceInfo[numGPUDevices];
        CHECK_ALLOCATION(devicesInfo, "Allocation failed(devicesInfo)!!");

        randBufs = new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(randBufs, "Allocation failed(randBufs)");

        randBufsAsync = new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(randBufsAsync,
                         "Allocation failed(randBufsAsync)");

        priceBufs = new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(priceBufs, "Allocation failed(priceBufs)");

        priceDerivBufs= new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(priceDerivBufs,
                         "Allocation failed(priceDerivBufs)");

        priceBufsAsync= new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(priceBufsAsync, "Allocation failed(priceBufsAsync)");

        priceDerivBufsAsync = new cl_mem[numGPUDevices];
        CHECK_ALLOCATION(priceDerivBufsAsync, "Allocation failed(priceDerivBufsAsync)");
    }

    /**
    * Used to store the device information in case of Single device
    */

    SDKDeviceInfo currentDeviceInfo;
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

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    cl_command_queue_properties prop = 0;

    if (!noMultiGPUSupport)
    {
        for (int i = 0; i < numGPUDevices; i++)
        {
            commandQueues[i] = CECL_CREATE_COMMAND_QUEUE(context,
                                                    gpuDeviceIDs[i],
                                                    prop,
                                                    &status);
            CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");
        }
    }
    else
    {
        // Create command queue
        commandQueue = CECL_CREATE_COMMAND_QUEUE(context,
                                            devices[sampleArgs->deviceId],
                                            0,
                                            &status);
        CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");
    }

    /**
    * Get the device info of all the devices by using SDKDeviceInfo structure and setDeviceInfo function
    **/

    if (!noMultiGPUSupport)
    {
        for (int i = 0; i < numGPUDevices; i++)
        {
            status = devicesInfo[i].setDeviceInfo(gpuDeviceIDs[i]);
            if(status != SDK_SUCCESS)
            {
                std::cout << "devicesInfo[i].setDeviceInfo failed"<<std::endl;
                return SDK_FAILURE;
            }
        }
    }

    status = currentDeviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    if(status != SDK_SUCCESS)
    {
        std::cout << "currentDeviceInfo.setDeviceInfo failed " << std::endl;
        return SDK_FAILURE;
    }

    maxWorkGroupSize = currentDeviceInfo.maxWorkGroupSize;

    maxDimensions = currentDeviceInfo.maxWorkItemDims;

    maxWorkItemSizes = (size_t*)malloc(maxDimensions * sizeof(size_t));

    memcpy(maxWorkItemSizes, currentDeviceInfo.maxWorkItemSizes,
           maxDimensions * sizeof(size_t));

    totalLocalMemory = currentDeviceInfo.localMemSize;

    /**
    * Call load balancing for work division
    **/
    if (!noMultiGPUSupport)
    {
        status = loadBalancing();
        CHECK_ERROR(status, SDK_SUCCESS, "loadBalancing failed!!");
    }

    if (!noMultiGPUSupport)
    {
        for (int i = 0; i < numGPUDevices; i++)
        {
            randBufs[i] = CECL_BUFFER(context,
                                         CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                         sizeof(cl_uint4) * width  * height,
                                         NULL,
                                         &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBufs[i]) failed.");

            randBufsAsync[i] = CECL_BUFFER(context,
                                              CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                              sizeof(cl_uint4) * width  * height,
                                              NULL,
                                              &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBufsAsync[i]) failed.");

            priceBufs[i] = CECL_BUFFER(context,
                                          CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                          sizeof(cl_float4) * width * height * 2,
                                          NULL,
                                          &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBufs[i]) failed.");

            priceDerivBufs[i] = CECL_BUFFER(context,
                                               CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                               sizeof(cl_float4) * width * height * 2,
                                               NULL,
                                               &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBufs[i]) failed.");

            priceBufsAsync[i] = CECL_BUFFER(context,
                                               CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                               sizeof(cl_float4) * width * height * 2,
                                               NULL,
                                               &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBufsAsync[i]) failed.");


            priceDerivBufsAsync[i] = CECL_BUFFER(context,
                                                    CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                    sizeof(cl_float4) * width * height * 2,
                                                    NULL,
                                                    &status);
            CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBufsAsync[i]) failed.");
        }
    }
    else
    {
        randBuf = CECL_BUFFER(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                 sizeof(cl_uint4) * width  * height,
                                 NULL,
                                 &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBuf) failed.");

        randBufAsync = CECL_BUFFER(context,
                                      CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                      sizeof(cl_uint4) * width  * height,
                                      NULL,
                                      &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(randBufAsync) failed.");


        priceBuf = CECL_BUFFER(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(cl_float4) * width * height * 2,
                                  NULL,
                                  &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBuf) failed.");


        priceDerivBuf = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       sizeof(cl_float4) * width * height * 2,
                                       NULL,
                                       &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBuf) failed.");

        priceBufAsync = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       sizeof(cl_float4) * width * height * 2,
                                       NULL,
                                       &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceBufAsync) failed.");


        priceDerivBufAsync = CECL_BUFFER(context,
                                            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                            sizeof(cl_float4) * width * height * 2,
                                            NULL,
                                            &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER(priceDerivBufAsync) failed.");
    }

    // Create a CL program using the kernel source
    buildProgramData buildData;

    std::string kernelPath = getPath();

    if(sampleArgs->isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs->loadBinary.c_str());
    }
    else
    {
        kernelPath.append("MonteCarloAsianMultiGPU_Kernels.cl");
    }

    if(!noMultiGPUSupport)
    {
        for (int i = 0; i < numGPUDevices; i++)
        {
            // create a CL program using the kernel source
            buildProgramData buildData;
            buildData.kernelName = std::string("MonteCarloAsianMultiGPU_Kernels.cl");
            buildData.devices = gpuDeviceIDs;
            buildData.deviceId = i;
            buildData.flagsStr = std::string("");
            if(sampleArgs->isLoadBinaryEnabled())
            {
                buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
            }

            if(sampleArgs->isComplierFlagsSpecified())
            {
                buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
            }

            int retValue = buildOpenCLProgram(programs[i],
                                              context,
                                              buildData);
            CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

            // get a kernel object handle for a kernel with the given name
            kernels[i] = CECL_KERNEL(programs[i],
                                        "calPriceVega",
                                        &status);
            CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

            // Check group-size against what is returned by kernel
            status = CECL_GET_KERNEL_WORK_GROUP_INFO(kernels[i],
                                              gpuDeviceIDs[i],
                                              CL_KERNEL_WORK_GROUP_SIZE,
                                              sizeof(size_t),
                                              &kernelWorkGroupSize,
                                              0);
            CHECK_OPENCL_ERROR(status, "CECL_GET_KERNEL_WORK_GROUP_INFO failed.");

            if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
            {
                if(!sampleArgs->quiet)
                {
                    std::cout << "Out of Resources!"
                              << std::endl;
                    std::cout << "Group Size specified : "
                              << blockSizeX * blockSizeY
                              << std::endl;
                    std::cout << "Max Group Size supported on the kernel : "
                              << kernelWorkGroupSize
                              << std::endl;
                    std::cout << "Falling back to "
                              << kernelWorkGroupSize
                              << std::endl;
                }

                // Three possible cases
                if(blockSizeX > kernelWorkGroupSize)
                {
                    blockSizeX = kernelWorkGroupSize;
                    blockSizeY = 1;
                }
            }
        }
    }
    else
    {
        buildData.kernelName = std::string("MonteCarloAsianMultiGPU_Kernels.cl");
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

        retValue = buildOpenCLProgram(program,
                                      context,
                                      buildData);
        CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

        // get a kernel object handle for a kernel with the given name
        kernel = CECL_KERNEL(program,
                                "calPriceVega",
                                &status);
        CHECK_OPENCL_ERROR(status, "CECL_KERNEL(calPriceVega) failed.");

        // Check group-size against what is returned by kernel
        status = CECL_GET_KERNEL_WORK_GROUP_INFO(kernel,
                                          devices[sampleArgs->deviceId],
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t),
                                          &kernelWorkGroupSize,
                                          0);
        CHECK_OPENCL_ERROR(status,
                           "CECL_GET_KERNEL_WORK_GROUP_INFO(CL_KERNEL_WORK_GROUP_SIZE) failed.");
        if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
        {
            if(!sampleArgs->quiet)
            {
                std::cout << "Out of Resources!"
                          << std::endl;
                std::cout << "Group Size specified : "
                          << blockSizeX * blockSizeY
                          << std::endl;
                std::cout << "Max Group Size supported on the kernel : "
                          << kernelWorkGroupSize
                          << std::endl;
                std::cout << "Falling back to "
                          << kernelWorkGroupSize
                          << std::endl;
            }

            // Three possible cases
            if(blockSizeX > kernelWorkGroupSize)
            {
                blockSizeX = kernelWorkGroupSize;
                blockSizeY = 1;
            }
        }
    }
    if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!"
                      << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY
                      << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize
                      << std::endl;
            std::cout << "Falling back to "
                      << kernelWorkGroupSize
                      << std::endl;
        }

        // Three possible cases
        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

    return SDK_SUCCESS;
}

/**
* This structure is used in multiing
*/

struct dataPerGPU
{
    MonteCarloAsianMultiGPU *mcaObj;
    int deviceNumber;
};

/**
* Thread run function per GPU
*/
void* threadFuncPerGPU(void *data1)
{
    dataPerGPU *data = (dataPerGPU *)data1;
    int deviceNumber = data->deviceNumber;
    MonteCarloAsianMultiGPU *mcaObj = data->mcaObj;

    cl_int status;
    size_t globalThreads[2] = {mcaObj->width, mcaObj->height};
    size_t localThreads[2] = {mcaObj->blockSizeX, mcaObj->blockSizeY};

    // Declare attribute structure
    MonteCarloAttrib attributes;

    if(localThreads[0] > mcaObj->maxWorkItemSizes[0] ||
            localThreads[1] > mcaObj->maxWorkItemSizes[1] ||
            (size_t)mcaObj->blockSizeX * mcaObj->blockSizeY > mcaObj->maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support requested" <<
                  ":number of work items.";
        return NULL;
    }

    // width - i.e number of elements in the array
    status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                            2,
                            sizeof(cl_uint),
                            (void*)&mcaObj->width);

    CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

    status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                            1,
                            sizeof(cl_int),
                            (void*)&mcaObj->noOfSum);
    CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

    float timeStep = mcaObj->maturity / (mcaObj->noOfSum - 1);

    // Initialize random number generator
    srand(1);

    int startIndex = mcaObj->cumulativeStepsPerGPU[deviceNumber] -
                     mcaObj->numStepsPerGPU[deviceNumber];
    int endIndex = mcaObj->cumulativeStepsPerGPU[deviceNumber];

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
    cl_int eventStatus = CL_QUEUED;

    size_t size = mcaObj->width * mcaObj->height * sizeof(cl_float4);

    for (int k = startIndex; k < endIndex ; k += 2)
    {
        // Map input buffer for kernel 1
        inMapPtr1 = CECL_MAP_BUFFER(
                        mcaObj->commandQueues[deviceNumber],
                        mcaObj->randBufs[deviceNumber],
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        size,
                        0,
                        NULL,
                        &inMapEvt1,
                        &status);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                       "CECL_MAP_BUFFER(randBufs[deviceNumber]) failed.");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed.!!");

        //Generate input for kernel 1
        for(int j = 0; j < (mcaObj->width * mcaObj->height * 4); j += 4)
        {
            mcaObj->randNum[j] = (cl_uint)rand();
            mcaObj->randNum[j + 1] = (cl_uint)rand();
            mcaObj->randNum[j + 2] = (cl_uint)rand();
            mcaObj->randNum[j + 3] = (cl_uint)rand();
        }

        // Wait for map of input of kernel 1
        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         inMapEvt1,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed.!!");
        }

        status = clReleaseEvent(inMapEvt1);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed.!!");

        memcpy(inMapPtr1,
               (void*) mcaObj->randNum,
               size);

        // Unmap of input buffer of kernel 1
        status = clEnqueueUnmapMemObject(
                     mcaObj->commandQueues[deviceNumber],
                     mcaObj->randBufs[deviceNumber],
                     inMapPtr1,
                     0,
                     NULL,
                     &inUnmapEvt1);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed.!!");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed.!!");

        //get data from output buffers of kernel2
        if (k != startIndex)
        {
            //Wait for kernel 2 to complete
            status = clWaitForEvents(1, &ndrEvt);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clWaitForEvents failed.!!");

            status = clReleaseEvent(ndrEvt);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed.!!");

            outMapPtr21 = CECL_MAP_BUFFER(
                              mcaObj->commandQueues[deviceNumber],
                              mcaObj->priceBufsAsync[deviceNumber],
                              CL_FALSE,
                              CL_MAP_READ,
                              0,
                              size * 2,
                              0,
                              NULL,
                              &outMapEvt21,
                              &status);
            CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                           "CECL_MAP_BUFFER failed(priceBufsAsync).!!");

            outMapPtr22 = CECL_MAP_BUFFER(
                              mcaObj->commandQueues[deviceNumber],
                              mcaObj->priceDerivBufsAsync[deviceNumber],
                              CL_FALSE,
                              CL_MAP_READ,
                              0,
                              size * 2,
                              0,
                              NULL,
                              &outMapEvt22,
                              &status);
            CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                           "CECL_MAP_BUFFER failed.(priceDerivBufAsyncs[deviceNumber])!!");

            status = clFlush(mcaObj->commandQueues[deviceNumber]);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush() failed.");
        }

        // Set up arguments required for kernel 1
        float c1 = (mcaObj->interest - 0.5f * mcaObj->sigma[k] * mcaObj->sigma[k]) *
                   timeStep;
        float c2 = mcaObj->sigma[k] * sqrt(timeStep);
        float c3 = (mcaObj->interest + 0.5f * mcaObj->sigma[k] * mcaObj->sigma[k]);

        const cl_float4 c1F4 = {c1, c1, c1, c1};
        attributes.c1 = c1F4;

        const cl_float4 c2F4 = {c2, c2, c2, c2};
        attributes.c2 = c2F4;

        const cl_float4 c3F4 = {c3, c3, c3, c3};
        attributes.c3 = c3F4;

        const cl_float4 initPriceF4 = {mcaObj->initPrice,
                                       mcaObj->initPrice,
                                       mcaObj->initPrice,
                                       mcaObj->initPrice
                                      };
        attributes.initPrice = initPriceF4;

        const cl_float4 strikePriceF4 = {mcaObj->strikePrice,
                                         mcaObj->strikePrice,
                                         mcaObj->strikePrice,
                                         mcaObj->strikePrice
                                        };
        attributes.strikePrice = strikePriceF4;

        const cl_float4 sigmaF4 = {mcaObj->sigma[k],
                                   mcaObj->sigma[k],
                                   mcaObj->sigma[k],
                                   mcaObj->sigma[k]
                                  };
        attributes.sigma = sigmaF4;

        const cl_float4 timeStepF4 = {timeStep,
                                      timeStep,
                                      timeStep,
                                      timeStep
                                     };
        attributes.timeStep = timeStepF4;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                0,
                                sizeof(attributes),
                                (void*)&attributes);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG(attributes) failed.");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                3,
                                sizeof(cl_mem),
                                (void*)&mcaObj->randBufs[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                4,
                                sizeof(cl_mem),
                                (void*)&mcaObj->priceBufs[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                       "CECL_SET_KERNEL_ARG(priceBufs[deviceNumber]) failed.");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                5,
                                sizeof(cl_mem),
                                (void*)&mcaObj->priceDerivBufs[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                       "CECL_SET_KERNEL_ARG(priceDerivBufs[deviceNumber]) failed.");

        // Wait for input of kernel 1 to complete
        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         inUnmapEvt1,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed.");
        }

        status = clReleaseEvent(inUnmapEvt1);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed.");

        inMapPtr1 = NULL;

        // Enqueue kernel 1
        status = CECL_ND_RANGE_KERNEL(mcaObj->commandQueues[deviceNumber],
                                        mcaObj->kernels[deviceNumber],
                                        2,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_ND_RANGE_KERNEL failed.");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed.");

        // Generate data of input for kernel 2
        // Fill data of input buffer for kernel 2
        if(k <= endIndex - 1)
        {
            // Map input buffer for kernel 1
            inMapPtr2 = CECL_MAP_BUFFER(
                            mcaObj->commandQueues[deviceNumber],
                            mcaObj->randBufsAsync[deviceNumber],
                            CL_FALSE,
                            CL_MAP_WRITE,
                            0,
                            size,
                            0,
                            NULL,
                            &inMapEvt2,
                            &status);

            CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                           "CECL_MAP_BUFFER(randBufAsyncs[i]) failed.");

            status = clFlush(mcaObj->commandQueues[deviceNumber]);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed.");

            // Generate data for input for kernel 1
            for(int j = 0; j < (mcaObj->width * mcaObj->height * 4); j += 4)
            {
                mcaObj->randNum[j] = (cl_uint)rand();
                mcaObj->randNum[j + 1] = (cl_uint)rand();
                mcaObj->randNum[j + 2] = (cl_uint)rand();
                mcaObj->randNum[j + 3] = (cl_uint)rand();
            }

            // Wait for map of input of kernel 1
            eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             inMapEvt2,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
            }

            status = clReleaseEvent(inMapEvt2);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

            memcpy(inMapPtr2,
                   (void*)mcaObj->randNum,
                   size);

            // Unmap of input buffer of kernel 1
            status = clEnqueueUnmapMemObject(
                         mcaObj->commandQueues[deviceNumber],
                         mcaObj->randBufsAsync[deviceNumber],
                         inMapPtr2,
                         0,
                         NULL,
                         &inUnmapEvt2);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed!!");

            status = clFlush(mcaObj->commandQueues[deviceNumber]);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");
        }

        // Wait for output buffers of kernel 2 to complete
        // Calculate the results from output of kernel 2
        if(k != startIndex)
        {
            // Wait for output buffers of kernel 2 to complete
            eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             outMapEvt21,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
            }

            status = clReleaseEvent(outMapEvt21);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

            eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             outMapEvt22,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
            }

            status = clReleaseEvent(outMapEvt22);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

            // Calculate the results from output of kernel 2
            ptr21 = (cl_float*)outMapPtr21;
            ptr22 = (cl_float*)outMapPtr22;

            for(int i = 0; i < mcaObj->noOfTraj * mcaObj->noOfTraj; i++)
            {
                mcaObj->price[k - 1] += ptr21[i];
                mcaObj->vega[k - 1] += ptr22[i];
            }

            mcaObj->price[k - 1] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);
            mcaObj->vega[k - 1] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);

            mcaObj->price[k - 1] = exp(-mcaObj->interest * mcaObj->maturity) *
                                   mcaObj->price[k - 1];
            mcaObj->vega[k  - 1] = exp(-mcaObj->interest * mcaObj->maturity) *
                                   mcaObj->vega[k - 1];

            // Unmap of output buffers of kernel 2
            status = clEnqueueUnmapMemObject(
                         mcaObj->commandQueues[deviceNumber],
                         mcaObj->priceBufsAsync[deviceNumber],
                         outMapPtr21,
                         0,
                         NULL,
                         &outUnmapEvt21);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed!!");

            status = clEnqueueUnmapMemObject(
                         mcaObj->commandQueues[deviceNumber],
                         mcaObj->priceDerivBufsAsync[deviceNumber],
                         outMapPtr22,
                         0,
                         NULL,
                         &outUnmapEvt22);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed!!");

            status = clFlush(mcaObj->commandQueues[deviceNumber]);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");

            eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             outUnmapEvt21,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
            }

            status = clReleaseEvent(outUnmapEvt21);
            CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                           "clReleaseEvent(outUnmapEvt21) failed!!");

            eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             outUnmapEvt22,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
            }

            status = clReleaseEvent(outUnmapEvt22);
            CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                           "clReleaseEvent(outUnmapEvt22) failed!!");
        }

        // Wait for kernel 1 to complete
        status = clWaitForEvents(1,
                                 &ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clWaitForEvents failed!!");

        status = clReleaseEvent(ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        // Get data from output buffers of kernel 1
        outMapPtr11 = CECL_MAP_BUFFER(
                          mcaObj->commandQueues[deviceNumber],
                          mcaObj->priceBufs[deviceNumber],
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          size * 2,
                          0,
                          NULL,
                          &outMapEvt11,
                          &status);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_MAP_BUFFER failed!!");

        outMapPtr12 = CECL_MAP_BUFFER(
                          mcaObj->commandQueues[deviceNumber],
                          mcaObj->priceDerivBufs[deviceNumber],
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          size * 2,
                          0,
                          NULL,
                          &outMapEvt12,
                          &status);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_MAP_BUFFER failed!!");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");

        // Set up arguments required for kernel 2
        float c21 = (mcaObj->interest - 0.5f * mcaObj->sigma[k + 1] *
                     mcaObj->sigma[k + 1]) * timeStep;
        float c22 = mcaObj->sigma[k + 1] * sqrt(timeStep);
        float c23 = (mcaObj->interest + 0.5f * mcaObj->sigma[k + 1] *
                     mcaObj->sigma[k + 1]);

        const cl_float4 c1F42 = {c21, c21, c21, c21};
        attributes.c1 = c1F42;

        const cl_float4 c2F42 = {c22, c22, c22, c22};
        attributes.c2 = c2F42;

        const cl_float4 c3F42 = {c23, c23, c23, c23};
        attributes.c3 = c3F42;

        const cl_float4 initPriceF42 = {mcaObj->initPrice,
                                        mcaObj->initPrice,
                                        mcaObj->initPrice,
                                        mcaObj->initPrice
                                       };
        attributes.initPrice = initPriceF42;

        const cl_float4 strikePriceF42 = {mcaObj->strikePrice,
                                          mcaObj->strikePrice,
                                          mcaObj->strikePrice,
                                          mcaObj->strikePrice
                                         };
        attributes.strikePrice = strikePriceF42;

        const cl_float4 sigmaF42 = {mcaObj->sigma[k + 1],
                                    mcaObj->sigma[k + 1],
                                    mcaObj->sigma[k + 1],
                                    mcaObj->sigma[k + 1]
                                   };
        attributes.sigma = sigmaF42;

        const cl_float4 timeStepF42 = {timeStep,
                                       timeStep,
                                       timeStep,
                                       timeStep
                                      };
        attributes.timeStep = timeStepF42;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                0,
                                sizeof(attributes),
                                (void*)&attributes);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                3,
                                sizeof(cl_mem),
                                (void*)&mcaObj->randBufsAsync[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                4,
                                sizeof(cl_mem),
                                (void*)&mcaObj->priceBufsAsync[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

        status = CECL_SET_KERNEL_ARG(mcaObj->kernels[deviceNumber],
                                5,
                                sizeof(cl_mem),
                                (void*)&mcaObj->priceDerivBufsAsync[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_SET_KERNEL_ARG failed!!");

        // Wait for input of kernel 2 to complete
        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         inUnmapEvt2,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(inUnmapEvt2);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        inMapPtr2 = NULL;

        // Enqueue kernel 2
        status = CECL_ND_RANGE_KERNEL(mcaObj->commandQueues[deviceNumber],
                                        mcaObj->kernels[deviceNumber],
                                        2,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_ND_RANGE_KERNEL failed!!");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");

        // Wait for output buffers of kernel 1 to complete
        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outMapEvt11,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outMapEvt11);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outMapEvt12,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outMapEvt12);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        // Calculate the results from output of kernel 1
        ptr21 = (cl_float*)outMapPtr11;
        ptr22 = (cl_float*)outMapPtr12;

        for(int i = 0; i < mcaObj->noOfTraj * mcaObj->noOfTraj; i++)
        {
            mcaObj->price[k] += ptr21[i];
            mcaObj->vega[k] += ptr22[i];
        }

        mcaObj->price[k] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);
        mcaObj->vega[k] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);

        mcaObj->price[k] = exp(-mcaObj->interest * mcaObj->maturity) * mcaObj->price[k];
        mcaObj->vega[k] = exp(-mcaObj->interest * mcaObj->maturity) * mcaObj->vega[k];


        // Unmap of output buffers of kernel 1
        status = clEnqueueUnmapMemObject(
                     mcaObj->commandQueues[deviceNumber],
                     mcaObj->priceBufs[deviceNumber],
                     outMapPtr11,
                     0,
                     NULL,
                     &outUnmapEvt11);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed!!");

        status = clEnqueueUnmapMemObject(
                     mcaObj->commandQueues[deviceNumber],
                     mcaObj->priceDerivBufs[deviceNumber],
                     outMapPtr12,
                     0,
                     NULL,
                     &outUnmapEvt12);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clEnqueueUnmapMemObject failed!!");


        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");

        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outUnmapEvt11,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outUnmapEvt11);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                       "clReleaseEvent(outUnmapEvt11) failed!!");

        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outUnmapEvt12,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outUnmapEvt12);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,
                                       "clReleaseEvent(outUnmapEvt12) failed!!");
    }

    if ((endIndex - startIndex) % 2 == 0)
    {
        // Wait for kernel 1 to complete
        status = clWaitForEvents(1, &ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status,"clWaitForEvents failed!!");

        status = clReleaseEvent(ndrEvt);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        // Gather last kernel 2 execution here
        outMapPtr21 = CECL_MAP_BUFFER(
                          mcaObj->commandQueues[deviceNumber],
                          mcaObj->priceBufsAsync[deviceNumber],
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          size * 2,
                          0,
                          NULL,
                          &outMapEvt21,
                          &status);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_MAP_BUFFER failed!!");

        outMapPtr22 = CECL_MAP_BUFFER(
                          mcaObj->commandQueues[deviceNumber],
                          mcaObj->priceDerivBufsAsync[deviceNumber],
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          size * 2,
                          0,
                          NULL,
                          &outMapEvt22,
                          &status);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "CECL_MAP_BUFFER failed!!");

        status = clFlush(mcaObj->commandQueues[deviceNumber]);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clFlush failed!!");

        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outMapEvt21,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outMapEvt21);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        eventStatus = CL_QUEUED;
        while(eventStatus != CL_COMPLETE)
        {
            status = clGetEventInfo(
                         outMapEvt22,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &eventStatus,
                         NULL);
            CHECK_OPENCL_ERROR_RETURN_NULL(status, "clGetEventInfo failed!!");
        }

        status = clReleaseEvent(outMapEvt22);
        CHECK_OPENCL_ERROR_RETURN_NULL(status, "clReleaseEvent failed!!");

        // Calculate the results from output of kernel 2
        ptr21 = (cl_float*)outMapPtr21;
        ptr22 = (cl_float*)outMapPtr22;
        for(int i = 0; i < mcaObj->noOfTraj * mcaObj->noOfTraj; i++)
        {
            mcaObj->price[endIndex - 1] += ptr21[i];
            mcaObj->vega[endIndex - 1] += ptr22[i];
        }

        mcaObj->price[endIndex - 1] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);
        mcaObj->vega[endIndex - 1] /= (mcaObj->noOfTraj * mcaObj->noOfTraj);

        mcaObj->price[endIndex - 1] = exp(-mcaObj->interest * mcaObj->maturity) *
                                      mcaObj->price[endIndex - 1];
        mcaObj->vega[endIndex - 1] = exp(-mcaObj->interest * mcaObj->maturity) *
                                     mcaObj->vega[endIndex - 1];
    }
    return NULL;
}

int
MonteCarloAsianMultiGPU::runCLKernelsMultiGPU(void)
{
    SDKThread *threads = new SDKThread[numGPUDevices];
    CHECK_ALLOCATION(threads, "Allocation failed!!");

    dataPerGPU *data = new dataPerGPU[numGPUDevices];
    CHECK_ALLOCATION(data, "Allocation failed!!");

    for (int i = 0 ; i < numGPUDevices; i++)
    {
        data[i].deviceNumber = i;
        data[i].mcaObj = this;
        threads[i].create(threadFuncPerGPU,
                          (void *) &data[i]);
    }

    for (int i = 0; i < numGPUDevices; i++)
    {
        threads[i].join();
    }

    delete []threads;
    delete []data;
    return SDK_SUCCESS;
}

int
MonteCarloAsianMultiGPU::runCLKernels(void)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    size_t globalThreads[2] = {width, height};
    size_t localThreads[2] = {blockSizeX, blockSizeY};

    // Declare attribute structure
    MonteCarloAttrib attributes;

    if(localThreads[0] > maxWorkItemSizes[0] ||
            localThreads[1] > maxWorkItemSizes[1] ||
            (size_t)blockSizeX * blockSizeY > maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support requested"
                  << ":number of work items.";
        return SDK_FAILURE;
    }

    /* width - i.e number of elements in the array */
    status = CECL_SET_KERNEL_ARG(kernel,
                            2,
                            sizeof(cl_uint),
                            (void*)&width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(width) failed.");

    status = CECL_SET_KERNEL_ARG(kernel,
                            1,
                            sizeof(cl_int),
                            (void*)&noOfSum);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(noOfSum) failed.");

    float timeStep = maturity / (noOfSum - 1);

    // Initialize random number generator
    srand(1);

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

    size_t size = width * height * sizeof(cl_float4);
    for(int k = 0; k < steps / 2; k++)
    {
        // Map input buffer for kernel 1
        inMapPtr1 = CECL_MAP_BUFFER(
                        commandQueue,
                        randBuf,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        size,
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

        // Wait for map of input of kernel 1
        status = waitForEventAndRelease(&inMapEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");

        memcpy(inMapPtr1,
               (void*)randNum,
               size);

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
                              size * 2,
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
                              size * 2,
                              0,
                              NULL,
                              &outMapEvt22,
                              &status);
            CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceDerivBufAsync) failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush() failed.");
        }

        // Set up arguments required for kernel 1
        float c1 = (interest - 0.5f * sigma[k * 2] * sigma[k * 2]) * timeStep;
        float c2 = sigma[k * 2] * sqrt(timeStep);
        float c3 = (interest + 0.5f * sigma[k * 2] * sigma[k * 2]);

        const cl_float4 c1F4 = {c1, c1, c1, c1};
        attributes.c1 = c1F4;

        const cl_float4 c2F4 = {c2, c2, c2, c2};
        attributes.c2 = c2F4;

        const cl_float4 c3F4 = {c3, c3, c3, c3};
        attributes.c3 = c3F4;

        const cl_float4 initPriceF4 = {initPrice,
                                       initPrice,
                                       initPrice,
                                       initPrice
                                      };
        attributes.initPrice = initPriceF4;

        const cl_float4 strikePriceF4 = {strikePrice,
                                         strikePrice,
                                         strikePrice,
                                         strikePrice
                                        };
        attributes.strikePrice = strikePriceF4;

        const cl_float4 sigmaF4 = {sigma[k * 2], sigma[k * 2], sigma[k * 2], sigma[k * 2]};
        attributes.sigma = sigmaF4;

        const cl_float4 timeStepF4 = {timeStep,
                                      timeStep,
                                      timeStep,
                                      timeStep
                                     };
        attributes.timeStep = timeStepF4;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(kernel,
                                0,
                                sizeof(attributes),
                                (void*)&attributes);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributes) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                3,
                                sizeof(cl_mem),
                                (void*)&randBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                4,
                                sizeof(cl_mem),
                                (void*)&priceBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                5,
                                sizeof(cl_mem),
                                (void*)&priceDerivBuf);
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
                            size,
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

            memcpy(inMapPtr2, (void*)randNum, size);

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
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt21) Failed");

            status = waitForEventAndRelease(&outMapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt22) Failed");

            // Calculate the results from output of kernel 2
            ptr21 = (cl_float*)outMapPtr21;
            ptr22 = (cl_float*)outMapPtr22;
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
                        "WaitForEventAndRelease(outUnmapEvt21) Failed");

            status = waitForEventAndRelease(&outUnmapEvt22);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "WaitForEventAndRelease(outUnmapEvt22) Failed");
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
                          size * 2,
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
                          size * 2,
                          0,
                          NULL,
                          &outMapEvt12,
                          &status);
        CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(priceDerivBuf) failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");

        // Set up arguments required for kernel 2
        float c21 = (interest - 0.5f * sigma[k * 2 + 1] * sigma[k * 2 + 1]) * timeStep;
        float c22 = sigma[k * 2 + 1] * sqrt(timeStep);
        float c23 = (interest + 0.5f * sigma[k * 2 + 1] * sigma[k * 2 + 1]);

        const cl_float4 c1F42 = {c21, c21, c21, c21};
        attributes.c1 = c1F42;

        const cl_float4 c2F42 = {c22, c22, c22, c22};
        attributes.c2 = c2F42;

        const cl_float4 c3F42 = {c23, c23, c23, c23};
        attributes.c3 = c3F42;

        const cl_float4 initPriceF42 = {initPrice,
                                        initPrice,
                                        initPrice,
                                        initPrice
                                       };
        attributes.initPrice = initPriceF42;

        const cl_float4 strikePriceF42 = {strikePrice,
                                          strikePrice,
                                          strikePrice,
                                          strikePrice
                                         };
        attributes.strikePrice = strikePriceF42;

        const cl_float4 sigmaF42 = {sigma[k * 2 + 1],
                                    sigma[k * 2 + 1],
                                    sigma[k * 2 + 1],
                                    sigma[k * 2 + 1]
                                   };
        attributes.sigma = sigmaF42;

        const cl_float4 timeStepF42 = {timeStep,
                                       timeStep,
                                       timeStep,
                                       timeStep
                                      };
        attributes.timeStep = timeStepF42;

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(kernel,
                                0,
                                sizeof(attributes),
                                (void*)&attributes);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(attributes) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                3,
                                sizeof(cl_mem),
                                (void*)&randBufAsync);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(randBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                4,
                                sizeof(cl_mem),
                                (void*)&priceBufAsync);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(priceBuf) failed.");

        status = CECL_SET_KERNEL_ARG(kernel,
                                5,
                                sizeof(cl_mem),
                                (void*)&priceDerivBufAsync);
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
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt11) Failed");

        status = waitForEventAndRelease(&outMapEvt12);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt12) Failed");

        // Calculate the results from output of kernel 2
        ptr21 = (cl_float*)outMapPtr11;
        ptr22 = (cl_float*)outMapPtr12;
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
                      size * 2,
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
                      size * 2,
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
    ptr21 = (cl_float*)outMapPtr21;
    ptr22 = (cl_float*)outMapPtr22;
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
MonteCarloAsianMultiGPU::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    const int optionsCount = 5;
    Option *optionList = new Option[optionsCount];
    CHECK_ALLOCATION(optionList, "Failed to allocate memory (optionList)\n");

    optionList[0]._sVersion = "c";
    optionList[0]._lVersion = "steps";
    optionList[0]._description = "Steps of Monte carlo simulation";
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
    optionList[3]._description = "interest rate (Default value 0.06)";
    optionList[3]._type = CA_ARG_FLOAT;//STRING;
    optionList[3]._value = &interest;

    optionList[4]._sVersion = "m";
    optionList[4]._lVersion = "maturity";
    optionList[4]._description = "Maturity (Default value 1)";
    optionList[4]._type = CA_ARG_FLOAT;//STRING;
    optionList[4]._value = &maturity;

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

    return SDK_SUCCESS;
}

int MonteCarloAsianMultiGPU::setup()
{
    if (setupMonteCarloAsianMultiGPU() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if (setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int MonteCarloAsianMultiGPU::run()
{
    int status = 0;
    // Warmup
    for(int i = 0; i < 2; i++)
    {
        if(noMultiGPUSupport)
        {
            // Arguments are set and execution call is enqueued on command buffer
            if (runCLKernels() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
        else
        {
            if (runCLKernelsMultiGPU() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
    }

    std::cout<<"Executing kernel for " << iterations << " iterations " << std::endl;
    std::cout<<"-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        if(noMultiGPUSupport)
        {
            // Arguments are set and execution call is enqueued on command buffer
            if(runCLKernels() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
        else
        {
            if(runCLKernelsMultiGPU() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;


    if(!sampleArgs->quiet)
    {
        printArray<cl_float>("price", price, steps, 1);
        printArray<cl_float>("vega", vega, steps, 1);
    }

    return SDK_SUCCESS;
}

void
MonteCarloAsianMultiGPU::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[5] =
        {
            "Steps",
            "Time(sec)",
			"SetupTime(sec)",
            "[Transfer+kernel](sec)",
            "Samples used /sec"
        };
        std::string stats[5];

        sampleTimer->totalTime = setupTime + kernelTime;
        stats[0] = toString(steps, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
		stats[2] = toString(setupTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);
        stats[4] = toString((noOfTraj * (noOfSum - 1) * steps) /
                            kernelTime, std::dec);
        printStatistics(strArray, stats, 5);
    }
}

void
MonteCarloAsianMultiGPU::lshift128(unsigned int* input,
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
MonteCarloAsianMultiGPU::rshift128(unsigned int* input,
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
MonteCarloAsianMultiGPU::generateRand(unsigned int* seed,
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
MonteCarloAsianMultiGPU::calOutputs(float strikePrice,
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

void MonteCarloAsianMultiGPU::cpuReferenceImpl()
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

int MonteCarloAsianMultiGPU::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        cpuReferenceImpl();

        std::cout<<"CPU Price values "<<std::endl;

        for (int i = 0; i < steps; i++)
        {
            std::cout<<refPrice[i]<<" ";
        }

        std::cout<<std::endl;

        // compare the results and see if they match
        for(int i = 0; i < steps; ++i)
        {
            if(fabs(price[i] - refPrice[i]) > 0.2f)
            {
                std::cout << "Failed\n" << std::endl;
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

int MonteCarloAsianMultiGPU::cleanup()
{
    cl_int status;
    if (noMultiGPUSupport)
    {
        // Releases OpenCL resources (Context, Memory etc.)

        status = clReleaseMemObject(priceBuf);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBuf) failed.");

        status = clReleaseMemObject(priceDerivBuf);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceDerivBuf) failed.");

        status = clReleaseMemObject(randBuf);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBuf) failed.");

        status = clReleaseMemObject(priceBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBufAsync) failed.");

        status = clReleaseMemObject(priceDerivBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceDerivBufAsync) failed.");

        status = clReleaseMemObject(randBufAsync);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBufAsync) failed.");

        status = clReleaseKernel(kernel);
        CHECK_OPENCL_ERROR(status, "clReleaseKernel(kernel) failed.");

        status = clReleaseProgram(program);
        CHECK_OPENCL_ERROR(status, "clReleaseProgram(program) failed.");

        status = clReleaseCommandQueue(commandQueue);
        CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue(readKernel) failed.");

        status = clReleaseContext(context);
        CHECK_OPENCL_ERROR(status, "clReleaseContext(context) failed.");
    }
    else
    {
        for (int i = 0; i < numGPUDevices; i++)
        {
            // Releases OpenCL resources (Context, Memory etc.)

            status = clReleaseMemObject(priceBufs[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBufs[i]) failed.");

            status = clReleaseMemObject(priceDerivBufs[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceDerivBufs[i]) failed.");

            status = clReleaseMemObject(randBufs[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBufs[i]) failed.");

            status = clReleaseMemObject(priceBufsAsync[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseMemObject(priceBufsAsync[i]) failed.");

            status = clReleaseMemObject(priceDerivBufsAsync[i]);
            CHECK_OPENCL_ERROR(status,
                               "clReleaseMemObject(priceDerivBufsAsync[i]) failed.");

            status = clReleaseMemObject(randBufsAsync[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseMemObject(randBufsAsync[i]) failed.");

            status = clReleaseKernel(kernels[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseKernel(kernels[i]) failed.");

            status = clReleaseProgram(programs[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseProgram(programs[i]) failed.");

            status = clReleaseCommandQueue(commandQueues[i]);
            CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue(commandQueues[i]) failed.");

        }

        status = clReleaseContext(context);
        CHECK_OPENCL_ERROR(status, "clReleaseContext(context) failed.");

        if (commandQueues)
        {
            delete []commandQueues;
            commandQueues = NULL;
        }

        if (kernels)
        {
            delete []kernels;
            kernels = NULL;
        }

        if (programs)
        {
            delete []programs;
            programs = NULL;
        }

        if (randBufsAsync)
        {
            delete []randBufsAsync;
            randBufsAsync = NULL;
        }

        if (priceBufs)
        {
            delete []priceBufs;
            priceBufs = NULL;
        }

        if (priceBufsAsync)
        {
            delete []priceBufsAsync;
            priceBufsAsync = NULL;
        }

        if(randBufs)
        {
            delete []randBufs;
            randBufs = NULL;
        }

        if (priceDerivBufs)
        {
            delete []priceDerivBufs;
            priceDerivBufs = NULL;
        }

        if(priceDerivBufsAsync)
        {
            delete []priceDerivBufsAsync;
            priceDerivBufsAsync = NULL;
        }
    }

    // Release program resources (input memory etc.)

    FREE(sigma);
    FREE(price);
    FREE(vega);
    FREE(refPrice);
    FREE(refVega);

#if defined (_WIN32)
    ALIGNED_FREE(randNum);
#else
    FREE(randNum);
#endif

    FREE(priceVals);
    FREE(priceDeriv);
    FREE(priceValsAsync);
    FREE(priceDerivAsync);
    FREE(devices);

    if(gpuDeviceIDs)
    {
        delete []gpuDeviceIDs;
        gpuDeviceIDs = NULL;
    }

    if(numStepsPerGPU)
    {
        delete []numStepsPerGPU;
        numStepsPerGPU = NULL;
    }

    if(cumulativeStepsPerGPU)
    {
        delete []cumulativeStepsPerGPU;
        cumulativeStepsPerGPU = NULL;
    }

    if(peakGflopsGPU)
    {
        delete []peakGflopsGPU;
        peakGflopsGPU = NULL;
    }

    if(devicesInfo)
    {
        delete []devicesInfo;
        devicesInfo = NULL;
    }

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    cl_int status = 0;
    MonteCarloAsianMultiGPU clMonteCarloAsianMultiGPU;

    if (clMonteCarloAsianMultiGPU.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clMonteCarloAsianMultiGPU.sampleArgs->parseCommandLine(argc,
            argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clMonteCarloAsianMultiGPU.sampleArgs->isDumpBinaryEnabled())
    {
        return clMonteCarloAsianMultiGPU.genBinaryImage();
    }

    if (clMonteCarloAsianMultiGPU.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clMonteCarloAsianMultiGPU.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clMonteCarloAsianMultiGPU.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clMonteCarloAsianMultiGPU.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clMonteCarloAsianMultiGPU.printStats();
    return SDK_SUCCESS;
}
