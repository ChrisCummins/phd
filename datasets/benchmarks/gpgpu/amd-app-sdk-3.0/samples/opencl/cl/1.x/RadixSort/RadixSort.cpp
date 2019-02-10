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


#include "RadixSort.hpp"

#include <math.h>


int
RadixSort::hostRadixSort()
{
    cl_uint *histogram = (cl_uint*)malloc(RADICES * sizeof(cl_uint));
    CHECK_ALLOCATION(histogram, "Failed to allocate host memory. (histogram)");

    cl_uint *tempData = (cl_uint*)malloc(elementCount * sizeof(cl_uint));
    CHECK_ALLOCATION(tempData, "Failed to allocate host memory. (tempData)");

    if(histogram != NULL && tempData != NULL)
    {
        memcpy(tempData, unsortedData, elementCount * sizeof(cl_uint));
        for(int bits = 0; bits < sizeof(cl_uint) * RADIX ; bits += RADIX)
        {
            // Initialize histogram bucket to zeros
            memset(histogram, 0, RADICES * sizeof(cl_uint));

            // Calculate 256 histogram for all element
            for(int i = 0; i < elementCount; ++i)
            {
                cl_uint element = tempData[i];
                cl_uint value = (element >> bits) & RADIX_MASK;
                histogram[value]++;
            }

            // Prescan the histogram bucket
            cl_uint sum = 0;
            for(int i = 0; i < RADICES; ++i)
            {
                cl_uint val = histogram[i];
                histogram[i] = sum;
                sum += val;
            }

            // Rearrange  the elements based on prescaned histogram
            for(int i = 0; i < elementCount; ++i)
            {
                cl_uint element = tempData[i];
                cl_uint value = (element >> bits) & RADIX_MASK;
                cl_uint index = histogram[value];
                hSortedData[index] = tempData[i];
                histogram[value] = index + 1;
            }

            // Copy to tempData for further use
            if(bits != RADIX * 3)
            {
                memcpy(tempData, hSortedData, elementCount * sizeof(cl_uint));
            }
        }
    }

    FREE(tempData);
    FREE(histogram);
    return SDK_SUCCESS;
}

template<typename T>
int RadixSort::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
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
RadixSort::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
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
RadixSort::setupRadixSort()
{
    /*
     * Map cl_mem origUnsortedDataBuf to host for writing
     * Note the usage of CL_MAP_WRITE_INVALIDATE_REGION flag
     * This flag indicates the runtime that whole buffer is mapped for writing and
     * there is no need of device->host transfer. Hence map call will be faster
     */
    int status = mapBuffer( origUnsortedDataBuf, unsortedData,
                            (elementCount * sizeof(cl_uint)),
                            CL_MAP_WRITE_INVALIDATE_REGION );
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(origUnsortedDataBuf)");

    for(int i=0; i < elementCount; i++)
    {
        unsortedData[i] = (cl_uint)rand();
    }

    /* Unmaps cl_mem origUnsortedDataBuf from host
     * host->device transfer happens if device exists in different address-space
     */
    status = unmapBuffer(origUnsortedDataBuf, unsortedData);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(origUnsortedDataBuf)");

    hSortedData = (cl_uint*)malloc(elementCount * sizeof(cl_uint));
    CHECK_ALLOCATION(hSortedData, "Failed to allocate host memory. (hSortedData)");

    return SDK_SUCCESS;
}

int
RadixSort::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("RadixSort_Kernels.cl");
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
RadixSort::setupCL(void)
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
    CHECK_OPENCL_ERROR( status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
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
        CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUE failed.");
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Check particular extension
    if(!strstr(deviceInfo.extensions, "cl_khr_byte_addressable_store"))
    {
        byteRWSupport = false;
        OPENCL_EXPECTED_ERROR("Device does not support cl_khr_byte_addressable_store extension!");
    }

    if (RADICES > deviceInfo.maxWorkItemSizes[0])
    {
        OPENCL_EXPECTED_ERROR("Device does not support requested number of work items.");
    }

    // Input buffer

    origUnsortedDataBuf = CECL_BUFFER(
                              context,
                              CL_MEM_READ_ONLY,
                              sizeof(cl_uint) * elementCount,
                              this->unsortedData,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (origUnsortedDataBuf)");

    partiallySortedBuf = CECL_BUFFER(
                             context,
                             CL_MEM_READ_ONLY,
                             sizeof(cl_uint) * elementCount,
                             NULL,
                             &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (partiallySortedBuf)");


    // Output for histogram kernel
    histogramBinsBuf = CECL_BUFFER(
                           context,
                           CL_MEM_READ_WRITE,
                           numGroups * groupSize * RADICES * sizeof(cl_uint),
                           NULL,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (histogramBinsBuf)");

    // Input for permute kernel
    scanedHistogramBinsBuf = CECL_BUFFER(
                                 context,
                                 CL_MEM_READ_WRITE,
                                 numGroups * groupSize * RADICES * sizeof(cl_uint),
                                 NULL,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (scanedHistogramBinsBuf)");

    // Final output
    sortedDataBuf = CECL_BUFFER(
                        context,
                        CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY,
                        elementCount * sizeof(cl_uint),
                        NULL,
                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (sortedDataBuf)");

    //add four buffers
    sumBufferin = CECL_BUFFER(
                      context,
                      CL_MEM_READ_WRITE,
                      (elementCount/GROUP_SIZE)*sizeof(cl_uint),
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (sumBufferin)");

    sumBufferout = CECL_BUFFER(
                       context,
                       CL_MEM_READ_WRITE,
                       (elementCount/GROUP_SIZE)*sizeof(cl_uint),
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (sumBufferout)");

    summaryBUfferin = CECL_BUFFER(
                          context,
                          CL_MEM_READ_WRITE,
                          RADICES*sizeof(cl_uint),
                          NULL,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (summaryBUfferin)");

    summaryBUfferout = CECL_BUFFER(
                           context,
                           CL_MEM_READ_WRITE,
                           RADICES*sizeof(cl_uint),
                           NULL,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (summaryBUfferout)");


    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("RadixSort_Kernels.cl");
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
    histogramKernel = CECL_KERNEL(program, "histogram", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    status = kernelInfoHistogram.setKernelWorkGroupInfo(histogramKernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,
                "kernelInfoHistogram.setKernelWorkGroupInfo() failed");

    permuteKernel = CECL_KERNEL(program, "permute", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    status = kernelInfoPermute.setKernelWorkGroupInfo(permuteKernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,
                "kernelInfoPermute.setKernelWorkGroupInfo() failed");

    //add five kernels
    scanArrayKerneldim1 = CECL_KERNEL(program,"ScanArraysdim1",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    scanArrayKerneldim2 = CECL_KERNEL(program,"ScanArraysdim2",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    prefixSumKernel = CECL_KERNEL(program,"prefixSum",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    blockAdditionKernel = CECL_KERNEL(program,"blockAddition",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    FixOffsetkernel = CECL_KERNEL(program,"FixOffset",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");



    // Find minimum of all kernel's work-group sizes
    size_t temp = min(kernelInfoHistogram.kernelWorkGroupSize,
                      kernelInfoPermute.kernelWorkGroupSize);

    // If groupSize exceeds the minimum
    if((size_t)groupSize > temp)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << temp << std::endl;
            std::cout << "Falling back to " << temp << std::endl;
        }
        groupSize = (cl_int)temp;
    }

    return SDK_SUCCESS;
}

int
RadixSort::runHistogramKernel(int bits)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    cl_event ndrEvt;

    size_t globalThreads = elementCount;
    size_t localThreads = 256;

    if(localThreads > deviceInfo.maxWorkItemSizes[0] ||
            localThreads > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not"
                  "support requested number of work items.";
        return SDK_FAILURE;
    }


    // Setup kernel arguments
    status = CECL_SET_KERNEL_ARG(histogramKernel,
                            0,
                            sizeof(cl_mem),
                            (void*) ((firstLoopIter)? &origUnsortedDataBuf: &partiallySortedBuf) );
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (partiallySortedBuf)");

    status = CECL_SET_KERNEL_ARG(histogramKernel,
                            1,
                            sizeof(cl_mem),
                            (void *)&histogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (histogramBinsBuf)");

    status = CECL_SET_KERNEL_ARG(histogramKernel,
                            2,
                            sizeof(cl_int),
                            (void *)&bits);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (bits)");

    status = CECL_SET_KERNEL_ARG(histogramKernel,
                            3,
                            (256 * 1 * sizeof(cl_uint)),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (local memory)");

    if(kernelInfoHistogram.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient"
                  "local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    /*
    * Enqueue a kernel run call.
    */
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 histogramKernel,
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
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");



    return SDK_SUCCESS;
}


int
RadixSort::runPermuteKernel(int bits)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    cl_event ndrEvt;

    size_t bufferSize = numGroups * groupSize * RADICES * sizeof(cl_uint);

    size_t globalThreads = elementCount/RADICES;
    size_t localThreads = groupSize;

    if(localThreads > deviceInfo.maxWorkItemSizes[0] ||
            localThreads > deviceInfo.maxWorkGroupSize)
    {
        std::cout<<"Unsupported: Device does not"
                 "support requested number of work items.";
        return SDK_FAILURE;
    }

    // Whether sort is to be in increasing order. CL_TRUE implies increasing
    status = CECL_SET_KERNEL_ARG(permuteKernel,
                            0,
                            sizeof(cl_mem),
                            (void*) ((firstLoopIter)? &origUnsortedDataBuf: &partiallySortedBuf) );
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (partiallySortedBuf)");

    status = CECL_SET_KERNEL_ARG(permuteKernel,
                            1,
                            sizeof(cl_mem),
                            (void *)&scanedHistogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (scanedHistogramBinsBuf)");

    status = CECL_SET_KERNEL_ARG(permuteKernel,
                            2,
                            sizeof(cl_int),
                            (void *)&bits);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (bits)");

    status = CECL_SET_KERNEL_ARG(permuteKernel,
                            3,
                            (groupSize * RADICES * sizeof(cl_ushort)),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (local memory)");

    status = CECL_SET_KERNEL_ARG(permuteKernel,
                            4,
                            sizeof(cl_mem),
                            (void *)&sortedDataBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sortedDataBuf)");

    if(kernelInfoPermute.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient"
                  "local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    /*
    * Enqueue a kernel run call.
    */
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 permuteKernel,
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
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    // Enqueue the results to application pointe


    cl_event copyEvt;
    status = clEnqueueCopyBuffer(
                 commandQueue,
                 sortedDataBuf,
                 partiallySortedBuf,
                 0,
                 0,
                 elementCount * sizeof(cl_uint),
                 0,
                 NULL,
                 &copyEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&copyEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(readEvt) Failed");

    return SDK_SUCCESS;
}



int RadixSort::runFixOffsetKernel()
{

    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    cl_event ndrEvt,ndrEvt1,ndrEvt2,ndrEvt3,ndrEvt4;

    size_t numGroups = elementCount/256;
    size_t globalThreadsScan[2] = {numGroups,RADICES};
    size_t localThreadsScan[2] = {GROUP_SIZE,1};

    /* There are five kernels to be run ,but when numGroups/GROUP_SIZE is equal to 1,there are only 3 kernels
    to be run*/

    /*The first kernel: we can issue the scanedHistogramBinsBuf buffer is a 2-dimention buffer,and the range
    of the 2nd dimention is 0-255,we need scan this buffer in a workgroup .*/
    cl_uint arg = 0;

    // Setup kernel arguments
    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&scanedHistogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (scanedHistogramBinsBuf)");


    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&histogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (histogramBinsBuf)");

    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            GROUP_SIZE * sizeof(cl_uint),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (localBUffer)");
    cl_uint groupS = GROUP_SIZE;

    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            sizeof(cl_uint),
                            (void *)&groupS);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (GROUP_SIZE)");

    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            sizeof(cl_uint),
                            (void *)&numGroups);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (GROUP_SIZE)");


    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim2,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&sumBufferin);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sumBufferin)");
    /*
    * Enqueue a kernel run call.
    */

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 scanArrayKerneldim2,
                 2,
                 NULL,
                 globalThreadsScan,
                 localThreadsScan,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    /*If there is only one workgroup in the 1st dimention we needn't run the prefix kernel
      and the blockadition kernel*/
    if(numGroups/GROUP_SIZE != 1)
    {
        /*run prefix kernel:if there are more than workgroups in the 1st dimention , do an accumulation
         of the each group summary.*/
        size_t globalThredsPrefixSum[2] = {numGroups/GROUP_SIZE,RADICES};

        arg = 0;

        // Setup kernel arguments
        status = CECL_SET_KERNEL_ARG(prefixSumKernel,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&sumBufferout);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sumBufferout)");

        status = CECL_SET_KERNEL_ARG(prefixSumKernel,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&sumBufferin);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sumBufferout)");

        status = CECL_SET_KERNEL_ARG(prefixSumKernel,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&summaryBUfferin);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferin)");

        cl_uint pstride = (cl_uint)numGroups/GROUP_SIZE;

        status = CECL_SET_KERNEL_ARG(prefixSumKernel,
                                arg++,
                                sizeof(cl_uint),
                                (void *)&pstride);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferin)");


        /*
        * Enqueue a kernel run call.
        */
        status = CECL_ND_RANGE_KERNEL(
                     commandQueue,
                     prefixSumKernel,
                     2,
                     NULL,
                     globalThredsPrefixSum,
                     NULL,
                     0,
                     NULL,
                     &ndrEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");

        status = waitForEventAndRelease(&ndrEvt1);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

        /*run blockAddition kernel: for each element of the current group adds the accumulation of the previous
         group coputed by the prefix kernel*/
        size_t globalThreadsAdd[2] = {numGroups,RADICES};
        size_t localThreadsAdd[2] = {GROUP_SIZE,1};

        arg = 0;

        // Setup kernel arguments
        status = CECL_SET_KERNEL_ARG(blockAdditionKernel,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&sumBufferout);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sumBufferout)");

        status = CECL_SET_KERNEL_ARG(blockAdditionKernel,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&scanedHistogramBinsBuf);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (sumBufferout)");

        status = CECL_SET_KERNEL_ARG(blockAdditionKernel,
                                arg++,
                                sizeof(cl_uint),
                                (void *)&pstride);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferin)");

        /*
        * Enqueue a kernel run call.
        */
        status = CECL_ND_RANGE_KERNEL(
                     commandQueue,
                     blockAdditionKernel,
                     2,
                     NULL,
                     globalThreadsAdd,
                     localThreadsAdd,
                     0,
                     NULL,
                     &ndrEvt2);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");

        status = waitForEventAndRelease(&ndrEvt2);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");
    }

    /*run ScanArraysdim1 kernel:now we have 256 values which are the summary of each row.
      we shoule do a accumulation of this 256 value*/
    size_t globalThreadsScan1[1] = {RADICES};
    size_t localThreadsScan1[1] = {RADICES};

    arg = 0;

    // Setup kernel arguments
    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim1,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&summaryBUfferout);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferout)");

    if(numGroups/GROUP_SIZE != 1)
    {
        status = CECL_SET_KERNEL_ARG(scanArrayKerneldim1,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&summaryBUfferin);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferin)");

    }
    else
    {
        status = CECL_SET_KERNEL_ARG(scanArrayKerneldim1,
                                arg++,
                                sizeof(cl_mem),
                                (void *)&sumBufferin);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferin)");
    }


    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim1,
                            arg++,
                            RADICES * sizeof(cl_uint),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (localmemory)");
    groupS = RADICES;

    status = CECL_SET_KERNEL_ARG(scanArrayKerneldim1,
                            arg++,
                            sizeof(cl_uint),
                            (void *)&groupS);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (block_size)");

    /*
    * Enqueue a kernel run call.
    */
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 scanArrayKerneldim1,
                 1,
                 NULL,
                 globalThreadsScan1,
                 localThreadsScan1,
                 0,
                 NULL,
                 &ndrEvt3);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt3);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");


    /*run fixoffset kernel: for each row of the 2nd dimention ,add the summary of the previous
    row computed by the ScanArraysdim1 kenel ,(the 1st row needn't be changed!), now the correct
    position has been computed out*/
    size_t globalThreadsFixOffset[2]= {numGroups,RADICES};

    arg = 0;

    // Setup kernel arguments
    status = CECL_SET_KERNEL_ARG(FixOffsetkernel,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&summaryBUfferout);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (summaryBUfferout)");

    status = CECL_SET_KERNEL_ARG(FixOffsetkernel,
                            arg++,
                            sizeof(cl_mem),
                            (void *)&scanedHistogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (scanedHistogramBinsBuf)");


    /*
    * Enqueue a kernel run call.
    */
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 FixOffsetkernel,
                 2,
                 NULL,
                 globalThreadsFixOffset,
                 NULL,
                 0,
                 NULL,
                 &ndrEvt4);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt4);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    return SDK_SUCCESS;

}



int
RadixSort::runCLKernels(void)
{
    firstLoopIter = true;
    for(int bits = 0; bits < sizeof(cl_uint) * RADIX; bits += RADIX)
    {
        // Calculate thread-histograms
        runHistogramKernel(bits);

        // Scan the histogram
        runFixOffsetKernel();

        // Permute the element to appropriate place
        runPermuteKernel(bits);

        // Current output now becomes the next input. Buffer was copied in runPermuteKernel()
        firstLoopIter = false;
    }

    return SDK_SUCCESS;
}

int
RadixSort::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Now add customized options
    Option* array_length = new Option;
    if(!array_length)
    {
        std::cout << "Memory allocation error.\n";
        return SDK_FAILURE;
    }

    array_length->_sVersion = "x";
    array_length->_lVersion = "count";
    array_length->_description = "Element count";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &elementCount;
    sampleArgs->AddOption(array_length);
    delete array_length;


    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        std::cout << "Memory Allocation error.\n";
        return SDK_FAILURE;
    }
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    return SDK_SUCCESS;
}

int RadixSort::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        return SDK_FAILURE;
    }

    groupSize = GROUP_SIZE;
    elementCount = roundToPowerOf2<cl_uint>(elementCount);

    //element count must be multiple of GROUP_SIZE * RADICES
    int mulFactor = groupSize * RADICES;

    if(elementCount < mulFactor)
    {
        elementCount = mulFactor;
    }
    else
    {
        elementCount = (elementCount / mulFactor) * mulFactor;
    }

    if(elementCount > 65536)
    {

        printf("\nBecause the local memory is limited to 32K,So the input elementCount is limited to 65536!\n");
        elementCount = 65536;
    }

    numGroups = elementCount / mulFactor;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = setupRadixSort();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample SetUp Resources Failed");

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int RadixSort::run()
{
    int status = 0;
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " <<
              iterations<<" iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    totalKernelTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}

int
RadixSort::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        /*
         * Map cl_mem origUnsortedDataBuf to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        int status = mapBuffer( origUnsortedDataBuf, unsortedData,
                                (elementCount * sizeof(cl_uint)),
                                CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to map device buffer.(origUnsortedDataBuf)");


        /* Rreference implementation on host device
        * Sorted by radix sort
        */
        status = hostRadixSort();
        CHECK_ERROR(status, SDK_SUCCESS, "Host Implementation Failed");

        /*
         * Unmap cl_mem origUnsortedDataBuf from host
         * there will be no data-transfers since cl_mem inputBuffer was mapped for reading
         */
        status = unmapBuffer(origUnsortedDataBuf, unsortedData);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(origUnsortedDataBuf)");

        /*
         * Map cl_mem sortedDataBuf to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        status = mapBuffer( sortedDataBuf, dSortedData,
                            (elementCount * sizeof(cl_uint)),
                            CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(sortedDataBuf)");

        // compare the results and see if they match
        bool result = true;
        int failedCount = 0;
        for(int i = 0; i < elementCount; ++i)
        {
            if(dSortedData[i] != hSortedData[i])
            {
                result = false;
                failedCount++;
                printf("Element(%d) -  %u = %u \n",
                       i, dSortedData[i], hSortedData[i]);
            }
        }

        if(!sampleArgs->quiet)
        {
            printArray<cl_uint>("dSortedData", dSortedData, elementCount, 1);
        }

        /*
         * Unmap cl_mem sortedDataBuf from host
         * there will be no data-transfers since cl_mem outputBuffer was mapped for reading
         */
        status = unmapBuffer(sortedDataBuf, dSortedData);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(outputBuffer)");

        if(result)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout <<" Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}

void
RadixSort::printStats()
{
    if(sampleArgs->timing)
    {
        if(!byteRWSupport)
        {
            return;
        }

        std::string strArray[4] = {"Elements", "Setup time (sec)", "Avg. kernel time (sec)", "Elements/sec"};
        std::string stats[4];

        cl_double avgTime =  (totalKernelTime / iterations);

        stats[0]  = toString(elementCount, std::dec);
        stats[1]  = toString(setupTime, std::dec);
        stats[2]  = toString(avgTime, std::dec);
        stats[3]  = toString((elementCount / avgTime), std::dec);

        printStatistics(strArray, stats, 4);
    }
}


int
RadixSort::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(origUnsortedDataBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(origUnsortedDataBuf)");

    status = clReleaseMemObject(partiallySortedBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(partiallySortedBuf)");

    status = clReleaseMemObject(histogramBinsBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(histogramBinsBuf)");

    status = clReleaseMemObject(scanedHistogramBinsBuf);
    CHECK_OPENCL_ERROR(status,
                       "clReleaseMemObject failed.(scanedHistogramBinsBuf)");

    status = clReleaseMemObject(sortedDataBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(sortedDataBuf)");


    status = clReleaseMemObject(sumBufferin);
    CHECK_OPENCL_ERROR(status,
                       "clReleaseMemObject failed.(scanedHistogramBinsBuf)");

    status = clReleaseMemObject(sumBufferout);
    CHECK_OPENCL_ERROR(status,
                       "clReleaseMemObject failed.(scanedHistogramBinsBuf)");

    status = clReleaseMemObject(summaryBUfferin);
    CHECK_OPENCL_ERROR(status,
                       "clReleaseMemObject failed.(scanedHistogramBinsBuf)");

    status = clReleaseMemObject(summaryBUfferout);
    CHECK_OPENCL_ERROR(status,
                       "clReleaseMemObject failed.(scanedHistogramBinsBuf)");

    status = clReleaseKernel(histogramKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(histogramKernel)");

    status = clReleaseKernel(permuteKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");

    status = clReleaseKernel(scanArrayKerneldim1);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");

    status = clReleaseKernel(scanArrayKerneldim2);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");

    status = clReleaseKernel(prefixSumKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");

    status = clReleaseKernel(blockAdditionKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");

    status = clReleaseKernel(FixOffsetkernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(permuteKernel)");


    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // Release program resources (input memory etc.)
    FREE(hSortedData);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    cl_int status = 0;
    RadixSort clRadixSort;

    if (clRadixSort.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRadixSort.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clRadixSort.sampleArgs->isDumpBinaryEnabled())
    {
        return clRadixSort.genBinaryImage();
    }

    status = clRadixSort.setup();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    if (clRadixSort.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRadixSort.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRadixSort.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clRadixSort.printStats();

    return SDK_SUCCESS;
}
