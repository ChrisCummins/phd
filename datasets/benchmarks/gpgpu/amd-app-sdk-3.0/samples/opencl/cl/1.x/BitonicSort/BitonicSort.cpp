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


#include "BitonicSort.hpp"

int BitonicSort::setupBitonicSort()
{
    cl_uint inputSizeBytes;

    // allocate and init memory used by host
    inputSizeBytes = length * sizeof(cl_uint);

    int status = mapBuffer( inputBuffer, input, inputSizeBytes,
                            CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    // random initialisation of input
    fillRandom<cl_uint>(input, length, 1, 0, 255);

    if(sampleArgs->verify)
    {
        verificationInput = (cl_uint *) malloc(length * sizeof(cl_int));
        CHECK_ALLOCATION(verificationInput,
                         "Failed to allocate host memory. (verificationInput)");
        memcpy(verificationInput, input, length * sizeof(cl_int));
    }

    /*
     * Unless sampleArgs->quiet mode has been enabled, print the INPUT array.
     * No more than 256 values are printed because it clutters the screen
     * and it is not practical to manually compare a large set of numbers
     */
    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>(
            "Unsorted Input",
            input,
            length,
            1);
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    return SDK_SUCCESS;
}


template<typename T>
int BitonicSort::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
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
BitonicSort::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
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

/**
 * genBinary Image function is used to when we want to create the binary
 * for the kernel and run it at a later time. This is useful where offline
 * compilation is the preferred mechanism.
 */
int
BitonicSort::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("BitonicSort_Kernels.cl");
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
BitonicSort::setupCL(void)
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
                               sampleArgs-> isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

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

    retValue = getDevices(context, &devices, sampleArgs->deviceId,
                          sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");
    }

    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_uint) * length,
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("BitonicSort_Kernels.cl");
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
    kernel = CECL_KERNEL(program, "bitonicSort", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}


int
BitonicSort::runCLKernels(void)
{
    cl_int status;
    cl_uint numStages = 0;
    cl_uint temp;

    cl_uint stage;
    cl_uint passOfStage;

    size_t globalThreads[1] = {length / 2};
    size_t localThreads[1] = {GROUP_SIZE};

    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, 0, "setKernelWrkGroupInfo failed");

    if((cl_uint)(localThreads[0]) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << localThreads[0] << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Changing the group size to " << kernelInfo.kernelWorkGroupSize 
                      << "\n";
        }
        localThreads[0] = kernelInfo.kernelWorkGroupSize;
    }

    /*
     * This algorithm is run as NS stages. Each stage has NP passes.
     * so the total number of times the kernel call is enqueued is NS * NP.
     *
     * For every stage S, we have S + 1 passes.
     * eg: For stage S = 0, we have 1 pass.
     *     For stage S = 1, we have 2 passes.
     *
     * if length is 2^N, then the number of stages (numStages) is N.
     * Do keep in mind the fact that the algorithm only works for
     * arrays whose size is a power of 2.
     *
     * here, numStages is N.
     *
     * For an explanation of how the algorithm works, please go through
     * the documentation of this sample.
     */

    /*
     * 2^numStages should be equal to length.
     * i.e the number of times you halve length to get 1 should be numStages
     */
    for(temp = length; temp > 1; temp >>= 1)
    {
        ++numStages;
    }

    // Set appropriate arguments to the kernel

    // the input array - also acts as output for this pass (input for next)
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    // whether sort is to be in increasing order. CL_TRUE implies increasing
    if(sortOrder.compare("asc")==0)
    {
        sortFlag = 1;
    }
    else if(sortOrder.compare("desc")==0)
    {
        sortFlag = 0;
    }
    else
    {
        std::cout << "Please input asc or desc,the default sort order is desc!" <<
                  std::endl;
        sortFlag = 0;
    }

    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 3,
                 sizeof(cl_uint),
                 (void *)&sortFlag);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (increasing)");

    for(stage = 0; stage < numStages; ++stage)
    {
        // stage of the algorithm
        status = CECL_SET_KERNEL_ARG(
                     kernel,
                     1,
                     sizeof(cl_uint),
                     (void *)&stage);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (stage)");

        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {
            // pass of the current stage
            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         2,
                         sizeof(cl_uint),
                         (void *)&passOfStage);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (passOfStage)");

            /*
             * Enqueue a kernel run call.
             * For simplicity, the groupsize used is 1.
             *
             * Each thread writes a sorted pair.
             * So, the number of  threads (global) is half the length.
             */
            cl_event ndrEvt;
            status = CECL_ND_RANGE_KERNEL(
                         commandQueue,
                         kernel,
                         1,
                         NULL,
                         globalThreads,
                         localThreads,
                         0,
                         NULL,
                         &ndrEvt);

            CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush failed.");

            status = waitForEventAndRelease(&ndrEvt);
            CHECK_ERROR(status, 0, "WaitForEventAndRelease(ndrEvt) Failed");
        }
    }
    return SDK_SUCCESS;
}

void
BitonicSort::swapIfFirstIsGreater(cl_uint *a, cl_uint *b)
{
    if(*a > *b)
    {
        cl_uint temp = *a;
        *a = *b;
        *b = temp;
    }
}


/*
 * sorts the input array (in place) using the bitonic sort algorithm
 * sorts in increasing order if sortIncreasing is CL_TRUE
 * else sorts in decreasing order
 * length specifies the length of the array
 */
void
BitonicSort::bitonicSortCPUReference(
    cl_uint * input,
    const cl_uint length,
    const cl_bool sortIncreasing)
{
    const cl_uint halfLength = length/2;

    cl_uint i;
    for(i = 2; i <= length; i *= 2)
    {
        cl_uint j;
        for(j = i; j > 1; j /= 2)
        {
            cl_bool increasing = sortIncreasing;
            const cl_uint half_j = j/2;

            cl_uint k;
            for(k = 0; k < length; k += j)
            {
                const cl_uint k_plus_half_j = k + half_j;
                cl_uint l;

                if(i < length)
                {
                    if((k == i) || ((k % i) == 0) && (k != halfLength))
                    {
                        increasing = !increasing;
                    }
                }

                for(l = k; l < k_plus_half_j; ++l)
                {
                    if(increasing)
                    {
                        swapIfFirstIsGreater(&input[l], &input[l + half_j]);
                    }
                    else
                    {
                        swapIfFirstIsGreater(&input[l + half_j], &input[l]);
                    }
                }
            }
        }
    }
}

int BitonicSort::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initilization failed");

    // Now add customized options
    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error.\n");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the array to be sorted";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);

    delete array_length;

    Option* sort_order = new Option;
    CHECK_ALLOCATION(sort_order, "Memory allocation error.\n");

    sort_order->_sVersion = "s";
    sort_order->_lVersion = "sort";
    sort_order->_description = "Sort in descending/ascending order[desc/asc]";
    sort_order->_type = CA_ARG_STRING;
    sort_order->_value = &sortOrder;
    sampleArgs->AddOption(sort_order);

    delete sort_order;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);

    delete num_iterations;

    return SDK_SUCCESS;
}

int BitonicSort::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int i = length & (length - 1);
    if(i != 0)
    {
        std::cout<<"\nThe input lentgh must be a power of 2\n"<<std::endl;
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupCL())
    {
        return SDK_FAILURE;
    }

    if(setupBitonicSort() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);

    setupTime = (cl_double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int BitonicSort::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels())
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
        if(runCLKernels())
        {
            return SDK_FAILURE;
        }
    }
    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer));


    if(!sampleArgs->quiet)
    {
        int status = mapBuffer( inputBuffer, input, length, CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to map device buffer.(inputBuffer in run())");

        printArray<cl_uint>("Output", input, length, 1);

        status = unmapBuffer( inputBuffer, input);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(inputBuffer in run())");
    }

    return SDK_SUCCESS;
}

int BitonicSort::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
          * reference implementation
          * it overwrites the input array with the output
          */
        int refTimer = sampleTimer->createTimer();
        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        bitonicSortCPUReference(verificationInput, length, sortFlag);
        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        int status = mapBuffer( inputBuffer, input, length * sizeof(cl_uint),
                                CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to map device buffer.(inputBuffer in verifyResults())");

        // compare the results and see if they match
        if(memcmp(input, verificationInput, length*sizeof(cl_uint)) == 0)
        {
            std::cout<<"Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }
        status = unmapBuffer( inputBuffer, input );
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(inputBuffer in verifyResults())");

    }
    return SDK_SUCCESS;
}

void BitonicSort::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Setup Time (sec)", "Avg. Kernel Time (sec)", "Elements/sec"};
        std::string stats[4];

        sampleTimer->totalTime = ( totalKernelTime/ iterations );

        stats[0]  = toString(length, std::dec);
        stats[1]  = toString(setupTime, std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(( length/sampleTimer->totalTime ), std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int BitonicSort::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)
    FREE(verificationInput);

    FREE(devices);

    return SDK_SUCCESS;
}



int
main(int argc, char * argv[])
{
    BitonicSort clBitonicSort;

    if(clBitonicSort.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBitonicSort.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBitonicSort.sampleArgs->isDumpBinaryEnabled() != SDK_SUCCESS)
    {
        return clBitonicSort.genBinaryImage();
    }
    else
    {
        if(clBitonicSort.setup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBitonicSort.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBitonicSort.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBitonicSort.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clBitonicSort.printStats();
    }
    return SDK_SUCCESS;
}
