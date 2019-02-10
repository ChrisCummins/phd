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


#include "HistogramAtomics.hpp"

#include <math.h>

int
Histogram::calculateHostBin()
{
    // compute CPU histogram

    int status = mapBuffer( inputBuffer, input, inputNBytes, CL_MAP_READ);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    cl_int *p = (cl_int*)input;

    memset(cpuhist, 0, sizeof(cl_uint) * NBINS);

    for(unsigned int i = 0; i < inputNBytes / sizeof(cl_uint); i++)
    {
        cpuhist[ (p[i] >> 24) & 0xff ]++;
        cpuhist[ (p[i] >> 16) & 0xff ]++;
        cpuhist[ (p[i] >> 8) & 0xff ]++;
        cpuhist[ (p[i] >> 0) & 0xff ]++;
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

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
    cl_int status =     0;

    // random initialization of input
    time_t ltime;
    time(&ltime);
    cl_uint a = (cl_uint) ltime, b = (cl_uint) ltime;

    status = mapBuffer( inputBuffer, input, inputNBytes,
                        CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    cl_uint *p = (cl_uint *) input;

    for(unsigned int i = 0; i < inputNBytes / sizeof(cl_uint); i++)
    {
        p[i] = ( b = ( a * ( b & 65535 )) + ( b >> 16 ));
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    return SDK_SUCCESS;
}

int
Histogram::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("HistogramAtomics_Kernels.cl");
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
    CHECK_ERROR(retValue, 0, "SDKDeviceInfo::setDeviceInfo() failed");

    // If both options were ambiguously specified, use default vector-width for the device
    int vectorWidth = 0;
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

    nThreads =          64 * 1024 * ((vectorWidth == 1) ? VECTOR_SIZE : 1);
    nVectors =          2048 * 2048 * ((vectorWidth == 1) ? VECTOR_SIZE : 1);
    nVectorsPerThread = nVectors / nThreads;
    inputNBytes =       nVectors * ((vectorWidth == 1) ? sizeof(cl_uint) : sizeof(
                                        cl_uint4));

    // Check if byte-addressable store is supported
    if(!strstr(deviceInfo.extensions, "cl_khr_local_int32_base_atomics"))
    {
        reqdExtSupport = false;
        OPENCL_EXPECTED_ERROR("Device does not support local_int32_base_atomics extension!");
    }

    // Create input buffer
    inputBuffer = CECL_BUFFER(context,
                                 CL_MEM_READ_ONLY,
                                 inputNBytes,
                                 NULL,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("HistogramAtomics_Kernels.cl");
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
    if (vectorWidth == 1)
    {
        histogram = CECL_KERNEL(program, "histogramKernel_Scalar", &status);
    }
    else
    {
        histogram = CECL_KERNEL(program, "histogramKernel_Vector", &status);
    }
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    // get a kernel object handle for a kernel with the given name
    reduce = CECL_KERNEL(program, "reduceKernel", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    status =  kernelInfoReduce.setKernelWorkGroupInfo(reduce,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    status =  kernelInfoHistogram.setKernelWorkGroupInfo(histogram,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    // histogramKernel work group size must be integer multiple of 256
    if(kernelInfoHistogram.kernelWorkGroupSize % 256 != 0)
    {
        OPENCL_EXPECTED_ERROR("Device does not support work-group size of 256 on histogram kernel. Exiting!");
    }

	nThreadsPerGroup =   (cl_uint) kernelInfoHistogram.kernelWorkGroupSize;
    nGroups =            nThreads / nThreadsPerGroup;
    outputNBytes =       nGroups * NBINS * sizeof(cl_uint);

    // Create output Buffer
    outputBuffer = CECL_BUFFER( context,
                                   CL_MEM_READ_WRITE,
                                   outputNBytes,
                                   0,
                                   &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    return SDK_SUCCESS;
}

int
Histogram::runCLKernels(void)
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;
    size_t globalThreads[3] = {1};
    size_t localThreads[3] = {1};
    size_t globalThreadsReduce = NBINS;
    size_t localThreadsReduce = min((int)nThreadsPerGroup, NBINS);

    globalThreads[0] = nThreads;
    localThreads[0]  = nThreadsPerGroup;

    int Arg = 0;

    // __global input & output
    status = CECL_SET_KERNEL_ARG(histogram,
                            Arg++,
                            sizeof(cl_mem),
                            (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    status |= CECL_SET_KERNEL_ARG(histogram,
                             Arg++,
                             sizeof(cl_mem),
                             (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    status |= CECL_SET_KERNEL_ARG(histogram,
                             Arg++,
                             sizeof(nVectorsPerThread),
                             (void *)&nVectorsPerThread);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (nVectorsPerThread)");

    // reduceKernel
    Arg = 0;
    status |= CECL_SET_KERNEL_ARG(reduce,
                             Arg++,
                             sizeof(cl_mem),
                             (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    status |= CECL_SET_KERNEL_ARG(reduce,
                             Arg++,
                             sizeof(nGroups),
                             (void *)&nGroups);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (nGroups)");

    /*
    * Enqueue a kernel run call.
    */
    cl_event ndrEvt1;
    cl_event ndrEvt2;

    status = CECL_ND_RANGE_KERNEL(commandQueue,
                                    histogram,
                                    1,
                                    NULL,
                                    globalThreads,
                                    localThreads,
                                    0,
                                    NULL,
                                    &ndrEvt1);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed. (histogram)");

	status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = CECL_ND_RANGE_KERNEL(commandQueue,
                                    reduce,
                                    1,
                                    NULL,
                                    &globalThreadsReduce,
                                    &localThreadsReduce,
                                    1,
                                    &ndrEvt1,
                                    &ndrEvt2);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed. (reduce)");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

	status = clReleaseEvent(ndrEvt1);
	CHECK_OPENCL_ERROR(status, "clReleaseEvent ndrEvt1 Failed with Error Code:");

	status = clReleaseEvent(ndrEvt2);
	CHECK_OPENCL_ERROR(status, "clReleaseEvent ndrEvt2 Failed with Error Code:");

    return SDK_SUCCESS;
}

int
Histogram::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* option = new Option;
    CHECK_ALLOCATION(option, "Memory allocation error.\n");

    option->_sVersion = "";
    option->_lVersion = "scalar";
    option->_description =
        "Run scalar version of the kernel(--scalar and --vector options are mutually exclusive)";
    option->_type = CA_NO_ARGUMENT;
    option->_value = &useScalarKernel;
    sampleArgs->AddOption(option);

    option->_sVersion = "";
    option->_lVersion = "vector";
    option->_description =
        "Run vector version of the kernel(--scalar and --vector options are mutually exclusive)";
    option->_type = CA_NO_ARGUMENT;
    option->_value = &useVectorKernel;
    sampleArgs->AddOption(option);

    option->_sVersion = "i";
    option->_lVersion = "iterations";
    option->_description = "Number of iterations to execute kernel";
    option->_type = CA_ARG_INT;
    option->_value = &iterations;
    sampleArgs->AddOption(option);

    delete option;
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

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = setupHistogram();
    if(status != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));


    return SDK_SUCCESS;
}

int
Histogram::run()
{
    if(!reqdExtSupport)
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


    std::cout << "Executing kernel for "
              << iterations << " iterations" << std::endl;
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
    kernelTimeGlobal = (double)(sampleTimer->readTimer(timer));

    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    return SDK_SUCCESS;
}

int
Histogram::verifyResults()
{
    int status;
    if(!reqdExtSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        /* reference implementation on host device
         * calculates the histogram bin on host
         */
        calculateHostBin();

        status = mapBuffer( outputBuffer, output, sizeof(cl_uint) * NBINS, CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuff)");

        // compare the results and see if they match
        bool flag = true;
        for(cl_uint i = 0; i < NBINS; ++i)
        {
            if(cpuhist[i] != output[i])
            {
                flag = false;
                break;
            }
        }

        if(!sampleArgs->quiet)
        {
            printArray<cl_uint>("Output", output, NBINS, 1);
        }

        status = unmapBuffer( outputBuffer, output );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuff)");


        if(flag)
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
        if(!reqdExtSupport)
        {
            return;
        }

        // calculate total time
        double avgKernelTime = (kernelTimeGlobal / iterations);

        std::string strArray[4] =
        {
            "Elements",
            "Setup Time (sec)",
            "Avg. kernel time (sec)",
            "Elements/sec"
        };
        std::string stats[4];

        cl_int elem = inputNBytes / sizeof(cl_uint);
        stats[0] = toString(elem, std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString(avgKernelTime, std::dec);
        stats[3] = toString((elem/avgKernelTime), std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int Histogram::cleanup()
{
    if(!reqdExtSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer )");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

    status = clReleaseKernel(histogram);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(histogram)");

    status = clReleaseKernel(reduce);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(reduce)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    // Create HistogramAtomics object
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
    int status = clHistogram.setup();
    if(status != SDK_SUCCESS)
    {
        if(status == SDK_EXPECTED_FAILURE)
        {
            return SDK_SUCCESS;
        }
        else
        {
            return SDK_FAILURE;
        }
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
