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

#include "QuasiRandomSequence.hpp"
#include <malloc.h>


/* Generate direction numbers
   v[j][32] : j dimensions each having 32 direction numbers
*/
void
QuasiRandomSequence::generateDirectionNumbers(cl_uint nDimensions,
        cl_uint* directionNumbers)
{
    cl_uint *v = directionNumbers;

    for (int dim = 0 ; dim < (int)(nDimensions); dim++)
    {
        // First dimension is a special case
        if (dim == 0)
        {
            for (int i = 0 ; i < N_DIRECTIONS ; i++)
            {
                // All m's are 1
                v[i] = 1 << (31 - i);
            }
        }
        else
        {
            int d = sobolPrimitives[dim].degree;
            /* The first direction numbers (up to the degree of the polynomial)
             are simply   v[i] = m[i] / 2^i  (stored in Q0.32 format) */
            for (int i = 0 ; i < d ; i++)
            {
                v[i] = sobolPrimitives[dim].m[i] << (31 - i);
            }

            for (int i = d ; i < N_DIRECTIONS ; i++)
            {

                v[i] = v[i - d] ^ (v[i - d] >> d);
                /*
                Note that the coefficients a[] are zero or one and for compactness in
                 the input tables they are stored as bits of a single integer. To extract
                 the relevant bit we use right shift and mask with 1.
                 For example, for a 10 degree polynomial there are ten useful bits in a,
                 so to get a[2] we need to right shift 7 times (to get the 8th bit into
                 the LSB) and then mask with 1.*/
                for (int j = 1 ; j < d ; j++)
                {
                    v[i] ^= (((sobolPrimitives[dim].a >> (d - 1 - j)) & 1) * v[i - j]);
                }
            }
        }
        v += N_DIRECTIONS;
    }
}


/*
*  Host Initialization
*    Allocate and initialize memory on the host.
*    Print input array.
*/

int
QuasiRandomSequence::setupQuasiRandomSequence()
{
    // Check for dimensions
    if(nDimensions > MAX_DIMENSIONS)
    {
        std::cout << "Max allowed dimension is 10200!\n";
        return SDK_FAILURE;
    }

    /*
     * Map cl_mem inputBuffer to host for writing
     * Note the usage of CL_MAP_WRITE_INVALIDATE_REGION flag
     * This flag indicates the runtime that whole buffer is mapped for writing and
     * there is no need of device->host transfer. Hence map call will be faster
     */
    int status = mapBuffer( inputBuffer, input,
                            (nDimensions * N_DIRECTIONS * sizeof(cl_uint)),
                            CL_MAP_WRITE_INVALIDATE_REGION );
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    // initialize sobol direction numbers
    generateDirectionNumbers(nDimensions, input);

    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>(
            "Input",
            input,
            N_DIRECTIONS,
            nDimensions);
    }

    /* Unmaps cl_mem inputBuffer from host
     * host->device transfer happens if device exists in different address-space
     */
    status = unmapBuffer(inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    // If verification is enabled
    if(sampleArgs->verify)
    {
        // Allocate memory for verification output array
        verificationOutput = (cl_float*)malloc(nVectors * nDimensions * sizeof(
                cl_float));
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verify)");

        memset(verificationOutput,
               0,
               nVectors * nDimensions * sizeof(cl_float));
    }

    return SDK_SUCCESS;
}

int
QuasiRandomSequence::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("QuasiRandomSequence_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

template<typename T>
int QuasiRandomSequence::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
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
QuasiRandomSequence::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
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
QuasiRandomSequence::setupCL(void)
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
        //The block is to move the declaration of prop closer to its use
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
    if(useScalarKernel && useVectorKernel)
    {
        std::cout <<
                  "Ignoring \"--scalar\" & \"--vector\" options. Using default vector-width for the device"
                  << std::endl;
        // Always use vector-width of 1 or 4
        vectorWidth = (deviceInfo.preferredFloatVecWidth == 1)? 1: 4;
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
        // Always use vector-width of 1 or 4
        vectorWidth = (deviceInfo.preferredFloatVecWidth == 1)? 1: 4;
    }

    if(vectorWidth == 1)
    {
        std::cout << "Selecting scalar kernel" << std::endl;
    }
    else
    {
        std::cout << "Selecting vector kernel" << std::endl;
    }

    // Round nVectors to nearest multiple of vectorWidth
    nVectors = (nVectors / vectorWidth)? ((nVectors / vectorWidth) * vectorWidth):
               vectorWidth;

    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY,
                      sizeof(cl_uint) * nDimensions * N_DIRECTIONS,
                      0,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY,
                       sizeof(cl_float) * nVectors * nDimensions,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("QuasiRandomSequence_Kernels.cl");
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
    const char *kernelName = (vectorWidth == 1)? "QuasiRandomSequence_Scalar"
                             : "QuasiRandomSequence_Vector";
    kernel = CECL_KERNEL(program, kernelName, &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "KernelInfo.setKernelWorkGroupInfo() failed");

    if((nVectors/vectorWidth) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified: " << (nVectors/vectorWidth) << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize<<std::endl;
            std::cout << "Falling back to "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
        }
        nVectors = (cl_uint)kernelInfo.kernelWorkGroupSize * vectorWidth;
    }
    return SDK_SUCCESS;
}



int
QuasiRandomSequence::runCLKernels(void)
{
    cl_int   status;

    // set total threads and block size
    size_t globalThreads[1]= {nDimensions * (nVectors / vectorWidth)};
    size_t localThreads[1] = {(nVectors / vectorWidth)};

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[0] > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support"
                  << "requested number of work items." << std::endl;
        return SDK_FAILURE;
    }

    // Set appropriate arguments to the kernel

    // 1st argument to the kernel - outputBuffer
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    // 2nd argument to the kernel - inputBuffer
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 1,
                 sizeof(cl_mem),
                 (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    // 3rd argument to the kernel - localBuffer(shared memory)
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 2,
                 32 * sizeof(cl_uint),
                 NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (localBuffer)");

    availableLocalMemory = deviceInfo.localMemSize - kernelInfo.localMemoryUsed;

    neededLocalMemory = 32 * sizeof(cl_uint);

    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient"
                  "local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    // Enqueue a kernel run call
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernel,
                 1,                              // work_dim
                 NULL,                           // global_work_offset
                 globalThreads,                  // global_work_size
                 localThreads,                   // local_work_size
                 0,                              // num_events in wait list
                 NULL,                           // event_wait_list
                 NULL);                          // event

    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.(commandQueue)");

    return SDK_SUCCESS;
}


int
QuasiRandomSequence::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error.\n");

    array_length->_sVersion = "x";
    array_length->_lVersion = "width";
    array_length->_description = "Number of vectors";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &nVectors;
    sampleArgs->AddOption(array_length);

    array_length->_sVersion = "y";
    array_length->_lVersion = "height";
    array_length->_description = "Number of dimensions";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &nDimensions;
    sampleArgs->AddOption(array_length);

    delete array_length;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory Allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* vs = new Option;
    CHECK_ALLOCATION(vs, "Memory Allocation error.\n");

    vs->_sVersion = "";
    vs->_lVersion = "scalar";
    vs->_description =
        "Run scalar version of the kernel (--scalar and --vector options are mutually exclusive)";
    vs->_type = CA_NO_ARGUMENT;
    vs->_value = &useScalarKernel;
    sampleArgs->AddOption(vs);

    vs->_sVersion = "";
    vs->_lVersion = "vector";
    vs->_description =
        "Run vector version of the kernel (--scalar and --vector options are mutually exclusive)";
    vs->_type = CA_NO_ARGUMENT;
    vs->_value = &useVectorKernel;
    sampleArgs->AddOption(vs);

    delete vs;

    return SDK_SUCCESS;
}

int
QuasiRandomSequence::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if (setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (setupQuasiRandomSequence() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int QuasiRandomSequence::run()
{
    // Warm up
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
    std::cout<<"-------------------------------------------"<<std::endl;

    // create and initialize timers
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


void
QuasiRandomSequence::quasiRandomSequenceCPUReference()
{

    for(int j=0; j < (int)nDimensions; j++)
    {
        for(int i=0; i < (int)nVectors; i++)
        {
            unsigned int temp = 0;
            for(int k=0; k < 32; k++)
            {
                int mask = (int)(pow(2, (double)k));
                temp ^= ((i & mask) >> k) * input[j * 32 + k];
            }

            if(i==0 && j==0)
            {
                verificationOutput[j * nVectors + i] = 0;
            }
            else
            {
                verificationOutput[j * nVectors + i] =
                    (cl_float)(temp / pow(2, (double)32));
            }
        }
    }
}


int
QuasiRandomSequence::verifyResults()
{

    if(sampleArgs->verify)
    {
        /*
         * Map cl_mem inputBuffer to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        int status = mapBuffer( inputBuffer, input,
                                (nDimensions * N_DIRECTIONS * sizeof(cl_uint)),
                                CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

        // Reference implementation
        quasiRandomSequenceCPUReference();

        /*
         * Unmap cl_mem inputBuffer from host
         * there will be no data-transfers since cl_mem inputBuffer was mapped for reading
         */
        status = unmapBuffer(inputBuffer, input);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

        /*
         * Map cl_mem outputBuffer to host for reading
         * device->host transfer happens if device exists in different address-space
         */
        status = mapBuffer( outputBuffer, output,
                            (nVectors * nDimensions * sizeof(cl_float)),
                            CL_MAP_READ );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuffer)");

        // compare the results and see if they match
        bool pass = compare(output, verificationOutput, nDimensions * nVectors);

        if(!sampleArgs->quiet)
        {
            printArray<cl_float>("Output",
                                 output,
                                 nVectors, nDimensions);
        }

        /*
         * Unmap cl_mem outputBuffer from host
         * there will be no data-transfers since cl_mem outputBuffer was mapped for reading
         */
        status = unmapBuffer(outputBuffer, output);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(outputBuffer)");

        if(pass)
        {
            std::cout<<"Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
QuasiRandomSequence::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Setup time (sec)", "Avg. kernel time (sec)", "Elements/sec"};
        std::string stats[4];

        int length = nDimensions * nVectors;
        double avgTime = totalKernelTime / iterations;

        stats[0]  = toString(length, std::dec);
        stats[1]  = toString(setupTime, std::dec);
        stats[2]  = toString(avgTime, std::dec);
        stats[3]  = toString((length / avgTime), std::dec);

        printStatistics(strArray, stats, 4);
    }

}

int
QuasiRandomSequence::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

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

    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

/* Main routine */
int
main(int argc, char * argv[])
{
    QuasiRandomSequence clQuasiRandomSequence;

    if (clQuasiRandomSequence.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clQuasiRandomSequence.sampleArgs->parseCommandLine(argc,
            argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clQuasiRandomSequence.sampleArgs->isDumpBinaryEnabled())
    {
        return clQuasiRandomSequence.genBinaryImage();
    }

    if (clQuasiRandomSequence.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clQuasiRandomSequence.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clQuasiRandomSequence.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clQuasiRandomSequence.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clQuasiRandomSequence.printStats();
    return SDK_SUCCESS;
}
