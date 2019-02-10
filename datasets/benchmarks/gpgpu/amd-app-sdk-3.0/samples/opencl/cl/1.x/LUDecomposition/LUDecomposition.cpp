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


#include "LUDecomposition.hpp"
#define VECTOR_SIZE 4
#define SIZE effectiveDimension * effectiveDimension * sizeof(double)

int LUD::setupLUD()
{
    /*
     *Changing dimension to a multiple of block if it is not.
     *Currently execution will be done on the effective dimention
     *and results will be shown upto what user requests
     */


    if(actualDimension % VECTOR_SIZE != 0)
        effectiveDimension = actualDimension
                             - (actualDimension % VECTOR_SIZE)
                             + VECTOR_SIZE;
    else
    {
        effectiveDimension = actualDimension;
    }

    blockSize = effectiveDimension / VECTOR_SIZE;

#ifdef _WIN32
    input = static_cast<double*>(_aligned_malloc(SIZE, 4096));
#else
    input = static_cast<double*>(memalign(4096, SIZE));
#endif
    CHECK_ALLOCATION(input, "Unable to allocate input memory");

    //initialize with random double type elements
    fillRandom<double>(
        input,
        effectiveDimension,
        effectiveDimension,
        1,
        2,
        1);

#ifdef _WIN32
    matrixGPU = static_cast<double*>(_aligned_malloc(SIZE, 4096));
#else
    matrixGPU = static_cast<double*>(memalign(4096, SIZE));
#endif
    CHECK_ALLOCATION(matrixGPU, "Unable to allocate memory for GPU input");

    if(sampleArgs->verify)
    {
        matrixCPU = static_cast<double*>(malloc(SIZE));
        CHECK_ALLOCATION(matrixCPU,
                         "Unable to allocate memory for refernce implementation");
        memcpy((void*)matrixCPU, (const void*)input, SIZE);
    }

    if(!sampleArgs->quiet)
    {
        printArray<double>(
            "Input Matrix",
            input,
            actualDimension,
            actualDimension);
    }

    return SDK_SUCCESS;
}

int LUD::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("LUDecomposition_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int LUD::setupCL(void)
{
    cl_int status = CL_SUCCESS;
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
#ifdef CL_VERSION_2_0
		cl_queue_properties *props = NULL;
		commandQueue = CECL_CREATE_COMMAND_QUEUEWithProperties(
						   context,
						   devices[sampleArgs->deviceId],
						   props,
						   &status);
		CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUEWithProperties failed.");
#else
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUE failed.");
#endif
    }

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
        // Check if cl_amd_fp64 extension is supported
        if(!strstr(deviceInfo.extensions, "cl_amd_fp64"))
        {
            OPENCL_EXPECTED_ERROR("Device does not support cl_amd_fp64 extension!");
        }
    }

    //Local Memory Required in worst case
    localMemoryNeeded = (cl_ulong)((deviceInfo.maxWorkGroupSize / 2) * sizeof(
                                       cl_double));

    status = setupLUD();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    //Creating Buffers
    inplaceBuffer = CECL_BUFFER(
                        context,
                        CL_MEM_READ_WRITE,
                        sizeof(double) * effectiveDimension * effectiveDimension,
                        NULL,
                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    inputBuffer2 = CECL_BUFFER(
                       context,
                       CL_MEM_READ_WRITE,
                       sizeof(double) * effectiveDimension * effectiveDimension,
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("LUDecomposition_Kernels.cl");
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
    kernelLUD = CECL_KERNEL(program, "kernelLUDecompose", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed. : kernelLUD");

    kernelCombine = CECL_KERNEL(program, "kernelLUCombine", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed. : kernelCombine");

    return SDK_SUCCESS;
}

int LUD::runCLKernels(void)
{
    cl_int   status;
    cl_int eventStatus = CL_QUEUED;
    cl_event inMapEvt;
    cl_event inUnmapEvt;
    cl_event outMapEvt;
    cl_event outUnmapEvt;
    void* inMapPtr = NULL;
    void* outMapPtr = NULL;

    inMapPtr = CECL_MAP_BUFFER(
                   commandQueue,
                   inplaceBuffer,
                   CL_FALSE,
                   CL_MAP_WRITE,
                   0,
                   SIZE,
                   0,
                   NULL,
                   &inMapEvt,
                   &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed. (inplaceBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inMapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt) Failed");

    memcpy(inMapPtr, input, SIZE);
    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inplaceBuffer,
                 inMapPtr,
                 0,
                 NULL,
                 &inUnmapEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inplaceBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");

    status = waitForEventAndRelease(&inUnmapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inUnmapEvt) Failed");

    size_t globalThreads[2]= {blockSize, effectiveDimension};
    size_t localThreads[2] = {blockSize, 1};
    size_t offset[2] = {0, 0};

    /*
     *Querying some important device paramneters
     *Finding out the appropritate Work group size for the kernel
     * See if local memory can be used
     * Check group size against kernelWorkGroupSize
     */

    status =  kernelInfo.setKernelWorkGroupInfo(kernelLUD,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    /*
     *Setting Common Kernel Arguments
     */
    {
        //kernelLUD
        // 1st kernel argument - output
        status = CECL_SET_KERNEL_ARG(
                     kernelLUD,
                     0,
                     sizeof(cl_mem),
                     (void *)&inputBuffer2);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. :kernelLUD(inputBuffer2)");

        // 2nd kernel argument - input
        status = CECL_SET_KERNEL_ARG(
                     kernelLUD,
                     1,
                     sizeof(cl_mem),
                     (void *)&inplaceBuffer);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. :kernelLUD(inplaceBuffer)");
    }

    for(int index = 0; index < effectiveDimension - 1; ++index)
    {
        /*
         *  code to find :
         *  1. new offset
         *  2. Local Work group size
         *  3. Global ND range
         */
        if(index % VECTOR_SIZE == 0)
        {
            offset[0] = (index / VECTOR_SIZE);
            offset[1] = VECTOR_SIZE * (index / VECTOR_SIZE);

            if(!index)
            {
                globalThreads[0] += 1;
                globalThreads[1] += VECTOR_SIZE;
            }
            globalThreads[0] -= 1;
            globalThreads[1] -= VECTOR_SIZE;

            if(globalThreads[0] <= kernelInfo.kernelWorkGroupSize)
            {
                localThreads[0] = globalThreads[0];
            }
            else
            {
                size_t temp = (int)kernelInfo.kernelWorkGroupSize;
                for(; temp > 1; temp--)
                {
                    if(globalThreads[0] % temp == 0)
                    {
                        break;
                    }
                }
                localThreads[0] = temp;
            }

            if( globalThreads[1] < kernelInfo.kernelWorkGroupSize / localThreads[0])
            {
                localThreads[1] = globalThreads[1];
            }
            else
            {
                size_t temp = kernelInfo.kernelWorkGroupSize / localThreads[0];
                for(; temp > 1; temp--)
                {
                    if(globalThreads[1] % temp == 0)
                    {
                        break;
                    }
                }
                localThreads[1] = temp;
            }
        }

        // 3rd kernel argument - index for the matrix
        status = CECL_SET_KERNEL_ARG(
                     kernelLUD,
                     2,
                     sizeof(cl_uint),
                     &index);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. :kernelLUD(index)");

        // 4th kernel argument - local spcae for holding ratio
        status = CECL_SET_KERNEL_ARG(
                     kernelLUD,
                     3,
                     sizeof(cl_double) * localThreads[1],
                     NULL);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. :kernelLUD(localBuffer)");

        /*
         * Enqueue a kernel run call.
         * kernelLUD
         */
        status = CECL_ND_RANGE_KERNEL(
                     commandQueue,
                     kernelLUD,
                     2,
                     offset,
                     globalThreads,
                     localThreads,
                     0,
                     NULL,
                     NULL);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

        // wait for the kernel call to finish execution
        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");
    }

    /*
     * This will combine the L & U matrices at the GPU side
     * so that they can be bought in CPU space as a single matrix
     */

    // 2nd kernel argument - L matrix
    status = CECL_SET_KERNEL_ARG(
                 kernelCombine,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer2);
    CHECK_OPENCL_ERROR(status,
                       "CECL_SET_KERNEL_ARG failed. :kernelCombine(inputBuffer2)");

    // 2nd kernel argument - inplace matrix
    status = CECL_SET_KERNEL_ARG(
                 kernelCombine,
                 1,
                 sizeof(cl_mem),
                 (void *)&inplaceBuffer);
    CHECK_OPENCL_ERROR(status,
                       "CECL_SET_KERNEL_ARG failed. :kernelCombine(inplaceBuffer)");

    globalThreads[0] = effectiveDimension;
    globalThreads[1] = effectiveDimension;

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernelCombine,
                 2,
                 NULL,
                 globalThreads,
                 NULL,
                 0,
                 NULL,
                 NULL);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed. ");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    // Get final matrix data
    outMapPtr = CECL_MAP_BUFFER(
                    commandQueue,
                    inplaceBuffer,
                    CL_FALSE,
                    CL_MAP_READ,
                    0,
                    SIZE,
                    0,
                    NULL,
                    &outMapEvt,
                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed. (inplaceBuffer)");
	 
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outMapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outMapEvt) Failed");

    memcpy(matrixGPU, outMapPtr, SIZE);

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inplaceBuffer,
                 outMapPtr,
                 0,
                 NULL,
                 &outUnmapEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inplaceBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outUnmapEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outUnmapEvt) Failed");

    return SDK_SUCCESS;
}

void LUD::LUDCPUReference(double* matrixCPU, const cl_uint effectiveDimension)
{
    for(unsigned int d = 0 ; d < effectiveDimension-1 ; d++)
    {
        for(unsigned i = d + 1 ; i < effectiveDimension; i++)
        {
            double ratio = matrixCPU[i * effectiveDimension + d]
                           / matrixCPU[d * effectiveDimension + d];
            for(unsigned int j = d; j < effectiveDimension; j++)
            {
                matrixCPU[ i * effectiveDimension + j]
                -= matrixCPU[d * effectiveDimension + j] * ratio;
                if(j == d)
                {
                    matrixCPU[i * effectiveDimension + j] = ratio;
                }
            }
        }
    }
}

int LUD::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // add command line option for matrix actualDimension
    Option* xParam = new Option;
    CHECK_ALLOCATION(xParam, "Memory Allocation error.\n");

    xParam->_sVersion = "x";
    xParam->_lVersion = "dimension";
    xParam->_description = "actualDimension of input matrix";
    xParam->_type     = CA_ARG_INT;
    xParam->_value    = &actualDimension;

    sampleArgs->AddOption(xParam);
    delete xParam;

    Option* iter = new Option;
    CHECK_ALLOCATION(iter, "Memory allocation for Option failed\n");

    iter->_sVersion = "i";
    iter->_lVersion = "iterations";
    iter->_description = "Number of iteration of kernel";
    iter->_type = CA_ARG_INT;
    iter->_value = &iterations;

    sampleArgs->AddOption(iter);
    delete iter;

    return SDK_SUCCESS;
}


int LUD::setup()
{
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    cl_int status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    setupTime = (cl_double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int LUD::run()
{
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    std::cout << "Executing kernel for "
              << iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------"
              << std::endl;

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
        printArray<double>(
            "LU Matrix GPU implementation",
            matrixGPU,
            actualDimension,
            actualDimension);

    return SDK_SUCCESS;
}

int LUD::verifyResults()
{
    if(sampleArgs->verify)
    {
        //reference implementation
        int refTimer = sampleTimer->createTimer();
        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        LUDCPUReference(matrixCPU, effectiveDimension);
        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        if(!sampleArgs->quiet)
            printArray<double>(
                "LU Matrix CPU Reference",
                matrixCPU,
                actualDimension,
                actualDimension);

        // compare the results and see if they match
        if(compare(
                    matrixGPU,
                    matrixCPU,
                    effectiveDimension * effectiveDimension))
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

void LUD::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"WxH" , "Time(sec)", "[Transfer+Kernel]Time(sec)"};
        std::string stats[3];

        sampleTimer->totalTime = setupTime + totalKernelTime;

        stats[0]  = toString(actualDimension, std::dec)
                    + "x" + toString(actualDimension, std::dec);
        stats[1]  = toString(sampleTimer->totalTime, std::dec);
        stats[2]  = toString(totalKernelTime, std::dec);

        printStatistics(strArray, stats, 3);
    }
}

int LUD::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernelLUD);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernelLUD)");

    status = clReleaseKernel(kernelCombine);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernelCombine)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inplaceBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inplaceBuffer)");

    status = clReleaseMemObject(inputBuffer2);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer2)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)
#ifdef WIN32
    ALIGNED_FREE(input);

    ALIGNED_FREE(matrixGPU);
#else
    FREE(input);

    FREE(matrixGPU);
#endif

    FREE(matrixCPU);
    FREE(devices);

    return SDK_SUCCESS;
}


int main(int argc, char * argv[])
{
    LUD clLUDecompose;

    // Initialize
    if(clLUDecompose.initialize()!= SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clLUDecompose.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clLUDecompose.sampleArgs->isDumpBinaryEnabled())
    {
        return clLUDecompose.genBinaryImage();
    }
    else
    {
        // Setup
        int status = clLUDecompose.setup();
        if(status != SDK_SUCCESS)
        {
            return (status == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;
        }

        // Run
        if(clLUDecompose.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // VerifyResults
        if(clLUDecompose.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Cleanup
        if(clLUDecompose.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
        clLUDecompose.printStats();
    }

    return SDK_SUCCESS;
}

