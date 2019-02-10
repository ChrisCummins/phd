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


#include "DCT.hpp"

using namespace dct;

int
DCT::setupDCT()
{
    cl_uint inputSizeBytes;

    // allocate and init memory used by host
    inputSizeBytes = width * height * sizeof(cl_float);
    input = (cl_float *) malloc(inputSizeBytes);
    CHECK_ALLOCATION(input, "Failed to allocate host memory. (input)");

    cl_uint outputSizeBytes =  width * height * sizeof(cl_float);
    output = (cl_float *)malloc(outputSizeBytes);
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

    // random initialisation of input
    fillRandom<cl_float>(input, width, height, 0, 255);
	
	//Get the dct8x8 transpose
	for(cl_uint j=0; j < blockWidth ; ++j)
        for(cl_uint i = 0; i < blockWidth ; ++i)
        {
			dct8x8_trans[j*blockWidth + i] = dct8x8[i*blockWidth + j];
		}

    // Unless sampleArgs->quiet mode has been enabled, print the INPUT array.

    if(!sampleArgs->quiet)
    {
        printArray<cl_float>(
            "Input",
            input,
            width,
            1);
    }

    return SDK_SUCCESS;
}

int
DCT::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("DCT_Kernels.cl");
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
DCT::setupCL(void)
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
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY,
                      sizeof(cl_float) * width * height,
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

    outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY,
                       sizeof(cl_float) * width * height,
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");

    dctBuffer = CECL_BUFFER(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(cl_float) * blockSize,
                    dct8x8,
                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (dctBuffer)");

	dct_transBuffer = CECL_BUFFER(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(cl_float) * blockSize,
                    dct8x8_trans,
                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (dct_transBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("DCT_Kernels.cl");
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
    kernel = CECL_KERNEL(program, "DCT", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}


int
DCT::runCLKernels(void)
{
    cl_int   status;

    size_t globalThreads[2] = {width, height};
    size_t localThreads[2] = {blockWidth, blockWidth};

    status =  kernelInfo.setKernelWorkGroupInfo(kernel,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    availableLocalMemory = deviceInfo.localMemSize - kernelInfo.localMemoryUsed;

    neededLocalMemory    = blockWidth * blockWidth* sizeof(cl_float);

    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    if(localThreads[0]                 > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[1]                 > deviceInfo.maxWorkItemSizes[1] ||
            localThreads[0] * localThreads[1] > deviceInfo.maxWorkGroupSize)
    {
        std::cout <<
                  "Unsupported: Device does not support requested number of work items."<<std::endl;
        return SDK_FAILURE;
    }

    if((cl_uint)(localThreads[0] * localThreads[1]) >
            kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size required : {"
                      << localThreads[0] << "x" << localThreads[1] << "}" << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Exiting!\n";
        }
        return SDK_EXPECTED_FAILURE;
    }

    // Set input data
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 inputBuffer,
                 CL_FALSE,
                 0,
                 sizeof(cl_float) * width * height,
                 input,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (inputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&writeEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(writeEvt) Failed");


    // Set appropriate arguments to the kernel

    /**
     * 1st argument to the kernel which stores the output of the DCT
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    /**
     * 2nd argument to the kernel , the input matrix
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 1,
                 sizeof(cl_mem),
                 (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    /**
     * 3rd argument to the kernel , the dct8x8 cosine basis function matrix
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 2,
                 sizeof(cl_mem),
                 (void *)&dctBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (dctBuffer)");

    /**
     * 4th argument to the kernel , the dct8x8 cosine basis function matrix
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 3,
                 sizeof(cl_mem),
                 (void *)&dct_transBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (dct_transBuffer)");

    /**
     * 5th argument to the kernel , local memory which stores intermediate values
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 4,
                 blockWidth * blockWidth * sizeof(cl_float),
                 NULL);
    CHECK_OPENCL_ERROR(status,
                       "CECL_SET_KERNEL_ARG failed. (Intermediate local buffer)");

    /**
     * 6th argument to the kernel , width of the input matrix
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 5,
                 sizeof(cl_uint),
                 (void *)&width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (width)");

    /**
     * 7th argument to the kernel , width of the block = 8
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 6,
                 sizeof(cl_uint),
                 (void *)&blockWidth);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (blockWidth)");

    /**
     * 8th argument to the kernel , flag to perform inverse DCT
     */
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 7,
                 sizeof(cl_uint),
                 (void *)&inverse);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inverse)");

    /**
     * Enqueue a kernel run call.
     */
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernel,
                 2,
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
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    // Enqueue readBuffer
    cl_event readEvt;
    status = CECL_READ_BUFFER(
                 commandQueue,
                 outputBuffer,
                 CL_FALSE,
                 0,
                 width * height * sizeof(cl_float),
                 output,
                 0,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&readEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(readEvt) Failed");

    return SDK_SUCCESS;
}



cl_uint
DCT::getIdx(cl_uint blockIdx, cl_uint blockIdy, cl_uint localIdx,
            cl_uint localIdy, cl_uint blockWidth, cl_uint globalWidth)
{
    cl_uint globalIdx = blockIdx * blockWidth + localIdx;
    cl_uint globalIdy = blockIdy * blockWidth + localIdy;
    return (globalIdy * globalWidth  + globalIdx);
}


/*
 * Reference implementation of the Discrete Cosine Transform on the CPU
 */
void
DCT::DCTCPUReference( cl_float * verificationOutput,
                      const cl_float * input ,
                      const cl_float * dct8x8 ,
                      const cl_uint    width,
                      const cl_uint    height,
                      const cl_uint   numBlocksX,
                      const cl_uint   numBlocksY,
                      const cl_uint    inverse)
{
    cl_float * temp = (cl_float *)malloc(width*height*sizeof(cl_float));

    // for each block in the image
    for(cl_uint blockIdy=0; blockIdy < numBlocksY; ++blockIdy)
        for(cl_uint blockIdx=0; blockIdx < numBlocksX; ++blockIdx)
        {
            //  First calculate A^T * X for FDCT or A * X for iDCT
            for(cl_uint j=0; j < blockWidth ; ++j)
                for(cl_uint i = 0; i < blockWidth ; ++i)
                {
                    cl_uint index = getIdx(blockIdx, blockIdy, i, j, blockWidth, width);
                    cl_float tmp = 0.0f;
                    for(cl_uint k=0; k < blockWidth; ++k)
                    {
                        // multiply with dct (j,k) 
                        cl_uint index1 =  j*blockWidth +k;
                        //input(i,k)
						cl_uint index2 = getIdx(blockIdx, blockIdy, i, k, blockWidth, width);
						
                        if (inverse)
							tmp += dct8x8[index1]*input[index2];
						else
							tmp += dct8x8_trans[index1]*input[index2];
                    }
                    temp[index] = tmp;
                }


            // And now for FDCT, multiply the result of previous step with A i.e. calculate (A^T * X) * A for FDCT or 
			// for iDCT multiply the result of previous step with A^T i.e. calculate (A * X) * A^T 
            for(cl_uint j=0; j < blockWidth ; ++j)
                for(cl_uint i = 0; i < blockWidth ; ++i)
                {
                    cl_uint index = getIdx(blockIdx, blockIdy, i, j, blockWidth, width);
                    cl_float tmp = 0.0f;
                    for(cl_uint k=0; k < blockWidth; ++k)
                    {
                        //input(j,k)
                        cl_uint index1 = getIdx(blockIdx, blockIdy, k,j, blockWidth, width);

                        // multiply with dct (k,i) 
                        cl_uint index2 =  k*blockWidth + i;
						
						if (inverse)
							tmp += temp[index1]*dct8x8_trans[index2];
						else
							tmp += temp[index1]*dct8x8[index2];
                    }
                    verificationOutput[index] = tmp;
                }
        }

    free(temp);
}

int DCT::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initialization failed");
    Option* width_option = new Option;
    CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

    width_option->_sVersion = "x";
    width_option->_lVersion = "width";
    width_option->_description = "Width of the input matrix";
    width_option->_type = CA_ARG_INT;
    width_option->_value = &width;

    sampleArgs->AddOption(width_option);
    delete width_option;

    Option* height_option = new Option;
    CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

    height_option->_sVersion = "y";
    height_option->_lVersion = "height";
    height_option->_description = "Height of the input matrix";
    height_option->_type = CA_ARG_INT;
    height_option->_value = &height;

    sampleArgs->AddOption(height_option);
    delete height_option;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

	Option* isInverse = new Option;
    CHECK_ALLOCATION(isInverse, "Memory allocation error.\n");

    isInverse->_sVersion = "inv";
    isInverse->_lVersion = "inverse";
    isInverse->_description = "Run inverse DCT";
    isInverse->_type = CA_ARG_INT;
    isInverse->_value = &inverse;

    sampleArgs->AddOption(isInverse);
    delete isInverse;

    return SDK_SUCCESS;
}

int DCT::setup()
{
    // Make sure the width is a multiple of blockWidth 8 here
    if(width % blockWidth != 0)
    {
        width = (width/blockWidth + 1)*blockWidth;
    }

    // Make sure the height is a multiple of blockWidth 8 here
    if(height%blockWidth !=0)
    {
        height = (height/blockWidth + 1) * blockWidth;
    }
    CHECK_ERROR(setupDCT(), SDK_SUCCESS, "OpenCL setupDCT failed");
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);
    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    sampleTimer->stopTimer(timer);

    setupTime = (cl_double)sampleTimer->readTimer(timer);
    return SDK_SUCCESS;
}


int DCT::run()
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

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
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
    totalKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<cl_float>("Output", output, width,1);
    }

    return SDK_SUCCESS;
}

int DCT::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationOutput = (cl_float *) malloc(width*height*sizeof(cl_float));
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");

        /**
         * reference implementation
         */
        int refTimer = sampleTimer->createTimer();

        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        DCTCPUReference(verificationOutput, input, dct8x8, width, height,
                        width/blockWidth, height/blockWidth, inverse);

        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        // compare the results and see if they match
		if(compare(output, verificationOutput, width*height))
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

void DCT::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Width", "Height", "Time(sec)", "[Transfer+Kernel]Time(sec)"};
        std::string stats[4];

        sampleTimer->totalTime = setupTime + totalKernelTime;

        stats[0]  = toString(width    , std::dec);
        stats[1]  = toString(height   , std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(totalKernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int DCT::cleanup()
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
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(dctBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(dctBuffer)");

	status = clReleaseMemObject(dct_transBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(dct_transBuffer)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)
    FREE(input);
    FREE(output);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    DCT clDCT;
    if(clDCT.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clDCT.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clDCT.sampleArgs->isDumpBinaryEnabled())
    {
        return clDCT.genBinaryImage();
    }

    if(clDCT.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clDCT.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clDCT.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clDCT.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clDCT.printStats();

    return SDK_SUCCESS;
}
