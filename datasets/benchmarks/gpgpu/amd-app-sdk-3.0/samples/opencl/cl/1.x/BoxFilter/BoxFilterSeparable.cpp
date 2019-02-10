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


#include "BoxFilterSeparable.hpp"
#include <cmath>


int
BoxFilterSeparable::readInputImage(std::string inputImageName)
{

    // load input bitmap image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    inputBitmap.load(filePath.c_str());
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // allocate memory for input & output image data
    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");

    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");

    // initializa the Image data to NULL
    memset(outputImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();
    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (cl_uchar4*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * pixelSize);

    return SDK_SUCCESS;
}

int
BoxFilterSeparable::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!";
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}


int
BoxFilterSeparable::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("BoxFilter_Kernels.cl");
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
BoxFilterSeparable::setupCL()
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

    // Create and initialize memory objects

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create memory object for input Image
    inputImageBuffer = CECL_BUFFER(
                           context,
                           inMemFlags,
                           width * height * pixelSize,
                           NULL,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputImageBuffer)");

    // Create memory object for temp Image
    tempImageBuffer = CECL_BUFFER(
                          context,
                          CL_MEM_READ_WRITE,
                          width * height * pixelSize,
                          0,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (tempImageBuffer)");

    // Create memory objects for output Image
    outputImageBuffer = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY,
                                       width * height * pixelSize,
                                       NULL,
                                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputImageBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("BoxFilter_Kernels.cl");
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
    verticalKernel = CECL_KERNEL(program,
                                    "box_filter_vertical",
                                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed. (vertical)");

#ifdef USE_LDS
    horizontalKernel = CECL_KERNEL(program,
                                      "box_filter_horizontal_local",
                                      &status);
#else
    horizontalKernel = CECL_KERNEL(program,
                                      "box_filter_horizontal",
                                      &status);
#endif
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed. (horizontal)");

    status =  kernelInfoH.setKernelWorkGroupInfo(horizontalKernel,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    status =  kernelInfoV.setKernelWorkGroupInfo(verticalKernel, devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    if((blockSizeX * blockSizeY) > kernelInfoV.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfoV.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfoV.kernelWorkGroupSize << std::endl;
        }

        // Three possible cases
        if(blockSizeX > kernelInfoV.kernelWorkGroupSize)
        {
            blockSizeX = kernelInfoV.kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }
    return SDK_SUCCESS;
}

int
BoxFilterSeparable::runCLKernels()
{
    cl_int status;
    cl_int eventStatus = CL_QUEUED;

    // Set input data
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 inputImageBuffer,
                 CL_FALSE,
                 0,
                 width * height * pixelSize,
                 inputImageData,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (inputImageBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&writeEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(writeEvt) Failed");

    // Set appropriate arguments to the kernel

    // input buffer image
    status = CECL_SET_KERNEL_ARG(
                 horizontalKernel,
                 0,
                 sizeof(cl_mem),
                 &inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputImageBuffer)");

    // outBuffer imager
    status = CECL_SET_KERNEL_ARG(
                 horizontalKernel,
                 1,
                 sizeof(cl_mem),
                 &tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputImageBuffer)");

    // filter width
    status = CECL_SET_KERNEL_ARG(
                 horizontalKernel,
                 2,
                 sizeof(cl_int),
                 &filterWidth);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (filterWidth)");

#ifdef USE_LDS
    // shared memory
    status = CECL_SET_KERNEL_ARG(
                 horizontalKernel,
                 3,
                 (GROUP_SIZE + filterWidth - 1) * sizeof(cl_uchar4),
                 0);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (local memory)");
#endif
    // Enqueue a kernel run call.
    size_t globalThreads[] = {width, height};
    size_t localThreads[] = {blockSizeX, blockSizeY};

    cl_event ndrEvt1;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 horizontalKernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &ndrEvt1);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt1);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt1) Failed");

    // Do vertical pass

    // Set appropriate arguments to the kernel

    // input buffer image
    status = CECL_SET_KERNEL_ARG(
                 verticalKernel,
                 0,
                 sizeof(cl_mem),
                 &tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (inputImageBuffer)");

    // outBuffer imager
    status = CECL_SET_KERNEL_ARG(
                 verticalKernel,
                 1,
                 sizeof(cl_mem),
                 &outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputImageBuffer)");

    // filter width
    status = CECL_SET_KERNEL_ARG(
                 verticalKernel,
                 2,
                 sizeof(cl_int),
                 &filterWidth);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (filterWidth)");

    // Enqueue a kernel run call.
    cl_event ndrEvt2;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 verticalKernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &ndrEvt2);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&ndrEvt2);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt2) Failed");

    // Enqueue readBuffer
    cl_event readEvt;
    status = CECL_READ_BUFFER(
                 commandQueue,
                 outputImageBuffer,
                 CL_FALSE,
                 0,
                 width * height * pixelSize,
                 outputImageData,
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



int
BoxFilterSeparable::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory Allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* filter_width = new Option;
    CHECK_ALLOCATION(filter_width, "Memory Allocation error.\n");

    filter_width->_sVersion = "x";
    filter_width->_lVersion = "width";
    filter_width->_description = "Filter width";
    filter_width->_type = CA_ARG_INT;
    filter_width->_value = &filterWidth;

    sampleArgs->AddOption(filter_width);
    delete filter_width;

    return SDK_SUCCESS;
}

int
BoxFilterSeparable::setup()
{
    // Allocate host memory and read input image
    if(readInputImage(INPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer =sampleTimer->createTimer();
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

int
BoxFilterSeparable::run()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer =sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    if(writeOutputImage(OUTPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
BoxFilterSeparable::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(verticalKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(vertical)");

    status = clReleaseKernel(horizontalKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(Horizontal)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)
    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}


int
BoxFilterSeparable::boxFilterCPUReference()
{
    std::cout << "Verifying results...";
    int t = (filterWidth - 1) / 2;
    int filterSize = filterWidth;

    cl_uchar4 *tempData = (cl_uchar4*)malloc(width * height * 4);
    if(tempData == NULL)
    {
        std::cout << "Memory Allocation error.\n";
        return SDK_FAILURE;
    }

    memset(tempData, 0, width * height * sizeof(cl_uchar4));

    // Horizontal filter
    for(int y = 0; y < (int)height; y++)
    {
        for(int x = 0; x < (int)width; x++)
        {
            // Only threads inside horizontal apron will calculate output value
            if(x >= t && x < (int)(width - t))
            {
                cl_int4 sum = {0, 0, 0, 0};
                for(int x1 = -t; x1 <= t; x1++)
                {
                    sum.s[0] += inputImageData[x + x1 + y * width].s[0];
                    sum.s[1] += inputImageData[x + x1 + y * width].s[1];
                    sum.s[2] += inputImageData[x + x1 + y * width].s[2];
                    sum.s[3] += inputImageData[x + x1 + y * width].s[3];
                }
                tempData[x + y * width].s[0] = sum.s[0] / filterSize;
                tempData[x + y * width].s[1] = sum.s[1] / filterSize;
                tempData[x + y * width].s[2] = sum.s[2] / filterSize;
                tempData[x + y * width].s[3] = sum.s[3] / filterSize;
            }
        }
    }

    // Vertical filter
    for(int y = 0; y < (int)height; y++)
    {
        for(int x = 0; x < (int)width; x++)
        {
            // Only threads inside vertical apron will calculate output value
            if(y >= t && y < (int)(height - t))
            {
                cl_int4 sum = {0, 0, 0, 0};
                for(int y1 = -t; y1 <= t; y1++)
                {
                    sum.s[0] += tempData[x + (y + y1) * width].s[0];
                    sum.s[1] += tempData[x + (y + y1) * width].s[1];
                    sum.s[2] += tempData[x + (y + y1) * width].s[2];
                    sum.s[3] += tempData[x + (y + y1) * width].s[3];
                }
                verificationOutput[x + y * width].s[0] = sum.s[0] / filterSize;
                verificationOutput[x + y * width].s[1] = sum.s[1] / filterSize;
                verificationOutput[x + y * width].s[2] = sum.s[2] / filterSize;
                verificationOutput[x + y * width].s[3] = sum.s[3] / filterSize;
            }
        }
    }

    FREE(tempData);
    return SDK_SUCCESS;
}


int
BoxFilterSeparable::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        // reference implementation
        boxFilterCPUReference();

        int j = 0;

        // Compare between outputImageData and verificationOutput
        if(!memcmp(outputImageData,
                   verificationOutput,
                   width * height * sizeof(cl_uchar4)))
        {
            std::cout << "Passed!\n" <<std::endl;
        }
        else
        {
            std::cout << "Failed!\n" <<std::endl;
        }
    }

    return SDK_SUCCESS;
}

void
BoxFilterSeparable::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int BoxFilterSeparable::runSeparableVersion(int argc, char * argv[]) 
{
              
 if(initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(sampleArgs->isDumpBinaryEnabled())
    {
        return genBinaryImage();
    }
    else
    {
        if(setup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        printStats();

    }
    return SDK_SUCCESS;

}