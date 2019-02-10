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


#include "SimpleImage.hpp"
#include <cmath>


int
SimpleImage::readInputImage(std::string inputImageName)
{

    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        error("Failed to load input image!");
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // allocate memory for input & outputimage data
    inputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(inputImageData,"Failed to allocate memory! (inputImageData)");

    // allocate memory for 2D-copy output image data
    outputImageData2D = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(outputImageData2D,
                     "Failed to allocate memory! (outputImageData)");

    // allocate memory for 3D-copy output image data
    outputImageData3D = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(outputImageData3D,
                     "Failed to allocate memory! (outputImageData)");

    // initializa the Image data to NULL
    memset(outputImageData2D, 0, width * height * pixelSize);
    memset(outputImageData3D, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();
    CHECK_ALLOCATION(pixelData,"Failed to read pixel Data!");

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (cl_uchar*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(pixelData,"verificationOutput heap allocation failed!");

    // initialize the data to NULL
    //memset(verificationOutput, 0, width * height * pixelSize);
    memcpy(verificationOutput, inputImageData, width * height * pixelSize);

    return SDK_SUCCESS;

}


int
SimpleImage::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData2D, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!";
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

int
SimpleImage::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("SimpleImage_Kernels.cl");
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
SimpleImage::setupCL()
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
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    status = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "deviceInfo.setDeviceInfo failed");

    if(!deviceInfo.imageSupport)
    {
        OPENCL_EXPECTED_ERROR(" Expected Error: Device does not support Images");
    }

    // Create command queue
    cl_command_queue_properties prop = 0;
    commandQueue = CECL_CREATE_COMMAND_QUEUE(
                       context,
                       devices[sampleArgs->deviceId],
                       prop,
                       &status);
    CHECK_OPENCL_ERROR(status,"CECL_CREATE_COMMAND_QUEUE failed.");

    // Create and initialize image objects
    cl_image_desc imageDesc;
    memset(&imageDesc, '\0', sizeof(cl_image_desc));
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = width;
    imageDesc.image_height = height;

    // Create 2D input image
    inputImage2D = clCreateImage(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 &imageFormat,
                                 &imageDesc,
                                 inputImageData,
                                 &status);
    CHECK_OPENCL_ERROR(status,"clCreateImage failed. (inputImage2D)");

    // Create 2D output image
    outputImage2D = clCreateImage(context,
                                  CL_MEM_WRITE_ONLY,
                                  &imageFormat,
                                  &imageDesc,
                                  0,
                                  &status);
    CHECK_OPENCL_ERROR(status,"clCreateImage failed. (outputImage2D)");

    // Writes to 3D images not allowed in spec currently
    outputImage3D = clCreateImage(context,
                                  CL_MEM_WRITE_ONLY,
                                  &imageFormat,
                                  &imageDesc,
                                  0,
                                  &status);
    CHECK_OPENCL_ERROR(status,"clCreateImage failed. (outputImage3D)");

    // Create 3D input image
    memset(&imageDesc, '\0', sizeof(cl_image_desc));
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
    imageDesc.image_width = width;
    imageDesc.image_height = height / 2;
    imageDesc.image_depth = 2;

    inputImage3D = clCreateImage(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 &imageFormat,
                                 &imageDesc,
                                 inputImageData,
                                 &status);
    CHECK_OPENCL_ERROR(status,"clCreateImage failed. (inputImage3D)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("SimpleImage_Kernels.cl");
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
    kernel2D = CECL_KERNEL(program, "image2dCopy", &status);
    CHECK_OPENCL_ERROR(status,"CECL_KERNEL failed.(kernel2D)");

    kernel3D = CECL_KERNEL(program, "image3dCopy", &status);
    CHECK_OPENCL_ERROR(status,"CECL_KERNEL failed.(kernel3D)");

    // Check group size against group size returned by kernel
    status = CECL_GET_KERNEL_WORK_GROUP_INFO(kernel2D,
                                      devices[sampleArgs->deviceId],
                                      CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(size_t),
                                      &kernel2DWorkGroupSize,
                                      0);
    CHECK_OPENCL_ERROR(status,"CECL_GET_KERNEL_WORK_GROUP_INFO  failed.");

    // Check group size against group size returned by kernel
    status = CECL_GET_KERNEL_WORK_GROUP_INFO(kernel3D,
                                      devices[sampleArgs->deviceId],
                                      CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(size_t),
                                      &kernel3DWorkGroupSize,
                                      0);
    CHECK_OPENCL_ERROR(status,"CECL_GET_KERNEL_WORK_GROUP_INFO  failed.");

    cl_uint temp = (cl_uint)min(kernel2DWorkGroupSize, kernel3DWorkGroupSize);

    if((blockSizeX * blockSizeY) > temp)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel(s) : "
                      << temp << std::endl;
            std::cout << "Falling back to " << temp << std::endl;
        }

        if(blockSizeX > temp)
        {
            blockSizeX = temp;
            blockSizeY = 1;
        }
    }
    return SDK_SUCCESS;
}

int
SimpleImage::runCLKernels()
{
    cl_int status;

    // Set appropriate arguments to the kernel2D

    // input buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernel2D,
                 0,
                 sizeof(cl_mem),
                 &inputImage2D);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (inputImage2D)");

    // outBuffer image
    status = CECL_SET_KERNEL_ARG(
                 kernel2D,
                 1,
                 sizeof(cl_mem),
                 &outputImage2D);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (outputImage2D)");

    // Set appropriate arguments to the kernel3D

    // input buffer image
    status = CECL_SET_KERNEL_ARG(kernel3D,
                            0,
                            sizeof(cl_mem),
                            &inputImage3D);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (inputImage3D)");

    // outBuffer image
    status = CECL_SET_KERNEL_ARG(
                 kernel3D,
                 1,
                 sizeof(cl_mem),
                 &outputImage3D);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (outputImage3D)");

    // Enqueue a kernel run call.
    size_t globalThreads[] = {width, height};
    size_t localThreads[] = {blockSizeX, blockSizeY};

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernel2D,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 0);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernel3D,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 0);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status,"clFinish failed.");

    // Enqueue Read Image
    size_t origin[] = {0, 0, 0};
    size_t region[] = {width, height, 1};

    // Read output of 2D copy
    status = clEnqueueReadImage(commandQueue,
                                outputImage2D,
                                1,
                                origin,
                                region,
                                0,
                                0,
                                outputImageData2D,
                                0, 0, 0);
    CHECK_OPENCL_ERROR(status,"clEnqueueReadImage failed.");

    // Read output of 3D copy
    status = clEnqueueReadImage(commandQueue,
                                outputImage3D,
                                1,
                                origin,
                                region,
                                0,
                                0,
                                outputImageData3D,
                                0, 0, 0);
    CHECK_OPENCL_ERROR(status,"clEnqueueReadImage failed.");

    // Wait for the read buffer to finish execution
    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status,"clFinish failed.(commandQueue)");

    return SDK_SUCCESS;
}

int
SimpleImage::initialize()
{

    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option,
                     "Memory Allocation error. (iteration_option)");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);

    delete iteration_option;

    return SDK_SUCCESS;
}

int
SimpleImage::setup()
{
    int status = 0;
    // Allocate host memoryF and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    status = readInputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "Read Input Image failed");

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

int
SimpleImage::run()
{
    int status;
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    std::cout << "Executing kernel for " << iterations <<
              " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        status = runCLKernels();
        CHECK_ERROR(status, SDK_SUCCESS, "OpenCL run Kernel failed");
    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    status = writeOutputImage(OUTPUT_IMAGE);
    CHECK_ERROR(status, SDK_SUCCESS, "write Output Image Failed");

    return SDK_SUCCESS;
}

int
SimpleImage::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel2D);
    CHECK_OPENCL_ERROR(status,"clReleaseKernel failed.(kernel2D)");

    status = clReleaseKernel(kernel3D);
    CHECK_OPENCL_ERROR(status,"clReleaseKernel failed.(kernel3D)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inputImage2D);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(inputImage2D)");

    status = clReleaseMemObject(outputImage2D);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(outputImage2D)");

    status = clReleaseMemObject(inputImage3D);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(inputImage3D)");

    status = clReleaseMemObject(outputImage3D);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(outputImage3D)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status,"clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status,"clReleaseContext failed.(context)");

    // release program resources (input memory etc.)

    FREE(inputImageData);
    FREE(outputImageData2D);
    FREE(outputImageData3D);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}


void
SimpleImage::simpleImageCPUReference()
{

}


int
SimpleImage::verifyResults()
{
    if(sampleArgs->verify)
    {
        std::cout << "Verifying 2D copy result - ";
        // compare the results and see if they match
        if(!memcmp(inputImageData, outputImageData2D, width * height * 4))
        {
            std::cout << "Passed!\n" << std::endl;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }

        std::cout << "Verifying 3D copy result - ";

        // compare the results and see if they match
        if(!memcmp(inputImageData, outputImageData3D, width * height * 4))
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

void
SimpleImage::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "kernelTime(sec)"
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


int
main(int argc, char * argv[])
{
    int status = 0;
    SimpleImage clSimpleImage;

    if (clSimpleImage.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleImage.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSimpleImage.sampleArgs->isDumpBinaryEnabled())
    {
        return clSimpleImage.genBinaryImage();
    }

    status = clSimpleImage.setup();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    if (clSimpleImage.run() !=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleImage.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleImage.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clSimpleImage.printStats();
    return SDK_SUCCESS;
}
