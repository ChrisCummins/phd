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


#include "ImageOverlap.hpp"
#include <cmath>


int
ImageOverlap::readImage(std::string mapImageName,std::string verifyImageName)
{

    // load input bitmap image
    mapBitmap.load(mapImageName.c_str());
    verifyBitmap.load(verifyImageName.c_str());
    // error if image did not load
    if(!mapBitmap.isLoaded())
    {
        error("Failed to load input image!");
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = mapBitmap.getHeight();
    width = mapBitmap.getWidth();
    image_desc.image_width=width;
    image_desc.image_height=height;
    // allocate memory for map image data
    mapImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(mapImageData,"Failed to allocate memory! (mapImageData)");

    // allocate memory for fill image data
    fillImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(fillImageData,"Failed to allocate memory! (fillImageData)");

    // initializa the Image data to NULL
    memset(fillImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = mapBitmap.getPixels();
    CHECK_ALLOCATION(pixelData,"Failed to read mapBitmap pixel Data!");

    // Copy pixel data into mapImageData
    memcpy(mapImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationImageData = (cl_uchar4*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(pixelData,"verificationOutput heap allocation failed!");

    pixelData = verifyBitmap.getPixels();
    CHECK_ALLOCATION(pixelData,"Failed to read verifyBitmap pixel Data!");

    // Copy pixel data into verificationOutput
    memcpy(verificationImageData, pixelData, width * height * pixelSize);

    return SDK_SUCCESS;
}


int
ImageOverlap::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("ImageOverlap_Kernels.cl");
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
ImageOverlap::setupCL()
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

    blockSizeX = deviceInfo.maxWorkGroupSize<GROUP_SIZE
                 ?deviceInfo.maxWorkGroupSize:GROUP_SIZE;

    // Create command queue
    cl_command_queue_properties prop = 0;
    for(int i=0; i<3; i++)
    {
        commandQueue[i] = CECL_CREATE_COMMAND_QUEUE(
                              context,
                              devices[sampleArgs->deviceId],
                              prop,
                              &status);
        CHECK_OPENCL_ERROR(status,"CECL_CREATE_COMMAND_QUEUEfailed.");
    }

    // Create and initialize image objects

    // Create map image
    mapImage = clCreateImage(context,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             &imageFormat,
                             &image_desc,
                             mapImageData,
                             &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed. (mapImage)");
    int color[4] = {0,0,80,255};
    size_t origin[3] = {300,300,0};
    size_t region[3] = {100,100,1};
    status = clEnqueueFillImage(commandQueue[0], mapImage, color, origin, region,
                                0, NULL, &eventlist[0]);

    // Create fill image
    fillImage = clCreateImage(context,
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              &imageFormat,
                              &image_desc,
                              fillImageData,
                              &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed. (fillImage)");

    color[0] = 80;
    color[1] = 0;
    color[2] = 0;
    color[3] = 0;
    origin[0] = 50;
    origin[1] = 50;
    status = clEnqueueFillImage(commandQueue[1], fillImage, color, origin, region,
                                0, NULL, &eventlist[1]);

    //Create output image
    outputImage = clCreateImage(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                &imageFormat,
                                &image_desc,
                                NULL,
                                &status);
    CHECK_OPENCL_ERROR(status,"CECL_BUFFER failed. (outputImage)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("ImageOverlap_Kernels.cl");
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
    kernelOverLap = CECL_KERNEL(program, "OverLap", &status);
    CHECK_OPENCL_ERROR(status,"CECL_KERNEL failed.(OverLap)");

    return SDK_SUCCESS;
}


int
ImageOverlap::runCLKernels()
{
    cl_int status;

    //wait for fill end
    status=clEnqueueMarkerWithWaitList(commandQueue[2],2,eventlist,&enqueueEvent);
    CHECK_OPENCL_ERROR(status,
                       "clEnqueueMarkerWithWaitList failed.(commandQueue[2])");

    // Set appropriate arguments to the kernelOverLap

    // map buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelOverLap,
                 0,
                 sizeof(cl_mem),
                 &mapImage);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (mapImage)");

    // fill Buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelOverLap,
                 1,
                 sizeof(cl_mem),
                 &fillImage);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (fillImage)");

    // fill Buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelOverLap,
                 2,
                 sizeof(cl_mem),
                 &outputImage);
    CHECK_OPENCL_ERROR(status,"CECL_SET_KERNEL_ARG failed. (outputImage)");

    // Enqueue a kernel run call.
    size_t globalThreads[] = {width, height};
    size_t localThreads[] = {blockSizeX, blockSizeY};

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue[2],
                 kernelOverLap,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 1,
                 &enqueueEvent,
                 NULL);
    CHECK_OPENCL_ERROR(status,"CECL_ND_RANGE_KERNEL failed.");

    // Enqueue Read Image
    size_t origin[] = {0, 0, 0};
    size_t region[] = {width, height, 1};
    size_t  rowPitch;
    size_t  slicePitch;
    // Read copy
    outputImageData = (cl_uchar4*)clEnqueueMapImage( commandQueue[2],
                      outputImage,
                      CL_FALSE,
                      mapFlag,
                      origin, region,
                      &rowPitch, &slicePitch,
                      0, NULL,
                      NULL,
                      &status );
    CHECK_OPENCL_ERROR(status,"clEnqueueMapImage failed.(commandQueue[2])");

    clFlush(commandQueue[0]);
    clFlush(commandQueue[1]);

    status = clEnqueueUnmapMemObject(commandQueue[2],outputImage,
                                     (void*)outputImageData,0,0,NULL);
    CHECK_OPENCL_ERROR(status,"clEnqueueUnmapMemObject failed.(outputImage)");

    // Wait for the read buffer to finish execution
    status = clFinish(commandQueue[2]);
    CHECK_OPENCL_ERROR(status,"clFinish failed.(commandQueue[2])");


    return SDK_SUCCESS;
}


int
ImageOverlap::initialize()
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
ImageOverlap::setup()
{
    int status = 0;
    // Allocate host memoryF and read input image
    std::string filePath = getPath() + std::string(MAP_IMAGE);
    std::string  verifyfilePath = getPath() + std::string(MAP_VERIFY_IMAGE);
    status = readImage(filePath,verifyfilePath);
    CHECK_ERROR(status, SDK_SUCCESS, "Read Map Image failed");

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
ImageOverlap::run()
{
    int status;

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

    return SDK_SUCCESS;
}


int
ImageOverlap::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;
    status = clReleaseEvent(eventlist[0]);
    CHECK_OPENCL_ERROR(status,"clReleaseEvent failed.(eventlist[0])");

    status = clReleaseEvent(eventlist[1]);
    CHECK_OPENCL_ERROR(status,"clReleaseEvent failed.(eventlist[1])");

    status = clReleaseEvent(enqueueEvent);
    CHECK_OPENCL_ERROR(status,"clReleaseEvent failed.(enqueueEvent)");

    status = clReleaseKernel(kernelOverLap);
    CHECK_OPENCL_ERROR(status,"clReleaseKernel failed.(kernelOverLap)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status,"clReleaseProgram failed.(program)");

    status = clReleaseMemObject(mapImage);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(mapImage)");

    status = clReleaseMemObject(fillImage);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(fillImage)");

    status = clReleaseMemObject(outputImage);
    CHECK_OPENCL_ERROR(status,"clReleaseMemObject failed.(outputImage)");

    for (int i=0; i<3; i++)
    {
        status = clReleaseCommandQueue(commandQueue[i]);
        CHECK_OPENCL_ERROR(status,"clReleaseCommandQueue failed.(commandQueue)");
    }

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status,"clReleaseContext failed.(context)");

    // release program resources (input memory etc.)

    FREE(mapImageData);
    FREE(fillImageData);
    FREE(verificationImageData);
    FREE(devices);

    return SDK_SUCCESS;
}


void
ImageOverlap::ImageOverlapCPUReference()
{

}


int
ImageOverlap::verifyResults()
{
    if(sampleArgs->verify)
    {
        std::cout << "Verifying result - ";
        // compare the results and see if they match
        if(!memcmp(outputImageData, verificationImageData, width * height * 4))
        {
            std::cout << "Passed!\n" << std::endl;
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
ImageOverlap::printStats()
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
    ImageOverlap clImageOverlap;

    if (clImageOverlap.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clImageOverlap.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clImageOverlap.sampleArgs->isDumpBinaryEnabled())
    {
        return clImageOverlap.genBinaryImage();
    }

    status = clImageOverlap.setup();
    if(status != SDK_SUCCESS)
    {
        return (status == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;
    }

    if (clImageOverlap.run() !=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clImageOverlap.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clImageOverlap.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clImageOverlap.printStats();
    return SDK_SUCCESS;
}
