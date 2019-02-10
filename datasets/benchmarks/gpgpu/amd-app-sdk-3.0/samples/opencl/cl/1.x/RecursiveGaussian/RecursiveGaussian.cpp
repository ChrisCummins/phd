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


#include "RecursiveGaussian.hpp"
#include <cmath>


int
RecursiveGaussian::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // Check width against blockSizeX
    if(width % GROUP_SIZE || height % GROUP_SIZE)
    {
        char err[2048];
        sprintf(err, "Width should be a multiple of %d \n", GROUP_SIZE);
        std::cout << err;
        return SDK_FAILURE;
    }

    // allocate memory for input & output image data
    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");
    verificationInput = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(verificationInput,
                     "Failed to allocate memory! (verificationInput)");

    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");

    // initialize the Image data to NULL
    memset(outputImageData, 0, width * height * sizeof(cl_uchar4));

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();
    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * sizeof(cl_uchar4));
    memcpy(verificationInput, pixelData, width * height * sizeof(cl_uchar4));

    // allocate memory for verification output
    verificationOutput = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(verificationOutput,
                     "Failed to allocate memory! (verificationOutput)");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * sizeof(cl_uchar4));

    return SDK_SUCCESS;

}


int
RecursiveGaussian::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * sizeof(cl_uchar4));

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        error("Failed to write output image!");
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


void
RecursiveGaussian::computeGaussParms(float fSigma, int iOrder, GaussParms* pGP)
{
    // pre-compute filter coefficients
    pGP->nsigma =
        fSigma; // note: fSigma is range-checked and clamped >= 0.1f upstream
    pGP->alpha = 1.695f / pGP->nsigma;
    pGP->ema = exp(-pGP->alpha);
    pGP->ema2 = exp(-2.0f * pGP->alpha);
    pGP->b1 = -2.0f * pGP->ema;
    pGP->b2 = pGP->ema2;
    pGP->a0 = 0.0f;
    pGP->a1 = 0.0f;
    pGP->a2 = 0.0f;
    pGP->a3 = 0.0f;
    pGP->coefp = 0.0f;
    pGP->coefn = 0.0f;

    switch (iOrder)
    {
    case 0:
    {
        const float k = (1.0f - pGP->ema)*(1.0f - pGP->ema)/(1.0f +
                        (2.0f * pGP->alpha * pGP->ema) - pGP->ema2);
        pGP->a0 = k;
        pGP->a1 = k * (pGP->alpha - 1.0f) * pGP->ema;
        pGP->a2 = k * (pGP->alpha + 1.0f) * pGP->ema;
        pGP->a3 = -k * pGP->ema2;
    }
    break;
    case 1:
    {
        pGP->a0 = (1.0f - pGP->ema) * (1.0f - pGP->ema);
        pGP->a1 = 0.0f;
        pGP->a2 = -pGP->a0;
        pGP->a3 = 0.0f;
    }
    break;
    case 2:
    {
        const float ea = exp(-pGP->alpha);
        const float k = -(pGP->ema2 - 1.0f)/(2.0f * pGP->alpha * pGP->ema);
        float kn = -2.0f * (-1.0f + (3.0f * ea) - (3.0f * ea * ea) + (ea * ea * ea));
        kn /= (((3.0f * ea) + 1.0f + (3.0f * ea * ea) + (ea * ea * ea)));
        pGP->a0 = kn;
        pGP->a1 = -kn * (1.0f + (k * pGP->alpha)) * pGP->ema;
        pGP->a2 = kn * (1.0f - (k * pGP->alpha)) * pGP->ema;
        pGP->a3 = -kn * pGP->ema2;
    }
    break;
    default:
        // note: iOrder is range-checked and clamped to 0-2 upstream
        return;
    }
    pGP->coefp = (pGP->a0 + pGP->a1)/(1.0f + pGP->b1 + pGP->b2);
    pGP->coefn = (pGP->a2 + pGP->a3)/(1.0f + pGP->b1 + pGP->b2);
}


int
RecursiveGaussian::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("RecursiveGaussian_Kernels.cl");
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
RecursiveGaussian::setupCL()
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

    // Create and initialize memory objects

    // Set Persistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_WRITE;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create memory object for input Image
    inputImageBuffer = CECL_BUFFER(
                           context,
                           inMemFlags,
                           width * height * pixelSize,
                           0,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputImageBuffer)");

    // Create memory objects for output Image
    outputImageBuffer = CECL_BUFFER(context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       width * height * pixelSize,
                                       NULL,
                                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputImageBuffer)");

    // create memory object for temp buffer
    tempImageBuffer = CECL_BUFFER(context,
                                     CL_MEM_READ_WRITE,
                                     width * height * pixelSize,
                                     0,
                                     &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (tempImageBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("RecursiveGaussian_Kernels.cl");
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

    // kernel object for transpose kernel
    kernelTranspose = CECL_KERNEL(program,
                                     "transpose_kernel",
                                     &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(transpose_kernel)");

    // kernel object for recursive gaussian kernel
    kernelRecursiveGaussian = CECL_KERNEL(program,
                              "RecursiveGaussian_kernel",
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(RecursiveGaussian_kernel)");

    status = transposeKernelInfo.setKernelWorkGroupInfo(kernelTranspose,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,
                "transposeKernelInfo.setKernelWorkGroupInfo() failed");

    status = RGKernelInfo.setKernelWorkGroupInfo(kernelRecursiveGaussian,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,
                "RGKernelInfo.setKernelWorkGroupInfo() failed");

    // Calculate block size according to required work-group size by kernel
    if((blockSizeX * blockSizeY) > RGKernelInfo.kernelWorkGroupSize)
    {
        blockSizeX = RGKernelInfo.kernelWorkGroupSize;
        blockSizeY = 1;
    }

    // Calculate 2D block size according to required work-group size by transpose kernel
    blockSize = (size_t)sqrt((float)transposeKernelInfo.kernelWorkGroupSize);
    //blockSize should a multiple of power of 2
    blockSize = (size_t)pow((float)2, (int)(log((float)blockSize) / log((float)2)));

    return SDK_SUCCESS;
}

int
RecursiveGaussian::runCLKernels()
{
    cl_int status = CL_SUCCESS;
    cl_int eventStatus = CL_QUEUED;

    // initialize Gaussian parameters
    float fSigma = 10.0f;               // filter sigma (blur factor)
    int iOrder = 0;                     // filter order

    // compute gaussian parameters
    computeGaussParms(fSigma, iOrder, &oclGP);

    // Write inputImageData to inputImageBuffer on device
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(commandQueue,
                                  inputImageBuffer,
                                  CL_FALSE,
                                  0,
                                  width * height * pixelSize,
                                  inputImageData,
                                  0,
                                  NULL,
                                  &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(inputImageBuffer) failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = waitForEventAndRelease(&writeEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(writeEvt) Failed");

    // Set appropriate arguments to the kernel (Recursive Gaussian)

    // input : input buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelRecursiveGaussian,
                 0,
                 sizeof(cl_mem),
                 &inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(inputImageBuffer) failed.");

    // output : temp Buffer
    status = CECL_SET_KERNEL_ARG(
                 kernelRecursiveGaussian,
                 1,
                 sizeof(cl_mem),
                 &tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(tempImageBuffer) failed.");

    // image width
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            2,
                            sizeof(cl_int),
                            &width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(width) failed.");

    // image height
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            3,
                            sizeof(cl_int),
                            &height);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(height) failed.");

    // gaussian parameter : a0
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            4,
                            sizeof(cl_float),
                            &oclGP.a0);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.a0) failed.");

    // gaussian parameter : a1
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            5,
                            sizeof(cl_float),
                            &oclGP.a1);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.a1) failed.");


    // gaussian parameter : a2
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            6,
                            sizeof(cl_float),
                            &oclGP.a2);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.a2) failed.");

    // gaussian parameter : a3
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            7,
                            sizeof(cl_float),
                            &oclGP.a3);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.a3) failed.");

    // gaussian parameter : b1
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            8,
                            sizeof(cl_float),
                            &oclGP.b1);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.b1) failed.");

    // gaussian parameter : b2
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            9,
                            sizeof(cl_float),
                            &oclGP.b2);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.b2) failed.");

    // gaussian parameter : coefp
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            10,
                            sizeof(cl_float),
                            &oclGP.coefp);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.coefp) failed.");

    // gaussian parameter : coefn
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            11,
                            sizeof(cl_float),
                            &oclGP.coefn);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(oclGP.coefn) failed.");

    // set global index and group size
    size_t globalThreads[] = {width, 1};
    size_t localThreads[] = {blockSizeX, blockSizeY};

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[0] > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not"
                  "support requested number of work items.";
        return SDK_FAILURE;
    }

    // Enqueue a kernel run call.
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernelRecursiveGaussian,
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

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    // Set appropriate arguments to the kernel (Transpose)

    // output : input buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelTranspose,
                 0,
                 sizeof(cl_mem),
                 &inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(inputImageBuffer) failed.");

    // input : temp Buffer
    status = CECL_SET_KERNEL_ARG(
                 kernelTranspose,
                 1,
                 sizeof(cl_mem),
                 &tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(tempImageBuffer) failed.");

    // local memory for block transpose
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            2,
                            blockSize * blockSize * sizeof(cl_uchar4),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(Local) failed.");

    // image width
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            3,
                            sizeof(cl_int),
                            &width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(width) failed.");

    // image height
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            4,
                            sizeof(cl_int),
                            &height);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(height) failed.");

    // block_size
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            5,
                            sizeof(cl_int),
                            &blockSize);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(blockSize) failed.");

    // group dimensions for transpose kernel
    size_t localThreadsT[] = {blockSize, blockSize};
    size_t globalThreadsT[] = {width, height};

    if(localThreadsT[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreadsT[1] > deviceInfo.maxWorkItemSizes[1] ||
            localThreadsT[0] * localThreadsT[1] > deviceInfo.maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support requested"
                  "number of work items.";
        return SDK_FAILURE;
    }

    if(transposeKernelInfo.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient"
                  "local memory on device." << std::endl;
        return SDK_FAILURE;
    }

    // Enqueue Transpose Kernel
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernelTranspose,
                 2,
                 NULL,
                 globalThreadsT,
                 localThreadsT,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "kernelTranspose() failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    /* Set Arguments for Recursive Gaussian Kernel
    Image is now transposed
    new_width = height
    new_height = width */

    // image width : swap with height
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            2,
                            sizeof(cl_int),
                            &height);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(height) failed.");

    // image height
    status = CECL_SET_KERNEL_ARG(kernelRecursiveGaussian,
                            3,
                            sizeof(cl_int),
                            &width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(width) failed.");

    // Set new global index
    globalThreads[0] = height;
    globalThreads[1] = 1;

    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernelRecursiveGaussian,
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

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");
    // Set Arguments to Transpose Kernel

    // output : output buffer image
    status = CECL_SET_KERNEL_ARG(
                 kernelTranspose,
                 0,
                 sizeof(cl_mem),
                 &outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(outputImageBuffer) failed.");

    // input : temp Buffer
    status = CECL_SET_KERNEL_ARG(
                 kernelTranspose,
                 1,
                 sizeof(cl_mem),
                 &tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(tempImageBuffer) failed.");

    // local memory for block transpose
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            2,
                            blockSize * blockSize * sizeof(cl_uchar4),
                            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(Local) failed.");

    // image width : is height actually as the image is currently transposed
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            3,
                            sizeof(cl_int),
                            &height);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(height) failed.");

    // image height
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            4,
                            sizeof(cl_int),
                            &width);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(width) failed.");

    // block_size
    status = CECL_SET_KERNEL_ARG(kernelTranspose,
                            5,
                            sizeof(cl_int),
                            &blockSize);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(blockSize) failed.");

    // group dimensions for transpose kernel
    globalThreadsT[0] = height;
    globalThreadsT[1] = width;

    // Enqueue final Transpose Kernel
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 kernelTranspose,
                 2,
                 NULL,
                 globalThreadsT,
                 localThreadsT,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL() failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = waitForEventAndRelease(&ndrEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

    // Enqueue read output buffer to outputImageData
    cl_event readEvt;
    status = CECL_READ_BUFFER(commandQueue,
                                 outputImageBuffer,
                                 CL_FALSE,
                                 0,
                                 width * height * sizeof(cl_uchar4),
                                 outputImageData,
                                 0,
                                 NULL,
                                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(outputImageBuffer) failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = waitForEventAndRelease(&readEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(readEvt) Failed");

    return SDK_SUCCESS;
}



int
RecursiveGaussian::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        error("Memory Allocation error.\n");
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

int
RecursiveGaussian::setup()
{
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    std::cout << "Searching for input Image at following location : " <<
              filePath << std::endl;
    int status = readInputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Read Input Image Failed");

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

int
RecursiveGaussian::run()
{
    int status = 0;
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    // create and initialize timers
    std::cout << "Executing kernel for " <<
              iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if (runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    std::string filePath = std::string(OUTPUT_IMAGE);
    status = writeOutputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Write Output Image Failed");

    return SDK_SUCCESS;
}

int
RecursiveGaussian::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernelRecursiveGaussian);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel(kernelRecursiveGaussian) failed.");

    status = clReleaseKernel(kernelTranspose);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel(kernelTranspose) failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(inputImageBuffer) failed.");

    status = clReleaseMemObject(outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(outputImageBuffer) failed.");

    status = clReleaseMemObject(tempImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(tempImageBuffer) failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)

    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationInput);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

void
RecursiveGaussian::recursiveGaussianCPU(cl_uchar4* input, cl_uchar4* output,
                                        const int width, const int height,
                                        const float a0, const float a1,
                                        const float a2, const float a3,
                                        const float b1, const float b2,
                                        const float coefp, const float coefn)
{

    // outer loop over all columns within image
    for (int X = 0; X < width; X++)
    {
        // start forward filter pass
        float xp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous input
        float yp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output
        float yb[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output by 2

        float xc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int Y = 0; Y < height; Y++)
        {
            // output position to write
            int pos = Y * width + X;

            // convert input element to float4
            xc[0] = input[pos].s[0];
            xc[1] = input[pos].s[1];
            xc[2] = input[pos].s[2];
            xc[3] = input[pos].s[3];

            yc[0] = (a0 * xc[0]) + (a1 * xp[0]) - (b1 * yp[0]) - (b2 * yb[0]);
            yc[1] = (a0 * xc[1]) + (a1 * xp[1]) - (b1 * yp[1]) - (b2 * yb[1]);
            yc[2] = (a0 * xc[2]) + (a1 * xp[2]) - (b1 * yp[2]) - (b2 * yb[2]);
            yc[3] = (a0 * xc[3]) + (a1 * xp[3]) - (b1 * yp[3]) - (b2 * yb[3]);

            // convert float4 element to output
            output[pos].s[0] = (cl_uchar)yc[0];
            output[pos].s[1] = (cl_uchar)yc[1];
            output[pos].s[2] = (cl_uchar)yc[2];
            output[pos].s[3] = (cl_uchar)yc[3];

            for (int i = 0; i < 4; i++)
            {
                xp[i] = xc[i];
                yb[i] = yp[i];
                yp[i] = yc[i];
            }
        }

        // start reverse filter pass: ensures response is symmetrical
        float xn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float xa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ya[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        float fTemp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int Y = height - 1; Y > -1; Y--)
        {
            int pos = Y * width + X;

            // convert uchar4 to float4
            xc[0] = input[pos].s[0];
            xc[1] = input[pos].s[1];
            xc[2] = input[pos].s[2];
            xc[3] = input[pos].s[3];

            yc[0] = (a2 * xn[0]) + (a3 * xa[0]) - (b1 * yn[0]) - (b2 * ya[0]);
            yc[1] = (a2 * xn[1]) + (a3 * xa[1]) - (b1 * yn[1]) - (b2 * ya[1]);
            yc[2] = (a2 * xn[2]) + (a3 * xa[2]) - (b1 * yn[2]) - (b2 * ya[2]);
            yc[3] = (a2 * xn[3]) + (a3 * xa[3]) - (b1 * yn[3]) - (b2 * ya[3]);

            for (int i = 0; i< 4; i++)
            {
                xa[i] = xn[i];
                xn[i] = xc[i];
                ya[i] = yn[i];
                yn[i] = yc[i];
            }

            // convert uhcar4 to float4
            fTemp[0] = output[pos].s[0];
            fTemp[1] = output[pos].s[1];
            fTemp[2] = output[pos].s[2];
            fTemp[3] = output[pos].s[3];

            fTemp[0] += yc[0];
            fTemp[1] += yc[1];
            fTemp[2] += yc[2];
            fTemp[3] += yc[3];

            // convert float4 to uchar4
            output[pos].s[0] = (cl_uchar)fTemp[0];
            output[pos].s[1] = (cl_uchar)fTemp[1];
            output[pos].s[2] = (cl_uchar)fTemp[2];
            output[pos].s[3] = (cl_uchar)fTemp[3];
        }
    }

}

void
RecursiveGaussian::transposeCPU(cl_uchar4* input,
                                cl_uchar4* output,
                                const int width,
                                const int height)
{
    // transpose matrix
    for(int Y = 0; Y < height; Y++)
    {
        for(int X = 0; X < width; X++)
        {
            output[Y + X * height] = input[X + Y * width];
        }
    }
}

void
RecursiveGaussian::recursiveGaussianCPUReference()
{

    // Create a temp uchar4 array
    cl_uchar4* temp = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    if(temp == NULL)
    {
        error("Failed to allocate host memory! (temp)");
        return;
    }

    // Call recursive Gaussian CPU
    recursiveGaussianCPU(verificationInput, temp, width, height,
                         oclGP.a0, oclGP.a1, oclGP.a2, oclGP.a3,
                         oclGP.b1, oclGP.b2, oclGP.coefp, oclGP.coefn);

    // Transpose the temp buffer
    transposeCPU(temp, verificationOutput, width, height);

    // again Call recursive Gaussian CPU
    recursiveGaussianCPU(verificationOutput, temp, height, width,
                         oclGP.a0, oclGP.a1, oclGP.a2, oclGP.a3,
                         oclGP.b1, oclGP.b2, oclGP.coefp, oclGP.coefn);

    // Do a final Transpose
    transposeCPU(temp, verificationOutput, height, width);

    if(temp)
    {
        free(temp);
    }

}

// convert uchar4 data to uint
unsigned int rgbaUchar4ToUint(const cl_uchar4 rgba)
{
    unsigned int uiPackedRGBA = 0U;
    uiPackedRGBA |= 0x000000FF & (unsigned int)rgba.s[0];
    uiPackedRGBA |= 0x0000FF00 & (((unsigned int)rgba.s[1]) << 8);
    uiPackedRGBA |= 0x00FF0000 & (((unsigned int)rgba.s[2]) << 16);
    uiPackedRGBA |= 0xFF000000 & (((unsigned int)rgba.s[3]) << 24);
    return uiPackedRGBA;
}


int
RecursiveGaussian::verifyResults()
{

    if(sampleArgs->verify)
    {
        recursiveGaussianCPUReference();

        float *outputDevice = new float[width * height * 4];
        CHECK_ALLOCATION(outputDevice,
                         "Failed to allocate host" "memory! (outputDevice)");

        float *outputReference = new float[width * height * 4];
        CHECK_ALLOCATION(outputReference,
                         "Failed to allocate host" "memory! (outputReference)");

        int m = 0;

        // copy uchar4 data to float array
        for(int i=0; i < (int)(width * height); i++)
        {
            outputDevice[4 * i + 0] = outputImageData[i].s[0];
            outputDevice[4 * i + 1] = outputImageData[i].s[1];
            outputDevice[4 * i + 2] = outputImageData[i].s[2];
            outputDevice[4 * i + 3] = outputImageData[i].s[3];

            outputReference[4 * i + 0] = verificationOutput[i].s[0];
            outputReference[4 * i + 1] = verificationOutput[i].s[1];
            outputReference[4 * i + 2] = verificationOutput[i].s[2];
            outputReference[4 * i + 3] = verificationOutput[i].s[3];
        }


        // compare the results and see if they match
        if(compare(outputReference,
                   outputDevice,
                   width * height,
                   (float)0.0001))
        {
            std::cout <<"Passed!\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
RecursiveGaussian::printStats()
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

        stats[0]  = toString(width, std::dec);
        stats[1]  = toString(height, std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}


int
main(int argc, char * argv[])
{

    RecursiveGaussian clRecursiveGaussian;

    if (clRecursiveGaussian.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRecursiveGaussian.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clRecursiveGaussian.sampleArgs->isDumpBinaryEnabled())
    {
        return clRecursiveGaussian.genBinaryImage();
    }

    if (clRecursiveGaussian.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRecursiveGaussian.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRecursiveGaussian.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clRecursiveGaussian.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clRecursiveGaussian.printStats();

    return SDK_SUCCESS;
}