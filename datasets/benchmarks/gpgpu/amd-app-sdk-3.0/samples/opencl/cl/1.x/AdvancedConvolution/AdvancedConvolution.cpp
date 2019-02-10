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


#include "AdvancedConvolution.hpp"

int
AdvancedConvolution::readInputImage(std::string inputImageName)
{
	// Check whether isLds is zero or one 
	if(useLDSPass1 != 0 && useLDSPass1 != 1)
	{
		std::cout << "isLds should be either 0 or 1" << std::endl;
		return SDK_EXPECTED_FAILURE;
	}

	// initialization of mask 
	if(filterSize != 3 && filterSize != 5)
	{
		std::cout << "Filter Size should be either 3 or 5" << std::endl;
		return SDK_EXPECTED_FAILURE;
	}

    if (filterType !=0 && filterType != 1 && filterType !=2)
    {
        std::cout << "Filter Type can only be 0, 1 or 2 for Sobel, Box and Gaussian filters respectively." << std::endl;
		return SDK_EXPECTED_FAILURE;
    }

    switch (filterType)
    {
    case 0: /* Sobel Filter */
        if(filterSize == 3)
	    {
		    mask = SOBEL_FILTER_3x3;
		    rowFilter = SOBEL_FILTER_3x3_pass1;
		    colFilter = SOBEL_FILTER_3x3_pass2;
	    }
	    else
	    {
		    mask = SOBEL_FILTER_5x5;
		    rowFilter = SOBEL_FILTER_5x5_pass1;
		    colFilter = SOBEL_FILTER_5x5_pass2;
	    }
        break;

    case 1: /* Box Filter */
        if(filterSize == 3)
	    {
		    mask = BOX_FILTER_3x3;
		    rowFilter = BOX_FILTER_3x3_pass1;
		    colFilter = BOX_FILTER_3x3_pass2;
	    }
	    else
	    {
		    mask = BOX_FILTER_5x5;
		    rowFilter = BOX_FILTER_5x5_pass1;
		    colFilter = BOX_FILTER_5x5_pass2;
	    }
        break;

    case 2: /* Gaussian Filter */
        if(filterSize == 3)
	    {
		    mask = GAUSSIAN_FILTER_3x3;
		    rowFilter = GAUSSIAN_FILTER_3x3_pass1;
		    colFilter = GAUSSIAN_FILTER_3x3_pass2;
	    }
	    else
	    {
		    mask = GAUSSIAN_FILTER_5x5;
		    rowFilter = GAUSSIAN_FILTER_5x5_pass1;
		    colFilter = GAUSSIAN_FILTER_5x5_pass2;
	    }
        break;
    }
	

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

    // allocate memory for input image data to host
	inputImage2D = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(inputImage2D,"Failed to allocate memory! (inputImage2D)");
	
	// get the pointer to pixel data
    pixelData = inputBitmap.getPixels();
    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData2D
    memcpy(inputImage2D, pixelData, width * height * pixelSize);

	// allocate and initalize memory for padded input image data to host
	filterRadius = filterSize - 1;
	paddedHeight = height + filterRadius;
	paddedWidth = width + filterRadius;

    paddedInputImage2D = (cl_uchar4*)malloc(paddedWidth * paddedHeight * sizeof(cl_uchar4));
    CHECK_ALLOCATION(paddedInputImage2D,"Failed to allocate memory! (paddedInputImage2D)");
	memset(paddedInputImage2D, 0, paddedHeight*paddedWidth*sizeof(cl_uchar4));
	for(cl_uint i = filterRadius; i < height + filterRadius; i++)
	{
		for(cl_uint j = filterRadius; j < width + filterRadius; j++)
		{
			paddedInputImage2D[i * paddedWidth + j] = inputImage2D[(i - filterRadius) * width + (j - filterRadius)];		
		}
	}

	// allocate memory for output image data for Non-Separable Filter to host
    nonSepOutputImage2D = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(nonSepOutputImage2D,"Failed to allocate memory! (nonSepOutputImage2D)");
	memset(nonSepOutputImage2D, 0, width * height * pixelSize);

	// allocate memory for output image data for Separable Filter to host
	sepOutputImage2D = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(sepOutputImage2D,"Failed to allocate memory! (sepOutputImage2D)");
	memset(sepOutputImage2D, 0, width * height * pixelSize);
	
	// allocate memory for verification output
	nonSepVerificationOutput = (cl_uchar*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(nonSepVerificationOutput,"Failed to allocate memory! (verificationOutput)");

    sepVerificationOutput = (cl_uchar*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(sepVerificationOutput,"Failed to allocate memory! (verificationOutput)");

	memset(nonSepVerificationOutput, 0, width * height * pixelSize);
    memset(sepVerificationOutput, 0, width * height * pixelSize);

	// set local work-group size
	localThreads[0] = blockSizeX; 
	localThreads[1] = blockSizeY;

	// set global work-group size, padding work-items do not need to be considered
	globalThreads[0] = (width + localThreads[0] - 1) / localThreads[0];
    globalThreads[0] *= localThreads[0];
    globalThreads[1] = (height + localThreads[1] - 1) / localThreads[1];
    globalThreads[1] *= localThreads[1];

    return SDK_SUCCESS;
}

int
AdvancedConvolution::writeOutputImage(std::string outputImageName, cl_uchar4 *outputImageData)
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
AdvancedConvolution::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("AdvancedConvolution_Kernels.cl");
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
AdvancedConvolution::setupCL(void)
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

	retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
	CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

	if (localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
		localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
		(localThreads[0] * localThreads[1]) > deviceInfo.maxWorkGroupSize)
	{
		std::cout << "Unsupported: Device does not support requested"
			<< ":number of work items.";
		return SDK_EXPECTED_FAILURE;
	}

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

    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
					  pixelSize * paddedWidth * paddedHeight,
					  paddedInputImage2D,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer)");

	outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY,
					   pixelSize * width * height,
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR( status,  "CECL_BUFFER failed. (outputBuffer)");

	maskBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
					 sizeof(cl_float ) * filterSize * filterSize,
                     mask,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (maskBuffer)");

	rowFilterBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(cl_float ) * filterSize,
                     rowFilter,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (rowFilterBuffer)");

	colFilterBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(cl_float ) * filterSize,
                     colFilter,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (colFilterBuffer)");

    // create a CL program using the kernel source
	char option[256];
    sprintf(option, "-DFILTERSIZE=%d -DLOCAL_XRES=%d -DLOCAL_YRES=%d -DUSE_LDS=%d",
                    filterSize, LOCAL_XRES, LOCAL_YRES, useLDSPass1);

    buildProgramData buildData;
    buildData.kernelName = std::string("AdvancedConvolution_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string(option);
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

    // get a kernel object handle for a Non-Separable Filter
    nonSeparablekernel = CECL_KERNEL(program, "advancedNonSeparableConvolution", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed (advancedNonSeparableConvolution).");

	// get a kernel object handle for Separable Filter
    separablekernel = CECL_KERNEL(program, "advancedSeparableConvolution", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed (advancedSeparableConvolution).");

    return SDK_SUCCESS;
}

int
AdvancedConvolution::runNonSeparableCLKernels(void)
{
    cl_int   status;
    cl_event events[1];

    // Set appropriate arguments to the kernel
    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 1,
                 sizeof(cl_mem),
                 (void *)&maskBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (maskBuffer)");

	status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 2,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 3,
                 sizeof(cl_uint),
				 (void *)&width);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (width)");

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 4,
                 sizeof(cl_uint),
				 (void *)&height);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (height)");

	 status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 5,
                 sizeof(cl_uint),
				 (void *)&paddedWidth);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (paddedWidth)");

    // Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 nonSeparablekernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    CHECK_OPENCL_ERROR( status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status,"clFlush() failed");

    status = waitForEventAndRelease(&events[0]);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(events[0]) Failed");
    

    return SDK_SUCCESS;
}

int
AdvancedConvolution::runSeparableCLKernels(void)
{
    cl_int   status;
    cl_event events[1];

    // Set appropriate arguments to the kernel
    status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 1,
                 sizeof(cl_mem),
                 (void *)&rowFilterBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (rowFilterBuffer)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 2,
                 sizeof(cl_mem),
                 (void *)&colFilterBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (colFilterBuffer)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 3,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 4,
                 sizeof(cl_uint),
                 (void *)&width);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (width)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 5,
                 sizeof(cl_uint),
				 (void *)&height);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (height)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernel,
                 6,
                 sizeof(cl_uint),
				 (void *)&paddedWidth);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (paddedWidth)");

    // Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 separablekernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    CHECK_OPENCL_ERROR( status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status,"clFlush() failed");

    status = waitForEventAndRelease(&events[0]);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(events[0]) Failed");

	return SDK_SUCCESS;
}

/**
 * Reference CPU implementation of Advanced Convolution kernel
 * for performance comparison
 */
void
AdvancedConvolution::CPUReference()
{	
    for(cl_uint i = 0; i < height; ++i)
    {
        for(cl_uint j = 0; j < width; ++j)
        {
			cl_float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
			for(cl_uint m = 0; m < filterSize; m++)
			{
				for(cl_uint n = 0; n < filterSize; n++)
				{
					cl_uint maskIndex = m*filterSize+n;
					cl_uint inputIndex = (i+m)*paddedWidth + (j+n);

					// copy uchar4 data to float4
					sum.s[0] += (cl_float)(paddedInputImage2D[inputIndex].s[0]) * (mask[maskIndex]);
					sum.s[1] += (cl_float)(paddedInputImage2D[inputIndex].s[1]) * (mask[maskIndex]);
					sum.s[2] += (cl_float)(paddedInputImage2D[inputIndex].s[2]) * (mask[maskIndex]);
					sum.s[3] += (cl_float)(paddedInputImage2D[inputIndex].s[3]) * (mask[maskIndex]);
				}
			}

			// calculating cpu reference for advanced convolution kernel
			nonSepVerificationOutput[((i*width + j) * 4) + 0] = (cl_uchar)((sum.s[0] < 0) ? 0 : ((sum.s[0] > 255.0) ? 255 : sum.s[0]));
			nonSepVerificationOutput[((i*width + j) * 4) + 1] = (cl_uchar)((sum.s[1] < 0) ? 0 : ((sum.s[1] > 255.0) ? 255 : sum.s[1]));
			nonSepVerificationOutput[((i*width + j) * 4) + 2] = (cl_uchar)((sum.s[2] < 0) ? 0 : ((sum.s[2] > 255.0) ? 255 : sum.s[2]));
			nonSepVerificationOutput[((i*width + j) * 4) + 3] = (cl_uchar)((sum.s[3] < 0) ? 0 : ((sum.s[3] > 255.0) ? 255 : sum.s[3]));
		}
	}	
}

int AdvancedConvolution::initialize()
{
    // Call base class Initialize to get default configuration
    if  (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Now add customized options
	Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");
    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;
    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

	Option* use_lds = new Option;
    CHECK_ALLOCATION(use_lds, "Memory allocation error.\n");
	useLDSPass1 = 1;
    use_lds->_sVersion = "l";
    use_lds->_lVersion = "useLDSPass1";
    use_lds->_description = "Use LDS for Pass1 of Separable Filter";
    use_lds->_type = CA_ARG_INT;
    use_lds->_value = &useLDSPass1;
    sampleArgs->AddOption(use_lds);
    delete use_lds;

    Option* mask_width = new Option;
    CHECK_ALLOCATION(mask_width, "Memory allocation error.\n");
	filterSize = 3;
    mask_width->_sVersion = "m";
    mask_width->_lVersion = "Filter Size";
    mask_width->_description = "Dimension of Convolution Filter - Supported values: 3 and 5";
    mask_width->_type = CA_ARG_INT;
    mask_width->_value = &filterSize;
    sampleArgs->AddOption(mask_width);
    delete mask_width;

    Option* filter_type = new Option;
    CHECK_ALLOCATION(filter_type, "Memory allocation error.\n");
	filterType = 0;
    filter_type->_sVersion = "f";
    filter_type->_lVersion = "Filter Type";
    filter_type->_description = "Filter Type- 0: Sobel Filter, 1: Box Filter, 2: Gaussian Filter";
    filter_type->_type = CA_ARG_INT;
    filter_type->_value = &filterType;
    sampleArgs->AddOption(filter_type);
    delete filter_type;

    return SDK_SUCCESS;
}

int AdvancedConvolution::setup()
{
    // Allocate host memory and read input image
	std::string filePath = getPath() + std::string(INPUT_IMAGE);
    int status = readInputImage(filePath);
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

    setupTime = (cl_double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int AdvancedConvolution::run()
{
	int status;

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
		// run non-separable implementation of convolution
		if (runNonSeparableCLKernels() != SDK_SUCCESS)
		{
		     return SDK_FAILURE;
		}

		// Enqueue readBuffer for non-separable filter
		status = CECL_READ_BUFFER(
						commandQueue,
						outputBuffer,
						CL_TRUE,
						0,
						width * height * pixelSize,
						nonSepOutputImage2D,
						0,
						NULL,
						NULL);
		CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER(nonSepOutputImage2D) failed.");

		// run separable version implementation of convolution
		if (runSeparableCLKernels() != SDK_SUCCESS)
		{
			 return SDK_FAILURE;
		}

		// Enqueue readBuffer for separable filter
		status = CECL_READ_BUFFER(
					 commandQueue,
					 outputBuffer,
					 CL_TRUE,
					 0,
					 width * height * pixelSize,
					 sepOutputImage2D,
					 0,
					 NULL,
					 NULL);
		CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER (sepOutputImage2D) failed.");
    }

	std::cout << "Executing kernel for " << iterations <<
              " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

	// create and initialize timers
    int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

	// running non-separable filter
	for(int i = 0; i < iterations; i++)
    {
        status = runNonSeparableCLKernels();
        CHECK_ERROR(status, SDK_SUCCESS, "OpenCL run Kernel failed for Separable Filter");
    }

    sampleTimer->stopTimer(timer);
    totalNonSeparableKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

	// Enqueue readBuffer for non-separable filter
	status = CECL_READ_BUFFER(
					commandQueue,
					outputBuffer,
					CL_TRUE,
					0,
					width * height * pixelSize,
					nonSepOutputImage2D,
					0,
					NULL,
					NULL);
	CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER(nonSepOutputImage2D) failed.");

	// write the non-separable filter output image to bitmap file
    status = writeOutputImage(OUTPUT_IMAGE_NON_SEPARABLE, nonSepOutputImage2D);
    CHECK_ERROR(status, SDK_SUCCESS, "non-Separable Filter Output Image Failed");

	sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

	// running non-separable filter
	for(int i = 0; i < iterations; i++)
    {
		status = runSeparableCLKernels();
        CHECK_ERROR(status, SDK_SUCCESS, "OpenCL run Kernel failed for Non-Separable Filter");
	}

	sampleTimer->stopTimer(timer);
	totalSeparableKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

	// Enqueue readBuffer for separable filter
	status = CECL_READ_BUFFER(
					commandQueue,
					outputBuffer,
					CL_TRUE,
					0,
					width * height * pixelSize,
					sepOutputImage2D,
					0,
					NULL,
					NULL);
	CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER (sepOutputImage2D) failed.");
	
	// write the Separable filter output image to bitmap file
    status = writeOutputImage(OUTPUT_IMAGE_SEPARABLE, sepOutputImage2D);
    CHECK_ERROR(status, SDK_SUCCESS, "Separable Filter Output Image Failed");

    return SDK_SUCCESS;
}

int AdvancedConvolution::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
		* Reference implementation on host device
		*/
		CPUReference();

		float *outputDevice = new float[width * height * pixelSize];
        CHECK_ALLOCATION(outputDevice,"Failed to allocate host memory! (outputDevice)");
	
		float *outputReference = new float[width * height * pixelSize];
		CHECK_ALLOCATION(outputReference, "Failed to allocate host memory! (outputReference)");

		std::cout << "Verifying advanced non-Separable Convolution Kernel result - ";

		for(int i = 0; i < (int)(width * height); i++)
		{
			// copy uchar data to float array from verificationConvolutionOutput
			outputReference[i * 4 + 0] = nonSepVerificationOutput[i * 4 + 0];
			outputReference[i * 4 + 1] = nonSepVerificationOutput[i * 4 + 1];
			outputReference[i * 4 + 2] = nonSepVerificationOutput[i * 4 + 2];
			outputReference[i * 4 + 3] = nonSepVerificationOutput[i * 4 + 3];

			// copy uchar data to float array from global kernel
			outputDevice[i * 4 + 0] = nonSepOutputImage2D[i].s[0];
			outputDevice[i * 4 + 1] = nonSepOutputImage2D[i].s[1];
			outputDevice[i * 4 + 2] = nonSepOutputImage2D[i].s[2];
			outputDevice[i * 4 + 3] = nonSepOutputImage2D[i].s[3];
		}		

		// compare the results and see if they match
        if(compare(outputDevice, outputReference, width * height * 4))
        {
            std::cout << "Passed!\n" << std::endl;
        }
        else
        {
			delete[] outputDevice;
			delete[] outputReference;
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }

		std::cout << "Verifying advanced Separable Convolution Kernel result - ";

		memset(outputDevice, 0, width*height*4);
        for(int i = 0; i < (int)(width * height); i++)
        {
			// copy uchar data to float array from global kernel
            outputDevice[i * 4 + 0] = sepOutputImage2D[i].s[0];
            outputDevice[i * 4 + 1] = sepOutputImage2D[i].s[1];
            outputDevice[i * 4 + 2] = sepOutputImage2D[i].s[2];
            outputDevice[i * 4 + 3] = sepOutputImage2D[i].s[3];
        }

        // compare the results and see if they match
        if(compare(outputDevice, outputReference, width * height * 4))
        {
			delete[] outputDevice;
			delete[] outputReference;
            std::cout << "Passed!\n" << std::endl;
        }
        else
        {
			delete[] outputDevice;
			delete[] outputReference;
            std::cout << "Failed!\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void AdvancedConvolution::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Width", "Height", "Filter Size", "KernelTime(sec)"};
        std::string stats[4];

		std::cout << "\n Non-Separable Filter Timing Measurement!" << std::endl;
        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
		stats[2] = toString(filterSize, std::dec);
        stats[3] = toString(totalNonSeparableKernelTime, std::dec);
        printStatistics(strArray, stats, 4);

		std::cout << "\n Separable Filter Timing Measurement!" << std::endl;
        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
		stats[2] = toString(filterSize, std::dec);
        stats[3] = toString(totalSeparableKernelTime, std::dec);
        printStatistics(strArray, stats, 4);
    }
}

int AdvancedConvolution::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

	if (nonSeparablekernel != NULL)
	{
		status = clReleaseKernel(nonSeparablekernel);
		CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(nonSeparablekernel)");
	}

	if (separablekernel)
	{
		status = clReleaseKernel(separablekernel);
		CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(separablekernl)");
	}

	if (program)
	{
		status = clReleaseProgram(program);
		CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");
	}

	if (inputBuffer)
	{
		status = clReleaseMemObject(inputBuffer);
		CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer)");
	}

	if (outputBuffer)
	{
		status = clReleaseMemObject(outputBuffer);
		CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");
	}

	if (maskBuffer)
	{
		status = clReleaseMemObject(maskBuffer);
		CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(maskBuffer)");
	}

	if (rowFilterBuffer)
	{
		status = clReleaseMemObject(rowFilterBuffer);
		CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(rowFilterBuffer)");
	}

	if(colFilterBuffer)
	{
		status = clReleaseMemObject(colFilterBuffer);
		CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(colFilterBuffer)");
	}

	if (commandQueue)
	{
		status = clReleaseCommandQueue(commandQueue);
		CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");
	}

	if (context)
	{
		status = clReleaseContext(context);
		CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");
	}

    // release program resources (input memory etc.)
	FREE(inputImage2D);
	FREE(paddedInputImage2D);
	FREE(nonSepOutputImage2D);
	FREE(sepOutputImage2D);
    FREE(nonSepVerificationOutput);
    FREE(sepVerificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    AdvancedConvolution clAdvancedConvolution;

    if (clAdvancedConvolution.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clAdvancedConvolution.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clAdvancedConvolution.sampleArgs->isDumpBinaryEnabled())
    {
        return clAdvancedConvolution.genBinaryImage();
    }

	int status = clAdvancedConvolution.setup();
    if (status != SDK_SUCCESS)
    {
		clAdvancedConvolution.cleanup();
        return status;
    }

    if (clAdvancedConvolution.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clAdvancedConvolution.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clAdvancedConvolution.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

	clAdvancedConvolution.printStats();
    return SDK_SUCCESS;
}
