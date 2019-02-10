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


#include "SimpleConvolution.hpp"

int SimpleConvolution::setupSimpleConvolution()
{
    cl_uint inputSizeBytes;

	if(maskWidth != 3 && maskWidth != 5)
	{
		std::cout << "Mask width should be either 3 or 5" << std::endl;
		return SDK_EXPECTED_FAILURE;
	}

	// initialisation of mask 
	if(maskWidth == 3)
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

    if(width * height < 256)
    {
        width = 64;
        height = 64;
    }

    // allocate and init memory used by host
    inputSizeBytes = width * height * sizeof(cl_uint);
    input  = (cl_uint *) malloc(inputSizeBytes);
    CHECK_ALLOCATION(input, "Failed to allocate host memory. (input)");
	output = (cl_int *) malloc(width*height*sizeof(cl_int));
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");
	outputSep = (cl_int  *) malloc(width*height*sizeof(cl_int));
    CHECK_ALLOCATION(outputSep, "Failed to allocate host memory. (outputSep)");

	// random initialisation of input1
    fillRandom<cl_uint >(input, width, height, 0, 255);

	// allocate and initalize memory for padded input data to host
	filterRadius = filterSize/2;
	paddedHeight = height + (filterRadius*2);
	paddedWidth = width + (filterRadius*2);
	int paddedSize = paddedHeight*paddedWidth;

	paddedInput = (cl_uint *)calloc(paddedWidth*paddedHeight, sizeof(cl_uint));
	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++) {
			paddedInput[(i+filterRadius)*paddedWidth + (j+filterRadius)] = input[i*width+j];
		}

	tmpOutput = (cl_float  *)calloc(width*paddedHeight, sizeof(cl_float));
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (tmpOutput)");

    return SDK_SUCCESS;
}


int
SimpleConvolution::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("SimpleConvolution_Kernels.cl");
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
SimpleConvolution::setupCL(void)
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

	if (localSize > deviceInfo.maxWorkGroupSize)
	{
		std::cout << std::endl;
		std::cout << "Group Size specified is greater than device limit " << std::endl;
		std::cout << "Resetting group size to " << deviceInfo.maxWorkGroupSize << std::endl;
		std::cout << std::endl;
		localSize = deviceInfo.maxWorkGroupSize;
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

	// Create Input buffer on device
    inputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_ONLY,
					  sizeof(cl_uint ) * paddedHeight * paddedWidth,
					  NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (inputBuffer1)");

	// Send the padded input data to device
	status = CECL_WRITE_BUFFER(commandQueue, inputBuffer,
						CL_TRUE, 0, paddedHeight*paddedWidth*sizeof(cl_uint), 
						paddedInput, 0, NULL, NULL);
    CHECK_OPENCL_ERROR(status, "Error in CECL_WRITE_BUFFER. (inputBuffer1)");

	// Create a temporary output buffer on device
	tmpOutputBuffer = CECL_BUFFER(
                      context,
                      CL_MEM_READ_WRITE,
					  sizeof(cl_float) * paddedHeight * width,
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (tmpOutputBuffer)");

	// Create a Non-Separable Output buffer on device
	outputBuffer = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY,
					   sizeof(cl_int) * height * width,  
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR( status,  "CECL_BUFFER failed. (outputBuffer)");

	// Create a Separable Output buffer on device
    outputBufferSep = CECL_BUFFER(
                       context,
                       CL_MEM_WRITE_ONLY,
					   sizeof(cl_int ) * height * width,
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR( status,  "CECL_BUFFER failed. (outputBufferSep)");

	// Create a mask buffer on device
    maskBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(cl_float ) * maskWidth * maskHeight,
                     mask,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (maskBuffer)");

	// Create a row-wise filter buffer on device
	rowFilterBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(cl_float ) * filterSize,
                     rowFilter,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (rowFilterBuffer)");

	// Create a column-wise filter buffer on device
	colFilterBuffer = CECL_BUFFER(
                     context,
                     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(cl_float ) * filterSize,
                     colFilter,
                     &status);
    CHECK_OPENCL_ERROR( status, "CECL_BUFFER failed. (colFilterBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("SimpleConvolution_Kernels.cl");
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

    // get a kernel object handle for a nonSeparable convolution
    nonSeparablekernel = CECL_KERNEL(program, "simpleNonSeparableConvolution", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed (nonSeparablekernel).");

	// get a kernel object handle for first pass Separable convolution
    separablekernelPass1 = CECL_KERNEL(program, "simpleSeparableConvolutionPass1", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed (separablekernelPass1).");

	// get a kernel object handle for second pass Separable convolution
    separablekernelPass2 = CECL_KERNEL(program, "simpleSeparableConvolutionPass2", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed (separablekernelPass2).");

    return SDK_SUCCESS;
}

int
SimpleConvolution::runNonSeparableCLKernels(void)
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

    cl_uint2 inputDimensions = {width, height};
    cl_uint2 maskDimensions  = {maskWidth, maskHeight};

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 3,
                 sizeof(cl_uint2),
                 (void *)&inputDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputDimensions)");

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 4,
                 sizeof(cl_uint2),
                 (void *)&maskDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (maskDimensions)");

    status = CECL_SET_KERNEL_ARG(
                 nonSeparablekernel,
                 5,
                 sizeof(cl_uint),
				 (void *)&paddedWidth);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (paddedWidth)");

	//Set global and local work-group size, global work-group size should be multiple of local work-group size
	localThreads[0] = localSize;
	globalThreads[0] = (width*height + localThreads[0] - 1) / localThreads[0];
    globalThreads[0] *= localThreads[0];

	// Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 nonSeparablekernel,
                 1,
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
SimpleConvolution::runSeparableCLKernels(void)
{
    cl_int   status;
    cl_event events[1];

	// Run first pass filter
    // Set appropriate arguments to the kernel
    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer); 
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 1,
                 sizeof(cl_mem),
                 (void *)&rowFilterBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (rowFilterBuffer)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 2,
                 sizeof(cl_mem),
                 (void *)&tmpOutputBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (tmpOutputBuffer)");

    cl_uint2 inputDimensions = {width, height};

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 3,
                 sizeof(cl_uint2),
                 (void *)&inputDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputDimensions)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 4,
                 sizeof(cl_uint),
                 (void *)&filterSize);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (filterSize)");

	cl_uint2 paddedInputDimensions = {paddedWidth, paddedHeight};
	status = CECL_SET_KERNEL_ARG(
                 separablekernelPass1,
                 5,
                 sizeof(cl_uint2),
				 (void *)&paddedInputDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (paddedInputDimensions)");

	// Setting global work-group for pass1
	size_t globalSizePass1 = (width*paddedHeight);
	localThreads[0] = localSize;
	globalThreads[0] = (globalSizePass1 + localThreads[0] - 1)/localThreads[0];
	globalThreads[0] *= localThreads[0];

    // Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 separablekernelPass1,
                 1,
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

	// Run Second pass filter
	// Set appropriate arguments to the separablekernelPass2 kernel
    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 0,
                 sizeof(cl_mem),
                 (void *)&tmpOutputBuffer); /*tmpOutputBuffer is a input buffer for second kernel*/
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (tmpOutputBuffer)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 1,
                 sizeof(cl_mem),
                 (void *)&colFilterBuffer);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (colFilterBuffer)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 2,
                 sizeof(cl_mem),
                 (void *)&outputBufferSep);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (outputBufferSep)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 3,
                 sizeof(cl_uint2),
                 (void *)&inputDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (inputDimensions)");

    status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 4,
                 sizeof(cl_uint),
                 (void *)&filterSize);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (filterSize)");

	status = CECL_SET_KERNEL_ARG(
                 separablekernelPass2,
                 5,
                 sizeof(cl_uint2),
				 (void *)&paddedInputDimensions);
    CHECK_OPENCL_ERROR( status, "CECL_SET_KERNEL_ARG failed. (paddedInputDimensions)");

	// Setting global work-group size for pass2
	size_t globalSizePass2 = (width*height);
	globalThreads[0] = (globalSizePass2 + localThreads[0] - 1)/localThreads[0];
	globalThreads[0] *= localThreads[0];

    // Enqueue a kernel run call.
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 separablekernelPass2,
                 1,
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
 * Reference CPU implementation of Simple Convolution
 * for performance comparison
 */
void
SimpleConvolution::CPUReference()
{	
    for(cl_int y = 0; y < height; y++)
		for(cl_int x = 0; x < width; x++)
        {
            cl_float sum = 0.0f;
			for(cl_uint m = 0; m < filterSize; m++)
			{
				for(cl_uint n = 0; n < filterSize; n++)
				{
					cl_uint maskIndex = m*filterSize+n;
					cl_uint inputIndex = (y+m)*paddedWidth + (x+n);

					// applying convolution operation
					sum += (cl_float)(paddedInput[inputIndex]) * (mask[maskIndex]);
				}
			}
			sum += 0.5f;
			verificationOutput[(y*width + x)] = (cl_int)sum;
        }
}

int SimpleConvolution::initialize()
{
    // Call base class Initialize to get default configuration
    if  (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Now add customized options
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

    Option* mask_width = new Option;
    CHECK_ALLOCATION(mask_width, "Memory allocation error.\n");
    maskWidth = 3;
    mask_width->_sVersion = "m";
    mask_width->_lVersion = "masksize";
    mask_width->_description = "Width of the mask matrix";
    mask_width->_type = CA_ARG_INT;
    mask_width->_value = &maskWidth;
    sampleArgs->AddOption(mask_width);
    delete mask_width;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");
    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;
    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

	Option* group_size = new Option;
    CHECK_ALLOCATION(group_size, "Memory allocation error.\n");
    group_size->_sVersion = "l";
    group_size->_lVersion = "localSize";
    group_size->_description = "Size of work-group";
    group_size->_type = CA_ARG_INT;
    group_size->_value = &localSize;
    sampleArgs->AddOption(group_size);
    delete group_size;

    return SDK_SUCCESS;
}

int SimpleConvolution::setup()
{
    if(!isPowerOf2(width))
    {
        width = roundToPowerOf2(width);
    }
    if(!isPowerOf2(height))
    {
        height = roundToPowerOf2(height);
    }

    filterSize = maskHeight = maskWidth;

    if(!(maskWidth%2))
    {
        maskWidth++;
    }
    if(!(maskHeight%2))
    {
        maskHeight++;
    }

	int status = setupSimpleConvolution();
    if (status != SDK_SUCCESS)
    {
        return status;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if (setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);

    setupTime = (cl_double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int SimpleConvolution::run()
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
						width * height * sizeof(cl_int),
						output,
						0,
						NULL,
						NULL);
		CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER failed.");

		// run separable version implementation of convolution
		if (runSeparableCLKernels() != SDK_SUCCESS)
		{
			 return SDK_FAILURE;
		}

		// Enqueue readBuffer for separable filter
		status = CECL_READ_BUFFER(
					 commandQueue,
					 outputBufferSep,
					 CL_TRUE,
					 0,
					 width * height * sizeof(cl_int),
					 outputSep,
					 0,
					 NULL,
					 NULL);
		CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER failed.");
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
					width * height * sizeof(cl_int),
					output,
					0,
					NULL,
					NULL);
	CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER failed.");
	
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
					outputBufferSep,
					CL_TRUE,
					0,
					width * height * sizeof(cl_int),
					outputSep,
					0,
					NULL,
					NULL);
	CHECK_OPENCL_ERROR( status, "CECL_READ_BUFFER failed.");

    return SDK_SUCCESS;
}

int SimpleConvolution::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationOutput = (cl_int *) malloc(width * height * sizeof(cl_int));
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");

        /*
         * reference implementation
         */
        CPUReference();

		std::cout << "Verifying non-Separable Convolution Kernel result - ";
        // compare the results and see if they match
        if(memcmp(output, verificationOutput, height*width*sizeof(cl_int)) == 0)
        {
            std::cout<<"Passed!\n" << std::endl;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }

		std::cout << "Verifying Separable Convolution Kernel result - ";
        // compare the results and see if they match
        if(memcmp(outputSep, verificationOutput, height*width*sizeof(cl_int)) == 0)
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

void SimpleConvolution::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Width", "Height", "mask Size", "KernelTime(sec)"};
        std::string stats[4];

		std::cout << "\n Non-Separable Filter Timing Measurement!" << std::endl;
        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
		stats[2] = toString(maskWidth, std::dec);
        stats[3] = toString(totalNonSeparableKernelTime, std::dec);
        printStatistics(strArray, stats, 4);

		std::cout << "\n Separable Filter Timing Measurement!" << std::endl;
        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
		stats[2] = toString(maskWidth, std::dec);
        stats[3] = toString(totalSeparableKernelTime, std::dec);
        printStatistics(strArray, stats, 4);
    }
}

int SimpleConvolution::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(nonSeparablekernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(nonSeparablekernel)");

	status = clReleaseKernel(separablekernelPass1);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(separablekernelPass1)");

	status = clReleaseKernel(separablekernelPass2);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(separablekernelPass2)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer)");

	status = clReleaseMemObject(tmpOutputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(tmpOutputBuffer)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

    status = clReleaseMemObject(maskBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(maskBuffer)");

	status = clReleaseMemObject(rowFilterBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(rowFilterBuffer)");

	status = clReleaseMemObject(colFilterBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(colFilterBuffer)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)
    FREE(input);
	FREE(paddedInput);
    FREE(output);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    SimpleConvolution clSimpleConvolution;

    if (clSimpleConvolution.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleConvolution.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSimpleConvolution.sampleArgs->isDumpBinaryEnabled())
    {
        return clSimpleConvolution.genBinaryImage();
    }

	int status = clSimpleConvolution.setup();
    if (status != SDK_SUCCESS)
    {
        return status;
    }

    if (clSimpleConvolution.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleConvolution.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clSimpleConvolution.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
	clSimpleConvolution.printStats();
    return SDK_SUCCESS;
}
