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


#ifndef SIMPLECONVOLUTION_H_
#define SIMPLECONVOLUTION_H_

/**
 * Header Files
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "FilterCoeff.h"

#define GROUP_SIZE 256
#define SAMPLE_VERSION "AMD-APP-SDK-vx.y.z.s"

using namespace appsdk;

/**
 * SimpleConvolution
 * Class implements OpenCL SimpleConvolution sample
 */

class SimpleConvolution
{
        cl_uint      seed;               /**< Seed value for random number generation */
        cl_double    setupTime;          /**< Time for setting up OpenCL */
        cl_double    totalNonSeparableKernelTime;    /**< Time for Non-Separable kernel execution */
		cl_double    totalSeparableKernelTime;		 /**< Time for Separable kernel execution */

        cl_int       width;              /**< Width of the Input array */
        cl_int       height;             /**< Height of the Input array */
		cl_int		 paddedWidth;		 /**< Padded Width of the Input array */
		cl_int		 paddedHeight;		 /**< Padded Height of the Input array */
        cl_uint      *input;			 /**< Input array */
		cl_uint		 *paddedInput;		 /**< Padded Input array */
		cl_float     *tmpOutput;         /**< Temporary Output array to store result of first pass kernel */
        cl_int		*output;             /**< Non-Separable Output array */
		cl_int		*outputSep;          /**< Separable Output array */
        cl_float     *mask;              /**< mask array */
        cl_uint      maskWidth;          /**< mask dimensions */
        cl_uint      maskHeight;         /**< mask dimensions */
		cl_float     *rowFilter;		 /**< Row-wise filter for pass1 */
		cl_float     *colFilter;		 /**< Column-wise filter for pass2 */
		cl_uint		 filterSize;		 /**< FilterSize */
		cl_int		 filterRadius;		 /**< FilterRadius */
        cl_int  	*verificationOutput;/**< Output array for reference implementation */

        cl_context   context;            /**< CL context */
        cl_device_id *devices;           /**< CL device list */
        cl_mem       inputBuffer;        /**< CL memory input buffer */
		cl_mem       tmpOutputBuffer;    /**< CL memory temporary output buffer */
        cl_mem       outputBuffer;       /**< CL memory output buffer for Non-Separable kernel */
		cl_mem       outputBufferSep;    /**< CL memory output buffer for Separable Kernel */
        cl_mem       maskBuffer;         /**< CL memory mask buffer */
		cl_mem       rowFilterBuffer;    /**< CL memory row filter buffer */
		cl_mem       colFilterBuffer;    /**< CL memory col filter buffer */
        cl_command_queue commandQueue;   /**< CL command queue */
        cl_program   program;            /**< CL program  */
        cl_kernel    nonSeparablekernel; /**< CL kernel for nonSeparable implementation*/
		cl_kernel    separablekernelPass1; /**< CL kernel for Separable implementation of first pass*/
		cl_kernel    separablekernelPass2; /**< CL kernel for Separable implementation of second pass*/

        size_t       globalThreads[1];   /**< global NDRange */
        size_t       localThreads[1];    /**< Local Work Group Size */
		int			 localSize;			 /**< User-specified Local Work Group Size */
        int          iterations;         /**< Number of iterations to execute kernel */
        SDKDeviceInfo deviceInfo;        /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfo;  /**< Structure to store kernel related info */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        SimpleConvolution()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            seed = 123;
            input = NULL;
            output = NULL;
			tmpOutput = NULL;
			outputSep = NULL;
            mask   = NULL;
            verificationOutput = NULL;
            width = 512;
            height = 512;
            setupTime = 0;
			totalNonSeparableKernelTime = 0;
            totalSeparableKernelTime = 0;
            iterations = 1;
			localSize = GROUP_SIZE;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setupSimpleConvolution();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         * @return 0 on success and nonzero on failure
         */
        int setWorkGroupSize();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setupCL();

        /**
		* Call both non-separable and separable OpenCL implementation of 
		* Convolution
		* @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
		*/
        int runCLKernels();

		/**
         * Set values for Non-Separable kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
		int runNonSeparableCLKernels();

		/**
         * Set values for Separable kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
		int runSeparableCLKernels();

        /**
         * Reference CPU implementation of Simple Convolution
         * for performance comparison
         */
        void CPUReference();

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int initialize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int genBinaryImage();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run OpenCL SimpleConvolution kernel
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int verifyResults();
};



#endif
