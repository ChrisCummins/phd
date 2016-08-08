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


#ifndef REDUCTION_H_
#define REDUCTION_H_

/**
 * Header Files
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#include <malloc.h>

#define SAMPLE_VERSION "AMD-APP-SDK-vx.y.z.s"

#define GROUP_SIZE 256
#define VECTOR_SIZE 4
#define MULTIPLY  2  //Require because of extra addition before loading to local memory

using namespace appsdk;

/**
 * Reduction
 * Class implements OpenCL  Reduction sample
 */

class Reduction
{
        cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;           /**< time taken to run kernel and read result back */

        size_t globalThreads[1];        /**< Global NDRange for the kernel */
        size_t localThreads[1];         /**< Local WorkGroup for kernel */

        cl_uint length;                 /**< length of the input array */
        int numBlocks;                  /**< Number of groups */
        cl_uint *input;                 /**< Input array */
        cl_uint *outputPtr;             /**< Output array */
        cl_uint output;                 /**< Output result */
        cl_uint refOutput;              /**< Reference result */
        cl_context context;             /**< CL context */
        cl_device_id *devices;          /**< CL device list */
        cl_mem inputBuffer;             /**< CL memory buffer */
        cl_mem outputBuffer;             /**< CL memory buffer */
        cl_command_queue commandQueue;  /**< CL command queue */
        cl_program program;             /**< CL program  */
        cl_kernel kernel;               /**< CL kernel */
        size_t groupSize;               /**< Work-group size */
        int iterations;                 /**< Number of iterations for kernel execution*/
        SDKDeviceInfo
        deviceInfo;            /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;      /**< Structure to store kernel related info */
        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:
        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        explicit Reduction()
            : input(NULL),
              outputPtr(NULL),
              output(0),
              refOutput(0),
              devices(NULL)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            length = 64;
            groupSize = GROUP_SIZE;
            iterations = 1;
        }

        ~Reduction();

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupReduction();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setWorkGroupSize();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCL();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runCLKernels();

        /**
         * Reference CPU implementation of Reduction
         * for performance comparison
         * @param input the input array
         * @param length length of the array
         * @param output value
         */
        void reductionCPUReference(
            cl_uint * input,
            const cl_uint length,
            cl_uint& output);

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
         * Override from SDKSample
         * Run OpenCL Reduction
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();
};

#endif // REDUCTION_H_
