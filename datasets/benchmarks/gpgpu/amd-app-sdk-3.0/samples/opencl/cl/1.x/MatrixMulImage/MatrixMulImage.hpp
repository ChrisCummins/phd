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


#ifndef MATRIXMULTIPLICATION_IMAGE_H_
#define MATRIXMULTIPLICATION_IMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

using namespace appsdk;

/**
 * MatrixMulImage
 * Class implements OpenCL Matrix Multiplication sample using Images
 */

class MatrixMulImage
{
        cl_uint
        seed;      /**< Seed value for random number generation */
        cl_double           setupTime;      /**< Time for setting up OpenCL */
        cl_double     totalKernelTime;      /**< Time for kernel execution */
        cl_double    totalProgramTime;      /**< Time for program execution */
        cl_double referenceKernelTime;      /**< Time for reference implementation */
        cl_float              *input0;      /**< Input array */
        cl_int                 width0;      /**< width of input Array */
        cl_int                height0;      /**< height of input Array */
        cl_float              *input1;      /**< Input array */
        cl_int                 width1;      /**< width of Input Array */
        cl_int                height1;      /**< height of Input Array */
        cl_float              *output;      /**< Output Array */
        cl_float  *verificationOutput;      /**< Output array for reference implementation */
        cl_uint
        blockSize;      /**< Size of the block used for shared memory */
        cl_context            context;      /**< CL context */
        cl_device_id         *devices;      /**< CL device list */
        cl_mem           inputBuffer0;      /**< CL memory buffer  for matrix input0*/
        cl_mem           inputBuffer1;      /**< CL memory buffer  for matrix input1*/
        cl_mem
        outputBuffer;      /**< CL memory buffer  for storing the output*/
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program            program;      /**< CL program  */
        cl_kernel              kernel;      /**< CL kernel */

        cl_int                      n;      /**< for command line args */
        cl_int                      m;
        cl_int                      k;

        //size_t       maxWorkGroupSize;      /**< Device Specific Information */
        //size_t    kernelWorkGroupSize;      /**< Group Size returned by kernel */
        //cl_uint         maxDimensions;
        //size_t *     maxWorkItemSizes;
        /*cl_ulong     totalLocalMemory;
        cl_ulong      usedLocalMemory; */
        cl_ulong availableLocalMemory;
        cl_ulong    neededLocalMemory;
        int
        iterations;      /**< Number of iterations for kernel execution */
        bool imageSupport;
        SDKDeviceInfo deviceInfo;/**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfo;/**< Structure to store kernel related info */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        MatrixMulImage()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            seed   = 123;
            input0 = NULL;
            input1 = NULL;
            output = NULL;
            verificationOutput = NULL;
            n = 128;
            m = 128;
            k = 128;
            blockSize = 8;
            setupTime = 0;
            totalKernelTime = 0;
            iterations = 1;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupMatrixMulImage();

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
         * Reference CPU implementation of Matrix Multiplication
         * @param output stores the output of the multipliced matrices depthxheight
         * @param input0 input matrix of size width x height
         * @param input1 input matrix of size depth x width
         * @param height height of the output matrix
         * @param width  length of the common dimension of the matrices input0 and input1
         * @param depth  width  of the output matrix
         */
        void MatrixMulImageCPUReference(
            cl_float * output,
            cl_float * input0,
            cl_float * input1,
            const cl_uint height,
            const cl_uint width,
            const cl_uint depth);
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
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run OpenCL Matrix Multiplication
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
#endif
