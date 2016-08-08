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
#ifndef LUDECOMPOSITION_HPP_
#define LUDECOMPOSITION_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

#define KERNELFILE "LUDecomposition_Kernels.cl"

/**
 * LUD
 * Class implements OpenCL LU Decomposition sample

 */
class LUD
{
        cl_uint
        seed;      /**< Seed value for random number generation */
        cl_double           setupTime;      /**< Time for setting up OpenCL */
        cl_double     totalKernelTime;      /**< Time for kernel execution */
        cl_double    totalProgramTime;      /**< Time for program execution */
        cl_double referenceKernelTime;      /**< Time for reference implementation */
        cl_int
        effectiveDimension;      /**< effectiveDimension(square matrix) of the input matrix */
        cl_int
        actualDimension;      /**< actual dimension (might not be in 2^n form) of input matrix */
        cl_double              *input;      /**< Input array */
        cl_double
        *matrixCPU;      /**< Inplace Array for CPU for reference implementation */
        cl_double          *matrixGPU;      /**< Inplace Array for GPU */
        cl_int              blockSize;       /**< actual dimension / vector size */
        cl_context            context;      /**< CL context */
        cl_device_id         *devices;      /**< CL device list */
        cl_mem          inplaceBuffer;      /**< CL memory buffer */
        cl_mem           inputBuffer2;      /**< CL memory output Buffer */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program            program;      /**< CL program  */
        cl_kernel           kernelLUD;      /**< CL kernel LU Decomposition*/
        cl_kernel       kernelCombine;      /**< CL Kerenl Combine */
        cl_ulong    localMemoryNeeded;
        bool                   useLDS;
        int
        iterations;    /**< Number of iterations for kernel execution */
        SDKDeviceInfo
        deviceInfo;            /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;      /**< Structure to store kernel related info */


        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        LUD()
        {
            seed                = 123;
            input               = NULL;
            matrixCPU           = NULL;
            matrixGPU           = NULL;
            devices             = NULL;
            actualDimension     = 16;
            effectiveDimension  = 16;
            blockSize           = effectiveDimension/ 4;
            setupTime           = 0;
            totalKernelTime     = 0;
            iterations          = 1;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupLUD();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

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
         * Reference CPU implementation of matrix transpose
         * @param output stores the transpose of the input
         * @param input  input matrix
         * @param width  width of the input matrix
         * @param height height of the array
         */
        void LUDCPUReference(
            double *matrixCPU,
            const cl_uint effectiveDimension);

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
         * Override from SDKSample
         * Run OpenCL LU Decompose
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

