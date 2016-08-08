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


#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define NBINS        256
#define BITS_PER_PIX 8
#define VECTOR_SIZE 4

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

using namespace appsdk;

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

/**
* Histogram
* Class implements 256 Histogram bin implementation
*/

class Histogram
{
        cl_uint inputNBytes;
        cl_uint outputNBytes;

        cl_uint nLoops;
        cl_uint nThreads;
        cl_uint nThreadsPerGroup;
        cl_uint nGroups;
        cl_uint nVectors;
        cl_uint nVectorsPerThread;
        cl_uint nBins;
        cl_uint nBytesLDSPerGrp;

        cl_uint     *input;
        cl_uint     *output;
        cl_mem   inputBuffer;
        cl_mem   outputBuffer;

        cl_uint cpuhist[NBINS];

        cl_context          context;
        cl_device_id        *devices;
        cl_command_queue    commandQueue;

        cl_program program;
        cl_kernel  histogram;
        cl_kernel  reduce;

        cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTimeGlobal;     /**< time taken to run kernel and read result back */

        cl_ulong totalLocalMemory;      /**< Max local memory allowed */
        cl_ulong usedLocalMemory;       /**< Used local memory by kernel */

        int iterations;                 /**< Number of iterations for kernel execution */
        cl_bool reqdExtSupport;
        bool useScalarKernel;
        bool useVectorKernel;
        size_t KernelCompileWorkGroupSize[3];
        SDKDeviceInfo deviceInfo;    /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfoReduce,
                            kernelInfoHistogram;  /**< Structure to store kernel related info */
        int vectorWidth;

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Constructor
        * Initialize member variables
        */
        Histogram()
            : input(NULL),
              output(NULL),
              devices(NULL),
              setupTime(0),
              kernelTimeGlobal(0),
              iterations(1),
              reqdExtSupport(true),
              useScalarKernel(false),
              useVectorKernel(false),
              vectorWidth(0)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        ~Histogram()
        {
            FREE(devices);
        }

        /**
        * Allocate and initialize required host memory with appropriate values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupHistogram();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * OpenCL related initializations.
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
        * Run OpenCL Black-Scholes
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

    private:

        /**
         *  Calculate histogram bin on host
         */
        int calculateHostBin();
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);

        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);

};

#endif
