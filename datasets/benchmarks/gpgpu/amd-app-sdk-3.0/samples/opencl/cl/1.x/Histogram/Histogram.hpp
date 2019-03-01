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


#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>


#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"
#define WIDTH 1024
#define HEIGHT 1024
#define BIN_SIZE 256
#define GROUP_SIZE 128
#define GROUP_ITERATIONS (BIN_SIZE / 2)//This is done to avoid overflow in the kernel
#define SUB_HISTOGRAM_COUNT ((WIDTH * HEIGHT) /(GROUP_SIZE * GROUP_ITERATIONS))


/**
* Histogram
* Class implements 256 Histogram bin implementation

*/

class Histogram
{

        cl_int binSize;             /**< Size of Histogram bin */
        cl_int groupSize;           /**< Number of threads in group */
        cl_int subHistgCnt;         /**< Sub histogram count */
        cl_uint *data;              /**< input data initialized with normalized(0 - binSize) random values */
        cl_int width;               /**< width of the input */
        cl_int height;              /**< height of the input */
        cl_uint *hostBin;           /**< Host result for histogram bin */
        cl_uint *midDeviceBin;      /**< Intermittent sub-histogram bins */
        cl_uint *deviceBin;         /**< Device result for histogram bin */

        cl_double setupTime;        /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;       /**< time taken to run kernel and read result back */

        cl_ulong totalLocalMemory;      /**< Max local memory allowed */
        cl_ulong usedLocalMemory;       /**< Used local memory by kernel */

        cl_context context;             /**< CL context */
        cl_device_id *devices;          /**< CL device list */

        cl_mem dataBuf;                 /**< CL memory buffer for data */
        cl_mem midDeviceBinBuf;         /**< CL memory buffer for intermittent device bin */

        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernel;                   /**< CL kernel */
        int iterations;                     /**< Number of iterations for kernel execution */
        bool scalar;                        /**< scalar kernel */
        bool vector;                        /**< vector kernel */
        int vectorWidth;                    /**< vector width used by the kernel*/
        size_t globalThreads;
        size_t localThreads ;
        int groupIterations;

        SDKDeviceInfo
        deviceInfo;            /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;      /**< Structure to store kernel related info */

        cl_bool byteRWSupport;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (string)
        */
        Histogram()
            :
            binSize(BIN_SIZE),
            groupSize(GROUP_SIZE),
            setupTime(0),
            kernelTime(0),
            subHistgCnt(SUB_HISTOGRAM_COUNT),
            groupIterations(GROUP_ITERATIONS),
            data(NULL),
            hostBin(NULL),
            midDeviceBin(NULL),
            deviceBin(NULL),
            devices(NULL),
            byteRWSupport(true),
            iterations(1),
            scalar(false),
            vector(false),
            vectorWidth(0)
        {
            /* Set default values for width and height */
            width = WIDTH;
            height = HEIGHT;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        ~Histogram()
        {
        }

        /**
        * Allocate and initialize required host memory with appropriate values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupHistogram();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         * @return 0 on success and nonzero on failure
         */
        int setWorkGroupSize();

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

        /**
        * CECL_MAP_BUFFER
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);

        /**
        * clEnqueueUnmapMemObject
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);
};
#endif
