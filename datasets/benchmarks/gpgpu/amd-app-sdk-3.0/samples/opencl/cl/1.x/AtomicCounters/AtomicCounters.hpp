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

#ifndef BUFFER_BANDWIDTH_H_
#define BUFFER_BANDWIDTH_H_

#define  GROUP_SIZE 256
#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

//Header Files
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"

using namespace appsdk;

/**
 * AtomicCounters
 * Class implements OpenCL AtomicCounters benchmark sample
 */

class AtomicCounters
{
        cl_double kTimeAtomCounter;    /**< time taken to run Atomic Counter kernel */
        cl_double kTimeAtomGlobal;     /**< time taken to run Global Atomic kernel */
        cl_uint length;                /**< length of the input array */
        cl_uint *input;                /**< Input array */
        cl_uint value;                 /**< value to be counted */
        cl_uint refOut;                /**< Reference output */
        cl_uint counterOut;            /**< Output from Atomic Counter kernel */
        cl_uint globalOut;             /**< Output from Global Atomic kernel */
        cl_uint initValue;             /**< Initial value for counter */
        cl_context context;            /**< CL context */
        cl_device_id *devices;         /**< CL device list */
        cl_mem inBuf;                  /**< CL memory buffer */
        cl_mem counterOutBuf;          /**< CL memory buffer */
        cl_mem globalOutBuf;           /**< CL memory buffer */
        cl_command_queue commandQueue; /**< CL command queue */
        cl_program program;            /**< CL program  */
        cl_kernel counterKernel;       /**< CL kernel */
        cl_kernel globalKernel;        /**< CL kernel */
        size_t counterWorkGroupSize;   /**< Work-group size for Counter Atomic kernel */
        size_t globalWorkGroupSize;    /**< Work-group size for global Atomic kernel*/
        int iterations;                /**< Number of iterations for kernel execution*/
        SDKDeviceInfo deviceInfo;      /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfoC,
                            kernelInfoG;   
							           /**< Structure to store kernel related info */
        SDKTimer    *sampleTimer;      /**< SDKTimer object */
    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        AtomicCounters()
            :kTimeAtomCounter(0),
             kTimeAtomGlobal(0),
             length(1024),
             input(NULL),
             refOut(0),
             counterOut(0),
             globalOut(0),
             initValue(0),
             devices(NULL),
             counterWorkGroupSize(GROUP_SIZE),
             globalWorkGroupSize(GROUP_SIZE),
             iterations(1)
        {
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupAtomicCounters();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCL();

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
         * of execution domain, perform all sample set-up
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Clean-up memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();


        /**
         * Prints data and performance results
         */
        void printStats();


        /**
         * Runs the Atomic counter kernel
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runAtomicCounterKernel();

        /**
         * Runs the Global Atomic kernel
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runGlobalAtomicKernel();

        /**
         * Reference implementation to find
         * the occurrences of a value in a given array
         */
        void cpuRefImplementation();
};
#endif
