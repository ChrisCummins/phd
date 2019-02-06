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

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"

#define  MAX_WAVEFRONT_SIZE 64     // Work group size

#include "Log.h"
#include "Timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

/**
 * TransferOverlap
 * Class implements OpenCL TransferOverlap benchmark sample
 */

class TransferOverlap
{
        bool correctness;     // Correctness status variable
        int nLoops;           // Overall number of timing loops
        int nSkip;            // To discount lazy allocation effects, etc.
        int nKLoops;          // Repeat inside kernel to show peak mem B/W,

        int nBytes;           // Input and output buffer size
        int nThreads;         // Number of GPU work items
        int nItems;           // Number of 32-bit 4-vectors for GPU kernel
        int nAlign;           // Safe bet for most PCs
        int nItemsPerThread;  // Number of 32-bit 4-vectors per GPU thread
        int nBytesResult;

        size_t globalWorkSize; // Global work items
        size_t localWorkSize;  // Local work items
        double testTime;         // Total time to complete

        bool printLog;       // Enable/Disable print log
        bool noOverlap;      // Disallow memset/kernel overlap
        int  numWavefronts;

        TestLog *timeLog;

        cl_command_queue queue;
        cl_context context;
        cl_program program;
        cl_kernel readKernel;
        cl_kernel writeKernel;
        cl_device_id  *devices;      // CL device list

        CPerfCounter t;

        cl_mem inputBuffer1;
        cl_mem inputBuffer2;
        cl_mem resultBuffer1;
        cl_mem resultBuffer2;

        cl_mem_flags inFlags;
        int inFlagsValue;
        SDKDeviceInfo deviceInfo;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        TransferOverlap()
            :nLoops(50),
             nSkip(3),
             nKLoops(45),
             nBytes(16 * 1024 * 1024),
             nThreads(MAX_WAVEFRONT_SIZE),
             nItems(2),
             nAlign(4096),
             nBytesResult(32 * 1024 * 1024),
             printLog(false),
             numWavefronts(7),
             timeLog(NULL),
             queue(NULL),
             context(NULL),
             readKernel(NULL),
             writeKernel(NULL),
             inputBuffer1(NULL),
             inputBuffer2(NULL),
             resultBuffer1(NULL),
             resultBuffer2(NULL),
             inFlags(0),
             inFlagsValue(0),
             noOverlap(false),
             correctness(true)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return 1 on success and 0 on failure
         */
        int setupTransferOverlap();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return 1 on success and 0 on failure
         */
        int setupCL();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         */
        int initialize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         */
        int genBinaryImage();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         */
        int setup();

        /**
         * Override from SDKSample
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         */
        int verifyResults()
        {
            return SDK_SUCCESS;
        };

        void printStats();

        /**
         * Parses Extra command line options and
         * calls SDKSample::parseCommandLine()
         */
        int parseExtraCommandLineOptions(int argc, char**argv);
        int verifyResultBuffer(cl_mem resultBuffer, bool firstLoop);
        int launchKernel(cl_mem inputBuffer, cl_mem resultBuffer, unsigned char v);
        void* launchMapBuffer(cl_mem buffer, cl_event *mapEvent);
        int fillBuffer(cl_mem buffer, cl_event *mapEvent, void *ptr, unsigned char v);
        int runOverlapTest();
};


#endif
