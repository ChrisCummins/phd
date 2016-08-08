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


#ifndef _SCANLARGEARRAYS_H_
#define _SCANLARGEARRAYS_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"


#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif


/**
* ScanLargerrays
* Class implements OpenCL Scan Large Arrays sample
*/

#define GROUP_SIZE 256

#define SAMPLE_VERSION "AMD-APP-SDK-vx.y.z.s"

using namespace appsdk;

class ScanLargeArrays
{
        cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;           /**< time taken to run kernel and read result back */
        cl_float            *input;                 /**< Input array */
        cl_float            *output;                /**< Output Array */
        cl_float
        *verificationOutput;    /**< Output array for reference implementation */
        cl_context          context;                /**< CL context */
        cl_device_id        *devices;               /**< CL device list */
        cl_mem              inputBuffer;            /**< CL memory buffer */
        cl_mem              *outputBuffer;          /**< Array of output buffers */
        cl_mem              *blockSumBuffer;        /**< Array of block sum buffers */
        cl_mem              tempBuffer;             /**< Temporary bufer */
        cl_command_queue    commandQueue;           /**< CL command queue */
        cl_program          program;                /**< CL program  */
        cl_kernel
        bScanKernel;            /**< CL kernel for block-wise scan */
        cl_kernel           bAddKernel;             /**< CL Kernel for block-wise add */
        cl_kernel           pScanKernel;            /**< CL Kernel for prefix sum */
        cl_uint             blockSize;              /**< Size of a block */
        cl_uint             length;                 /**< Length of output */
        cl_uint             pass;                   /**< Number of passes */
        int
        iterations;             /**< Number of iterations for kernel execution */
        SDKDeviceInfo deviceInfo;/**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfoBScan, kernelInfoBAdd,
                            kernelInfoPScan;/**< Structure to store kernel related info */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Constructor
        * Initialize member variables
        */
        ScanLargeArrays()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            input = NULL;
            output = NULL;
            verificationOutput = NULL;
            outputBuffer = NULL;
            blockSumBuffer = NULL;
            blockSize = GROUP_SIZE;
            length = 32768;
            kernelTime = 0;
            setupTime = 0;
            iterations = 1;
        }

        /**
        * Allocate and initialize host memory array with random values
        * Calculate number of pass required and allocate device buffers accordingly
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupScanLargeArrays();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
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
        * Enqueue bScan Kernel
        * Scans the inputBuffer block-wise and stores scanned elements in outputBuffer
        * and sum of blocks in blockSumBuffer
        * @param len size of input buffer
        * @param inputBuffer input buffer
        * @param outputBuffer output buffer
        * @param blockSumBuffer sum of blocks of inputbuffer
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int bScan(cl_uint len,
                  cl_mem *inputBuffer,
                  cl_mem *outputBuffer,
                  cl_mem *blockSumBuffer);

        /**
        * Enqueue pScan Kernel
        * Basic prefix sum
        * @param len size of input buffer
        * @param inputBuffer input buffer
        * @param outputBuffer output buffer
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int pScan(cl_uint len,
                  cl_mem *inputBuffer,
                  cl_mem *outputBuffer);

        /**
        * Enqueue bAddition Kernel
        * Elements of inputBuffer are added block-wise to outputBuffer
        * @param len size of output buffer
        * @param inputBuffer input buffer
        * @param outputBuffer output buffer
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int bAddition(cl_uint len,
                      cl_mem *inputBuffer,
                      cl_mem *outputBuffer);


        /**
        * Reference CPU implementation of Prefix Sum
        * @param output the array that stores the prefix sum
        * @param input the input array
        * @param length length of the input array
        */
        void scanLargeArraysCPUReference(cl_float * output,
                                         cl_float * input,
                                         const cl_uint length);
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
        * Run OpenCL FastWalsh Transform
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
        * A common function to map cl_mem object to host
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);

        /**
        * A common function to unmap cl_mem object from host
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);

};
#endif //_SCANLARGEARRAYS_H_
