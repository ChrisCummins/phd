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


#ifndef DCT_H_
#define DCT_H_

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

using namespace appsdk;

#if !defined(M_PI)
#define M_PI (3.14159265358979323846f)
#endif

namespace dct
{
const cl_float a = cos(M_PI/16)/2;
const cl_float b = cos(M_PI/8 )/2;
const cl_float c = cos(3*M_PI/16)/2;
const cl_float d = cos(5*M_PI/16)/2;
const cl_float e = cos(3*M_PI/8)/2;
const cl_float f = cos(7*M_PI/16)/2;
const cl_float g = 1.0f/sqrt(8.0f);

/**
 * DCT8x8 mask that is used to calculate Discrete Cosine Transform
 * of an 8x8 matrix
 */
cl_float dct8x8[64] =
{
    g,  a,  b,  c,  g,  d,  e,  f,
    g,  c,  e, -f, -g, -a, -b, -d,
    g,  d, -e, -a, -g,  f,  b,  c,
    g,  f, -b, -d,  g,  c, -e, -a,
    g, -f, -b,  d,  g, -c, -e,  a,
    g, -d, -e,  a, -g, -f,  b, -c,
    g, -c,  e,  f, -g,  a, -b,  d,
    g, -a,  b, -c,  g, -d,  e,  -f
};

/**
* DCT
* Class implements OpenCL Discrete Cosine Transform

*/

class DCT
{
        cl_uint
        seed;    /**< Seed value for random number generation */
        cl_double              setupTime;    /**< Time for setting up OpenCL */
        cl_double        totalKernelTime;    /**< Time for kernel execution */
        cl_double       totalProgramTime;    /**< Time for program execution */
        cl_double    referenceKernelTime;    /**< Time for reference implementation */
        cl_int                     width;    /**< Width of the input array */
        cl_int                    height;    /**< height of the input array */
        cl_float                  *input;    /**< Input array */
        cl_float                 *output;    /**< Output array */
        cl_uint               blockWidth;    /**< width of the blockSize */
        cl_uint                blockSize;    /**< size of the block */
        cl_uint                  inverse;    /**< flag for inverse DCT */
        cl_float
        *verificationOutput;    /**< Input array for reference implementation */
        cl_context               context;    /**< CL context */
        cl_device_id            *devices;    /**< CL device list */
        cl_mem               inputBuffer;    /**< CL memory buffer */
        cl_mem              outputBuffer;    /**< CL memory buffer */
        cl_mem                 dctBuffer;    /**< CL memory buffer */
		cl_mem                 dct_transBuffer;    /**< CL memory buffer */
        cl_command_queue    commandQueue;    /**< CL command queue */
        cl_program               program;    /**< CL program  */
        cl_kernel                 kernel;    /**< CL kernel */
        cl_ulong    availableLocalMemory;
        cl_ulong       neededLocalMemory;
		cl_float	dct8x8_trans[64];
        int
        iterations;    /**< Number of iteration for kernel execution */
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
        DCT()
        {
            seed = 123;
            input = NULL;
            verificationOutput = NULL;
            width = 64;
            height = 64;
            blockWidth = 8;
            blockSize  = blockWidth * blockWidth;
            inverse = 0;
            setupTime = 0;
            totalKernelTime = 0;
            iterations  = 1;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupDCT();

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
         * Given the blockindices and localIndicies this
         * function calculate the global index
         * @param blockIdx index of the block horizontally
         * @param blockIdy index of the block vertically
         * @param localidx index of the element relative to the block horizontally
         * @param localIdy index of the element relative to the block vertically
         * @param blockWidth width of each block which is 8
         * @param globalWidth Width of the input matrix
         * @return ID in x dimension
         */
        cl_uint getIdx(cl_uint blockIdx, cl_uint blockIdy, cl_uint localIdx,
                       cl_uint localIdy, cl_uint blockWidth, cl_uint globalWidth);

        /**
         * Reference CPU implementation of Discrete Cosine Transform
         * for performance comparison
         * @param output output of the DCT8x8 transform
         * @param input  input array
         * @param dct8x8 8x8 cosine function base used to calculate DCT8x8
         * @param width width of the input matrix
         * @param height height of the input matrix
         * @param numBlocksX number of blocks horizontally
         * @param numBlocksY number of blocks vertically
         * @param inverse  flag to perform inverse DCT
         */
        void DCTCPUReference( cl_float * output,
                              const cl_float * input ,
                              const cl_float * dct8x8 ,
                              const cl_uint    width,
                              const cl_uint    height,
                              const cl_uint   numBlocksX,
                              const cl_uint   numBlocksY,
                              const cl_uint    inverse);
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
         * Run OpenCL DCT
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
};
} //namespace DCT

#endif
