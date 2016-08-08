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


#ifndef RADIXSORT_H_
#define RADIXSORT_H_

/**
 * Heaer Files
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#define ELEMENT_COUNT (8192)
#define RADIX 8
#define RADICES (1 << RADIX)    //Values handeled by each work-item?
#define RADIX_MASK (RADICES - 1)
#define GROUP_SIZE 64
#define NUM_GROUPS (ELEMENT_COUNT / (GROUP_SIZE * RADICES))

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

using namespace appsdk;

/**
* RadixSort
* Class implements 256 RadixSort bin implementation
*/

class RadixSort
{
        cl_int  elementCount;           /**< Size of RadixSort bin */
        cl_int  groupSize;              /**< Number of threads in a Group */
        cl_int  numGroups;              /**< Number of groups */
        cl_ulong neededLocalMemory;     /**< Local memory need by application which set from host */
        cl_bool byteRWSupport;          /**< Flag for byte-addressable store */
        int iterations;                 /**< Number of iterations for kernel execution */

        //Host buffers
        cl_uint *unsortedData;          /**< unsorted elements */
        cl_uint *dSortedData;           /**< device sorted elements */
        cl_uint *hSortedData;           /**< host sorted elements */


        //Device buffers
        cl_mem origUnsortedDataBuf;     /**< CL memory buffer to store input unsorted data */
        cl_mem partiallySortedBuf;      /**< CL memory buffer to store partially sorted data */
        cl_mem histogramBinsBuf;        /**< CL memory buffer for prescaneduckets --input */
        cl_mem scanedHistogramBinsBuf;  /**< CL memory buffer for prescaneduckets -- output */
        cl_mem sortedDataBuf;           /**< CL memory buffer for sorted data */
        //add four buffers
        cl_mem sumBufferin;
        cl_mem sumBufferout;
        cl_mem summaryBUfferin;
        cl_mem summaryBUfferout;

        cl_double totalKernelTime;      /**< Total time for kernel execution and memory transfers */
        cl_double setupTime;            /**< Time for OpenCL initializations */

        //CL objects
        cl_context context;             /**< CL context */
        cl_device_id *devices;          /**< CL device list */
        cl_command_queue commandQueue;  /**< CL command queue */
        cl_program program;             /**< CL program  */
        cl_kernel histogramKernel;      /**< CL kernel for histogram */
        cl_kernel permuteKernel;        /**< CL kernel for permute */
        //add four kernels
        cl_kernel scanArrayKerneldim2;
        cl_kernel scanArrayKerneldim1;
        cl_kernel prefixSumKernel;
        cl_kernel blockAdditionKernel;
        cl_kernel FixOffsetkernel;
        //end

        bool firstLoopIter;             /**< Indicates if the bit loop in runCLKernels() is running for first time */

        SDKDeviceInfo deviceInfo;/**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfoHistogram,
                            kernelInfoPermute;/**< Structure to store kernel related info */
        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:
        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Constructor
        * Initialize member variables
        */
        RadixSort()
            : elementCount(ELEMENT_COUNT),
              groupSize(GROUP_SIZE),
              numGroups(NUM_GROUPS),
              totalKernelTime(0),
              setupTime(0),
              unsortedData(NULL),
              dSortedData(NULL),
              hSortedData(NULL),
              devices(NULL),
              byteRWSupport(true),
              iterations(1)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        ~RadixSort()
        {}

        /**
        * Allocate and initialize required host memory with appropriate values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupRadixSort();

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
        *  Host Radix sort
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int hostRadixSort();

        /**
        *  Runs Histogram Kernel
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runHistogramKernel(int bits);

        /**
        *  Runs Permute Kernel
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runPermuteKernel(int bits);

        int runStaticKernel();

        int runFixOffsetKernel();

    private:

        /**
        * A common function to map cl_mem object to host
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);

        /**
        * A common function to unmap cl_mem onject from host
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);

};



#endif  //RADIXSORT_H_
