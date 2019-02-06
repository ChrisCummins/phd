/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

• Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
• Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef STRINGSEARCH_H_
#define STRINGSEARCH_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

#define LOCAL_SIZE      256
#define COMPARE(x,y)    ((caseSensitive) ? (x==y) : (toupper(x) == toupper(y)))
#define SEARCH_BYTES_PER_WORKITEM   512

enum KERNELS
{
    KERNEL_NAIVE = 0,
    KERNEL_LOADBALANCE = 1
};

/**
* StringSearch
* Class implements StringSearch implementation
*/
class StringSearch
{
        cl_uchar *text;
        cl_uint  textLength;
        std::string subStr;
        std::string file;
        std::vector<cl_uint> devResults;
        std::vector<cl_uint> cpuResults;

        cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;           /**< time taken to run kernel and read result back */

        cl_context context;             /**< CL context */
        cl_device_id *devices;          /**< CL device list */

        cl_mem textBuf;                 /**< CL memory buffer for text */
        cl_mem subStrBuf;               /**< CL memory buffer for pattern */
        cl_mem resultCountBuf;          /**< CL memory buffer for result counts per WG */
        cl_mem resultBuf;               /**< CL memory buffer for result match positions */

        cl_command_queue commandQueue;  /**< CL command queue */
        cl_program program;             /**< CL program  */
        cl_kernel kernelLoadBalance;    /**< CL kernel */
        cl_kernel kernelNaive;          /**< CL kernel */
        int iterations;                 /**< Number of iterations for kernel execution */
        SDKDeviceInfo
        deviceInfo;            /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;      /**< Structure to store kernel related info */

        cl_bool byteRWSupport;
        cl_uint workGroupCount;
        cl_uint availableLocalMemory;
        cl_uint searchLenPerWG;
        cl_kernel* kernel;
        int kernelType;
        bool caseSensitive;
        bool enable2ndLevelFilter;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        *******************************************************************************
        * @fn Constructor
        * @brief Initialize member variables
        *
        *******************************************************************************
        */
        StringSearch()
            : text(NULL),
              textLength(0),
              subStr("if there is a failure to allocate resources required by the"),
              file("StringSearch_Input.txt"),
              setupTime(0),
              kernelTime(0),
              devices(NULL),
              iterations(1),
              byteRWSupport(true),
              workGroupCount(0),
              availableLocalMemory(0),
              searchLenPerWG(0),
              kernel(&kernelNaive),
              kernelType(KERNEL_NAIVE),
              caseSensitive(false),
              enable2ndLevelFilter(false)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        /**
        *******************************************************************************
        * @fn Destructor
        * @brief Cleanup the member objects.
        *******************************************************************************
        */
        ~StringSearch()
        {
            FREE(text);
            devResults.clear();
            cpuResults.clear();
        }

        /**
        *******************************************************************************
        * @fn initialize
        * @brief Override from SDKSample. Initialize command line parser, add custom options.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int initialize();

        /**
        *******************************************************************************
        * @fn genBinaryImage
        * @brief Override from SDKSample, Generate binary image of given kernel and
        *        exit application.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int genBinaryImage();

        /**
        *******************************************************************************
        * @fn setup
        * @brief Override from SDKSample, adjust width and height of execution domain,
        *        perform all sample setup.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int setup();

        /**
        *******************************************************************************
        * @fn run
        * @brief Override from SDKSample. Run OpenCL StringSearch kernel.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int run();

        /**
        *******************************************************************************
        * @fn verifyResults
        * @brief Override from SDKSample. Verify against CPU reference implementation.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int verifyResults();

        /**
        *******************************************************************************
        * @fn cleanup
        * @brief Override from SDKSample. Cleanup memory allocations.
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int cleanup();

        /**
        *******************************************************************************
        * @fn printStats
        * @brief Override from SDKSample. Print sample stats.
        *******************************************************************************
        */
        void printStats();

    private:
        /**
        *******************************************************************************
        * @fn setupStringSearch
        * @brief Allocate and initialize required host memory with appropriate values
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int setupStringSearch();

        /**
        *******************************************************************************
        * @fn setupCL
        * @brief OpenCL related initialisations. Set up Context, Device list, Command Queue,
        *         Memory buffers. Build CL kernel program executable
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int setupCL();

        /**
        *******************************************************************************
        * @fn runKernel
        * @brief Run the specific version of OpenCL kernel, verify results and print statistics.
        *
        * @param[in] kernelName : Kernel name
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int runKernel(std::string kernelName);

        /**
        *******************************************************************************
        * @fn runCLKernels
        * @brief Set values for kernels' arguments, enqueue calls to the kernels on to
        *        the command queue, wait till end of kernel execution. Get kernel start
        *        and end time if timing is enabled
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int runCLKernels();

        /**
        *******************************************************************************
        * @fn cpuReferenceImpl
        * @brief CPU reference stringSearch implementation based on "Boyer-Moore-Horspool" Algorithm
        *******************************************************************************
        */
        void cpuReferenceImpl();

        /**
        *******************************************************************************
        * @fn mapBuffer
        * @brief A common function to map cl_mem object to host
        *
        * @param[in] deviceBuffer : Device buffer
        * @param[out] hostPointer : Host pointer
        * @param[in] sizeInBytes : Number of bytes to map
        * @param[in] flags : map flags
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags);

        /**
        *******************************************************************************
        * @fn unmapBuffer
        * @brief A common function to unmap cl_mem object from host
        *
        * @param[in] deviceBuffer : Device buffer
        * @param[in] hostPointer : Host pointer
        *
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *******************************************************************************
        */
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);
};
#endif
