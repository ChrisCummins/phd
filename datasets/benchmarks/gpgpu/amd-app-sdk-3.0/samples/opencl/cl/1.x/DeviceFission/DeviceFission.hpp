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


#ifndef REDUCTION_H_
#define REDUCTION_H_

//Headers Files
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>
#include <string.h>

#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"
#define REMOVE_GPU  1
#define GROUP_SIZE 256
#define DEFAULT_INPUT_SIZE 1024
#define VALUES_PRINTED 20

// Init extension function pointers
#define INIT_CL_EXT_FCN_PTR(name) \
    if(!pfn_##name) { \
        pfn_##name = (name##_fn) clGetExtensionFunctionAddress(#name); \
        if(!pfn_##name) { \
            std::cout << "Cannot get pointer to ext. fcn. " #name << std::endl; \
            return SDK_FAILURE; \
        } \
    }

/**
 * DeviceFission
 * Class implements OpenCL  DeviceFission sample

 */

class DeviceFission
{
        cl_uint length;                 /**< Length of the input array */
        cl_uint half_length;            /**< Half length of the input array */
        cl_int *input;                  /**< Input array */

        cl_int *subOutput;              /**< Output for sub-devices */

        cl_context rContext;            /**< CL context for all devices */
        cl_device_id *Devices;          /**< CL device list */
        cl_device_id cpuDevice;         /**< CPU CL device */

        cl_uint numSubDevices;          /**< Number of sub-devices */
        cl_device_id *subDevices;       /**< CL Sub device list */

        cl_mem InBuf;                   /**< CL memory buffers for sub-devices */
        cl_mem *subOutBuf;              /**< CL memory buffers for sub-devices */

        cl_command_queue *subCmdQueue;  /**< CL command queue for sub-devices */

        cl_program subProgram;          /**< CL program for sub-devices */
        cl_program gpuProgram;          /**< CL program for gpu device */

        cl_kernel *subKernel;           /**< CL kernel for sub-devices */

        size_t kernelWorkGroupSize;     /**< Group size returned by kernel */
        size_t groupSize;               /**< Work-group size */
        SDKDeviceInfo deviceInfo;/**< Structure to store device information */

        size_t deviceListSize;          /**< Size of all device-list */
        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        DeviceFission()
            :  input(NULL),
               subOutput(NULL),
               Devices(NULL),
               subDevices(NULL),
               cpuDevice(NULL),
               subOutBuf(NULL),
               subCmdQueue(NULL),
               subKernel(NULL),
               numSubDevices(2),
               length(DEFAULT_INPUT_SIZE),
               groupSize(GROUP_SIZE),
               deviceListSize(0)
        {
            sampleArgs = new CLCommandArgs(true) ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        ~DeviceFission();

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupDeviceFission();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list and Device properties.
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCLPlatform();

        /**
         * OpenCL related initialisations.
         * Create of command queue, buffers, Building program and kernel
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCLRuntime();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runCLALLKerenls();


        /**
         * Creat and initialize timer for runCLALLKerenls
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
         * Run OpenCL DeviceFission
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

#endif // REDUCTION_H_
