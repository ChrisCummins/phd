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


#ifndef DWTHAAR1D_H_
#define DWTHAAR1D_H_




/**
 * Header Files
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"

using namespace appsdk;

#define SIGNAL_LENGTH (1 << 10)
#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

/**
 * DwtHaar1D
 * Class implements One-dimensional Haar wavelet decomposition

 */

class DwtHaar1D
{

        cl_uint signalLength;           /**< Signal length (Must be power of 2)*/
        cl_float *inData;               /**< input data */
        cl_float *dOutData;             /**< output data */
        cl_float *dPartialOutData;      /**< paritial decomposed signal */
        cl_float *hOutData;             /**< output data calculated on host */

        cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;           /**< time taken to run kernel and read result back */

        cl_context context;             /**< CL context */
        cl_device_id *devices;          /**< CL device list */

        cl_mem inDataBuf;               /**< CL memory buffer for input data */
        cl_mem dOutDataBuf;             /**< CL memory buffer for output data */
        cl_mem dPartialOutDataBuf;      /**< CL memory buffer for paritial decomposed signal */

        cl_command_queue commandQueue;  /**< CL command queue */
        cl_program program;             /**< CL program  */
        cl_kernel kernel;               /**< CL kernel for histogram */
        cl_uint maxLevelsOnDevice;      /**< Maximum levels to be computed on device */
        int iterations;                 /**< Number of iterations to be executed on kernel */
        size_t        globalThreads;    /**< global NDRange */
        size_t         localThreads;    /**< Local Work Group Size */
        SDKDeviceInfo deviceInfo;        /**< Structure to store device information*/
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
        DwtHaar1D()
            :
            signalLength(SIGNAL_LENGTH),
            setupTime(0),
            kernelTime(0),
            inData(NULL),
            dOutData(NULL),
            dPartialOutData(NULL),
            hOutData(NULL),
            devices(NULL),
            iterations(1)
        {
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }



        ~DwtHaar1D()
        {
        }

        /**
         * Allocate and initialize required host memory with appropriate values
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setupDwtHaar1D();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setWorkGroupSize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int genBinaryImage();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setupCL();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int runCLKernels();

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setup();

        /**
         * Override from SDKSample
         * Run OpenCL DwtHaar1D
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int verifyResults();

    private:

        cl_int groupSize;       /**< Work items in a group */
        cl_int totalLevels;     /**< Total decomposition levels required for given signal length */
        cl_int curSignalLength; /**< Length of signal for given iteration */
        cl_int levelsDone;      /**< levels done */

        /**
         * @brief   Get number of decomposition levels to perform a full decomposition
         *          and also check if the input signal size is suitable
         * @return  returns the number of decomposition levels if they could be detrmined
         *          and the signal length is supported by the implementation,
         *          otherwise it returns SDK_FAILURE
         * @param   length  Length of input signal
         * @param   levels  Number of decoposition levels neessary to perform a full
         *                  decomposition
         *
         */
        int getLevels(unsigned int length, unsigned int* levels);

        /**
         * @brief   Runs the dwtHaar1D kernel to calculate
         *          the approximation coefficients on device
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int runDwtHaar1DKernel();

        /**
        * @brief   Reference implementation to calculates
        *          the approximation coefficients on host
        *          by normalized decomposition
        * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
        */
        int calApproxFinalOnHost();

};

#endif
