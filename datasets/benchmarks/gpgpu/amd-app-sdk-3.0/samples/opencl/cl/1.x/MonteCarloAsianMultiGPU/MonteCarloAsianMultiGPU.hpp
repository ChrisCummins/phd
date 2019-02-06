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


#ifndef MONTECARLOASIAN_H_
#define MONTECARLOASIAN_H_

#define GROUP_SIZE 256

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

/*!
*  Header declarations
*/


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKThread.hpp"

using namespace appsdk;

#define CHECK_OPENCL_ERROR_RETURN_NULL(actual, msg) \
    if(actual != CL_SUCCESS) \
    { \
        std::cout<<"Error :"<<msg<<" Error Code :"<<actual<<std::endl; \
        std::cout << "Location : " << __FILE__ << ":" << __LINE__<< std::endl; \
        return NULL; \
    }

/**
 * MonteCarloAsianMultiGPU
 * Class implements OpenCL  Monte Carlo Simution sample for Asian Option pricing
 */

class MonteCarloAsianMultiGPU
{
    public:
        cl_int steps;                           /**< Steps for Asian Monte Carlo simution */
        cl_float initPrice;                     /**< Initial price */
        cl_float strikePrice;                   /**< Strike price */
        cl_float interest;                      /**< Interest rate */
        cl_float maturity;                      /**< maturity */

        cl_int noOfSum;                         /**< Number of excersize points */
        cl_int noOfTraj;                        /**< Number of samples */

        cl_double setupTime;                    /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;                   /**< time taken to run kernel and read result back */

        size_t maxWorkGroupSize;                /**< Max allowed work-items in a group */
        cl_uint maxDimensions;                  /**< Max group dimensions allowed */
        size_t *maxWorkItemSizes;               /**< Max work-items sizes in each dimensions */

        cl_float *sigma;                        /**< Array of sigma values */
        cl_float *price;                        /**< Array of price values */
        cl_float *vega;                         /**< Array of vega values */

        cl_float *refPrice;                     /**< Array of reference price values */
        cl_float *refVega;                      /**< Array of reference vega values */

        cl_uint *randNum;                       /**< Array of random numbers */

        cl_float *priceVals;                    /**< Array of price values for given samples */
        cl_float *priceDeriv;                   /**< Array of price derivative values for given samples */

        cl_context context;                     /**< CL context */
        cl_device_id *devices;                  /**< CL device list */

        cl_mem priceBuf;                        /**< CL memory buffer for sigma */
        cl_mem priceDerivBuf;                   /**< CL memory buffer for price */
        cl_mem randBuf;                         /**< CL memory buffer for random number */

        cl_command_queue commandQueue;          /**< CL command queue */
        cl_program program;                     /**< CL program  */
        cl_kernel kernel;                       /**< CL kernel */

        cl_int width;
        cl_int height;

        size_t kernelWorkGroupSize;             /**< Group size returned by kernel */
        size_t blockSizeX;                      /**< Group-size in x-direction */
        size_t blockSizeY;                      /**< Group-size in y-direction */

        int iterations;                         /**< Number of iterations for kernel execution */

        // Required only when async enabled
        cl_mem priceBufAsync;                    /**< CL memory buffer for sigma */
        cl_mem priceDerivBufAsync;               /**< CL memory buffer for price */
        cl_mem randBufAsync;                     /**< CL memory buffer for random number */

        // Required only when async and mapping enabled
        cl_float *priceValsAsync;                /**< Array of price values for given samples */
        cl_float *priceDerivAsync;               /**< Array of price derivative values for given samples */
        cl_bool noMultiGPUSupport;               /**< MultiGPU support is there or not*/
        int numGPUDevices;                       /**< Number of GPU devices **/
        int numCPUDevices;                       /**< Number of CPU devices **/
        int numDevices;                          /**< Number of devices **/
        cl_device_id *gpuDeviceIDs;              /**< Array to store device ids*/
        cl_command_queue *commandQueues;         /**< Array to store command queues*/
        cl_kernel *kernels;                      /**< Array of kernels**/
        cl_program *programs;                    /**< Array of programs**/
        cl_double *peakGflopsGPU;                /**< Peak Gflops of a GPU device */
        SDKDeviceInfo
        *devicesInfo;              /**< Array to store the device information */
        cl_int *numStepsPerGPU;                  /**< Array to store the number of steps per GPU*/
        cl_ulong totalLocalMemory;               /**< Max local memory allowed */
        cl_ulong usedLocalMemory;                /**< Used local memory */
        cl_mem *randBufs;                        /**< Array to store cl_mem objects*/
        cl_mem *priceBufs;                       /**< Array to store cl_mem objects*/
        cl_mem *priceDerivBufs;                  /**< Array to store cl_mem objects*/
        cl_mem *randBufsAsync;                   /**< Array to store cl_mem objects*/
        cl_mem *priceBufsAsync;                  /**< Array to store cl_mem objects*/
        cl_mem *priceDerivBufsAsync;             /**< Array to store cl_mem objects*/
        cl_int *cumulativeStepsPerGPU;
        SDKDeviceInfo
        deviceInfo;                /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;          /**< Structure to store kernel related info */
        SDKTimer *sampleTimer;                   /**< SDKTimer object */

        CLCommandArgs   *sampleArgs;             /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        MonteCarloAsianMultiGPU()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            steps = 8;
            initPrice = 50.f;
            strikePrice = 55.f;
            interest = 0.06f;
            maturity = 1.f;

            setupTime = 0;
            kernelTime = 0;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;

            sigma = NULL;
            price = NULL;
            vega = NULL;
            refPrice = NULL;
            refVega = NULL;
            randNum = NULL;
            priceVals = NULL;
            priceDeriv = NULL;
            devices = NULL;
            iterations = 1;
            priceValsAsync = NULL;
            priceDerivAsync = NULL;
            noMultiGPUSupport = false;
            kernels = NULL;
            programs = NULL;
            commandQueues = NULL;
            peakGflopsGPU = NULL;
            devicesInfo = NULL;
            numStepsPerGPU = NULL;
            randBufs = NULL;
            randBufsAsync = NULL;
            priceBufs = NULL;
            priceBufsAsync = NULL;
            priceDerivBufsAsync = NULL;
            priceDerivBufs = NULL;
            cumulativeStepsPerGPU = NULL;
            gpuDeviceIDs = NULL;
        }

        /**
         * Destructor
         */
        ~MonteCarloAsianMultiGPU()
        {
            FREE(sigma);
            FREE(price);
            FREE(vega);
            FREE(refPrice);
            FREE(refVega);

#ifdef _WIN32
            ALIGNED_FREE(randNum);
#else
            FREE(randNum);
#endif

            FREE(priceVals);
            FREE(priceDeriv);
            FREE(devices);
        }

        /**
         * Allocate and initialize host memory with appropriate values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupMonteCarloAsianMultiGPU();

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
         * Run OpenCL Bitonic Sort
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
         * @brief Left shift
         * @param input input to be shifted
         * @param shift shifting count
         * @param output result after shifting input
         */
        void lshift128(unsigned int* input, unsigned int shift, unsigned int* output);

        /**
         * @brief Right shift
         * @param input input to be shifted
         * @param shift shifting count
         * @param output result after shifting input
         */
        void rshift128(unsigned int* input, unsigned int shift, unsigned int* output);


        /**
         * @brief Generates gaussian random numbers by using
         *        Mersenenne Twister algo and box muller transformation
         * @param seedArray  seed
         * @param gaussianRand1 gaussian random number generated
         * @param gaussianRand2 gaussian random number generated
         * @param nextRand  generated seed for next usage
         */
        void generateRand(unsigned int* seed,
                          float *gaussianRand1,
                          float *gaussianRand2,
                          unsigned int* nextRand);

        /**
         * @brief   calculates the  price and vega for all trajectories
         */
        void calOutputs(float strikePrice, float* meanDeriv1,
                        float*  meanDeriv2, float* meanPrice1,
                        float* meanPrice2, float* pathDeriv1,
                        float* pathDeriv2, float* priceVec1, float* priceVec2);

        /**
         * @brief   Reference implementation for Monte Carlo simuation for
         *          Asian Option pricing
         */
        void cpuReferenceImpl();
        /**
        * Function: loadBalancing()
        * calculates the peak Gflops for each device and shares the work accordingly.
        **/
        int loadBalancing();

        int runCLKernelsMultiGPU(void);
};

#endif
