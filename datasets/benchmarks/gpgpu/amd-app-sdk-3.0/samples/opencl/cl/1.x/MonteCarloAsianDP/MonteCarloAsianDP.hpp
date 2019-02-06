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


#ifndef MONTECARLOASIANDP_H_
#define MONTECARLOASIANDP_H_

#define GROUP_SIZE 256

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

/**
 * MonteCarloAsian
 * Class implements OpenCL  Monte Carlo Simution sample for Asian Option pricing using double
 */

class MonteCarloAsianDP
{

        cl_int steps;                       /**< Steps for Asian Monte Carlo simution */
        cl_double initPrice;                 /**< Initial price */
        cl_double strikePrice;               /**< Strike price */
        cl_double interest;                  /**< Interest rate */
        cl_double maturity;                  /**< maturity */

        cl_int noOfSum;                     /**< Number of excersize points */
        cl_int noOfTraj;                    /**< Number of samples */

        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */

        cl_double *sigma;                    /**< Array of sigma values */
        cl_double *price;                    /**< Array of price values */
        cl_double *vega;                     /**< Array of vega values */

        cl_double *refPrice;                 /**< Array of reference price values */
        cl_double *refVega;                  /**< Array of reference vega values */

        cl_uint *randNum;                   /**< Array of random numbers */

        cl_double *priceVals;                /**< Array of price values for given samples */
        cl_double *priceDeriv;               /**< Array of price derivative values for given samples */

        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */

        cl_mem priceBuf;                    /**< CL memory buffer for sigma */
        cl_mem priceDerivBuf;               /**< CL memory buffer for price */
        cl_mem randBuf;                     /**< CL memory buffer for random number */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernel;                   /**< CL kernel */

        cl_int width;
        cl_int height;

        //size_t kernelWorkGroupSize;         /**< Group size returned by kernel */
        size_t blockSizeX;                  /**< Group-size in x-direction */
        size_t blockSizeY;                  /**< Group-size in y-direction */

        int iterations;                     /**< Number of iterations for kernel execution */

        cl_mem priceBufAsync;                    /**< CL memory buffer for sigma */
        cl_mem priceDerivBufAsync;               /**< CL memory buffer for price */
        cl_mem randBufAsync;                     /**< CL memroy buffer for random number */
        SDKDeviceInfo deviceInfo;     /**<Structure to store device related info */
        KernelWorkGroupInfo kernelInfo;/**< Structure to store kernel related info */
        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        MonteCarloAsianDP()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            steps = 10;
            initPrice = 50.0;
            strikePrice = 55.0;
            interest = 0.06;
            maturity = 1.0;

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

        }

        /**
         * Destructor
         */
        ~MonteCarloAsianDP()
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
        int setupMonteCarloAsianDP();


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
                          double *gaussianRand1,
                          double *gaussianRand2,
                          unsigned int* nextRand);

        /**
         * @brief   calculates the  price and vega for all trajectories
         */
        void calOutputs(double strikePrice, double* meanDeriv1,
                        double*  meanDeriv2, double* meanPrice1,
                        double* meanPrice2, double* pathDeriv1,
                        double* pathDeriv2, double* priceVec1, double* priceVec2);

        /**
         * @brief   Reference implementation for Monte Carlo simuation for
         *          Asian Option pricing
         */
        void cpuReferenceImpl();
};

#endif


