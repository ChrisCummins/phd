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

#ifndef RECURSIVE_GAUSSIAN_H_
#define RECURSIVE_GAUSSIAN_H_

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define INPUT_IMAGE "RecursiveGaussian_Input.bmp"
#define OUTPUT_IMAGE "RecursiveGaussian_Output.bmp"

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

#define GROUP_SIZE 256

/**
* Custom type for gaussian parameters
* precomputation
*/
typedef struct _GaussParms
{
    float nsigma;
    float alpha;
    float ema;
    float ema2;
    float b1;
    float b2;
    float a0;
    float a1;
    float a2;
    float a3;
    float coefp;
    float coefn;
} GaussParms, *pGaussParms;



/**
* Recursive Gaussian
* Class implements OpenCL Recursive Gaussian sample
*/

class RecursiveGaussian
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */

        cl_uchar4* inputImageData;          /**< Input bitmap data to device */
        cl_uchar4* outputImageData;         /**< Output from device */
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_mem inputImageBuffer;            /**< CL memory buffer for input Image*/
        cl_mem tempImageBuffer;             /**< CL memory buffer for storing the transpose of the image*/
        cl_mem outputImageBuffer;           /**< CL memory buffer for Output Image*/
        cl_uchar4*
        verificationInput;       /**< Input array for reference implementation */
        cl_uchar4*
        verificationOutput;      /**< Output array for reference implementation */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernelTranspose;          /**< CL kernel for transpose*/
        cl_kernel kernelRecursiveGaussian;  /**< CL Kernel for gaussian filter */
        SDKBitMap inputBitmap;              /**< Bitmap class object */
        uchar4* pixelData;                  /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        GaussParms
        oclGP;                   /**< instance of struct to hold gaussian parameters */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                     /**< Height of image */
        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
        size_t blockSize;                   /**< block size for transpose kernel */
        int iterations;                     /**< Number of iterations for kernel execution */
        SDKDeviceInfo deviceInfo;/**< Structure to store device information*/
        KernelWorkGroupInfo transposeKernelInfo,
                            RGKernelInfo;/**< Structure to store kernel related info */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readInputImage(std::string inputImageName);

        /**
        * Write output to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Preprocess gaussian parameters
        * @param fSigma sigma value
        * @param iOrder order
        * @param pGp pointer to gaussian parameter object
        */
        void computeGaussParms(float fSigma, int iOrder, GaussParms* pGP);

        /**
        * RecursiveGaussian on CPU (for verification)
        * @param input input image
        * @param output output image
        * @param width width of image
        * @param height height of image
        * @param a0..a3, b1, b2, coefp, coefn gaussian parameters
        */
        void recursiveGaussianCPU(cl_uchar4* input, cl_uchar4* output,
                                  const int width, const int height,
                                  const float a0, const float a1,
                                  const float a2, const float a3,
                                  const float b1, const float b2,
                                  const float coefp, const float coefn);

        /**
        * Transpose on CPU (for verification)
        * @param input input image
        * @param output output image
        * @param width width of input image
        * @param height height of input image
        */
        void transposeCPU(cl_uchar4* input, cl_uchar4* output,
                          const int width, const int height);

        /**
        * Constructor
        * Initialize member variables
        */
        RecursiveGaussian()
            : inputImageData(NULL),
              outputImageData(NULL),
              verificationOutput(NULL)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            blockSize = 1;
            iterations = 1;
        }

        ~RecursiveGaussian()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupRecursiveGaussian();

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
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        void recursiveGaussianCPUReference();

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
        * Run OpenCL Sobel Filter
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

#endif // RECURSIVE_GAUSSIAN_H_
