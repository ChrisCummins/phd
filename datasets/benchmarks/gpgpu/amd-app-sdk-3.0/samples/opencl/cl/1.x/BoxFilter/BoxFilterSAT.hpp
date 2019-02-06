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

#ifndef BOX_FILTER_SAT_H_
#define BOX_FILTER_SAT_H_



#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

#define INPUT_IMAGE "BoxFilter_Input.bmp"
#define OUTPUT_IMAGE "BoxFilter_Output.bmp"

#define GROUP_SIZE 256
#define FILTER 6          //Filter size : FILTER x FILTER
#define SAT_FETCHES 16     //Number of fetches in computing SAT

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

/**
* BoxFilter
* Class implements OpenCL Box Filter sample
*/

class BoxFilterSAT
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */
        cl_uchar4* inputImageData;          /**< Input bitmap data to device */
        cl_uchar4* outputImageData;         /**< Output from device */
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_mem inputImageBuffer;            /**< CL memory buffer for input Image*/
        cl_mem tempImageBuffer0;
        cl_mem tempImageBuffer1;
        cl_mem outputImageBuffer;           /**< CL memory buffer for Output Image*/
        cl_uchar4*
        verificationOutput;       /**< Output array for reference implementation */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernel;                   /**< CL kernel */
        SDKBitMap inputBitmap;   /**< Bitmap class object */
        uchar4* pixelData;       /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                     /**< Height of image */
        cl_bool byteRWSupport;
        size_t kernelWorkGroupSize;         /**< Group Size returned by kernel */
        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
        int iterations;                     /**< Number of iterations for kernel execution */
        cl_uint n;                          /**< Number of horizontal passes for SAT calculation */
        cl_uint m;                          /**< Number of vertical passes for SAT calculation */
        cl_mem *satHorizontalBuffer;        /**< Pointer to an array of sat horizontal-pass buffers */
        cl_mem *satVerticalBuffer;          /**< Pointer to an array of sat vertical-pass buffers */
        cl_uint rHorizontal;                /**< Number of fetches for a pixel in horizontal SAT computation */
        cl_uint rVertical;                  /**< Number of fetches for a pixel in vertical SAT computation */
        cl_kernel horizontalSAT0;           /**< first invocation of horizontal SAT kernel */
        cl_kernel horizontalSAT;            /**< Rest all kernels for horizontalSAT wil be same */
        cl_kernel verticalSAT;              /**< All kernels for vertical SAT computation are same */
        cl_uint filterWidth;                /**< Width of filter */
        SDKDeviceInfo deviceInfo;           /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfo,
                            kernelInfoHSAT0,
                            kernelInfoHSAT,
                            kernelInfoVSAT; /**< Structure to store kernel related info */

        SDKTimer    *sampleTimer;           /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;        /**< CLCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readInputImage(std::string inputImageName);

        /**
        * Write to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Constructor
        * Initialize member variables
        */
        BoxFilterSAT()
            :inputImageData(NULL),
             outputImageData(NULL),
             verificationOutput(NULL),
             byteRWSupport(true)
        {
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            rHorizontal = SAT_FETCHES;
            rVertical = SAT_FETCHES;
            satHorizontalBuffer = NULL;
            satVerticalBuffer = NULL;
            filterWidth = FILTER;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        ~BoxFilterSAT()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupBoxFilter();

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
        * Run a particular SAT computation kernel
        * @param kernel kernel to be executed
        * @param input input buffer to the kernel
        * @param output output buffer to the kernel
        * @param pass current pass number
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        **/
        int runSatKernel(cl_kernel kernel,
                         cl_mem *input,
                         cl_mem *output,
                         cl_uint pass,
                         cl_uint r);

        int runBoxFilterKernel();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void boxFilterCPUReference();

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

        int runSATversion(int argc, char * argv[]);
		
		unsigned char clampToUchar(int n);
};

#endif // SOBEL_FILTER_H_
