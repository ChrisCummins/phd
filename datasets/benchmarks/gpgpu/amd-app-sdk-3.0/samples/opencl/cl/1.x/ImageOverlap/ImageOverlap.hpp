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

#ifndef SIMPLE_IMAGE_H_
#define SIMPLE_IMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"

#define MAP_IMAGE "ImageOverlap_map.bmp"
#define MAP_VERIFY_IMAGE "ImageOverlap_verify_map.bmp"

#define GROUP_SIZE 256

#ifndef min
#define min(a, b)            (((a) < (b)) ? (a) : (b))
#endif

/**
* ImageOverlap
* Class implements OpenCL Simple Image sample

*/

class ImageOverlap
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */
        cl_uchar4* mapImageData;            /**< load bitmap data to device */
        cl_uchar4* fillImageData;           /**< Output from device for 2D copy*/
        cl_uchar4* outputImageData;         /**< Output from device for 3D copy*/
        cl_uchar4* verificationImageData;      /**< Verify Output */

        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */

        cl_mem fillImage;                   /**< CL image buffer for fill Image*/
        cl_mem mapImage;                    /**< CL image buffer for map Image*/
        cl_mem outputImage;                 /**< CL image buffer for output Image*/

        cl_command_queue commandQueue[3];   /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_event eventlist[2];              /**< CL event  */
        cl_event enqueueEvent;              /**< CL event  */
        cl_kernel kernelOverLap;            /**< CL kernel */

        SDKBitMap mapBitmap;     /**< Bitmap class object */
        SDKBitMap verifyBitmap;  /**< Bitmap class object */
        uchar4* pixelData;       /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                     /**< Height of image */

        size_t kernelOverLapWorkGroupSize;         /**< Group Size returned by kernel */
        size_t kernel3DWorkGroupSize;         /**< Group Size returned by kernel */

        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */

        int iterations;                     /**< Number of iterations for kernel execution */
        cl_bool imageSupport;               /**< Flag to check whether images are supported */
        cl_image_format imageFormat;        /**< Image format descriptor */
        cl_image_desc image_desc;
        cl_map_flags mapFlag;
        SDKDeviceInfo
        deviceInfo;                    /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;              /**< Structure to store kernel related info */

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readImage(std::string mapImageName,std::string verifyImageName);

        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (string)
        */
        ImageOverlap()
            :
            mapImageData(NULL),
            fillImageData(NULL),
            outputImageData(NULL),
            verificationImageData(NULL)
        {
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
            imageFormat.image_channel_order = CL_RGBA;
            image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            image_desc.image_width = width;
            image_desc.image_height = height;
            image_desc.image_depth = 0;
            image_desc.image_array_size = 0;
            image_desc.image_row_pitch = 0;
            image_desc.image_slice_pitch = 0;
            image_desc.num_mip_levels = 0;
            image_desc.num_samples = 0;
            image_desc.buffer = NULL;
            mapFlag = CL_MAP_READ | CL_MAP_WRITE;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }



        ~ImageOverlap()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupImageOverlap();

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
        * @return  SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void ImageOverlapCPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        */
        int initialize();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return  SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL ImageOverlap
        * @return  SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return  SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return  SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();
};

#endif // SIMPLE_IMAGE_H_
