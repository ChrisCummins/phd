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


#ifndef MANDELBROT_H_
#define MANDELBROT_H_


/**
 * Header Files
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>


#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

#define MAX_ITER 16384
#define MIN_ITER 32
#define MAX_DEVICES 4

/**
 * Mandelbrot
 * Class implements OpenCL Mandelbrot sample

 */

class Mandelbrot
{
        cl_uint
        seed;                      /**< Seed value for random number generation */
        cl_double
        setupTime;                      /**< Time for setting up Opencl */
        cl_double
        totalKernelTime;                      /**< Time for kernel execution */
        cl_double
        totalProgramTime;                      /**< Time for program execution */
        cl_uint
        *verificationOutput;                     /**< Output array from reference implementation */

        bool
        enableDouble;               /**< enables double data type */
        bool
        enableFMA;                  /**< enables Fused Multiply-Add */
        cl_double                xpos;                      /**< x-coordinate for set */
        cl_double                ypos;                      /**< y-coordinate for set */
        cl_double               xsize;                      /**< window width for set */
        cl_double
        ysize;                      /**< window height for set */
        cl_double               xstep;                      /**< x-increment for set */
        cl_double               ystep;                      /**< y-increment for set */

        cl_double               leftx;
        cl_double               topy;
        cl_double               topy0;

        std::string          xpos_str;
        std::string          ypos_str;
        std::string         xsize_str;
        cl_int
        maxIterations;                           /**< paramters of mandelbrot */
        cl_context            context;                          /**< CL context */
        cl_device_id         *devices;                          /**< CL device list */
        cl_uint
        numDevices;                          /**< Number of devices matching our needs */
        cl_mem           outputBuffer[MAX_DEVICES];             /**< CL memory buffer */
        cl_command_queue commandQueue[MAX_DEVICES];             /**< CL command queue */
        cl_program            program;                          /**< CL program  */
        cl_kernel       kernel_vector[MAX_DEVICES];             /**< CL kernel */
        cl_int
        width;                          /**< width of the output image */
        cl_int
        height;                          /**< height of the output image */
        size_t    kernelWorkGroupSize;                          /**< Group Size returned by kernel */
        int
        iterations;                          /**< Number of iterations for kernel execution */
        cl_int                  bench;                          /**< Get performance */
        cl_int
        benched;                          /**< Flag indicating we benchmarked last run */
        cl_double                time;                          /**< Elapsed time */
        cl_device_type
        dType;                          /**< OpenCL device type */
        size_t globalThreads;
        size_t localThreads ;
        SDKDeviceInfo
        deviceInfo;                    /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;              /**< Structure to store kernel related info */


        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        cl_uint *output;                                        /**< Output array */


        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        Mandelbrot()

        {
            seed = 123;
            output = NULL;
            verificationOutput = NULL;
            xpos = 0.0;
            ypos = 0.0;
            width = 256;
            height = 256;
            xstep = (4.0/(double)width);
            ystep = (-4.0/(double)height);
            xpos_str = "";
            ypos_str = "";
            xsize_str = "";
            maxIterations = 1024;
            setupTime = 0;
            totalKernelTime = 0;
            iterations = 1;
            bench = 0;
            benched = 0;
            enableDouble = false;
            enableFMA = false;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupMandelbrot();

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
         * Mandelbrot image generated with CPU reference implementation
         * @param verificationOutput mandelbrot images is stored in this
         * @param mandelbrotImage    mandelbrot images is stored in this
         * @param scale              Represents the distance from which the fractal
         *                           is being seen if this is greater more area and
         *                           less detail is seen
         * @param maxIterations      More iterations gives more accurate mandelbrot image
         * @param width              size of the image
         */
        void mandelbrotRefFloat(cl_uint * verificationOutput,
                                cl_float leftx,
                                cl_float topy,
                                cl_float xstep,
                                cl_float ystep,
                                cl_int maxIterations,
                                cl_int width,
                                cl_int bench);

        /**
         * Mandelbrot image generated with CPU reference implementation
         * @param verificationOutput Mandelbrot images is stored in this
         * @param mandelbrotImage    mandelbrot images is stored in this
         * @param scale              Represents the distance from which the fractal
         *                           is being seen if this is greater more area and
         *                           less detail is seen
         * @param maxIterations      More iterations gives more accurate Mandelbrot image
         * @param width              size of the image
         */
        void
        mandelbrotRefDouble(
            cl_uint * verificationOutput,
            cl_double posx,
            cl_double posy,
            cl_double stepSizeX,
            cl_double stepSizeY,
            cl_int maxIterations,
            cl_int width,
            cl_int bench);


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
         * Run OpenCL Mandelbrot Sample
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

        /*
         * get window Width
         */
        cl_uint getWidth(void);

        /*
         * get window Height
         */
        cl_uint getHeight(void);

        inline cl_uint getMaxIterations(void)
        {
            return maxIterations;
        }
        inline void setMaxIterations(cl_uint maxIter)
        {
            maxIterations = maxIter;
        }

        inline cl_double getXSize(void)
        {
            return xsize;
        }
        inline void setXSize(cl_double xs)
        {
            xsize = xs;
        }
        inline cl_double getXStep(void)
        {
            return xstep;
        }
        inline cl_double getYStep(void)
        {
            return ystep;
        }
        inline cl_double getXPos(void)
        {
            return xpos;
        }
        inline cl_double getYPos(void)
        {
            return ypos;
        }
        inline void setXPos(cl_double xp)
        {
            if (xp < -2.0)
            {
                xp = -2.0;
            }
            else if (xp > 2.0)
            {
                xp = 2.0;
            }
            xpos = xp;
        }
        inline void setYPos(cl_double yp)
        {
            if (yp < -2.0)
            {
                yp = -2.0;
            }
            else if (yp > 2.0)
            {
                yp = 2.0;
            }
            ypos = yp;
        }
        inline void setBench(cl_int b)
        {
            bench = b;
        }
        inline cl_int getBenched(void)
        {
            return benched;
        }
        inline cl_int getTiming(void)
        {
            return sampleArgs->timing;
        }

        /*
         * get pixels to be displayed
         */
        cl_uint * getPixels(void);

        /*
         * if showWindow returns true then a window with mandelbrot set is displayed
         */
        cl_bool showWindow(void);
};

#endif
