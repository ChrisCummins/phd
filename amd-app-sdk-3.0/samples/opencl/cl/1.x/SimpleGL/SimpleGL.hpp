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

#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#include <CL/cl_gl.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#ifdef _WIN32
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma warning( disable : 4996)
#endif

// Define DISPLAY_DEVICE_ACTIVE as it is not defined in MinGW
#ifdef _WIN32
#ifndef DISPLAY_DEVICE_ACTIVE
#define DISPLAY_DEVICE_ACTIVE    0x00000001
#endif
#endif

#define screenWidth  512
#define screenHeight 512

#define GROUP_SIZE 256

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    const cl_context_properties *properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

// Rename references to this dynamically linked function to avoid
// collision with static link version
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_proc
static clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;


/**
* SimpleGL
* Class implements OpenCL  SimpleGL sample
* Derived from SDKSample base class
*/


class SimpleGLSample
{

        cl_uint meshWidth;                  /**< mesh width */
        cl_uint meshHeight;                 /**< mesh height */
        cl_float* pos;                      /**< position vector */
        cl_float* refPos;                   /**< reference position vector */
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_mem posBuf;                      /**< CL Buffer for position vector */
        cl_program program;                 /**< CL program */
        cl_kernel kernel;                   /**< CL kernel */
        //size_t kernelWorkGroupSize;         /**< Group size returned by kernel */
        size_t groupSize;                   /**< Work-Group size */
        cl_device_id interopDeviceId;
        SDKDeviceInfo
        deviceInfo;                    /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;              /**< Structure to store kernel related info */
        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    private:

        int compareArray(const float* mat0, const float* mat1, unsigned int size);
    public:
        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        static SimpleGLSample *simpleGLSample;

        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (const char*)
        */
        explicit SimpleGLSample()
            : meshWidth(WINDOW_WIDTH),
              meshHeight(WINDOW_HEIGHT),
              pos(NULL),
              refPos(NULL),
              devices(NULL),
              groupSize(GROUP_SIZE)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        ~SimpleGLSample();

        /**
        * Timer functions
        */
        int createTimer()
        {
            return sampleTimer->createTimer();
        }

        int resetTimer(int handle)
        {
            return sampleTimer->resetTimer(handle);
        }

        int startTimer(int handle)
        {
            return sampleTimer->startTimer(handle);
        }

        double readTimer(int handle)
        {
            return sampleTimer->readTimer(handle);
        }

        /**
         * enableGLAndGLContext
         * creates a GL Context on a specified device and get its deviceId
         * @param hWnd Window Handle
         * @param hRC context of window
         * @param platform cl_platform_id selected
         * @param context associated cl_context
         * @param interopDevice cl_device_id of selected device
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
#ifdef _WIN32
        int enableGLAndGetGLContext(HWND hWnd,
                                    HDC &hDC,
                                    HGLRC &hRC,
                                    cl_platform_id platform,
                                    cl_context &context,
                                    cl_device_id &interopDevice);

        void disableGL(HWND hWnd, HDC hDC, HGLRC hRC);
#endif

        void displayFunc(void);

        void keyboardFunc( unsigned char key, int /*x*/, int /*y*/);

        /**
        * Allocate and initialize host memory array with random values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupSimpleGL();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();


        /**
        * Set values for kernels' arguments
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCLKernels();

        /**
        * Enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int executeKernel();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void SimpleGLCPUReference();

        /**
        * Initializing GL and get interoperable CL context
        * @param argc number of arguments
        * @param argv command line arguments
        * @
        * @return SDK_SUCCESS on success and SDK_FALIURE on failure.
        */
        int initializeGLAndGetCLContext(cl_platform_id platform,
                                        cl_context &context,
                                        cl_device_id &interopDevice);

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
        * Run OpenCL SimpleGL
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

        /**
        *  Compiles vertex, pixel shaders and create valid program object.
        *  @vsrc Vertex source string
        *  @psrc Pixel source string
        *  @return Returns valid program id or Zero
        */
        GLuint compileProgram(const char * vsrc, const char * psrc);

        /**
         *  Loads given Texture.
         *  @texture texture object
         *  @return SDK_SUCCESS true if loading is successful valid program id or Zero
         */
        int loadTexture(GLuint * texture);

    private:
};


