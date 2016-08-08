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


#ifndef NBODY_H_
#define NBODY_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define GROUP_SIZE 64

//For FLOPS calculation
#define KERNEL_FLOPS 20

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.2"

using namespace appsdk;

/**
* NBody
* Class implements OpenCL  NBody sample
*/

class NBody
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */

        cl_float delT;                      /**< dT (timestep) */
        cl_float espSqr;                    /**< Softening Factor*/
        cl_float* initPos;                  /**< initial position */
        cl_float* initVel;                  /**< initial velocity */
        cl_float* vel;                      /**< Output velocity */
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_mem particlePos[2];              // positions of particles
        cl_mem particleVel[2];              // velocity of particles
        int currentPosBufferIndex;
        unsigned int currentIterationCL;    // current iteration in the simulation
        float* mappedPosBuffer;             // mapped pointer of the position buffer
        int mappedPosBufferIndex;
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program */
        cl_kernel kernel;                   /**< CL kernel */
        size_t groupSize;                   /**< Work-Group size */

        int iterations;
        SDKDeviceInfo
        deviceInfo;                /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;          /**< Structure to store kernel related info */

        int fpsTimer;
        int timerNumFrames;

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    private:

        float random(float randMax, float randMin);

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        cl_uint numParticles;
        bool    isFirstLuanch;
        cl_event glEvent;
		cl_bool display;

        /**
        * Constructor
        * Initialize member variables
        */
        explicit NBody()
            : setupTime(0),
              kernelTime(0),
              delT(0.005f),
              espSqr(500.0f),
              initPos(NULL),
              initVel(NULL),
              vel(NULL),
              devices(NULL),
              groupSize(GROUP_SIZE),
              iterations(1),
              currentPosBufferIndex(0),
              currentIterationCL(0),
              mappedPosBuffer(NULL),
              fpsTimer(0),
              timerNumFrames(0),
              isFirstLuanch(true),
              glEvent(NULL),
			  display(false)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            numParticles = 1024;
        }

        ~NBody();

        /**
        * Allocate and initialize host memory array with random values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupNBody();

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
        * Set values for kernels' arguments
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCLKernels();

        /**
        * Enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void nBodyCPUReference(float* currentPos, float* currentVel
                               , float* newPos, float* newVel);


        float* getMappedParticlePositions();
        void releaseMappedParticlePositions();

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
        * Run OpenCL NBody
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


        // init the timer for FPS calculation
        void initFPSTimer()
        {
            timerNumFrames = 0;
            fpsTimer = sampleTimer->createTimer();
            sampleTimer->resetTimer(fpsTimer);
            sampleTimer->startTimer(fpsTimer);
        };

        // calculate FPS
        double getFPS()
        {
            sampleTimer->stopTimer(fpsTimer);
            double elapsedTime = sampleTimer->readTimer(fpsTimer);
            double fps = timerNumFrames/elapsedTime;
            timerNumFrames = 0;
            sampleTimer->resetTimer(fpsTimer);
            sampleTimer->startTimer(fpsTimer);
            return fps;
        };
};

#endif // NBODY_H_
