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


#ifndef FLUID_SIMULATION2D_H_
#define FLUID_SIMULATION2D_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"

#define GROUP_SIZE  256
#define LBWIDTH     256
#define LBHEIGHT    256

int winwidth = LBWIDTH;
int winheight = LBHEIGHT;


/**
* FluidSimulation2D
* Class implements OpenCL  FluidSimulation2D sample

*/

class FluidSimulation2D
{
        cl_double setupTime;                        /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;                       /**< time taken to run kernel and read result back */

        size_t maxWorkGroupSize;                    /**< Max allowed work-items in a group */
        cl_uint maxDimensions;                      /**< Max group dimensions allowed */
        size_t* maxWorkItemSizes;                   /**< Max work-items sizes in each dimensions */
        cl_ulong totalLocalMemory;                  /**< Max local memory allowed */
        cl_ulong usedLocalMemory;                   /**< Used local memory */

        int dims[2];                                /**< Dimension of LBM simulation area */

        // 2D Host buffers
        cl_double *rho;                              /**< Density */
        cl_double2 *u;                               /**< Velocity */
        cl_double *h_if0, *h_if1234, *h_if5678;      /**< Host input buffers */
        cl_double *h_of0, *h_of1234, *h_of5678;      /**< Host output buffers */

        cl_double *v_ef0, *v_ef1234,
                  *v_ef5678;      /**< Host Eq distribution buffers for verification */
        cl_double *v_of0, *v_of1234,
                  *v_of5678;      /**< Host output buffers for verification */

        cl_bool *h_type;                            /**< Cell Type - Boundary = 1 or Fluid = 0 */
        cl_double *h_weight;                         /**< Weights for each direction */
        cl_double8 dirX, dirY;                       /**< Directions */

        // Device buffers
        cl_mem d_if0, d_if1234, d_if5678;           /**< Input distributions */
        cl_mem d_of0, d_of1234, d_of5678;           /**< Output distributions */
        cl_mem type;                                /**< Constant bool array for position type = boundary or fluid */
        cl_mem weight;                              /**< Weights for each distribution */
        cl_mem velocity;                            /**< 2D Velocity vector buffer */

        //OpenCL objects
        cl_context          context;
        cl_device_id        *devices;
        cl_command_queue    commandQueue;
        cl_program program;
        cl_kernel  kernel;

        //size_t kernelWorkGroupSize;                     /**< Group size returned by kernel */
        size_t groupSize;                               /**< Work-Group size */
        int iterations;

        cl_bool reqdExtSupport;
        SDKDeviceInfo
        deviceInfo;            /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;      /**< Structure to store kernel related info */

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        void reset();
        cl_bool isBoundary(int x, int y);
        bool isFluid(int x, int y);
        cl_double2 getVelocity(int x, int y);
        void setSite(int x, int y, bool cellType, double u[2]);
        void setUOutput(int x, int y, double u[2]);

        // Host functions for verification
        void collide(int x, int y);
        void streamToNeighbors(int x, int y);

        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (string)
        */
        FluidSimulation2D()
            :
            setupTime(0),
            kernelTime(0),

            devices(NULL),
            maxWorkItemSizes(NULL),
            groupSize(GROUP_SIZE),
            iterations(1),
            reqdExtSupport(true)
        {
            dims[0] = LBWIDTH;
            dims[1] = LBHEIGHT;
            rho = NULL;
            u = NULL;
            h_if0 = NULL;
            h_if1234 = NULL;
            h_if1234 = NULL;
            h_of0 = NULL;
            h_of1234 = NULL;
            h_of1234 = NULL;
            v_ef0 = NULL;
            v_ef1234 = NULL;
            v_ef5678 = NULL;
            v_of0 = NULL;
            v_of1234 = NULL;
            v_of5678 = NULL;
            h_type = NULL;
            h_weight = NULL;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        ~FluidSimulation2D();

        /**
        * Allocate and initialize host memory array with random values
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int setupFluidSimulation2D();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * OpenCL related initializations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Set values for kernels' arguments
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int setupCLKernels();

        /**
        * Enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void CPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int initialize();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL FluidSimulation2D
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_UCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();

};

#endif // FLUID_SIMULATION2D_H_
