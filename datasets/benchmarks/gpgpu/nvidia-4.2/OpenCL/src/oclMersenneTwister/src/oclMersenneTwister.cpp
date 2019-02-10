#include <libcecl.h>
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

///////////////////////////////////////////////////////////////////////////////
// This sample implements Mersenne Twister random number generator
// and Cartesian Box-Muller transformation on the GPU
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>
#include "MersenneTwister.h"

// comment the below line if not doing Box-Muller transformation
#define DO_BOXMULLER
#define MAX_GPU_COUNT 8

// Reference CPU MT and Box-Muller transformation 
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Rand, int nPerRng, unsigned int seed);
#ifdef DO_BOXMULLER
extern "C" void BoxMullerRef(float *h_Rand, int nPerRng);
#endif

///////////////////////////////////////////////////////////////////////////////
//Load twister configurations
///////////////////////////////////////////////////////////////////////////////
void loadMTGPU(const char *fname, 
	       const unsigned int seed, 
	       mt_struct_stripped *h_MT,
	       const size_t size)
{
    FILE* fd = 0;
    #ifdef _WIN32
        // open the file for binary read
        errno_t err;
        if ((err = fopen_s(&fd, fname, "rb")) != 0)
    #else
        // open the file for binary read
        if ((fd = fopen(fname, "rb")) == 0)
    #endif
        {
            if(fd)
            {
                fclose (fd);
            }
	        oclCheckError(0, 1);
        }
  
    for (unsigned int i = 0; i < size; i++)
        fread(&h_MT[i], sizeof(mt_struct_stripped), 1, fd);
    fclose(fd);

    for(unsigned int i = 0; i < size; i++)
        h_MT[i].seed = seed;
}

///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    cl_context cxGPUContext;                        // OpenCL context
    cl_command_queue cqCommandQueue[MAX_GPU_COUNT]; // OpenCL command que
    cl_platform_id cpPlatform;                      // OpenCL platform
    cl_uint nDevice;                                // OpenCL device count
    cl_device_id* cdDevices;                        // OpenCL device list    
    cl_program cpProgram;                           // OpenCL program
    cl_kernel ckMersenneTwister = NULL;             // OpenCL kernel
    cl_kernel ckBoxMuller = NULL;                   // OpenCL kernel
    cl_mem *d_Rand, *d_MT;                          // OpenCL buffers
    cl_int ciErr1, ciErr2;                          // Error code var
    size_t globalWorkSize[1] = {MT_RNG_COUNT};      // 1D var for Total # of work items
    size_t localWorkSize[1] = {128};                // 1D var for # of work items in the work group	
    const int seed = 777;
    const int nPerRng = 5860;                       // # of recurrence steps, must be even if do Box-Muller transformation
    const int nRand = MT_RNG_COUNT * nPerRng;       // Output size
    const char *clSourcefile = "MersenneTwister.cl";// kernel file

    shrQAStart(argc, (char **)argv);

    shrSetLogFileName ("oclMersenneTwister.txt");
    shrLog("%s Starting, using %s...\n\n", argv[0], clSourcefile); 

    shrLog("Get platforms...\n");
    ciErr1 = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErr1, CL_SUCCESS);

    shrLog("Get devices...\n");
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &nDevice);
    oclCheckError(ciErr1, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(nDevice * sizeof(cl_device_id) );
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, nDevice, cdDevices, NULL);
    oclCheckError(ciErr1, CL_SUCCESS);

    shrLog("Create context...\n");
    cxGPUContext = CECL_CREATE_CONTEXT(0, nDevice, cdDevices, NULL, NULL, &ciErr1);
    oclCheckError(ciErr1, CL_SUCCESS);
    
    shrLog("CECL_CREATE_COMMAND_QUEUE\n"); 
    int id_device;
    if(shrGetCmdLineArgumenti(argc, argv, "device", &id_device)) // Set up command queue(s) for GPU specified on the command line
    {
        // get & log device index # and name
        cl_device_id cdDevice = cdDevices[id_device];

        // create a command que
        cqCommandQueue[0] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, 0, &ciErr1);
        oclCheckErrorEX(ciErr1, CL_SUCCESS, NULL);
        oclPrintDevInfo(LOGBOTH, cdDevice);
        nDevice = 1;   
    } 
    else 
    { // create command queues for all available devices        
        for (cl_uint i = 0; i < nDevice; i++) 
        {
            cqCommandQueue[i] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[i], 0, &ciErr1);
            oclCheckErrorEX(ciErr1, CL_SUCCESS, NULL);
        }
        for (cl_uint i = 0; i < nDevice; i++) oclPrintDevInfo(LOGBOTH, cdDevices[i]);
    }

    shrLog("\nUsing %d GPU(s)...\n\n", nDevice);

    shrLog("Initialization: load MT parameters and init host buffers...\n");
    mt_struct_stripped *h_MT = (mt_struct_stripped*)malloc(sizeof(mt_struct_stripped)*MT_RNG_COUNT); // MT para
    char *cDatPath = shrFindFilePath("MersenneTwister.dat", argv[0]);
    shrCheckError(cDatPath != NULL, shrTRUE);
    loadMTGPU(cDatPath, seed, h_MT, MT_RNG_COUNT);
    char *cRawPath = shrFindFilePath("MersenneTwister.raw", argv[0]);
    shrCheckError(cRawPath != NULL, shrTRUE);
    initMTRef(cRawPath);
    float **h_RandGPU = (float**)malloc(nDevice*sizeof(float*));
    float **h_RandCPU = (float**)malloc(nDevice*sizeof(float*));
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        h_RandGPU[iDevice] = (float*)malloc(sizeof(float)*nRand); // Host buffers for GPU output
        h_RandCPU[iDevice] = (float*)malloc(sizeof(float)*nRand); // Host buffers for CPU test
    }

    shrLog("Allocate memory...\n"); 
    d_MT = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    d_Rand = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        d_MT[iDevice] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(mt_struct_stripped)*MT_RNG_COUNT, NULL, &ciErr2);
        ciErr1 |= ciErr2;
        ciErr1 |= CECL_WRITE_BUFFER(cqCommandQueue[iDevice], d_MT[iDevice], CL_TRUE, 0, 
            sizeof(mt_struct_stripped)*MT_RNG_COUNT, h_MT, 0, NULL, NULL);
        d_Rand[iDevice] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * nRand, NULL, &ciErr2);
        ciErr1 |= ciErr2;
        oclCheckError(ciErr1, CL_SUCCESS); 
    }

    shrLog("Create and build program from %s...\n", clSourcefile);
    size_t szKernelLength; // Byte size of kernel code
    char *cSourcePath = shrFindFilePath(clSourcefile, argv[0]);
    shrCheckError(cSourcePath != NULL, shrTRUE);
    char *cMersenneTwister = oclLoadProgSource(cSourcePath, "// My comment\n", &szKernelLength);
    oclCheckError(cMersenneTwister != NULL, shrTRUE);
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cMersenneTwister, &szKernelLength, &ciErr1);
    ciErr1 |= CECL_PROGRAM(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, (double)ciErr1, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "MersenneTwister.ptx");
        oclCheckError(ciErr1, CL_SUCCESS); 
    }

    shrLog("Call Mersenne Twister kernel on GPU...\n\n"); 
    ckMersenneTwister = CECL_KERNEL(cpProgram, "MersenneTwister", &ciErr1);
#ifdef DO_BOXMULLER
    ckBoxMuller = CECL_KERNEL(cpProgram, "BoxMuller", &ciErr1);
#endif

#ifdef GPU_PROFILING
    int numIterations = 100;
    for (int i = -1; i < numIterations; i++)
    {
		if (i == 0) 
		{
			for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
			{
				clFinish(cqCommandQueue[iDevice]);
			}
			shrDeltaT(1);
		}
#endif
        for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
        {
            ciErr1 |= CECL_SET_KERNEL_ARG(ckMersenneTwister, 0, sizeof(cl_mem), (void*)&d_Rand[iDevice]);
            ciErr1 |= CECL_SET_KERNEL_ARG(ckMersenneTwister, 1, sizeof(cl_mem), (void*)&d_MT[iDevice]);
            ciErr1 |= CECL_SET_KERNEL_ARG(ckMersenneTwister, 2, sizeof(int),    (void*)&nPerRng);
            ciErr1 |= CECL_ND_RANGE_KERNEL(cqCommandQueue[iDevice], ckMersenneTwister, 1, NULL, 
                globalWorkSize, localWorkSize, 0, NULL, NULL);
            oclCheckError(ciErr1, CL_SUCCESS); 
        }
    	
    #ifdef DO_BOXMULLER 
        for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
        {
            ciErr1 |= CECL_SET_KERNEL_ARG(ckBoxMuller, 0, sizeof(cl_mem), (void*)&d_Rand[iDevice]);
            ciErr1 |= CECL_SET_KERNEL_ARG(ckBoxMuller, 1, sizeof(int),    (void*)&nPerRng);
            ciErr1 |= CECL_ND_RANGE_KERNEL(cqCommandQueue[iDevice], ckBoxMuller, 1, NULL, 
                globalWorkSize, localWorkSize, 0, NULL, NULL);
            oclCheckError(ciErr1, CL_SUCCESS); 
        }
    #endif
#ifdef GPU_PROFILING
    }
	for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clFinish(cqCommandQueue[iDevice]);
    }
    double gpuTime = shrDeltaT(1)/(double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclMersenneTwister, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n", 
           ((double)nDevice * (double)nRand * 1.0E-9 / gpuTime), gpuTime, nDevice * nRand, nDevice, localWorkSize[0]);    
#endif

    shrLog("\nRead back results...\n"); 
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ciErr1 |= CECL_READ_BUFFER(cqCommandQueue[iDevice], d_Rand[iDevice], CL_TRUE, 0, 
            sizeof(cl_float) * nRand, h_RandGPU[iDevice], 0, NULL, NULL);
        oclCheckError(ciErr1, CL_SUCCESS); 
    }

    shrLog("Compute CPU reference solution...\n");
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        RandomRef(h_RandCPU[iDevice], nPerRng, seed);
#ifdef DO_BOXMULLER
        BoxMullerRef(h_RandCPU[iDevice], nPerRng);
#endif
    }

    shrLog("Compare CPU and GPU results...\n");
    double sum_delta = 0;
    double sum_ref   = 0;
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        for(int i = 0; i < MT_RNG_COUNT; i++)
            for(int j = 0; j < nPerRng; j++) {
	        double rCPU = h_RandCPU[iDevice][i * nPerRng + j];
	        double rGPU = h_RandGPU[iDevice][i + j * MT_RNG_COUNT];
	        double delta = fabs(rCPU - rGPU);
	        sum_delta += delta;
	        sum_ref   += fabs(rCPU);
	    }
    }
    double L1norm = sum_delta / sum_ref;
    shrLog("L1 norm: %E\n\n", L1norm);

    // NOTE:  Most properly this should be done at any of the exit points above, but it is omitted elsewhere for clarity.
    shrLog("Release CPU buffers and OpenCL objects...\n"); 
    clReleaseKernel(ckMersenneTwister);
    #ifdef DO_BOXMULLER
        clReleaseKernel(ckBoxMuller);
    #endif    
    clReleaseProgram(cpProgram);    
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        free(h_RandGPU[iDevice]); 
        free(h_RandCPU[iDevice]);
        clReleaseMemObject(d_Rand[iDevice]);
        clReleaseMemObject(d_MT[iDevice]);        
        clReleaseCommandQueue(cqCommandQueue[iDevice]);
    }
    free(h_MT);
    free(h_RandGPU);
    free(h_RandCPU);
    free(d_Rand);
    free(d_MT);
    free(cMersenneTwister);
    free(cSourcePath);
    free(cRawPath);
    free(cDatPath);
    free(cdDevices);
    clReleaseContext(cxGPUContext);

    // finish
    shrQAFinishExit(argc, (const char **)argv, (L1norm < 1e-6) ? QA_PASSED : QA_FAILED);

    shrEXIT(argc, argv);
}
