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

#include <oclUtils.h>
#include <shrQATest.h>
#include "HMM.h"

#define MAX_GPU_COUNT 8

// forward declaractions
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit);
int ViterbiCPU(float &viterbiProb,
               int *viterbiPath,
               int *obs, 
               const int &nObs, 
               float *initProb,
               float *mtState, 
               const int &nState,
               float *mtEmit);



// main function
//*****************************************************************************
int main(int argc, const char **argv)
{
    cl_platform_id cpPlatform;      // OpenCL platform
    cl_uint nDevice;                // OpenCL device count
    cl_device_id* cdDevices;        // OpenCL device list    
    cl_context cxGPUContext;        // OpenCL context
    cl_command_queue cqCommandQue[MAX_GPU_COUNT];             // OpenCL command que
    cl_int ciErrNum = 1;            // Error code var

    shrQAStart(argc, (char **)argv);

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("clGetPlatformID...\n"); 

    //Get all the devices
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    // Get command line device options and config accordingly
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("\n  # of Compute Units = %u\n", uiNumComputeUnits); 
	
    shrSetLogFileName ("oclHiddenMarkovModel.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    shrLog("Get platform...\n");
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("Get devices...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &nDevice);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(nDevice * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, nDevice, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("CECL_CREATE_CONTEXT\n");
    cxGPUContext = CECL_CREATE_CONTEXT(0, nDevice, cdDevices, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("CECL_CREATE_COMMAND_QUEUE\n"); 
    int id_device;
    if(shrGetCmdLineArgumenti(argc, argv, "device", &id_device)) // Set up command queue(s) for GPU specified on the command line
    {
        // create a command que
        cqCommandQue[0] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[id_device], 0, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
        oclPrintDevInfo(LOGBOTH, cdDevices[id_device]);
        nDevice = 1;   
    } 
    else 
    { // create command queues for all available devices        
        for (cl_uint i = 0; i < nDevice; i++) 
        {
            cqCommandQue[i] = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[i], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
        }
        for (cl_uint i = 0; i < nDevice; i++) oclPrintDevInfo(LOGBOTH, cdDevices[i]);
    }

    shrLog("\nUsing %d GPU(s)...\n\n", nDevice);
	int wgSize;
	if (!shrGetCmdLineArgumenti(argc, argv, "work-group-size", &wgSize)) 
	{
		wgSize = 256;
	}

    shrLog("Init Hidden Markov Model parameters\n");
    int nState = 256*16; // number of states, must be a multiple of 256
    int nEmit  = 128; // number of possible observations
    
    float *initProb = (float*)malloc(sizeof(float)*nState); // initial probability
    float *mtState  = (float*)malloc(sizeof(float)*nState*nState); // state transition matrix
    float *mtEmit   = (float*)malloc(sizeof(float)*nEmit*nState); // emission matrix
    initHMM(initProb, mtState, mtEmit, nState, nEmit);

    // define observational sequence
    int nObs = 100; // size of observational sequence
    int **obs = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathCPU = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathGPU = (int**)malloc(nDevice*sizeof(int*));
    float *viterbiProbCPU = (float*)malloc(nDevice*sizeof(float)); 
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        obs[iDevice] = (int*)malloc(sizeof(int)*nObs);
        for (int i = 0; i < nObs; i++)
            obs[iDevice][i] = i % 15;
        viterbiPathCPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
        viterbiPathGPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
    }

    shrLog("# of states = %d\n# of possible observations = %d \nSize of observational sequence = %d\n\n",
        nState, nEmit, nObs);

    shrLog("Compute Viterbi path on GPU\n\n");

    HMM **oHmm = (HMM**)malloc(nDevice*sizeof(HMM*));
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        oHmm[iDevice] = new HMM(cxGPUContext, cqCommandQue[iDevice], initProb, mtState, mtEmit, nState, nEmit, nObs, argv[0], wgSize);
    }

    cl_mem *vProb = (cl_mem*)malloc(sizeof(cl_mem)*nDevice);
    cl_mem *vPath = (cl_mem*)malloc(sizeof(cl_mem)*nDevice);
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        vProb[iDevice] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &ciErrNum);
        vPath[iDevice] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int)*nObs, NULL, &ciErrNum);
    }

#ifdef GPU_PROFILING
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clFinish(cqCommandQue[iDevice]);;
    }
	shrDeltaT(1);
#endif

    size_t szWorkGroup;
	for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
	{
		szWorkGroup = oHmm[iDevice]->ViterbiSearch(vProb[iDevice], vPath[iDevice], obs[iDevice]);
	}

	for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
	{
	  clFinish(cqCommandQue[iDevice]);
	}

#ifdef GPU_PROFILING
    double dElapsedTime = shrDeltaT(1);
    shrLogEx(LOGBOTH | MASTER, 0, "oclHiddenMarkovModel, Throughput = %.4f GB/s, Time = %.5f s, Size = %u items, NumDevsUsed = %u, Workgroup = %u\n",
        (1.0e-9 * 2.0 * sizeof(float) * nDevice * nState * nState * (nObs-1))/dElapsedTime, dElapsedTime, (nDevice * nState * nObs), nDevice, szWorkGroup); 

#endif

    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ciErrNum = CECL_READ_BUFFER(cqCommandQue[iDevice], vPath[iDevice], CL_TRUE, 0, sizeof(int)*nObs, viterbiPathGPU[iDevice], 0, NULL, NULL);
    }

    shrLog("\nCompute Viterbi path on CPU\n");
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ciErrNum = ViterbiCPU(viterbiProbCPU[iDevice], viterbiPathCPU[iDevice], obs[iDevice], nObs, initProb, mtState, nState, mtEmit);
    }
    
    if (!ciErrNum)
    {
        shrEXIT(argc, argv);
    }

    bool pass = true;
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        for (int i = 0; i < nObs; i++)
        {
            if (viterbiPathCPU[iDevice][i] != viterbiPathGPU[iDevice][i]) 
            {
                pass = false;
                break;
            }
        }
    }
        
    // NOTE:  Most properly this should be done at any of the exit points above, but it is omitted elsewhere for clarity.
    shrLog("Release CPU buffers and OpenCL objects...\n"); 
    free(initProb);
    free(mtState);
    free(mtEmit);
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        free(obs[iDevice]);
        free(viterbiPathCPU[iDevice]);
        free(viterbiPathGPU[iDevice]);
        delete oHmm[iDevice];
        clReleaseCommandQueue(cqCommandQue[iDevice]);
    }
    free(obs);
    free(viterbiPathCPU);
    free(viterbiPathGPU);
    free(viterbiProbCPU);
    free(cdDevices);
    free(oHmm);
    clReleaseContext(cxGPUContext);

    // finish
    shrQAFinishExit(argc, (const char **)argv, pass ? QA_PASSED : QA_FAILED);

    shrEXIT(argc, argv);
}

// initialize initial probability, state transition matrix and emission matrix with random 
// numbers. Note that this does not satisfy the normalization property of the state matrix. 
// However, since the algorithm does not use this property, for testing purpose this is fine.
//*****************************************************************************
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit)
{
    if (nState <= 0 || nEmit <=0) return 0;

    // Initialize initial probability

    for (int i = 0; i < nState; i++) initProb[i] = rand();
    float sum = 0.0;
    for (int i = 0; i < nState; i++) sum += initProb[i];
    for (int i = 0; i < nState; i++) initProb[i] /= sum;

    // Initialize state transition matrix

    for (int i = 0; i < nState; i++) {
        for (int j = 0; j < nState; j++) {
            mtState[i*nState + j] = rand();
            mtState[i*nState + j] /= RAND_MAX;
        }
    }

    // init emission matrix

    for (int i = 0; i < nEmit; i++)
    {
        for (int j = 0; j < nState; j++) 
        {
            mtObs[i*nState + j] = rand();
        }
    }

    // normalize the emission matrix
    for (int j = 0; j < nState; j++) 
    {
        float sum = 0.0;
        for (int i = 0; i < nEmit; i++) sum += mtObs[i*nState + j];
        for (int i = 0; i < nEmit; i++) mtObs[i*nState + j] /= sum;
    }

    return 1;
}
