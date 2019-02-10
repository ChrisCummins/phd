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
#include <math.h>
#include "HMM.h"

double executionTime(cl_event &event);

// constructer
//*****************************************************************************
HMM::HMM(cl_context GPUContext, 
         cl_command_queue CommandQue, 
         float *initProb, 
         float *mtState, 
         float *mtEmit, 
         int numState, 
         int numEmit,
		 int numObs,
         const char *path,
		 int workgroupSize) :
         cxGPUContext(GPUContext),
		 cqCommandQue(CommandQue),
         nState(numState),
         nEmit(numEmit),
		 nObs(numObs),
         tKernel(0.0),
		 wgSize(workgroupSize)
{
    cl_int err;
    d_mtState  = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(float)*nState*nState, NULL, &err);
    err |= CECL_WRITE_BUFFER(cqCommandQue, d_mtState, CL_TRUE, 0, sizeof(float)*nState*nState, mtState, 0, NULL, NULL);
    d_mtEmit   = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, sizeof(float)*nEmit*nState, NULL, &err);
    err |= CECL_WRITE_BUFFER(cqCommandQue, d_mtEmit, CL_TRUE, 0, sizeof(float)*nEmit*nState, mtEmit, 0, NULL, NULL);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);

    size_t szKernelLength; // Byte size of kernel code
    char *cViterbi = oclLoadProgSource(shrFindFilePath("Viterbi.cl", path), "// My comment\n", &szKernelLength);
    oclCheckErrorEX(cViterbi == NULL, false, NULL);
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cViterbi, &szKernelLength, &err);
    err = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, (double)err, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "Viterbi.ptx");
		shrEXIT(0, NULL);
    }

    d_maxProbNew = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*nState, NULL, &err);
    d_maxProbOld = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*nState, NULL, &err);
    err |= CECL_WRITE_BUFFER(cqCommandQue, d_maxProbOld, CL_TRUE, 0, sizeof(float)*nState, initProb, 0, NULL, NULL);
    
    ckViterbiOneStep = CECL_KERNEL(cpProgram, "ViterbiOneStep", &err);
    ckViterbiPath    = CECL_KERNEL(cpProgram, "ViterbiPath", &err);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);
    
    cl_device_id device;
    err = clGetCommandQueueInfo(cqCommandQue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);
    size_t maxWgSize;
    err = CECL_GET_KERNEL_WORK_GROUP_INFO(ckViterbiOneStep, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWgSize, NULL);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);
    if (maxWgSize == 64) smallBlock = true;
    else smallBlock = false;
       
	d_path = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int)*(nObs-1)*nState, NULL, &err);

    free(cViterbi);
    
}

// destructor
//*****************************************************************************
HMM::~HMM()
{
    cl_int err;
    err  = clReleaseMemObject(d_mtState);
	err |= clReleaseMemObject(d_path);
    err |= clReleaseMemObject(d_mtEmit);
    err |= clReleaseMemObject(d_maxProbNew);
    err |= clReleaseMemObject(d_maxProbOld);
    err |= clReleaseKernel(ckViterbiOneStep);
    err |= clReleaseKernel(ckViterbiPath);
    err |= clReleaseProgram(cpProgram);
	oclCheckErrorEX(err, CL_SUCCESS, NULL);
}


// wrapper for the Viterbi kernel
//*****************************************************************************
size_t HMM::ViterbiOneStep(const int &obs, const int &iObs)
{
    cl_int err;
	size_t globalWorkSize[2] = {1,1}, localWorkSize[2] = {1,1};
    if (nState >= 256)
    {
		if (smallBlock) 
			localWorkSize[0] = 64;
		else 
			localWorkSize[0] = wgSize;
		globalWorkSize[0] = localWorkSize[0]*256;
		globalWorkSize[1] = nState/256;
    } else {
        localWorkSize[0] = (int)pow(2.0f, (int)(log((float)nState)/log(2.0f))); // the largest 2^n number smaller than nState
        globalWorkSize[0] = localWorkSize[0]*nState;
    }

    err  = CECL_SET_KERNEL_ARG(ckViterbiOneStep, 0, sizeof(cl_mem), (void*)&d_maxProbNew);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 1, sizeof(cl_mem), (void*)&d_path);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 2, sizeof(cl_mem), (void*)&d_maxProbOld);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 3, sizeof(cl_mem), (void*)&d_mtState);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 4, sizeof(cl_mem), (void*)&d_mtEmit);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 5, sizeof(float)*localWorkSize[0], NULL);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 6, sizeof(int)*localWorkSize[0], NULL);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 7, sizeof(int), (void*)&nState);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 8, sizeof(int), (void*)&obs);
    err |= CECL_SET_KERNEL_ARG(ckViterbiOneStep, 9, sizeof(int), (void*)&iObs);

	err |= CECL_ND_RANGE_KERNEL(cqCommandQue, ckViterbiOneStep, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	err = clEnqueueCopyBuffer(cqCommandQue, d_maxProbNew, d_maxProbOld, 0, 0, sizeof(float)*nState, 0, NULL, NULL);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);

    return localWorkSize[0];
}

// Main routine of the Viterbi algorithm.
// Calls the Viterbi search kernel for every given observations in the input sequence.
//*****************************************************************************
size_t HMM::ViterbiSearch(cl_mem vProb, cl_mem vPath, int *obs)
{
    size_t szWorkgroup;

    for (int iObs = 1; iObs < nObs; iObs++)
    {
        szWorkgroup = ViterbiOneStep(obs[iObs], iObs);
    }
   

    ViterbiPath(vProb, vPath);

    return szWorkgroup;
}

void HMM::ViterbiPath(cl_mem vProb, cl_mem vPath)
{
    cl_int err;
    size_t globalWorkSize[1] = {1}, localWorkSize[1] = {1};

    err  = CECL_SET_KERNEL_ARG(ckViterbiPath, 0, sizeof(cl_mem), (void*)&vProb);
    err |= CECL_SET_KERNEL_ARG(ckViterbiPath, 1, sizeof(cl_mem), (void*)&vPath);
    err |= CECL_SET_KERNEL_ARG(ckViterbiPath, 2, sizeof(cl_mem), (void*)&d_maxProbNew);
    err |= CECL_SET_KERNEL_ARG(ckViterbiPath, 3, sizeof(cl_mem), (void*)&d_path);
    err |= CECL_SET_KERNEL_ARG(ckViterbiPath, 4, sizeof(int), (void*)&nState);
    err |= CECL_SET_KERNEL_ARG(ckViterbiPath, 5, sizeof(int), (void*)&nObs);

    err |= CECL_ND_RANGE_KERNEL(cqCommandQue, ckViterbiPath, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    oclCheckErrorEX(err, CL_SUCCESS, NULL);
  
}
