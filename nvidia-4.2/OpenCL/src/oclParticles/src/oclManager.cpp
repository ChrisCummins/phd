#include <cecl.h>
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

#include "particleSystem_common.h"
#include "particleSystem_engine.h"

////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;

extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeBitonicSort(void);

//Context initialization/deinitialization
extern "C" void startupOpenCL(int argc, const char **argv){
    cl_platform_id cpPlatform;
    cl_uint uiNumDevices;
	cl_uint uiTargetDevice = 0;
	cl_device_id* cdDevices;
    cl_int ciErrNum;
    
    // Get the NVIDIA platform
    shrLog("oclGetPlatformID...\n\n"); 
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get the devices
    shrLog("clGetDeviceIDs...\n\n"); 
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );

	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiNumDevices, cdDevices, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Set target device and Query number of compute units on uiTargetDevice
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
	shrLog("Using Device %u, ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);  

	// Create the context
    shrLog("\n\nclCreateContext...\n\n"); 
    cxGPUContext = clCreateContext(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Create a command-queue
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n\n"); 
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

	free( cdDevices );

    initBitonicSort(cxGPUContext, cqCommandQueue, argv);
    initParticles(cxGPUContext, cqCommandQueue, argv);
}

extern "C" void shutdownOpenCL(void){
    cl_int ciErrNum;
    closeParticles();
    closeBitonicSort();
    ciErrNum  = clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//GPU buffer allocation
extern "C" void allocateArray(memHandle_t *memObj, size_t size){
    cl_int ciErrNum;
    shrLog(" CECL_BUFFER (GPU GMEM, %u bytes)...\n\n", size); 
    *memObj = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void freeArray(memHandle_t memObj){
    cl_int ciErrNum;
    ciErrNum = clReleaseMemObject(memObj);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//host<->device memcopies
extern "C" void copyArrayFromDevice(void *hostPtr, memHandle_t memObj, unsigned int vbo, size_t size){
    cl_int ciErrNum;
    assert( vbo == 0 );
    ciErrNum = CECL_READ_BUFFER(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size){
    cl_int ciErrNum;
    ciErrNum = CECL_WRITE_BUFFER(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//Register/unregister OpenGL buffer object to/from Compute context
extern "C" void registerGLBufferObject(uint vbo){
}

extern "C" void unregisterGLBufferObject(uint vbo){
}

//Map/unmap OpenGL buffer object to/from Compute buffer
extern "C" memHandle_t mapGLBufferObject(uint vbo){
    return NULL;
}

extern "C" void unmapGLBufferObject(uint vbo){
}
