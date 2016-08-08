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
 
 #ifndef PARTICLESYSTEM_COMMON_H
#define PARTICLESYSTEM_COMMON_H

#include <GL/glew.h>

#include <oclUtils.h>
#include "vector_types.h"

////////////////////////////////////////////////////////////////////////////////
// CPU/GPU common types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef cl_mem memHandle_t;

//Simulation parameters
typedef struct{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
extern "C" void startupOpenCL(int argc, const char **argv);
extern "C" void shutdownOpenCL(void);

extern "C" void allocateArray(memHandle_t *memObj, size_t size);
extern "C" void freeArray(memHandle_t memObj);

extern "C" void copyArrayFromDevice(void *hostPtr, const memHandle_t memObj, unsigned int vbo, size_t size);
extern "C" void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size);

extern "C" void registerGLBufferObject(unsigned int vbo);
extern "C" void unregisterGLBufferObject(unsigned int vbo);

extern "C" memHandle_t mapGLBufferObject(uint vbo);
extern "C" void unmapGLBufferObject(uint vbo);

#endif
