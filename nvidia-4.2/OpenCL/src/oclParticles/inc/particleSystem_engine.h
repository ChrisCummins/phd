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
 
 #ifndef PARTICLESYSTEM_ENGINE_H
#define PARTICLESYSTEM_ENGINE_H

#include "particleSystem_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL simulation engine
////////////////////////////////////////////////////////////////////////////////
extern "C" void initParticles(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeParticles(void);

extern "C" void setParameters(simParams_t *hostParams);

extern "C" void integrateSystem(
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    float deltaTime,
    uint numParticles
);

extern "C" void calcHash(
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    int numParticles
);

extern "C" void findCellBoundsAndReorder(
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    uint numParticles,
    uint numCells
);

extern "C" void collide(
    memHandle_t d_Vel,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Index,
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    uint   numParticles,
    uint   numCells
);



////////////////////////////////////////////////////////////////////////////////
// CPU simulation engine
////////////////////////////////////////////////////////////////////////////////
extern "C" void setParametersHost(simParams_t *hostParams);

extern "C" void integrateSystemHost(
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    float deltaTime,
    uint numParticles
);

extern "C" void calcHashHost(
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    int numParticles
);

extern "C" void findCellBoundsAndReorderHost(
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    uint numParticles,
    uint numCells
);

extern "C" void collideHost(
    memHandle_t d_Vel,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Index,
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    uint numParticles,
    uint numCells
);



#endif
