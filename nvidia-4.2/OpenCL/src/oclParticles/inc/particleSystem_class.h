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

#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#define DEBUG_GRID 0
#define DO_TIMING 0

#include "vector_types.h"
#include "particleSystem_common.h"
#include "particleSystem_engine.h"

// Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint3 gridSize, float fParticleRadius, float fColliderRadius, shrBOOL bQATest);
    ~ParticleSystem();

    enum ParticleConfig
    {
        CONFIG_RANDOM,
        CONFIG_GRID,
        _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
    };

    void update(float deltaTime);
    void reset(ParticleConfig config);

    float* getArray(ParticleArray array);
    void   setArray(ParticleArray array, const float* data, int start, int count);

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer() const { return m_colorVBO; }

    void dumpGrid();
    void dumpParticles(uint start, uint count);

    void setIterations(int i) { m_solverIterations = i; }
    void setDamping(float x) { m_params.globalDamping = x; }
    void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }
    void setCollideSpring(float x) { m_params.spring = x; }
    void setCollideDamping(float x) { m_params.damping = x; }
    void setCollideShear(float x) { m_params.shear = x; }
    void setCollideAttraction(float x) { m_params.attraction = x; }
    void setColliderPos(float3 x) { m_params.colliderPos = x; }

    float getParticleRadius() { return m_params.particleRadius; }
    float3 getColliderPos() { return m_params.colliderPos; }
    float getColliderRadius() { return m_params.colliderRadius; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }

    void addSphere(int index, float *pos, float *vel, int r, float spacing);

protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized;
    uint m_numParticles;
    shrBOOL m_bQATest;

    // CPU data
    float *m_hPos;
    float *m_hVel;
    float *m_hReorderedPos;
    float *m_hReorderedVel;
    uint *m_hCellStart;
    uint *m_hCellEnd;
    uint *m_hHash;
    uint *m_hIndex;

    // GPU data
    memHandle_t          m_dPos;
    memHandle_t          m_dVel;
    memHandle_t m_dReorderedPos;
    memHandle_t m_dReorderedVel;
    memHandle_t         m_dHash;
    memHandle_t        m_dIndex;
    memHandle_t    m_dCellStart;
    memHandle_t      m_dCellEnd;

    uint m_gridSortBits;
    uint       m_posVbo;
    uint     m_colorVBO;

    // params
    simParams_t m_params;
    uint3 m_gridSize;
    uint m_numGridCells;
    uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
