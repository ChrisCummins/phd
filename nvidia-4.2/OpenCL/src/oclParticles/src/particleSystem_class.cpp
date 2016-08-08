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
#include "particleSystem_class.h"

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float fParticleRadius, float fColliderRadius, shrBOOL bQATest):
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
    m_gridSize(gridSize),
    m_solverIterations(1),
    m_bQATest(bQATest)
{
    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;
    m_params.particleRadius = fParticleRadius; 
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = fColliderRadius;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem(){
    _finalize();
    m_numParticles = 0;
}

uint ParticleSystem::createVBO(uint size){
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerGLBufferObject(vbo);
    return vbo;
}

inline float lerp(float a, float b, float t){
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r){
    const int ncolors = 7;
    float c[ncolors][3] = {
        { 1.0f, 0.0f, 0.0f, },
        { 1.0f, 0.5f, 0.0f, },
        { 1.0f, 1.0f, 0.0f, },
        { 0.0f, 1.0f, 0.0f, },
        { 0.0f, 1.0f, 1.0f, },
        { 0.0f, 0.0f, 1.0f, },
        { 1.0f, 0.0f, 1.0f, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void ParticleSystem::_initialize(int numParticles){
    assert(!m_bInitialized);
    m_numParticles = numParticles;

    //Allocate host storage
    m_hPos          = (float *)malloc(m_numParticles * 4 * sizeof(float));
    m_hVel          = (float *)malloc(m_numParticles * 4 * sizeof(float));
    m_hReorderedPos = (float *)malloc(m_numParticles * 4 * sizeof(float));
    m_hReorderedVel = (float *)malloc(m_numParticles * 4 * sizeof(float));
    m_hHash         = (uint  *)malloc(m_numParticles * sizeof(uint));
    m_hIndex        = (uint  *)malloc(m_numParticles * sizeof(uint));
    m_hCellStart    = (uint  *)malloc(m_numGridCells * sizeof(uint));
    m_hCellEnd      = (uint  *)malloc(m_numGridCells * sizeof(uint));

    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));
    memset(m_hCellEnd,   0, m_numGridCells * sizeof(uint));

    //Allocate GPU data
    shrLog("Allocating GPU Data buffers...\n\n");
    allocateArray(&m_dPos,          m_numParticles * 4 * sizeof(float));
    allocateArray(&m_dVel,          m_numParticles * 4 * sizeof(float));
    allocateArray(&m_dReorderedPos, m_numParticles * 4 * sizeof(float));
    allocateArray(&m_dReorderedVel, m_numParticles * 4 * sizeof(float));
    allocateArray(&m_dHash,         m_numParticles * sizeof(uint));
    allocateArray(&m_dIndex,        m_numParticles * sizeof(uint));
    allocateArray(&m_dCellStart,    m_numGridCells * sizeof(uint));
    allocateArray(&m_dCellEnd,      m_numGridCells * sizeof(uint));

    if (!m_bQATest)
    {
        //Allocate VBO storage
        m_posVbo   = createVBO(m_numParticles * 4 * sizeof(float));
        m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));

        //Fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            float *ptr = data;
            for(uint i = 0; i < m_numParticles; i++){
                float t = (float)i / (float) m_numParticles;
                #if 0
                    *ptr++ = rand() / (float) RAND_MAX;
                    *ptr++ = rand() / (float) RAND_MAX;
                    *ptr++ = rand() / (float) RAND_MAX;
                #else
                    colorRamp(t, ptr);
                    ptr += 3;
                #endif
                *ptr++ = 1.0f;
            }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }

    setParameters(&m_params);
    setParametersHost(&m_params);
    m_bInitialized = true;
}

void ParticleSystem::_finalize(){
    assert(m_bInitialized);

    free(m_hPos);
    free(m_hVel);
    free(m_hCellStart);
    free(m_hCellEnd);
    free(m_hReorderedVel);
    free(m_hReorderedPos);
    free(m_hIndex);
    free(m_hHash);

    freeArray(m_dPos);
    freeArray(m_dVel);
    freeArray(m_dReorderedPos);
    freeArray(m_dReorderedVel);
    freeArray(m_dHash);
    freeArray(m_dIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (!m_bQATest)
    {
        unregisterGLBufferObject(m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    }
}

extern"C" void bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir
);

static int isNan(float f){
    unsigned int u = *(unsigned int*)&f;
    return ( (u & 0x7F800000U) == 0x7F800000U ) && ( (u & 0x007FFFFFU) != 0 );
}

//Step the simulation
void ParticleSystem::update(float deltaTime){
    assert(m_bInitialized);

    setParameters(&m_params);
    setParametersHost(&m_params);

    //Download positions from VBO
    memHandle_t pos; 
    if (!m_bQATest)
    {
        glBindBufferARB(GL_ARRAY_BUFFER, m_posVbo);
        pos = (memHandle_t)glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
        copyArrayToDevice(m_dPos, pos, 0, m_numParticles * 4 * sizeof(float));
    }

    integrateSystem(
        m_dPos,
        m_dVel,
        deltaTime,
        m_numParticles
    );

    calcHash(
        m_dHash,
        m_dIndex,
        m_dPos,
        m_numParticles
    );

    bitonicSort(NULL, m_dHash, m_dIndex, m_dHash, m_dIndex, 1, m_numParticles, 0);

    //Find start and end of each cell and
    //Reorder particle data for better cache coherency
    findCellBoundsAndReorder(
        m_dCellStart,
        m_dCellEnd,
        m_dReorderedPos,
        m_dReorderedVel,
        m_dHash,
        m_dIndex,
        m_dPos,
        m_dVel,
        m_numParticles,
        m_numGridCells
    );

    collide(
        m_dVel,
        m_dReorderedPos,
        m_dReorderedVel,
        m_dIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells
    );

    //Update buffers
    if (!m_bQATest)
    {
        copyArrayFromDevice(pos,m_dPos, 0, m_numParticles * 4 * sizeof(float));
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
}

void ParticleSystem::dumpGrid(){
    //Sump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint) * m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);
}

void ParticleSystem::dumpParticles(uint start, uint count)
{
    //Debug
    copyArrayFromDevice(m_hPos, 0, m_posVbo, sizeof(float) * 4 * count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);
}

float *ParticleSystem::getArray(ParticleArray array){
    assert(m_bInitialized);

    float *hdata = 0;
    memHandle_t ddata = 0;
    unsigned int vbo = 0;

    switch (array){
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            vbo = m_posVbo;
        break;
        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
        break;
    }

    copyArrayFromDevice(hdata, ddata, vbo, m_numParticles * 4 * sizeof(float));
    return hdata;
}

void ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count){
    assert(m_bInitialized);

    switch (array){
        default:
        case POSITION:
            unregisterGLBufferObject(m_posVbo);
            glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
            glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            registerGLBufferObject(m_posVbo);
        break;
        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        break;
    }
}

inline float frand(void){
    return (float)rand() / (float)RAND_MAX;
}

void ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles){
    srand(1973);
    for(uint z=0; z<size[2]; z++) 
    {
        for(uint y=0; y<size[1]; y++) 
        {
            for(uint x=0; x<size[0]; x++) 
            {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;
                if (i < numParticles) 
                {
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;
                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}

void ParticleSystem::reset(ParticleConfig config){
    switch(config){
        default:
        case CONFIG_RANDOM:
        {
            int p = 0, v = 0;
            for(uint i=0; i < m_numParticles; i++)
            {
                float point[3];
                point[0] = frand();
                point[1] = frand();
                point[2] = frand();
                m_hPos[p++] = 2.0f * (point[0] - 0.5f);
                m_hPos[p++] = 2.0f * (point[1] - 0.5f);
                m_hPos[p++] = 2.0f * (point[2] - 0.5f);
                m_hPos[p++] = 1.0f; // radius
                m_hVel[v++] = 0.0f;
                m_hVel[v++] = 0.0f;
                m_hVel[v++] = 0.0f;
                m_hVel[v++] = 0.0f;
            }
        }
        break;

        case CONFIG_GRID:
        {
            float jitter = m_params.particleRadius * 0.01f;
            uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
            uint gridSize[3];
            gridSize[0] = gridSize[1] = gridSize[2] = s;
            initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numParticles);
        }
        break;
    }

    if (!m_bQATest)
    {
        setArray(POSITION, m_hPos, 0, m_numParticles);
        setArray(VELOCITY, m_hVel, 0, m_numParticles);
    }
}

void ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing){
    uint index = start;
    for(int z = -r; z <= r; z++)
    {
        for(int y = -r; y <= r; y++)
        {
            for(int x = -r; x <= r; x++)
            {
                float dx = x * spacing;
                float dy = y * spacing;
                float dz = z * spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius * 0.01f;
                if ((l <= m_params.particleRadius * 2.0f * r) && (index < m_numParticles))
                {
                    m_hPos[index * 4]   =   pos[0] + dx + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 1] = pos[1] + dy + (frand() * 2.0f - 1.0f) * jitter; 
                    m_hPos[index * 4 + 2] = pos[2] + dz + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 3] = pos[3];

                    m_hVel[index * 4]     = vel[0];
                    m_hVel[index * 4 + 1] = vel[1];
                    m_hVel[index * 4 + 2] = vel[2];
                    m_hVel[index * 4 + 3] = vel[3];
                    index++;
                }
            }
        }
    }

    if (!m_bQATest)
    {
        setArray(POSITION, m_hPos, start, index);
        setArray(VELOCITY, m_hVel, start, index);
    }
}
