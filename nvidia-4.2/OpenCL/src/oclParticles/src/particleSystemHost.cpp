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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vector_math.h"
#include "particleSystem_common.h"
#include "particleSystem_engine.h"

static simParams_t h_params;

extern "C" void setParametersHost(simParams_t *host_params){
    memcpy(&h_params, host_params, sizeof(simParams_t));
}

static void collideDimensionHost(float& r, float& v){
    if(r <= -1.0f + h_params.particleRadius){
        r = -1.0f + h_params.particleRadius;
        v *= h_params.boundaryDamping;
    }

    if(r >= 1.0f - h_params.particleRadius){
        r = 1.0f - h_params.particleRadius;
        v *= h_params.boundaryDamping;
    }
}

extern "C" void integrateSystemHost(
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    float deltaTime,
    uint numParticles
){
    //In/out
    float4 *h_Pos = (float4 *)d_Pos;
    float4 *h_Vel = (float4 *)d_Vel;

    for(unsigned int index = 0; index < numParticles; index++){
        float4& posData = h_Pos[index];
        float4& velData = h_Vel[index];
        float3      pos = make_float3(posData.x, posData.y, posData.z);
        float3      vel = make_float3(velData.x, velData.y, velData.z);

        vel += h_params.gravity * deltaTime;
        vel *= h_params.globalDamping;
        pos += vel * deltaTime;

        //Box collide
        collideDimensionHost(pos.x, vel.x);
        collideDimensionHost(pos.y, vel.y);
        collideDimensionHost(pos.z, vel.z);

        //Store new position and velocity
        posData = make_float4(pos, posData.w);
        velData = make_float4(vel, velData.w);
    }
}

static int3 getGridPosHost(float3 p){
    int3 gridPos;
    gridPos.x = (int)floor((p.x - h_params.worldOrigin.x) / h_params.cellSize.x);
    gridPos.y = (int)floor((p.y - h_params.worldOrigin.y) / h_params.cellSize.y);
    gridPos.z = (int)floor((p.z - h_params.worldOrigin.z) / h_params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
static uint getGridHashHost(int3 gridPos){
    //Wrap addressing, assume power-of-two grid dimensions
    gridPos.x = gridPos.x & (h_params.gridSize.x-1);
    gridPos.y = gridPos.y & (h_params.gridSize.y-1);
    gridPos.z = gridPos.z & (h_params.gridSize.z-1);
    return (gridPos.z * h_params.gridSize.y + gridPos.y) * h_params.gridSize.x + gridPos.x;
}

extern "C" void calcHashHost(
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    int numParticles
){
    //Out
    uint  *h_Hash = (uint *)d_Hash;
    uint *h_Index = (uint *)d_Index;
    //In
    const float4 *h_Pos = (float4 *)d_Pos;

    for(int index = 0; index < numParticles; index++){
        const float4& p = h_Pos[index];

        //Get address in grid
        int3 gridPos = getGridPosHost(make_float3(p.x, p.y, p.z));
        uint hash = getGridHashHost(gridPos);

        //Store grid hash and particle index
        h_Hash[index] = hash;
        h_Index[index] = index;
    }
}

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
){
    //Out
    uint      *h_CellStart = (uint   *)d_CellStart;
    uint        *h_CellEnd = (uint   *)d_CellEnd;
    float4 *h_ReorderedPos = (float4 *)d_ReorderedPos;
    float4 *h_ReorderedVel = (float4 *)d_ReorderedVel;

    //In
    const uint  *h_Hash = (uint *)d_Hash;
    const uint *h_Index = (uint *)d_Index;
    const float4 *h_Pos = (float4 *)d_Pos;
    const float4 *h_Vel = (float4 *)d_Vel;

    //Clear storage
    memset(h_CellStart, 0xffffffff, numCells * sizeof(uint));

    for(unsigned int index = 0; index < numParticles; index++){
        uint  currentHash = h_Hash[index];

        //Border case
        if(index == 0)
            h_CellStart[currentHash] = 0;

        //Main case
        else{
            uint previousHash = h_Hash[index - 1];
            if(currentHash != previousHash)
                h_CellEnd[previousHash]  = h_CellStart[currentHash] = index;
        };

        //Another border case
        if(index == numParticles - 1)
            h_CellEnd[currentHash] = numParticles;

        //Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = h_Index[index];
        h_ReorderedPos[index] = h_Pos[sortedIndex];
        h_ReorderedVel[index] = h_Vel[sortedIndex];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////
static float3 collideSpheresHost(
    float3 posA,
    float3 posB,
    float3 velA,
    float3 velB,
    float radiusA,
    float radiusB,
    float spring,
    float damping,
    float shear,
    float attraction
){
    //Calculate relative position
    float3     relPos = posB - posA;
    float        dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0);
    if(dist < collideDist){
        //Normalized direction vector
        float3 norm = relPos / dist;
        //Relative velocity
        float3 relVel = velB - velA;
        //Relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        //Spring force (potential)
        force = -spring * (collideDist - dist) * norm;

        //Damping force (friction)
        force += damping * relVel;

        //Tangential shear force (friction)
        force += shear * tanVel;

        //Attraction force (potential)
        force += attraction * relPos;
    }

    return force;
}


extern "C" void collideHost(
    memHandle_t d_Vel,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Index,
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    uint   numParticles,
    uint   numCells
){
    //Out
    float4                *h_Vel = (float4 *)d_Vel;

    //In
    const float4 *h_ReorderedPos = (float4 *)d_ReorderedPos;
    const float4 *h_ReorderedVel = (float4 *)d_ReorderedVel;
    const uint          *h_Index = (uint   *)d_Index;
    const uint      *h_CellStart = (uint   *)d_CellStart;
    const uint        *h_CellEnd = (uint   *)d_CellEnd;

    for(unsigned index = 0; index < numParticles; index++){
        //Read particle data from reordered arrays
        float3 pos = make_float3(h_ReorderedPos[index]);
        float3 vel = make_float3(h_ReorderedVel[index]);

        //Get particle cell id
        int3 gridPos = getGridPosHost(pos);

        //Examine neighbouring cells
        float3 force = make_float3(0);
        for(int z = -1; z <= 1; z++)
            for(int y =-1; y <= 1; y++)
                for(int x = -1; x <= 1; x++){
                    int3 neighbourPos = gridPos + make_int3(x, y, z);
                    uint     gridHash = getGridHashHost(neighbourPos);

                    //Get start of bucket for this cell
                    uint startIndex = h_CellStart[gridHash];

                    //Skip empty cells
                    if(startIndex == 0xFFFFFFFFU)
                        continue;

                    uint endIndex = h_CellEnd[gridHash];

                    //Iterate over particles in this cell
                    for(uint j = startIndex; j < endIndex; j++){
                        //Avoid colliding a particle with itself
                        if(j == index)
                            continue;

                        float3 pos2 = make_float3(h_ReorderedPos[j]);
                        float3 vel2 = make_float3(h_ReorderedVel[j]);

                        //Collide two spheres
                        force += collideSpheresHost(
                            pos, pos2,
                            vel, vel2,
                            h_params.particleRadius, h_params.particleRadius,
                            h_params.spring, h_params.damping, h_params.shear, h_params.attraction
                        );
                    }
                }

        //Collide with cursor sphere
        force += collideSpheresHost(
            pos, h_params.colliderPos,
            vel, make_float3(0),
            h_params.particleRadius, h_params.colliderRadius,
            h_params.spring, h_params.damping, h_params.shear, 0
        );

        //Write new velocity back to original unsorted location
        uint originalIndex = h_Index[index];
        h_Vel[originalIndex] = make_float4(vel + force, 0);
    }
}



