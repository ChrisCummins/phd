/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

.   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
.   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

void atomicAddGlobal(volatile __global float *ptr, float value)
{
    unsigned int oldIntVal, newIntVal;
    float newFltVal;

    do
    {
        oldIntVal = *((volatile __global unsigned int *)ptr);
        newFltVal = ((*(float*)(&oldIntVal)) + value);
        newIntVal = *((unsigned int *)(&newFltVal));
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)ptr, oldIntVal, newIntVal) != oldIntVal);
}

void atomicAddLocal(volatile __local float *ptr, float value)
{
    unsigned int oldIntVal, newIntVal;
    float newFltVal;

    do
    {
        oldIntVal = *((volatile __local unsigned int *)ptr);
        newFltVal = ((*(float*)(&oldIntVal)) + value);
        newIntVal = *((unsigned int *)(&newFltVal));
    }
    while (atomic_cmpxchg((volatile __local unsigned int *)ptr, oldIntVal, newIntVal) != oldIntVal);
}

__kernel
void assignCentroid(
    __global float2 *pointPos,
    __global uint *KMeansCluster,
    __global float2 *centroidPos,
    __global float2 *globalClusterBin,          // size k, newCentroidPos
    __global unsigned int *globalClusterCount,
    __local float2 *localClusterBin,            // size k
    __local unsigned int *localClusterCount,
    uint k, uint numPoints)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    __local float* localBinPtr = (__local float*)localClusterBin;
    
    if(lid < k)
    {
        localClusterBin[lid] = (float2)0.0;
        localClusterCount[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Load 1 point
    float2 vPoint = pointPos[gid];
    float leastDist = FLT_MAX;
    uint closestCentroid = 0;
    
    for(int i=0; i<k; i++)
    {
        // Even component in float2 implies x co-ordinate, odd implies y co-ordinate
        #ifdef USE_POWN
        float dist = pown((vPoint.even - centroidPos[i].even), 2) + 
                          pown((vPoint.odd - centroidPos[i].odd), 2);
        #else
        float xDist = (vPoint.x - centroidPos[i].x);
        float yDist = (vPoint.y - centroidPos[i].y);
        float dist = (xDist * xDist) + (yDist * yDist);
        #endif
        leastDist = fmin( leastDist, dist );

        closestCentroid = (leastDist == dist) ? i : closestCentroid;
    }
    
    KMeansCluster[gid] = closestCentroid;

    atomicAddLocal( &localBinPtr[2 * closestCentroid], vPoint.x );
    atomicAddLocal( &localBinPtr[2 * closestCentroid + 1], vPoint.y );
    atomic_inc( &localClusterCount[closestCentroid] );
    barrier(CLK_LOCAL_MEM_FENCE);

    // Push back the local bin and count values to global
    if(lid < k)
    {
        atomicAddGlobal( ((__global float*)(globalClusterBin) + (2 * lid)), localClusterBin[lid].x );
        atomicAddGlobal( ((__global float*)(globalClusterBin) + (2 * lid) + 1), localClusterBin[lid].y );
        atomic_add( &globalClusterCount[lid], localClusterCount[lid] );
    }
}


__kernel void computeSilhouettes(__global float2* pointPos,
                                __global float2* centroidPos, 
                                __global unsigned int* KmeansCluster, 
                                __global unsigned int* globalClusterCount, 
                                __local int* lClusterCount, //reduce global access
                                int k, 
                                int numPoints, 
                                __local float* lSilhouetteValue, 
                                __global float* gSilhoutteValue)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    if(lid == 0)
    {
        lSilhouetteValue[0] = 0.f;
    }
    if(lid < k)
    {
        lClusterCount[lid] = globalClusterCount[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float silhScore = 0.f;
    float dissimilarities[MAX_CLUSTERS] = {0.0f};
    
    for(int i=0; i<numPoints; i++)
    {
        dissimilarities[KmeansCluster[i]] += (sqrt(pow(pointPos[i].s0 - pointPos[gid].s0, 2.0f)
                                             + pow(pointPos[i].s1 - pointPos[gid].s1, 2.0f)));
    }
    
    float a = dissimilarities[KmeansCluster[gid]] / lClusterCount[KmeansCluster[gid]];
    float b = FLT_MAX;
    for(int i=0; i<k; i++)
    {
        if(i != KmeansCluster[gid])
            b =  min(b, dissimilarities[i] / lClusterCount[i]);
    }
    
    silhScore = ((b - a) / max(a, b));
    
    atomicAddLocal(lSilhouetteValue, silhScore);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(lid == 0)
    {
        atomicAddGlobal(gSilhoutteValue, lSilhouetteValue[0]);
    }
}
