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

// Find the max in local memory array. 
// On return, the max value is in maxValue[0], the index of max element is in maxInd[0]
//*****************************************************************************
void maxOneBlock(__local float maxValue[],
                 __local int   maxInd[])
{
    uint localId   = get_local_id(0);
    uint localSize = get_local_size(0);
    int idx;
    float m1, m2, m3;

    for (uint s = localSize/2; s > 32; s >>= 1)
    {
        if (localId < s) 
        {
            m1 = maxValue[localId];
            m2 = maxValue[localId+s];
            m3 = (m1 >= m2) ? m1 : m2;
            idx = (m1 >= m2) ? localId : localId + s;
            maxValue[localId] = m3;
            maxInd[localId] = maxInd[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // unroll the final warp to reduce loop and sync overheads
    if (localId < 32)
    {
        m1 = maxValue[localId];
        m2 = maxValue[localId+32];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 32;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        
        m1 = maxValue[localId];
        m2 = maxValue[localId+16];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 16;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];
        
        m1 = maxValue[localId];
        m2 = maxValue[localId+8];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 8;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];
 
        m1 = maxValue[localId];
        m2 = maxValue[localId+4];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 4;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId+2];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 2;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];
        
        m1 = maxValue[localId];
        m2 = maxValue[localId+1];
        m3 = (m1 > m2) ? m1 : m2;
        idx = (m1 > m2) ? localId : localId + 1;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];
    }
}


__kernel void ViterbiOneStep(__global float *maxProbNew,
                             __global int   *path, 
                             __global float *maxProbOld,
                             __global float *mtState,
                             __global float *mtEmit,
                             __local  float maxValue[],
                             __local  int   maxInd[],
                             int nState,
                             int obs,
                             int iObs)
{
    uint groupId   = get_group_id(0) + get_group_id(1)*get_num_groups(0);
    uint localId   = get_local_id(0);
    uint localSize = get_local_size(0);

    uint iState = groupId;

    // loop through all previous states, calculating the max
    float mValue = -1.0f;
    int mInd = -1;
    float value;
    for (int i = localId; i < nState; i += localSize)
    {
		value = maxProbOld[i] + mtState[iState*nState + i];
		if (value > mValue)
		{
			mValue = value;
			mInd = i;
		}
    }
    maxValue[localId] = mValue;
    maxInd[localId] = mInd;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    maxOneBlock(maxValue, maxInd);
    
    // copy results from local to global memory
    if (localId == 0) 
    {
        maxProbNew[iState] = maxValue[0] + mtEmit[obs*nState + iState];
        path[(iObs-1)*nState + iState] = maxInd[0];
    }
}


__kernel void ViterbiPath(__global float *vProb,
                          __global int   *vPath,
                          __global float *maxProbNew,
                          __global int   *path,
                          int nState,
                          int nObs)
{
    // find the final most probable state
    if (get_global_id(0) == 0) 
    {
        float maxProb = 0.0;
        int maxState = -1;
        for (int i = 0; i < nState; i++) 
        {
            if (maxProbNew[i] > maxProb) 
            {
                maxProb = maxProbNew[i];
                maxState = i;
            }
        }
        *vProb = maxProb;

        // backtrace to find the Viterbi path
        vPath[nObs-1] = maxState;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        for (int t = nObs-2; t >= 0; t--) 
        {
            vPath[t] = path[t*nState + vPath[t+1]];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}