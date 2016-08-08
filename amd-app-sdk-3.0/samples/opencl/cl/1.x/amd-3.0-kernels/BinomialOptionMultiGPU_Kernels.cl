/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each group invocation of this kernel, calculates the option value for a 
 * given stoke price, option strike price, time to expiration date, risk 
 * free interest and volatility factor.
 *
 * Multiple groups calculate the same with different input values.
 * Number of work-items in each group is same as number of leaf nodes
 * So, the maximum number of steps is limited by loca memory size available
 *
 * Each work-item calculate the leaf-node value and update local memory. 
 * These leaf nodes are further reduced to the root of the tree (time step 0). 
 */

#define RISKFREE 0.02f
#define VOLATILITY 0.30f

__kernel
void 
binomial_options(
    int numSteps,
    const __global float4* randArray,
    __global float4* output,
    __local float4* callA,
    __local float4* callB)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);

    float4 inRand = randArray[bid];

    float4 s = (1.0f - inRand) * 5.0f + inRand * 30.f;
    float4 x = (1.0f - inRand) * 1.0f + inRand * 100.f;
    float4 optionYears = (1.0f - inRand) * 0.25f + inRand * 10.f; 
    float4 dt = optionYears * (1.0f / (float)numSteps);
    float4 vsdt = VOLATILITY * sqrt(dt);
    float4 rdt = RISKFREE * dt;
    float4 r = exp(rdt);
    float4 rInv = 1.0f / r;
    float4 u = exp(vsdt);
    float4 d = 1.0f / u;
    float4 pu = (r - d)/(u - d);
    float4 pd = 1.0f - pu;
    float4 puByr = pu * rInv;
    float4 pdByr = pd * rInv;

    float4 profit = s * exp(vsdt * (2.0f * tid - (float)numSteps)) - x;
    callA[tid].x = profit.x > 0 ? profit.x : 0.0f;
    callA[tid].y = profit.y > 0 ? profit.y : 0.0f;
    callA[tid].z = profit.z > 0 ? profit.z: 0.0f;
    callA[tid].w = profit.w > 0 ? profit.w: 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int j = numSteps; j > 0; j -= 2)
    {
        if(tid < j)
        {
            callB[tid] = puByr * callA[tid] + pdByr * callA[tid + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(tid < j - 1)
        {
            callA[tid] = puByr * callB[tid] + pdByr * callB[tid + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = callA[0];
}