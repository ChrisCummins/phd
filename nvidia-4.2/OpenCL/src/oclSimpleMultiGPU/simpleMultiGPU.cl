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

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA SDK sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__kernel void reduce(__global float *d_Result, __global float *d_Input, int N){
    const int tid = get_global_id(0);
    const int threadN = get_global_size(0);

    float sum = 0;

    for(int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}
