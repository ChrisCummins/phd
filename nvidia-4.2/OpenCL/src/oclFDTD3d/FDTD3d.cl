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

__kernel void FiniteDifferences(__global float * const output,
                                __global const float * const input,
                                __constant float * const coeff,
                                const int dimx,
                                const int dimy,
                                const int dimz,
                                const int padding)
{
    bool valid = true;
    const int gtidx = get_global_id(0);
    const int gtidy = get_global_id(1);
    const int ltidx = get_local_id(0);
    const int ltidy = get_local_id(1);
    const int workx = get_local_size(0);
    const int worky = get_local_size(1);
    __local float tile[MAXWORKY + 2 * RADIUS][MAXWORKX + 2 * RADIUS];
    
    const int stride_y = dimx + 2 * RADIUS;
    const int stride_z = stride_y * (dimy + 2 * RADIUS);

    int inputIndex  = 0;
    int outputIndex = 0;

    // Advance inputIndex to start of inner volume
    inputIndex += RADIUS * stride_y + RADIUS + padding;
    
    // Advance inputIndex to target element
    inputIndex += gtidy * stride_y + gtidx;

    float infront[RADIUS];
    float behind[RADIUS];
    float current;

	const int tx = ltidx + RADIUS;
	const int ty = ltidy + RADIUS;

    if (gtidx >= dimx)
        valid = false;
    if (gtidy >= dimy)
        valid = false;

    // For simplicity we assume that the global size is equal to the actual
    // problem size; since the global size must be a multiple of the local size
    // this means the problem size must be a multiple of the local size (or
    // padded to meet this constraint).
    // Preload the "infront" and "behind" data
    for (int i = RADIUS - 2 ; i >= 0 ; i--)
    {
        behind[i] = input[inputIndex];
        inputIndex += stride_z;
    }

    current = input[inputIndex];
    outputIndex = inputIndex;
    inputIndex += stride_z;

    for (int i = 0 ; i < RADIUS ; i++)
    {
        infront[i] = input[inputIndex];
        inputIndex += stride_z;
    }

    // Step through the xy-planes
    for (int iz = 0 ; iz < dimz ; iz++)
    {
        // Advance the slice (move the thread-front)
        for (int i = RADIUS - 1 ; i > 0 ; i--)
            behind[i] = behind[i - 1];
        behind[0] = current;
        current = infront[0];
        for (int i = 0 ; i < RADIUS - 1 ; i++)
            infront[i] = infront[i + 1];
        infront[RADIUS - 1] = input[inputIndex];

        inputIndex  += stride_z;
        outputIndex += stride_z;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Note that for the work items on the boundary of the problem, the
        // supplied index when reading the halo (below) may wrap to the
        // previous/next row or even the previous/next xy-plane. This is
        // acceptable since a) we disable the output write for these work
        // items and b) there is at least one xy-plane before/after the
        // current plane, so the access will be within bounds.

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < RADIUS)
        {
            tile[ltidy][tx]                  = input[outputIndex - RADIUS * stride_y];
            tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
        }
        // Halo left & right
        if (ltidx < RADIUS)
        {
            tile[ty][ltidx]                  = input[outputIndex - RADIUS];
            tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
        }
        tile[ty][tx] = current;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the output value
        float value = coeff[0] * current;
#pragma unroll RADIUS
        for (int i = 1 ; i <= RADIUS ; i++)
        {
            value += coeff[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (valid)
            output[outputIndex] = value;
    }
}
