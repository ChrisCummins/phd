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

 /* This example demonstrates how to use the OpenCL/OpenGL bindings  */

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__kernel void sine_wave(__global float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = (float4)(u, w, v, 1.0f);
}

