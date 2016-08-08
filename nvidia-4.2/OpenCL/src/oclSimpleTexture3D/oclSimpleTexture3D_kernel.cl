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

__kernel void render(__read_only image3d_t volume, sampler_t volumeSampler,  __global uint *d_output, uint imageW, uint imageH, float w)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

    // compute normalized texture coordinates
    float u = x / (float) imageW;
    float v = y / (float) imageH;

    // read from 3D texture
    float4 voxel = read_imagef(volume, volumeSampler, (float4)(u,v,w,1.0f));

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = (y * imageW) + x;
        d_output[i] = voxel.x*255;
    }
}
