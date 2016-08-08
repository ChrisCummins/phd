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

// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

// Row summation filter kernel with rescaling, using Image (texture)
// USETEXTURE switch passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
#ifdef USETEXTURE
    // Row summation filter kernel with rescaling, using Image (texture)
    __kernel void BoxRowsTex( __read_only image2d_t SourceRgbaTex, __global unsigned int* uiDest, sampler_t RowSampler, 
                              unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
    {
        // Row to process (note:  1 dimensional workgroup and ND range used for row kernel)
	    size_t globalPosY = get_global_id(0);
        size_t szBaseOffset = mul24(globalPosY, uiWidth);

        // Process the row as long as Y pos isn'f4Sum off the image
        if (globalPosY < uiHeight) 
        {
            // 4 fp32 accumulators
            float4 f4Sum = (float4)0.0f;

            // Do the left boundary
            for(int x = -iRadius; x <= iRadius; x++)     // (note:  clamping provided by Image (texture))
            {
                int2 pos = {x , globalPosY};
                f4Sum += convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  
            }
            uiDest[szBaseOffset] = rgbaFloat4ToUint(f4Sum, fScale);

            // Do the rest of the image
            int2 pos = {0, globalPosY};
            for(unsigned int x = 1; x < uiWidth; x++)           //  (note:  clamping provided by Image (texture)) 
            {
                // Accumulate the next rgba sub-pixel vals
                pos.x = x + iRadius;
                f4Sum += convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  

                // Remove the trailing rgba sub-pixel vals
                pos.x = x - iRadius - 1;
                f4Sum -= convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  

                // Write out to GMEM
                uiDest[szBaseOffset + x] = rgbaFloat4ToUint(f4Sum, fScale);
            }
        }
    }
#endif

// Row summation filter kernel with rescaling, using LMEM
// USELMEM switch passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
#ifdef USELMEM

    // Row summation filter kernel with rescaling, using LMEM
    __kernel void BoxRowsLmem( __global const uchar4* uc4Source, __global unsigned int* uiDest,
                               __local uchar4* uc4LocalData,
                               unsigned int uiWidth, unsigned int uiHeight, int iRadius, int iRadiusAligned, 
                               float fScale, unsigned int uiNumOutputPix)
    {
        // Compute x and y pixel coordinates from group ID and local ID indexes
        int globalPosX = ((int)get_group_id(0) * uiNumOutputPix) + (int)get_local_id(0) - iRadiusAligned;
        int globalPosY = (int)get_group_id(1);
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        // Read global data into LMEM
        if (globalPosX >= 0 && globalPosX < uiWidth)
        {
            uc4LocalData[get_local_id(0)] = uc4Source[iGlobalOffset];
        }
        else 
        {
            uc4LocalData[get_local_id(0)].xyzw = (uchar4)0; 
        }

        // Synchronize the read into LMEM
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute (if pixel plus apron is within bounds)
        if((globalPosX >= 0) && (globalPosX < uiWidth) && (get_local_id(0) >= iRadiusAligned) && (get_local_id(0) < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            // Init summation registers to zero
            float4 f4Sum = (float4)0.0f;

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            int iOffsetX = (int)get_local_id(0) - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(iOffsetX; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w; 
            }

            // Use inline function to scale and convert registers to packed RGBA values in a uchar4, and write back out to GMEM
            uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }
#endif


// Column kernel using coalesced global memory reads
//*****************************************************************
__kernel void BoxColumns(__global unsigned int* uiInputImage, __global unsigned int* uiOutputImage, 
                         unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
{
	size_t globalPosX = get_global_id(0);
    uiInputImage = &uiInputImage[globalPosX];
    uiOutputImage = &uiOutputImage[globalPosX];

    // do left edge
    float4 f4Sum;
    f4Sum = rgbaUintToFloat4(uiInputImage[0]) * (float4)(iRadius);
    for (int y = 0; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
    }
    uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
    for(int y = 1; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[0]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
    
    // main loop
    for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }

    // do right edge
    for (int y = uiHeight - iRadius; y < uiHeight; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
}
