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

// RGB Sobel gradient intensity filter kernel 
// Uses 32 bit GMEM reads into a block of LMEM padded for apron of radius = 1 (3x3 neighbor op)
// Gradient intensity is from RSS combination of H and V gradient components
// R, G and B gradient intensities are treated separately then combined with linear weighting
//
// Implementation below is equivalent to linear 2D convolutions for H and V compoonents with:
//	    Convo Coefs for Horizontal component {1,0,-1,   2,0,-2,  1,0,-1}
//	    Convo Coefs for Vertical component   {-1,-2,-1,  0,0,0,  1,2,1};
//*****************************************************************************
__kernel void ckSobel(__global uchar4* uc4Source, __global unsigned int* uiDest,
                      __local uchar4* uc4LocalData, int iLocalPixPitch, 
                      int iImageWidth, int iDevImageHeight, float fThresh)
{
    // Get parent image x and y pixel coordinates from global ID, and compute offset into parent GMEM data
    int iImagePosX = get_global_id(0);
    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX; 

    // Compute initial offset of current pixel within work group LMEM block
    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0) + 1;

    // Main read of GMEM data into LMEM
    if((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (iImagePosX < iImageWidth))
    {
        uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset];
    }
    else 
    {
        uc4LocalData[iLocalPixOffset] = (uchar4)0; 
    }

    // Work items with y ID < 2 read bottom 2 rows of LMEM 
    if (get_local_id(1) < 2)
    {
        // Increase local offset by 1 workgroup LMEM block height
        // to read in top rows from the next block region down
        iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

        // If source offset is within the image boundaries
        if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (iImagePosX < iImageWidth))
        {
            // Read in top rows from the next block region down
            uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset + mul24(get_local_size(1), get_global_size(0))];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }
    }

    // Work items with x ID at right workgroup edge will read Left apron pixel
    if (get_local_id(0) == (get_local_size(0) - 1))
    {
        // set local offset to read data from the next region over
        iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch);

        // If source offset is within the image boundaries and not at the leftmost workgroup
        if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (get_group_id(0) > 0))
        {
            // Read data into the LMEM apron from the GMEM at the left edge of the next block region over
            uc4LocalData[iLocalPixOffset] = uc4Source[mul24(iDevYPrime, (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }

        // If in the bottom 2 rows of workgroup block 
        if (get_local_id(1) < 2)
        {
            // Increase local offset by 1 workgroup LMEM block height
            // to read in top rows from the next block region down
            iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

            // If source offset in the next block down isn't off the image and not at the leftmost workgroup
            if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (get_group_id(0) > 0))
            {
                // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel)
                uc4LocalData[iLocalPixOffset] = uc4Source[mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1];
            }
            else 
            {
                uc4LocalData[iLocalPixOffset] = (uchar4)0; 
            }
        }
    } 
    else if (get_local_id(0) == 0) // Work items with x ID at left workgroup edge will read right apron pixel
    {
        // set local offset 
        iLocalPixOffset = mul24(((int)get_local_id(1) + 1), iLocalPixPitch) - 1;

        if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (mul24(((int)get_group_id(0) + 1), (int)get_local_size(0)) < iImageWidth))
        {
            // read in from GMEM (reaching left 1 pixel) if source offset is within image boundaries
            uc4LocalData[iLocalPixOffset] = uc4Source[mul24(iDevYPrime, (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0))];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }

        // Read bottom 2 rows of workgroup LMEM block
        if (get_local_id(1) < 2)
        {
            // increase local offset by 1 workgroup LMEM block height
            iLocalPixOffset += (mul24((int)get_local_size(1), iLocalPixPitch));

            if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (mul24((get_group_id(0) + 1), get_local_size(0)) < iImageWidth) )
            {
                // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel) if source offset is within image boundaries
                uc4LocalData[iLocalPixOffset] = uc4Source[mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0))];
            }
            else 
            {
                uc4LocalData[iLocalPixOffset] = (uchar4)0; 
            }
        }
    }

    // Synchronize the read into LMEM
    barrier(CLK_LOCAL_MEM_FENCE);

    // Init summation registers to zero
    float fTemp = 0.0f; 
    float fHSum [3] = {0.0f, 0.0f, 0.0f};
    float fVSum [3] = {0.0f, 0.0f, 0.0f};

    // set local offset
    iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

    // NW
	fHSum[0] += (float)uc4LocalData[iLocalPixOffset].x;    // horizontal gradient of Red
	fHSum[1] += (float)uc4LocalData[iLocalPixOffset].y;    // horizontal gradient of Green
	fHSum[2] += (float)uc4LocalData[iLocalPixOffset].z;    // horizontal gradient of Blue
    fVSum[0] -= (float)uc4LocalData[iLocalPixOffset].x;    // vertical gradient of Red
	fVSum[1] -= (float)uc4LocalData[iLocalPixOffset].y;    // vertical gradient of Green
	fVSum[2] -= (float)uc4LocalData[iLocalPixOffset++].z;  // vertical gradient of Blue

    // N
	fVSum[0] -= (float)(uc4LocalData[iLocalPixOffset].x << 1);  // vertical gradient of Red
	fVSum[1] -= (float)(uc4LocalData[iLocalPixOffset].y << 1);  // vertical gradient of Green
	fVSum[2] -= (float)(uc4LocalData[iLocalPixOffset++].z << 1);// vertical gradient of Blue

    // NE
	fHSum[0] -= (float)uc4LocalData[iLocalPixOffset].x;    // horizontal gradient of Red
	fHSum[1] -= (float)uc4LocalData[iLocalPixOffset].y;    // horizontal gradient of Green
	fHSum[2] -= (float)uc4LocalData[iLocalPixOffset].z;    // horizontal gradient of Blue
	fVSum[0] -= (float)uc4LocalData[iLocalPixOffset].x;    // vertical gradient of Red
	fVSum[1] -= (float)uc4LocalData[iLocalPixOffset].y;    // vertical gradient of Green
	fVSum[2] -= (float)uc4LocalData[iLocalPixOffset].z;    // vertical gradient of Blue

    // increment LMEM block to next row, and unwind increments
    iLocalPixOffset += (iLocalPixPitch - 2);    
                
    // W
	fHSum[0] += (float)(uc4LocalData[iLocalPixOffset].x << 1);  // vertical gradient of Red
	fHSum[1] += (float)(uc4LocalData[iLocalPixOffset].y << 1);  // vertical gradient of Green
	fHSum[2] += (float)(uc4LocalData[iLocalPixOffset++].z << 1);// vertical gradient of Blue

    // C
    iLocalPixOffset++;

    // E
	fHSum[0] -= (float)(uc4LocalData[iLocalPixOffset].x << 1);  // vertical gradient of Red
	fHSum[1] -= (float)(uc4LocalData[iLocalPixOffset].y << 1);  // vertical gradient of Green
	fHSum[2] -= (float)(uc4LocalData[iLocalPixOffset].z << 1);  // vertical gradient of Blue

    // increment LMEM block to next row, and unwind increments
    iLocalPixOffset += (iLocalPixPitch - 2);    

    // SW
	fHSum[0] += (float)uc4LocalData[iLocalPixOffset].x;    // horizontal gradient of Red
	fHSum[1] += (float)uc4LocalData[iLocalPixOffset].y;    // horizontal gradient of Green
	fHSum[2] += (float)uc4LocalData[iLocalPixOffset].z;    // horizontal gradient of Blue
	fVSum[0] += (float)uc4LocalData[iLocalPixOffset].x;    // vertical gradient of Red
	fVSum[1] += (float)uc4LocalData[iLocalPixOffset].y;    // vertical gradient of Green
	fVSum[2] += (float)uc4LocalData[iLocalPixOffset++].z;  // vertical gradient of Blue

    // S
	fVSum[0] += (float)(uc4LocalData[iLocalPixOffset].x << 1);  // vertical gradient of Red
	fVSum[1] += (float)(uc4LocalData[iLocalPixOffset].y << 1);  // vertical gradient of Green
	fVSum[2] += (float)(uc4LocalData[iLocalPixOffset++].z << 1);// vertical gradient of Blue

    // SE
	fHSum[0] -= (float)uc4LocalData[iLocalPixOffset].x;    // horizontal gradient of Red
	fHSum[1] -= (float)uc4LocalData[iLocalPixOffset].y;    // horizontal gradient of Green
	fHSum[2] -= (float)uc4LocalData[iLocalPixOffset].z;    // horizontal gradient of Blue
	fVSum[0] += (float)uc4LocalData[iLocalPixOffset].x;    // vertical gradient of Red
	fVSum[1] += (float)uc4LocalData[iLocalPixOffset].y;    // vertical gradient of Green
	fVSum[2] += (float)uc4LocalData[iLocalPixOffset].z;    // vertical gradient of Blue

	// Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
	fTemp =  0.30f * sqrt((fHSum[0] * fHSum[0]) + (fVSum[0] * fVSum[0]));
	fTemp += 0.55f * sqrt((fHSum[1] * fHSum[1]) + (fVSum[1] * fVSum[1]));
	fTemp += 0.15f * sqrt((fHSum[2] * fHSum[2]) + (fVSum[2] * fVSum[2]));

    // threshold and clamp
    if (fTemp < fThresh)
    {
        fTemp = 0.0f;
    }
    else if (fTemp > 255.0f)
    {
        fTemp = 255.0f;
    }

    // pack into a monochrome uint 
    unsigned int uiPackedPix = 0x000000FF & (unsigned int)fTemp;
    uiPackedPix |= 0x0000FF00 & (((unsigned int)fTemp) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)fTemp) << 16);

    // Write out to GMEM with restored offset
    if((iDevYPrime + 1 < iDevImageHeight) && (iImagePosX < iImageWidth))
    {
        uiDest[iDevGMEMOffset + get_global_size(0)] = uiPackedPix;
    }
}
