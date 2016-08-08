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

#include <shrUtils.h>

// export C interface
//*****************************************************************
extern "C" double BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                                unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(const float* rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba[0] * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba[1] * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba[2] * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba[3] * fScale)) << 24);
    return uiPackedPix;
}

// CPU row box filter implementation
//*****************************************************************
void BoxFilterHostX(unsigned char* uiInputImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // do all rows, one row at a time
    for (unsigned int y = 0; y < uiHeight; y++) 
    {
        // init with dark edges
        float f4Sum [4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // do left edge first 
        for (int x = 0; x <= r; x++) 
        {
            int iBase = (y * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];
        }
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);

        // then do pixels within/including a radius of left edge, using rolling method
        for(int x = 1; x <= r; x++) 
        {
            // add the next rolling pix
            int iBase = (y * uiWidth + x + r) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            // nothing to delete for trailing edge... fade up from dark edge

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
        
        // do main body of image (beyond a radius of left and right edges), using rolling method
        for(unsigned int x = r + 1; x < uiWidth - r; x++) 
        {
            // add the next rolling pix
            int iBase = (y * uiWidth + x + r) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            // delete the trailing pix
            iBase = (y * uiWidth + x - r - 1) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }

        // do pixels within a radius of right edge 
        for (unsigned int x = uiWidth - r; x < uiWidth; x++) 
        {
            // No additions for next rolling pix... fade out to dark edge

            // delete the trailing pix
            int iBase = (y * uiWidth + x - r - 1) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }
}

// CPU column box filter implementation
//*****************************************************************
void BoxFilterHostY(unsigned char* uiInputImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // do all columns, one column at a time
    for (unsigned int x = 0; x < uiWidth; x++) 
    {
        // init with dark edges
        float f4Sum [4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // do top edge first 
        for (int y = 0; y <= r; y++) 
        {
            int iBase = (y * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];
        }
        uiOutputImage[x] = rgbaFloat4ToUint(f4Sum, fScale);

        // then do pixels within/including a radius of top edge, using rolling method
        for(int y = 1; y <= r; y++) 
        {
            // add the next rolling pix
            int iBase = ((y + r) * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];


            // nothing to delete for trailing edge... fade up from dark edge

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
        
        // do main body of image (beyond a radius of top and bottom edges), using rolling method
        for(unsigned int y = r + 1; y < uiHeight - r; y++) 
        {
            // add the next rolling pix
            int iBase = ((y + r) * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            // delete the trailing pix
            iBase = ((y - r) * uiWidth + x - uiWidth) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale); 
        }

        // do bottom edge
        for (unsigned int y = uiHeight - r; y < uiHeight; y++) 
        {
            // No additions for next rolling pix... dark edges

            // delete the trailing pix
            int iBase = ((y - r) * uiWidth + x - uiWidth) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale); 
        }
    }
}

//*****************************************************************
//! Compute reference data set
//! @param uiInputImage     pointer to input data
//! @param uiTempImage      pointer to temporary store
//! @param uiOutputImage    pointer to output data
//! @param uiWidth          width of image
//! @param uiHeight         height of image
//! @param r                radius of filter
//! @param r                rescale factor
//*****************************************************************
double BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                     unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // start computation timer
    shrDeltaT(0);

    // Run the computations
    BoxFilterHostX((unsigned char*)uiInputImage, uiTempImage, uiWidth, uiHeight, r, fScale);
    BoxFilterHostY((unsigned char*)uiTempImage, uiOutputImage, uiWidth, uiHeight, r, fScale);

    // return computation elapsed time in seconds
    return shrDeltaT(0);
}
