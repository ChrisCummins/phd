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

// Demo specific includes and options
#include "oclRecursiveGaussian.h"

// Helper function to convert 32-bit unsigned integer to float[4] rgba color 
void rgbaUintToFloat4(const unsigned int uiPackedRGBA, float* rgba)
{
    rgba[0] = (float)(uiPackedRGBA & 0xff);
    rgba[1] = (float)((uiPackedRGBA >> 8) & 0xff);
    rgba[2] = (float)((uiPackedRGBA >> 16) & 0xff);
    rgba[3] = (float)((uiPackedRGBA >> 24) & 0xff);
}

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
unsigned int rgbaFloat4ToUint(float* rgba)
{
    // Clamp to zero 
    for(int i = 0; i < 4; ++i) 
    {
        if(rgba[i] < 0.0f)
        {
            rgba[i] = 0.0f;
        }        
    }

    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)rgba[0];
    uiPackedPix |= 0x0000FF00 & ((unsigned int)(rgba[1]) << 8);
    uiPackedPix |= 0x00FF0000 & ((unsigned int)(rgba[2]) << 16);
    uiPackedPix |= 0xFF000000 & ((unsigned int)(rgba[3]) << 24);
    return uiPackedPix;
}

// CPU/Host transpose function
void TransposeHost(const unsigned int* uiDataIn, unsigned int* uiDataOut, int iWidth, int iHeight) 
{
    // transpose matrix
    for(int Y = 0; Y < iHeight; Y++) 
    {
        int iBaseIn = Y * iWidth;
        for(int X = 0; X < iWidth; X++) 
        {
            uiDataOut[X * iHeight + Y] = uiDataIn[iBaseIn + X];
        }
    }  
}

// CPU/Host simple recursive Gaussian filter implementation
//*****************************************************************
//    - processes one image column per thread
//      parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a  - blur parameter
//*****************************************************************
void SimpleRecursiveGaussianRGBAHost(const unsigned int* uiDataIn, unsigned int* uiDataOut, 
                                     int iWidth, int iHeight, float a)
{
    float yp [4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float xc [4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float yc [4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float fTemp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int X = 0; X < iWidth; X++)
    {
        // forward pass
        rgbaUintToFloat4(uiDataIn[X], yp);  // previous output
        for (int Y = 0; Y < iHeight; Y++) 
        {
            int iOffSet = Y * iWidth + X;
            rgbaUintToFloat4(uiDataIn[iOffSet], xc);
            yc[0] = xc[0] + (yp[0] - xc[0]) * a;   
            yc[1] = xc[1] + (yp[1] - xc[1]) * a;   
            yc[2] = xc[2] + (yp[2] - xc[2]) * a;   
            yc[3] = xc[3] + (yp[3] - xc[3]) * a;   
		    uiDataOut[iOffSet] = rgbaFloat4ToUint(yc);
            yp[0] = yc[0];
            yp[1] = yc[1];
            yp[2] = yc[2];
            yp[3] = yc[3];
        }

        // reset to last element of column at this x position
        rgbaUintToFloat4(uiDataIn[(iHeight - 1) * iWidth + X], yp);  

        // reverse pass: ensures response is symmetrical
        for (int Y = iHeight - 1; Y > -1; Y--) 
        {
            int iOffSet = Y * iWidth + X;
            rgbaUintToFloat4(uiDataIn[iOffSet], xc);
            yc[0] = xc[0] + (yp[0] - xc[0]) * a;   
            yc[1] = xc[1] + (yp[1] - xc[1]) * a;   
            yc[2] = xc[2] + (yp[2] - xc[2]) * a;   
            yc[3] = xc[3] + (yp[3] - xc[3]) * a;   
            rgbaUintToFloat4(uiDataOut[iOffSet], fTemp);
   		    fTemp[0] = (fTemp[0] + yc[0]) * 0.5f;
   		    fTemp[1] = (fTemp[1] + yc[1]) * 0.5f;
   		    fTemp[2] = (fTemp[2] + yc[2]) * 0.5f;
   		    fTemp[3] = (fTemp[3] + yc[3]) * 0.5f;
            uiDataOut[iOffSet] = rgbaFloat4ToUint (fTemp); 
            yp[0] = yc[0];
            yp[1] = yc[1];
            yp[2] = yc[2];
            yp[3] = yc[3];
        }
    }
}

// CPU/Host recursive Gaussian filter implementation
//*****************************************************************
//  parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//*****************************************************************
void RecursiveGaussianRGBAHost(const unsigned int* uiDataIn, unsigned int* uiDataOut, 
                               int iWidth, int iHeight, 
                               float a0, float a1, float a2, float a3, 
                               float b1, float b2, float coefp, float coefn)
{

    // outer loop over all columns within image
    for (int X = 0; X < iWidth; X++)
    {
        // start forward filter pass
        float xp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous input
        float yp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output
        float yb[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output by 2

#ifdef CLAMP_TO_EDGE
        rgbaUintToFloat4(uiDataIn[X], xp); 
        for (int i = 0; i < 4; i++)
        {
            yb[i] = xp[i] * coefp; 
            yp[i] = yb[i];
        }
#endif

        float xc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
        for (int Y = 0; Y < iHeight; Y++) 
        {
            int iOffSet = Y * iWidth + X;
            rgbaUintToFloat4(uiDataIn[iOffSet], xc);
            yc[0] = (a0 * xc[0]) + (a1 * xp[0]) - (b1 * yp[0]) - (b2 * yb[0]);
            yc[1] = (a0 * xc[1]) + (a1 * xp[1]) - (b1 * yp[1]) - (b2 * yb[1]);
            yc[2] = (a0 * xc[2]) + (a1 * xp[2]) - (b1 * yp[2]) - (b2 * yb[2]);
            yc[3] = (a0 * xc[3]) + (a1 * xp[3]) - (b1 * yp[3]) - (b2 * yb[3]);
            uiDataOut[iOffSet] = rgbaFloat4ToUint(yc);
            for (int i = 0; i < 4; i++)
            {
                xp[i] = xc[i]; 
                yb[i] = yp[i]; 
                yp[i] = yc[i]; 
            }
        }
        
        // start reverse filter pass: ensures response is symmetrical
        float xn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float xa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ya[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#ifdef CLAMP_TO_EDGE
        // reset to last element of column
        rgbaUintToFloat4(uiDataIn[(iHeight - 1) * iWidth + X], xn);  
        for (int i = 0; i< 4; i++)
        {
            xa[i] = xn[i]; 
            yn[i] = xn[i] * coefn; 
            ya[i] = yn[i];
        }
#endif

        float fTemp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int Y = iHeight - 1; Y > -1; Y--) 
        {
            int iOffSet = Y * iWidth + X;
            rgbaUintToFloat4(uiDataIn[iOffSet], xc);  
            yc[0] = (a2 * xn[0]) + (a3 * xa[0]) - (b1 * yn[0]) - (b2 * ya[0]);
            yc[1] = (a2 * xn[1]) + (a3 * xa[1]) - (b1 * yn[1]) - (b2 * ya[1]);
            yc[2] = (a2 * xn[2]) + (a3 * xa[2]) - (b1 * yn[2]) - (b2 * ya[2]);
            yc[3] = (a2 * xn[3]) + (a3 * xa[3]) - (b1 * yn[3]) - (b2 * ya[3]);
            for (int i = 0; i< 4; i++)
            {
                xa[i] = xn[i]; 
                xn[i] = xc[i]; 
                ya[i] = yn[i]; 
                yn[i] = yc[i];
            }
            rgbaUintToFloat4(uiDataOut[iOffSet], fTemp); 
            fTemp[0] += yc[0];
            fTemp[1] += yc[1];
            fTemp[2] += yc[2];
            fTemp[3] += yc[3];
            uiDataOut[iOffSet] = rgbaFloat4ToUint(fTemp);
        }
    }
}

//*****************************************************************
//! Compute Recursive Gaussian filter on the host
//! @param uiInputImage     pointer to input data
//! @param uiTempImage      pointer to temporary store
//! @param uiOutputImage    pointer to temporary store
//! @param iWidth           width of image
//! @param iHeight          height of image
//! @param GaussParms* pGP  struct packet of filter coeffs
//*****************************************************************
double HostRecursiveGaussianRGBA(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                               int iWidth, int iHeight, GaussParms* pGP)
{
    // start computation timer
    shrDeltaT(0);

    // run the gaussian in one dimension
    #if USE_SIMPLE_FILTER
            SimpleRecursiveGaussianRGBAHost(uiInputImage, uiTempImage, iWidth, iHeight, pGP->ema);
    #else   
            RecursiveGaussianRGBAHost(uiInputImage, uiTempImage, iWidth, iHeight, 
                                    pGP->a0, pGP->a1, pGP->a2, pGP->a3, 
                                    pGP->b1, pGP->b2, pGP->coefp, pGP->coefn);
    #endif
    
    // transpose results
    TransposeHost(uiTempImage, uiOutputImage, iWidth, iHeight);

    // run the Gaussian in the other dimension
    // note width and height flipped due to transpose
    #if USE_SIMPLE_FILTER
        SimpleRecursiveGaussianRGBAHost(uiOutputImage, uiTempImage, iHeight, iWidth, pGP->ema);
    #else 
        RecursiveGaussianRGBAHost(uiOutputImage, uiTempImage, iHeight, iWidth, 
                                    pGP->a0, pGP->a1, pGP->a2, pGP->a3, 
                                    pGP->b1, pGP->b2, pGP->coefp, pGP->coefn);
    #endif

    // transpose the results back
    // note width and height flipped due to 1st transpose
    TransposeHost(uiTempImage, uiOutputImage, iHeight, iWidth);

    // return computation elapsed time in seconds
    return shrDeltaT(0);
}

// Function implementation to precompute gaussian filter parameters
//*****************************************************************
void PreProcessGaussParms (float fSigma, int iOrder, GaussParms* pGP)
{
    // pre-compute filter coefficients
    pGP->nsigma = fSigma; // note: fSigma is range-checked and clamped >= 0.1f upstream
    pGP->alpha = 1.695f / pGP->nsigma;
    pGP->ema = exp(-pGP->alpha);
    pGP->ema2 = exp(-2.0f * pGP->alpha);
    pGP->b1 = -2.0f * pGP->ema;
    pGP->b2 = pGP->ema2;
    pGP->a0 = 0.0f;
    pGP->a1 = 0.0f;
    pGP->a2 = 0.0f;
    pGP->a3 = 0.0f;
    pGP->coefp = 0.0f;
    pGP->coefn = 0.0f;
    switch (iOrder) 
    {
        case 0: 
            {
                const float k = (1.0f - pGP->ema)*(1.0f - pGP->ema)/(1.0f + (2.0f * pGP->alpha * pGP->ema) - pGP->ema2);
                pGP->a0 = k;
                pGP->a1 = k * (pGP->alpha - 1.0f) * pGP->ema;
                pGP->a2 = k * (pGP->alpha + 1.0f) * pGP->ema;
                pGP->a3 = -k * pGP->ema2;
            } 
            break;
        case 1: 
            {
                pGP->a0 = (1.0f - pGP->ema) * (1.0f - pGP->ema);
                pGP->a1 = 0.0f;
                pGP->a2 = -pGP->a0;
                pGP->a3 = 0.0f;
            } 
            break;
        case 2: 
            {
                const float ea = exp(-pGP->alpha);
                const float k = -(pGP->ema2 - 1.0f)/(2.0f * pGP->alpha * pGP->ema);
                float kn = -2.0f * (-1.0f + (3.0f * ea) - (3.0f * ea * ea) + (ea * ea * ea));
                kn /= (((3.0f * ea) + 1.0f + (3.0f * ea * ea) + (ea * ea * ea)));
                pGP->a0 = kn;
                pGP->a1 = -kn * (1.0f + (k * pGP->alpha)) * pGP->ema;
                pGP->a2 = kn * (1.0f - (k * pGP->alpha)) * pGP->ema;
                pGP->a3 = -kn * pGP->ema2;
            } 
            break;
        default:
            // note: iOrder is range-checked and clamped to 0-2 upstream
            return;
    }
    pGP->coefp = (pGP->a0 + pGP->a1)/(1.0f + pGP->b1 + pGP->b2);
    pGP->coefn = (pGP->a2 + pGP->a3)/(1.0f + pGP->b1 + pGP->b2);
}
