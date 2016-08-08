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

// Utililities and system includes
#include <oclUtils.h>

// if CLAMP_TO_EDGE defined, filter gives "fade-from-dark" edges (model where zero value pixels are being averaged in when within radius of edge)
// if not, filter gives "extrapolation" edges (model where pixels within radius are copied from the last value outside radius)
#define CLAMP_TO_EDGE

// 0 selects full version of filter function, 1 selects simple version 
#define USE_SIMPLE_FILTER 0

// custom type for Gaussian parameter precomputation
typedef struct _GaussParms
{
    float nsigma; 
    float alpha;
    float ema; 
    float ema2; 
    float b1; 
    float b2; 
    float a0; 
    float a1; 
    float a2; 
    float a3; 
    float coefp; 
    float coefn; 
} GaussParms, *pGaussParms;

// struct instance to hold all the Gaussian filter coefs
static GaussParms oclGP;               

// forward interface declaration for Gaussian parameter pre-processing function (for host and GPU proc)
extern "C" void PreProcessGaussParms (float fSigma, int iOrder, GaussParms* pGP);

// forward interface declaration for host Gaussian processing function
extern "C" double HostRecursiveGaussianRGBA(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                                          int iWidth, int iHeight, GaussParms* pGP);
