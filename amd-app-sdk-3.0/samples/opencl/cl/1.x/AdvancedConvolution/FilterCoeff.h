/*******************************************************************************
 Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1   Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 2   Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
/**
 ********************************************************************************
 * @file <filterCoeff.h>
 *
 * @brief Contains the filter coefficients.
 *
 ********************************************************************************
 */

#ifndef __SEPFILTERCOEFF__H
#define __SEPFILTERCOEFF__H

/********************************************************************************
 *
 * Sobel 3X3 filter:
 *                      1 0 -1
 *                      2 0 -2
 *                      1 0 -1
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.75984, -1.51967, -0.75984};
 * h = {-1.31607, 0.0,  1.31607};
 *********************************************************************************/
float SOBEL_FILTER_3x3[3*3] = 
							{
								 1.0f,  0.0f,  -1.0f,
								 2.0f,  0.0f,  -2.0f,
								 1.0f,  0.0f,  -1.0f
							};
float SOBEL_FILTER_3x3_pass1[3] = { -1.31607f, 0.0000f, 1.31607f };
float SOBEL_FILTER_3x3_pass2[3] = { -0.75984f, -1.51967f, -0.75984f };

/********************************************************************************
 *
 * Sobel 5X5 filter:
 *                       1 2 0 -2 -1; 
 *                       4 8 0 -8 -4;
 *                       6 12 0 -12 -6;
 *                       4 8 0 -8 -4;
 *                       1 2 0 -2 -1;
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.61479, -2.45915, -3.68873, -2.45915, -0.61479};
 * h = {-1.62658, -3.25315, -0.00000,  3.25315,  1.62658};
 *********************************************************************************/
float SOBEL_FILTER_5x5[5*5] = 
							{
								 1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f,
								 4.0f,  8.0f,   0.0f,  -8.0f,  -4.0f,
								 6.0f,  12.0f,  0.0f,  -12.0f, -6.0f,
								 4.0f,  8.0f,   0.0f,  -8.0f,  -4.0f,
								 1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f
							};
float SOBEL_FILTER_5x5_pass1[5] = { -1.62658f, -3.25315f, -0.00000f, 3.25315f, 1.62658f };
float SOBEL_FILTER_5x5_pass2[5] = { -0.61479f, -2.45915f, -3.68873f, -2.45915f, -0.61479f };

/********************************************************************************
 *
 * Box 3X3 filter:
 *          0.11, 0.11, 0.11
 *          0.11, 0.11, 0.11
 *          0.11, 0.11, 0.11
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.33166, -0.33166, -0.33166};
 * h = {-0.33166, -0.33166, -0.33166};
 *********************************************************************************/
float BOX_FILTER_3x3[3*3] = 
							{
								 0.11f, 0.11f, 0.11f,
								 0.11f, 0.11f, 0.11f,
								 0.11f, 0.11f, 0.11f
							};

float BOX_FILTER_3x3_pass1[3] = { -0.33166f, -0.33166f, -0.33166f };
float BOX_FILTER_3x3_pass2[3] = { -0.33166f, -0.33166f, -0.33166f };
                                          
/********************************************************************************
 *
 * Box 5X5 filter:
 *       0.040f, 0.040f, 0.040f, 0.040f, 0.040f, 
 *       0.040f, 0.040f, 0.040f, 0.040f, 0.040f, 
 *       0.040f, 0.040f, 0.040f, 0.040f, 0.040f, 
 *       0.040f, 0.040f, 0.040f, 0.040f, 0.040f, 
 *       0.040f, 0.040f, 0.040f, 0.040f, 0.040f
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.20000f,  -0.20000f,  -0.20000f,  -0.20000f,  -0.20000f};
 * h = {-0.20000f,  -0.20000f,  -0.20000f,  -0.20000f,  -0.20000f};
 *********************************************************************************/
float BOX_FILTER_5x5[5*5] = 
							{
								 0.040f, 0.040f, 0.040f, 0.040f, 0.040f,
								 0.040f, 0.040f, 0.040f, 0.040f, 0.040f,
								 0.040f, 0.040f, 0.040f, 0.040f, 0.040f,
								 0.040f, 0.040f, 0.040f, 0.040f, 0.040f,
								 0.040f, 0.040f, 0.040f, 0.040f, 0.040f
							};

float BOX_FILTER_5x5_pass1[5] = { -0.20000f, -0.20000f, -0.20000f, -0.20000f,-0.20000f };
float BOX_FILTER_5x5_pass2[5] = { -0.20000f, -0.20000f, -0.20000f, -0.20000f,-0.20000f };

/********************************************************************************
 *
 * Gaussian 3X3 filter:
 *          0.0625f, 0.1250f, 0.0625f,
 *          0.1250f, 0.2500f, 0.1250f,
 *          0.0625f, 0.1250f, 0.0625f
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.25000f, -0.50000f, -0.25000f};
 * h = {-0.25000f, -0.50000f, -0.25000f};
 *********************************************************************************/
float GAUSSIAN_FILTER_3x3[3*3] = 
							{
								 0.0625f, 0.1250f, 0.0625f,
								 0.1250f, 0.2500f, 0.1250f,
								 0.0625f, 0.1250f, 0.0625f
							};
float GAUSSIAN_FILTER_3x3_pass1[3] = {-0.25000f, -0.50000f, -0.25000f};
float GAUSSIAN_FILTER_3x3_pass2[3] = {-0.25000f, -0.50000f, -0.25000f};


/********************************************************************************
 *
 * Gaussian 5X5 filter:
 *       0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f, 
 *       0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f, 
 *       0.0234375f, 0.0937500f, 0.1406250f, 0.0937500f, 0.0234375f, 
 *       0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f, 
 *       0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {-0.062500f, -0.250000f, -0.375000f, -0.250000f, -0.062500f};
 * h = {-0.062500f, -0.250000f, -0.375000f, -0.250000f, -0.062500f};
 *********************************************************************************/
float GAUSSIAN_FILTER_5x5[5*5] = 
							{
								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f,
								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
								 0.0234375f, 0.0937500f, 0.1406250f, 0.0937500f, 0.0234375f,
								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f
							};
      
float GAUSSIAN_FILTER_5x5_pass1[5] = {-0.062500f, -0.250000f, -0.375000f, -0.250000f, -0.062500f};
float GAUSSIAN_FILTER_5x5_pass2[5] = {-0.062500f, -0.250000f, -0.375000f, -0.250000f, -0.062500f};

#endif
