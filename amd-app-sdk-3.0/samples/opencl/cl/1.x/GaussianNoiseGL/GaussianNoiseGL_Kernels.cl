/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#define IA 16807    			// a
#define IM 2147483647 			// m
#define AM (1.0/IM) 			// 1/m - To calculate floating point result
#define IQ 127773
#define IR 2836
#define NTAB 4
#define NDIV (1 + (IM - 1)/ NTAB)
#define EPS 1.2e-7
#define RMAX (1.0 - EPS)
#define FACTOR 60			// Deviation factor
#define GROUP_SIZE 64
#define PI 3.14

float ran1(int idum, __local int *iv);
float2 BoxMuller(float2 uniform);


__kernel void gaussian_transform(__global uchar4* inputImage, __write_only  image2d_t outputImage, int factor)
{
    /* Global threads in x-direction = ImageWidth / 2 */
    int pos0 = get_global_id(0) + 2 * get_global_size(0) * get_global_id(1);
    int pos1 = get_global_id(0) + get_global_size(0) + 2 * get_global_size(0) * get_global_id(1);

    /* Read 2 texels from image data */
    float4 texel0 = convert_float4(inputImage[pos0]);
    float4 texel1 = convert_float4(inputImage[pos1]);

    /* Compute the average value for each pixel */
    float avg0 = (texel0.x + texel0.y + texel0.z + texel0.w) / 4;
    float avg1 = (texel1.x + texel1.y + texel1.z + texel1.w) / 4;

    __local int iv0[NTAB * GROUP_SIZE];
    __local int iv1[NTAB * GROUP_SIZE];

    /* Compute uniform deviation for the pixel */
    float dev0 = ran1(-avg0, iv0);
    float dev1 = ran1(-avg1, iv1);

    /* Apply the box-muller transform */
    float2 gaussian = BoxMuller((float2)(dev0, dev1));

    float4 out0 = (texel0 + (float4)(gaussian.x * factor))/((float4)255);
    float4 out1 = (texel1 + (float4)(gaussian.y * factor))/((float4)255);

    int2 locate0 ,locate1;
    locate0.x = get_global_id(0);
    locate0.y = get_global_id(1);
    locate1.x = get_global_id(0) + get_global_size(0);
    locate1.y = get_global_id(1);


    write_imagef(outputImage, locate0,out0 );
    write_imagef(outputImage, locate1,out1 );


}

