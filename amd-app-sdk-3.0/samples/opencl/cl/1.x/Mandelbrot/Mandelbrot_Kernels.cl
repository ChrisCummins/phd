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

/**
*  A fractal generator that calculates the mandlebrot set 
*  http://en.wikipedia.org/wiki/Mandelbrot_set 
* @param mandelbrotImage    mandelbrot images is stored in this
* @param scale              Represents the distance from which the fractal 
*                           is being seen if this is greater more area and 
*                           less detail is seen           
* @param maxIterations      More iterations gives more accurate mandelbrot image 
* @param width              size of the image 
*/

__kernel void mandelbrot_vector_float(
				__global uchar4 * mandelbrotImage,
                const    float posx, 
                const    float posy, 
                const    float stepSizeX,
                const    float stepSizeY,
                const    uint maxIterations,
                const    int width,
                const    int bench)
{
    int tid = get_global_id(0);

    int i = tid % (width / 4);
    int j = tid / (width / 4);
    
    int4 veci = {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
    int4 vecj = {j, j, j, j};
    
    float4 x0;
    x0.s0 = (float)(posx + stepSizeX * (float)veci.s0);
    x0.s1 = (float)(posx + stepSizeX * (float)veci.s1);
    x0.s2 = (float)(posx + stepSizeX * (float)veci.s2);
    x0.s3 = (float)(posx + stepSizeX * (float)veci.s3);
    float4 y0;
    y0.s0 = (float)(posy + stepSizeY * (float)vecj.s0);
    y0.s1 = (float)(posy + stepSizeY * (float)vecj.s1);
    y0.s2 = (float)(posy + stepSizeY * (float)vecj.s2);
    y0.s3 = (float)(posy + stepSizeY * (float)vecj.s3);

    float4 x = x0;
    float4 y = y0;
    
    uint iter=0;
    float4 tmp;
    int4 stay;
    int4 ccount = 0;

    stay.s0 = (x.s0 * x.s0 + y.s0 * y.s0) <= 4.0f;
    stay.s1 = (x.s1 * x.s1 + y.s1 * y.s1) <= 4.0f;
    stay.s2 = (x.s2 * x.s2 + y.s2 * y.s2) <= 4.0f;
    stay.s3 = (x.s3 * x.s3 + y.s3 * y.s3) <= 4.0f;
    float4 savx = x;
    float4 savy = y;
    for(iter=0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) && (iter < maxIterations); iter+= 16)
    {
        x = savx;
        y = savy;

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);
        
        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * x, y, y0);
        x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);

        stay.s0 = (x.s0 * x.s0 + y.s0 * y.s0) <= 4.0f;
        stay.s1 = (x.s1 * x.s1 + y.s1 * y.s1) <= 4.0f;
        stay.s2 = (x.s2 * x.s2 + y.s2 * y.s2) <= 4.0f;
        stay.s3 = (x.s3 * x.s3 + y.s3 * y.s3) <= 4.0f;

        savx.s0 = (stay.s0 ? x.s0 : savx.s0);
        savx.s1 = (stay.s1 ? x.s1 : savx.s1);
        savx.s2 = (stay.s2 ? x.s2 : savx.s2);
        savx.s3 = (stay.s3 ? x.s3 : savx.s3);
        savy.s0 = (stay.s0 ? y.s0 : savy.s0);
        savy.s1 = (stay.s1 ? y.s1 : savy.s1);
        savy.s2 = (stay.s2 ? y.s2 : savy.s2);
        savy.s3 = (stay.s3 ? y.s3 : savy.s3);
        ccount += stay*16;
    }
    // Handle remainder
    if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))
    {
        iter = 16;
        do
        {
            x = savx;
            y = savy;
            stay.s0 = ((x.s0 * x.s0 + y.s0 * y.s0) <= 4.0f) && 
						   (ccount.s0 < maxIterations);
            stay.s1 = ((x.s1 * x.s1 + y.s1 * y.s1) <= 4.0f) && 
				       (ccount.s1 < maxIterations);
            stay.s2 = ((x.s2 * x.s2 + y.s2 * y.s2) <= 4.0f) && 
				       (ccount.s2 < maxIterations);
            stay.s3 = ((x.s3 * x.s3 + y.s3 * y.s3) <= 4.0f) && 
				       (ccount.s3 < maxIterations);
            tmp = x;
            x = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
            y = MUL_ADD(2.0f * tmp, y, y0);
            ccount += stay;
            iter--;
            savx.s0 = (stay.s0 ? x.s0 : savx.s0);
            savx.s1 = (stay.s1 ? x.s1 : savx.s1);
            savx.s2 = (stay.s2 ? x.s2 : savx.s2);
            savx.s3 = (stay.s3 ? x.s3 : savx.s3);
            savy.s0 = (stay.s0 ? y.s0 : savy.s0);
            savy.s1 = (stay.s1 ? y.s1 : savy.s1);
            savy.s2 = (stay.s2 ? y.s2 : savy.s2);
            savy.s3 = (stay.s3 ? y.s3 : savy.s3);
        } while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);
    }
    x = savx;
    y = savy;
    float4 fc = convert_float4(ccount);
    fc.s0 = (float)ccount.s0 + 1 - 
			           native_log2(native_log2(x.s0 * x.s0 + y.s0 * y.s0));
    fc.s1 = (float)ccount.s1 + 1 - 
		           native_log2(native_log2(x.s1 * x.s1 + y.s1 * y.s1));
    fc.s2 = (float)ccount.s2 + 1 - 
				   native_log2(native_log2(x.s2 * x.s2 + y.s2 * y.s2));
    fc.s3 = (float)ccount.s3 + 1 - 
				   native_log2(native_log2(x.s3 * x.s3 + y.s3 * y.s3));
				   
    float c = fc.s0 * 2.0f * 3.1416f / 256.0f;
    uchar4 color[4];
    color[0].s0 = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[0].s1 = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[0].s2 = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[0].s3 = 0xff;
    if (ccount.s0 == maxIterations)
    {
        color[0].s0 = 0;
        color[0].s1 = 0;
        color[0].s2 = 0;
    }
    if (bench)
    {
        color[0].s0 = ccount.s0 & 0xff;
        color[0].s1 = (ccount.s0 & 0xff00) >> 8;
        color[0].s2 = (ccount.s0 & 0xff0000) >> 16;
        color[0].s3 = (ccount.s0 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid] = color[0];    
    c = fc.s1 * 2.0f * 3.1416f / 256.0f;
    color[1].s0 = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[1].s1 = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[1].s2 = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[1].s3 = 0xff;
    if (ccount.s1 == maxIterations)
    {
        color[1].s0 = 0;
        color[1].s1 = 0;
        color[1].s2 = 0;
    }
    if (bench)
    {
        color[1].s0 = ccount.s1 & 0xff;
        color[1].s1 = (ccount.s1 & 0xff00) >> 8;
        color[1].s2 = (ccount.s1 & 0xff0000) >> 16;
        color[1].s3 = (ccount.s1 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 1] = color[1];    
    c = fc.s2 * 2.0f * 3.1416f / 256.0f;
    color[2].s0 = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[2].s1 = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[2].s2 = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[2].s3 = 0xff;
    if (ccount.s2 == maxIterations)
    {
        color[2].s0 = 0;
        color[2].s1 = 0;
        color[2].s2 = 0;
    }
    if (bench)
    {
        color[2].s0 = ccount.s2 & 0xff;
        color[2].s1 = (ccount.s2 & 0xff00) >> 8;
        color[2].s2 = (ccount.s2 & 0xff0000) >> 16;
        color[2].s3 = (ccount.s2 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 2] = color[2];    
    c = fc.s3 * 2.0f * 3.1416f / 256.0f;
    color[3].s0 = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[3].s1 = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[3].s2 = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[3].s3 = 0xff;
    if (ccount.s3 == maxIterations)
    {
        color[3].s0 = 0;
        color[3].s1 = 0;
        color[3].s2 = 0;
    }
    if (bench)
    {
        color[3].s0 = ccount.s3 & 0xff;
        color[3].s1 = (ccount.s3 & 0xff00) >> 8;
        color[3].s2 = (ccount.s3 & 0xff0000) >> 16;
        color[3].s3 = (ccount.s3 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 3] = color[3];
}

#define native_log2 log2
#define native_cos  cos

#ifdef ENABLE_DOUBLE
#ifdef KHR_DP_EXTENSION
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mandelbrot_vector_double(
				__global uchar4 * mandelbrotImage,
				const    double posx, 
				const    double posy, 
                const    double stepSizeX,
                const    double stepSizeY,
                const    uint maxIterations,
                const    int width,
                const    int bench)
{
    int tid = get_global_id(0);

    int i = tid % (width / 4);
    int j = tid / (width / 4);
    
    int4 veci = {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
    int4 vecj = {j, j, j, j};
    
    double4 x0;
    x0.s0 = (double)(posx + stepSizeX * (double)veci.s0);
    x0.s1 = (double)(posx + stepSizeX * (double)veci.s1);
    x0.s2 = (double)(posx + stepSizeX * (double)veci.s2);
    x0.s3 = (double)(posx + stepSizeX * (double)veci.s3);
    double4 y0;
    y0.s0 = (double)(posy + stepSizeY * (double)vecj.s0);
    y0.s1 = (double)(posy + stepSizeY * (double)vecj.s1);
    y0.s2 = (double)(posy + stepSizeY * (double)vecj.s2);
    y0.s3 = (double)(posy + stepSizeY * (double)vecj.s3);

    double4 x = x0;
    double4 y = y0;
    
    uint iter=0;
    double4 tmp;
    int4 stay;
    int4 ccount = 0;

    stay.s0 = (x.s0 * x.s0 + y.s0 * y.s0) <= 4.0;
    stay.s1 = (x.s1 * x.s1 + y.s1 * y.s1) <= 4.0;
    stay.s2 = (x.s2 * x.s2 + y.s2 * y.s2) <= 4.0;
    stay.s3 = (x.s3 * x.s3 + y.s3 * y.s3) <= 4.0;
    double4 savx = x;
    double4 savy = y;
    for(iter=0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) && (iter < maxIterations); iter+= 16)
    {
        x = savx;
        y = savy;

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); // tmp = x * x + x0 - y * y;
        y = MUL_ADD(2.0 * x, y, y0); //y = 2.0 * x * y + y0;
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));// x = tmp * tmp + x0 - y * y;
        y = MUL_ADD(2.0 * tmp, y, y0); //y = 2.0 * tmp * y + y0;

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0); 

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        // Two iterations
        tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
        y = MUL_ADD(2.0 * x, y, y0); 
		x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
        y = MUL_ADD(2.0 * tmp, y, y0);

        stay.s0 = (x.s0 * x.s0 + y.s0 * y.s0) <= 4.0;
        stay.s1 = (x.s1 * x.s1 + y.s1 * y.s1) <= 4.0;
        stay.s2 = (x.s2 * x.s2 + y.s2 * y.s2) <= 4.0;
        stay.s3 = (x.s3 * x.s3 + y.s3 * y.s3) <= 4.0;

        savx.s0 = (stay.s0 ? x.s0 : savx.s0);
        savx.s1 = (stay.s1 ? x.s1 : savx.s1);
        savx.s2 = (stay.s2 ? x.s2 : savx.s2);
        savx.s3 = (stay.s3 ? x.s3 : savx.s3);
        savy.s0 = (stay.s0 ? y.s0 : savy.s0);
        savy.s1 = (stay.s1 ? y.s1 : savy.s1);
        savy.s2 = (stay.s2 ? y.s2 : savy.s2);
        savy.s3 = (stay.s3 ? y.s3 : savy.s3);
        ccount += stay*16;
    }
    // Handle remainder
    if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))
    {
        iter = 16;
        do
        {
            x = savx;
            y = savy;
            stay.s0 = ((x.s0 * x.s0 + y.s0 * y.s0) <= 4.0) && 
						   (ccount.s0 < maxIterations);
            stay.s1 = ((x.s1 * x.s1 + y.s1 * y.s1) <= 4.0) && 
				       (ccount.s1 < maxIterations);
            stay.s2 = ((x.s2 * x.s2 + y.s2 * y.s2) <= 4.0) && 
				       (ccount.s2 < maxIterations);
            stay.s3 = ((x.s3 * x.s3 + y.s3 * y.s3) <= 4.0) && 
				       (ccount.s3 < maxIterations);
            tmp = x;
            x = MUL_ADD(-y, y, MUL_ADD(x, x, x0)); 
            y = MUL_ADD(2.0 * tmp, y, y0); //y = 2.0 * tmp * y + y0;
            ccount += stay;
            iter--;
            savx.s0 = (stay.s0 ? x.s0 : savx.s0);
            savx.s1 = (stay.s1 ? x.s1 : savx.s1);
            savx.s2 = (stay.s2 ? x.s2 : savx.s2);
            savx.s3 = (stay.s3 ? x.s3 : savx.s3);
            savy.s0 = (stay.s0 ? y.s0 : savy.s0);
            savy.s1 = (stay.s1 ? y.s1 : savy.s1);
            savy.s2 = (stay.s2 ? y.s2 : savy.s2);
            savy.s3 = (stay.s3 ? y.s3 : savy.s3);
        } while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);
    }
    x = savx;
    y = savy;
    double4 fc = convert_double4(ccount);
    fc.s0 = (double)ccount.s0 + 1 - 
			           native_log2(native_log2(x.s0 * x.s0 + y.s0 * y.s0));
    fc.s1 = (double)ccount.s1 + 1 - 
		           native_log2(native_log2(x.s1 * x.s1 + y.s1 * y.s1));
    fc.s2 = (double)ccount.s2 + 1 - 
				   native_log2(native_log2(x.s2 * x.s2 + y.s2 * y.s2));
    fc.s3 = (double)ccount.s3 + 1 - 
				   native_log2(native_log2(x.s3 * x.s3 + y.s3 * y.s3));
				   
    double c = fc.s0 * 2.0 * 3.1416 / 256.0;
    uchar4 color[4];
    color[0].s0 = ((1.0 + native_cos(c)) * 0.5) * 255;
    color[0].s1 = ((1.0 + native_cos(2.0 * c + 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[0].s2 = ((1.0 + native_cos(c - 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[0].s3 = 0xff;
    if (ccount.s0 == maxIterations)
    {
        color[0].s0 = 0;
        color[0].s1 = 0;
        color[0].s2 = 0;
    }
    if (bench)
    {
        color[0].s0 = ccount.s0 & 0xff;
        color[0].s1 = (ccount.s0 & 0xff00) >> 8;
        color[0].s2 = (ccount.s0 & 0xff0000) >> 16;
        color[0].s3 = (ccount.s0 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid] = color[0];    
    c = fc.s1 * 2.0 * 3.1416 / 256.0;
    color[1].s0 = ((1.0 + native_cos(c)) * 0.5) * 255;
    color[1].s1 = ((1.0 + native_cos(2.0 * c + 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[1].s2 = ((1.0 + native_cos(c - 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[1].s3 = 0xff;
    if (ccount.s1 == maxIterations)
    {
        color[1].s0 = 0;
        color[1].s1 = 0;
        color[1].s2 = 0;
    }
    if (bench)
    {
        color[1].s0 = ccount.s1 & 0xff;
        color[1].s1 = (ccount.s1 & 0xff00) >> 8;
        color[1].s2 = (ccount.s1 & 0xff0000) >> 16;
        color[1].s3 = (ccount.s1 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 1] = color[1];    
    c = fc.s2 * 2.0 * 3.1416 / 256.0;
    color[2].s0 = ((1.0 + native_cos(c)) * 0.5) * 255;
    color[2].s1 = ((1.0 + native_cos(2.0 * c + 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[2].s2 = ((1.0 + native_cos(c - 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[2].s3 = 0xff;
    if (ccount.s2 == maxIterations)
    {
        color[2].s0 = 0;
        color[2].s1 = 0;
        color[2].s2 = 0;
    }
    if (bench)
    {
        color[2].s0 = ccount.s2 & 0xff;
        color[2].s1 = (ccount.s2 & 0xff00) >> 8;
        color[2].s2 = (ccount.s2 & 0xff0000) >> 16;
        color[2].s3 = (ccount.s2 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 2] = color[2];    
    c = fc.s3 * 2.0 * 3.1416 / 256.0;
    color[3].s0 = ((1.0 + native_cos(c)) * 0.5) * 255;
    color[3].s1 = ((1.0 + native_cos(2.0 * c + 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[3].s2 = ((1.0 + native_cos(c - 2.0 * 3.1416 / 3.0)) * 0.5) * 255;
    color[3].s3 = 0xff;
    if (ccount.s3 == maxIterations)
    {
        color[3].s0 = 0;
        color[3].s1 = 0;
        color[3].s2 = 0;
    }
    if (bench)
    {
        color[3].s0 = ccount.s3 & 0xff;
        color[3].s1 = (ccount.s3 & 0xff00) >> 8;
        color[3].s2 = (ccount.s3 & 0xff0000) >> 16;
        color[3].s3 = (ccount.s3 & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 3] = color[3];
}
#endif // ENABLE_DOUBLE
