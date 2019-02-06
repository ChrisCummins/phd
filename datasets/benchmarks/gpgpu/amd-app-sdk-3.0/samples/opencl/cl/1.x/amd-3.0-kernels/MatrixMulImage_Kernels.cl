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

/* 4x4 tile version = In each iteration of loop, data is loaded into 4 registers from matrixA, 4 registers from matrixB and their
   multiplication is computed - Fetches are cache friendly and increases the ALU/TEX ratio */
/* Requires global threads = (widthC / 4, heightC / 4) */
#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

__kernel void mmmKernel(__read_only image2d_t matrixA,
            __read_only image2d_t matrixB,
            __write_only image2d_t matrixC,
            uint widthA, 
            uint widthB)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    widthB /= 4; 

    for(int i = 0; i < widthA; i=i+4)
    {
        float4 tempA0 = read_imagef(matrixA, imageSampler, (int2)(i/4, pos.y << TILEY_SHIFT));
        float4 tempA1 = read_imagef(matrixA, imageSampler, (int2)(i/4, (pos.y << TILEY_SHIFT) + 1));
        float4 tempA2 = read_imagef(matrixA, imageSampler, (int2)(i/4, (pos.y << TILEY_SHIFT) + 2));
        float4 tempA3 = read_imagef(matrixA, imageSampler, (int2)(i/4, (pos.y << TILEY_SHIFT) + 3));

        float4 tempB0 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i));
        float4 tempB1 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 1));
        float4 tempB2 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 2));
        float4 tempB3 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 3));

        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

    }
    write_imagef(matrixC, (int2)(pos.x, pos.y * 4), sum0);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 4 + 1), sum1);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 4 + 2), sum2);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 4 + 3), sum3);
}


/* Tile 4x8 = each thread computes 16 floats*/
/* Requires global threads = (widthC / 4, heightC / 8) */
#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 8
#define TILEY_SHIFT 3


__kernel void mmmKernel2(__read_only image2d_t matrixA,
            __read_only image2d_t matrixB,
            __write_only image2d_t matrixC,
            uint widthA, 
            uint widthB)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);
    float4 sum4 = (float4)(0);
    float4 sum5 = (float4)(0);
    float4 sum6 = (float4)(0);
    float4 sum7 = (float4)(0);

    widthB = widthB >> 2;

    for(int i = 0; i < widthA; i=i+4)
    {
        float4 tempA0 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, pos.y << TILEY_SHIFT));
        float4 tempA1 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 1));
        float4 tempA2 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 2));
        float4 tempA3 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 3));
        float4 tempA4 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 4));
        float4 tempA5 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 5));
        float4 tempA6 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 6));
        float4 tempA7 = read_imagef(matrixA, imageSampler, (int2)(i >> 2, (pos.y << TILEY_SHIFT) + 7));

        float4 tempB0 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i));
        float4 tempB1 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 1));
        float4 tempB2 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 2));
        float4 tempB3 = read_imagef(matrixB, imageSampler, (int2)(pos.x, i + 3));

        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

        sum4.x += tempA4.x * tempB0.x + tempA4.y * tempB1.x + tempA4.z * tempB2.x + tempA4.w * tempB3.x;
        sum4.y += tempA4.x * tempB0.y + tempA4.y * tempB1.y + tempA4.z * tempB2.y + tempA4.w * tempB3.y;
        sum4.z += tempA4.x * tempB0.z + tempA4.y * tempB1.z + tempA4.z * tempB2.z + tempA4.w * tempB3.z;
        sum4.w += tempA4.x * tempB0.w + tempA4.y * tempB1.w + tempA4.z * tempB2.w + tempA4.w * tempB3.w;

        sum5.x += tempA5.x * tempB0.x + tempA5.y * tempB1.x + tempA5.z * tempB2.x + tempA5.w * tempB3.x;
        sum5.y += tempA5.x * tempB0.y + tempA5.y * tempB1.y + tempA5.z * tempB2.y + tempA5.w * tempB3.y;
        sum5.z += tempA5.x * tempB0.z + tempA5.y * tempB1.z + tempA5.z * tempB2.z + tempA5.w * tempB3.z;
        sum5.w += tempA5.x * tempB0.w + tempA5.y * tempB1.w + tempA5.z * tempB2.w + tempA5.w * tempB3.w;

        sum6.x += tempA6.x * tempB0.x + tempA6.y * tempB1.x + tempA6.z * tempB2.x + tempA6.w * tempB3.x;
        sum6.y += tempA6.x * tempB0.y + tempA6.y * tempB1.y + tempA6.z * tempB2.y + tempA6.w * tempB3.y;
        sum6.z += tempA6.x * tempB0.z + tempA6.y * tempB1.z + tempA6.z * tempB2.z + tempA6.w * tempB3.z;
        sum6.w += tempA6.x * tempB0.w + tempA6.y * tempB1.w + tempA6.z * tempB2.w + tempA6.w * tempB3.w;

        sum7.x += tempA7.x * tempB0.x + tempA7.y * tempB1.x + tempA7.z * tempB2.x + tempA7.w * tempB3.x;
        sum7.y += tempA7.x * tempB0.y + tempA7.y * tempB1.y + tempA7.z * tempB2.y + tempA7.w * tempB3.y;
        sum7.z += tempA7.x * tempB0.z + tempA7.y * tempB1.z + tempA7.z * tempB2.z + tempA7.w * tempB3.z;
        sum7.w += tempA7.x * tempB0.w + tempA7.y * tempB1.w + tempA7.z * tempB2.w + tempA7.w * tempB3.w;

    }
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8), sum0);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 1), sum1);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 2), sum2);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 3), sum3);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 4), sum4);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 5), sum5);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 6), sum6);
    write_imagef(matrixC, (int2)(pos.x, pos.y * 8 + 7), sum7);

}


/* Optimized version over 4x8 tiled version */
#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 8
#define TILEY_SHIFT 3

float4 mat_mult_mini(float4 a, float4 b0, float4 b1, float4 b2, float4 b3, float4 c)
{
    float4 tmp = mad((float4)a.x, b0, c);
    tmp = mad((float4)a.y, b1, tmp);
    tmp = mad((float4)a.z, b2, tmp);
    tmp = mad((float4)a.w, b3, tmp);
    return tmp;
}
float4 mat_mult_pre(float4 a, float4 b0, float4 b1, float4 b2, float4 b3)
{
    float4 tmp = (float4)a.x * b0;
    tmp = mad((float4)a.y, b1, tmp);
    tmp = mad((float4)a.z, b2, tmp);
    tmp = mad((float4)a.w, b3, tmp);
    return tmp;
}

__kernel void mmmKernel3(__read_only image2d_t matrixA,
            __read_only image2d_t matrixB,
            __write_only image2d_t matrixC,
            uint widthA, 
            uint widthB)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 sum0;
    float4 sum1;
    float4 sum2;
    float4 sum3;
    float4 sum4;
    float4 sum5;
    float4 sum6;
    float4 sum7;

    widthB = widthB >> 2;

    int8 offsety = (int8)(0, 1, 2, 3, 4, 5, 6, 7);
    int4 offsetx = (int4)(0, 1, 2, 3);
    int xpos = pos.x;
    int ypos = pos.y;
    int8 ybs = (int8)(ypos << TILEY_SHIFT) + offsety;
    int j = 0;
    int ib4 = 0;
    int4 ioff = offsetx;
    // by pulling the first iteration out of the loop, we don't need to
    // worry about zero'ing out our accumulation variables, saving 8 cycles.
    float4 tempA0 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s0));
    float4 tempA1 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s1));
    float4 tempA2 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s2));
    float4 tempA3 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s3));
    float4 tempA4 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s4));
    float4 tempA5 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s5));
    float4 tempA6 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s6));
    float4 tempA7 = read_imagef(matrixA, imageSampler, (int2)(0, ybs.s7));
    float4 tempB0 = read_imagef(matrixB, imageSampler, (int2)(pos.x, 0));
    float4 tempB1 = read_imagef(matrixB, imageSampler, (int2)(pos.x, 1));
    float4 tempB2 = read_imagef(matrixB, imageSampler, (int2)(pos.x, 2));
    float4 tempB3 = read_imagef(matrixB, imageSampler, (int2)(pos.x, 3));
    sum0 = mat_mult_pre(tempA0, tempB0, tempB1, tempB2, tempB3);
    sum1 = mat_mult_pre(tempA1, tempB0, tempB1, tempB2, tempB3);
    sum2 = mat_mult_pre(tempA2, tempB0, tempB1, tempB2, tempB3);
    sum3 = mat_mult_pre(tempA3, tempB0, tempB1, tempB2, tempB3);
    sum4 = mat_mult_pre(tempA4, tempB0, tempB1, tempB2, tempB3);
    sum5 = mat_mult_pre(tempA5, tempB0, tempB1, tempB2, tempB3);
    sum6 = mat_mult_pre(tempA6, tempB0, tempB1, tempB2, tempB3);
    sum7 = mat_mult_pre(tempA7, tempB0, tempB1, tempB2, tempB3);
    for(int i = 4; i < widthA; i=i+4)
    {
        int ib4 = i >> 2;
        int4 ioff = (int4)(i) + offsetx;
        tempA0 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s0));
        tempA1 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s1));
        tempA2 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s2));
        tempA3 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s3));
        tempB0 = read_imagef(matrixB, imageSampler, (int2)(pos.x, ioff.s0));
        tempB1 = read_imagef(matrixB, imageSampler, (int2)(pos.x, ioff.s1));
        tempB2 = read_imagef(matrixB, imageSampler, (int2)(pos.x, ioff.s2));
        tempB3 = read_imagef(matrixB, imageSampler, (int2)(pos.x, ioff.s3));
        tempA4 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s4));
        tempA5 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s5));
        tempA6 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s6));
        tempA7 = read_imagef(matrixA, imageSampler, (int2)(ib4, ybs.s7));
        sum0 = mat_mult_mini(tempA0, tempB0, tempB1, tempB2, tempB3, sum0);
        sum1 = mat_mult_mini(tempA1, tempB0, tempB1, tempB2, tempB3, sum1);
        sum2 = mat_mult_mini(tempA2, tempB0, tempB1, tempB2, tempB3, sum2);
        sum3 = mat_mult_mini(tempA3, tempB0, tempB1, tempB2, tempB3, sum3);
        sum4 = mat_mult_mini(tempA4, tempB0, tempB1, tempB2, tempB3, sum4);
        sum5 = mat_mult_mini(tempA5, tempB0, tempB1, tempB2, tempB3, sum5);
        sum6 = mat_mult_mini(tempA6, tempB0, tempB1, tempB2, tempB3, sum6);
        sum7 = mat_mult_mini(tempA7, tempB0, tempB1, tempB2, tempB3, sum7);
    }
    ypos = pos.y * 8;
    int8 ypos8 = (int8)(ypos) + offsety;
    write_imagef(matrixC, (int2)(pos.x, ypos8.s0), sum0);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s1), sum1);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s2), sum2);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s3), sum3);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s4), sum4);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s5), sum5);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s6), sum6);
    write_imagef(matrixC, (int2)(pos.x, ypos8.s7), sum7);
}