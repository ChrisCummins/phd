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

#include <assert.h>
#include <math.h>
#include "oclDCT8x8_common.h"

////////////////////////////////////////////////////////////////////////////////
// Straightforward general-sized (i)DCT with O(N ** 2) complexity
// so that we don't forget what we're calculating :)
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979323846264338327950288f

static void naiveDCT(float *dst, float *src, uint N){
    float *buf = (float *)malloc(N * sizeof(float));

    for(uint k = 0; k < N; k++){
        buf[k] = 0;
        for(uint n = 0; n < N; n++)
            buf[k] += src[n] * cosf(PI / (float)N * ((float)n + 0.5f) * (float)k);
    }

    dst[0] = buf[0] * sqrtf(1.0f / (float)N);
    for(uint i = 1; i < N; i++)
        dst[i] = buf[i] * sqrtf(2.0f / (float)N);

    free(buf);
}

static void naiveIDCT(float *dst, float *src, uint N){
    float *buf = (float *)malloc(N * sizeof(float));

    for(uint k = 0; k < N; k++){
        buf[k] = sqrtf(0.5f) * src[0];
        for(uint n = 1; n < N; n++)
            buf[k] += src[n] * cosf(PI / (float)N * (float)n * ((float)k + 0.5f) );
    }

    for(uint i = 0; i < N; i++)
        dst[i] = buf[i] * sqrtf(2.0f / (float)N);

    free(buf);
}

////////////////////////////////////////////////////////////////////////////////
// Hardcoded unrolled fast 8-point (i)DCT
////////////////////////////////////////////////////////////////////////////////
#define C_a 1.3870398453221474618216191915664f       //a = sqrt(2) * cos(1 * pi / 16)
#define C_b 1.3065629648763765278566431734272f       //b = sqrt(2) * cos(2 * pi / 16)
#define C_c 1.1758756024193587169744671046113f       //c = sqrt(2) * cos(3 * pi / 16)
#define C_d 0.78569495838710218127789736765722f      //d = sqrt(2) * cos(5 * pi / 16)
#define C_e 0.54119610014619698439972320536639f      //e = sqrt(2) * cos(6 * pi / 16)
#define C_f 0.27589937928294301233595756366937f      //f = sqrt(2) * cos(7 * pi / 16)
#define C_norm 0.35355339059327376220042218105242f   //1 / sqrt(8)

static void DCT8(float *dst, float *src, uint ostride, uint istride){
    float X07P = src[0 * istride] + src[7 * istride];
    float X16P = src[1 * istride] + src[6 * istride];
    float X25P = src[2 * istride] + src[5 * istride];
    float X34P = src[3 * istride] + src[4 * istride];

    float X07M = src[0 * istride] - src[7 * istride];
    float X61M = src[6 * istride] - src[1 * istride];
    float X25M = src[2 * istride] - src[5 * istride];
    float X43M = src[4 * istride] - src[3 * istride];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    dst[0 * ostride] = C_norm * (X07P34PP + X16P25PP);
    dst[2 * ostride] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    dst[4 * ostride] = C_norm * (X07P34PP - X16P25PP);
    dst[6 * ostride] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    dst[1 * ostride] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    dst[3 * ostride] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    dst[5 * ostride] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    dst[7 * ostride] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

static void IDCT8(float *dst, float *src, uint ostride, uint istride){
    float Y04P   = src[0 * istride] + src[4 * istride];
    float Y2b6eP = C_b * src[2 * istride] + C_e * src[6 * istride];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * src[7 * istride] + C_a * src[1 * istride] + C_c * src[3 * istride] + C_d * src[5 * istride];
    float Y7a1fM3d5cMP = C_a * src[7 * istride] - C_f * src[1 * istride] + C_d * src[3 * istride] - C_c * src[5 * istride];

    float Y04M   = src[0*istride] - src[4*istride];
    float Y2e6bM = C_e * src[2*istride] - C_b * src[6*istride];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * src[1 * istride] - C_d * src[7 * istride] - C_f * src[3 * istride] - C_a * src[5 * istride];
    float Y1d7cP3a5fMM = C_d * src[1 * istride] + C_c * src[7 * istride] - C_a * src[3 * istride] + C_f * src[5 * istride];

    dst[0 * ostride] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    dst[7 * ostride] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    dst[4 * ostride] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    dst[3 * ostride] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    dst[1 * ostride] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    dst[5 * ostride] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    dst[2 * ostride] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    dst[6 * ostride] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

extern "C" void DCT8x8CPU(float *dst, float *src, uint stride, uint imageH, uint imageW, int dir){
    assert( (dir == DCT_FORWARD) || (dir == DCT_INVERSE) );

    for (uint i = 0; i + BLOCK_SIZE - 1 < imageH; i += BLOCK_SIZE){
        for (uint j = 0; j + BLOCK_SIZE - 1 < imageW; j += BLOCK_SIZE){
            //process rows
            for(uint k = 0; k < BLOCK_SIZE; k++)
                if(dir == DCT_FORWARD)
                    DCT8(dst + (i + k) * stride + j, src + (i + k) * stride + j, 1, 1);
                else
                    IDCT8(dst + (i + k) * stride + j, src + (i + k) * stride + j, 1, 1);

            //process columns
            for(uint k = 0; k < BLOCK_SIZE; k++)
                if(dir == DCT_FORWARD)
                    DCT8(dst + i * stride + (j + k), dst + i * stride + (j + k), stride, stride);
                else
                    IDCT8(dst + i * stride + (j + k), dst + i * stride + (j + k), stride, stride);
        }
    }
}
