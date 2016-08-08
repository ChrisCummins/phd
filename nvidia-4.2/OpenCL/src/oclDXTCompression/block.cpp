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

#include "block.h"

void BlockDXT1::decompress(Color32 * colors) const
{
    Color32 palette[4];
    
    // Does bit expansion before interpolation.
    palette[0].b = (col0.b << 3) | (col0.b >> 2);
    palette[0].g = (col0.g << 2) | (col0.g >> 4);
    palette[0].r = (col0.r << 3) | (col0.r >> 2);
    palette[0].a = 0xFF;
    
    palette[1].r = (col1.r << 3) | (col1.r >> 2);
    palette[1].g = (col1.g << 2) | (col1.g >> 4);
    palette[1].b = (col1.b << 3) | (col1.b >> 2);
    palette[1].a = 0xFF;
    
    if( col0.u > col1.u ) {
        // Four-color block: derive the other two colors.
        palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
        palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
        palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
        palette[2].a = 0xFF;
        
        palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
        palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
        palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
        palette[3].a = 0xFF;
    }
    else {
        // Three-color block: derive the other color.
        palette[2].r = (palette[0].r + palette[1].r) / 2;
        palette[2].g = (palette[0].g + palette[1].g) / 2;
        palette[2].b = (palette[0].b + palette[1].b) / 2;
        palette[2].a = 0xFF;

        palette[3].r = 0x00;
        palette[3].g = 0x00;
        palette[3].b = 0x00;
        palette[3].a = 0x00;
    }

    for (int i = 0; i < 16; i++)
    {
        colors[i] = palette[(indices >> (2*i)) & 0x3];
    }
}

int compareColors(const Color32 * b0, const Color32 * b1)
{
    int sum = 0;
    for (int i = 0; i < 16; i++)
    {
        int r = (b0[i].r - b1[i].r);
        int g = (b0[i].g - b1[i].g);
        int b = (b0[i].b - b1[i].b);
        sum += r*r + g*g + b*b;
    }
    return sum;
}

int compareBlock(const BlockDXT1 * b0, const BlockDXT1 * b1)
{
    Color32 colors0[16];
    Color32 colors1[16];

    if (memcmp(b0, b1, sizeof(BlockDXT1)) == 0)
    {
        return 0;
    }
    else
    {
        b0->decompress(colors0);
        b1->decompress(colors1);
    
        return compareColors(colors0, colors1);
    }
}

