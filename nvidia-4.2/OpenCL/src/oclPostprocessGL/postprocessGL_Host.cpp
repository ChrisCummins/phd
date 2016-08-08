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

#include <math.h>
#include <oclUtils.h>

template<class T>
T clamp(T x, T a, T b)
{
    return MAX(a, MIN(b, x));
}


// convert floating point rgb color to 8-bit integer
unsigned int rgbToInt(float r, float g, float b)
{
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  return (((unsigned int)b)<<16) + (((unsigned int)g)<<8) + ((unsigned int)r);
}

// get pixel from 2D image, with clamping to border
unsigned int getPixel(unsigned int *data, int x, int y, int width, int height)
{
    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);
    return data[y*width+x];
}

/*
    2D convolution 
    - operates on 8-bit RGB data stored in 32-bit uint
    - assumes kernel radius is less than or equal to block size
    - not optimized for performance
     _____________
    |   :     :   |
    |_ _:_____:_ _|
    |   |     |   |
    |   |     |   |
    |_ _|_____|_ _|
  r |   :     :   |
    |___:_____:___|
      r    bw   r
    <----tilew---->
*/

void postprocessHost(unsigned int* g_data, unsigned int* g_odata, int imgw, int imgh, int tilew, int radius, float threshold, float highlight)
{

    for( int y=0; y<imgh; ++y ) {
        for( int x=0; x<imgw; ++x ) {        

            // perform convolution
            float rsum = .0f;
            float gsum = 0.0f;
            float bsum = 0.0f;
            float samples = 0.0f;
            

            for(int iy=0; iy<=radius+radius+1; iy++) {
                for(int ix=0; ix<=radius+radius+1; ix++) {
                    int dx = ix - radius;
                    int dy = iy - radius;
                    
                    unsigned int pixel = getPixel(g_data, x+dx, y+dy, imgw, imgh);

                    // only sum pixels within disc-shaped kernel
                    float l = dx*dx + dy*dy;
                    if (l <= radius*radius) {
                        float r = (float)(pixel&0x0ff);
                        float g = (float)((pixel>>8)&0x0ff);
                        float b = (float)((pixel>>16)&0x0ff);

#if 1
                        // brighten highlights
                        float lum = (r + g + b) / (255*3);
                        if (lum > threshold) {
                            r *= highlight;
                            g *= highlight;
                            b *= highlight;
                        }
#endif
                        rsum += r;
                        gsum += g;
                        bsum += b;
                        samples += 1.0f;
                    }
                }
            }
            
            rsum /= samples;
            gsum /= samples;
            bsum /= samples;
            
            g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
        }
    }
}

