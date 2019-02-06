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
#include "ColorScale.h"

ColorScale::ColorScale(int num)
{
    scale = new double[num][4];
    c = 0;
    n = num;
}

ColorScale::~ColorScale()
{
    delete[] scale;
}

// note: points should be added in ascending order
void ColorScale::AddPoint(double v, double r, double g, double b)
{
    if (c < n)
    {
        scale[c][0] = v;
        scale[c][1] = r;
        scale[c][2] = g;
        scale[c][3] = b;
        c++;
    }
}

// note: expects scale array to be sorted in ascending order
void ColorScale::GetColor(double v, double& r, double& g, double& b)
{
    int i = 0;
    double w;

    if (v < scale[0][0])
    {
        r = scale[0][1];
        g = scale[0][2];
        b = scale[0][3];
    }
    else if (v > scale[c-1][0])
    {
        r = scale[c-1][1];
        g = scale[c-1][2];
        b = scale[c-1][3];
    }
    else
    {
        for (i = 1; i <= c; i++)
            if (scale[i-1][0] <= v && v < scale[i][0])
                break;

        // linear interpolation
        w = (v - scale[i-1][0]) / (scale[i][0] - scale[i-1][0]);

        r = (1-w)*scale[i-1][1] + w*scale[i][1];
        g = (1-w)*scale[i-1][2] + w*scale[i][2];
        b = (1-w)*scale[i-1][3] + w*scale[i][3];
    }
}

