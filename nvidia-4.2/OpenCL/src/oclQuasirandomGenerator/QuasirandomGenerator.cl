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

#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
__kernel void QuasirandomGenerator(__global float *d_Output,
                                   __constant unsigned int *c_Table,
				                   unsigned int seed,
				                   unsigned int N)
{
    unsigned int globalID_x   = get_global_id(0);
    unsigned int localID_y    = get_local_id(1);
    unsigned int globalSize_x = get_global_size(0);
 
    for (unsigned int pos = globalID_x; pos < N; pos += globalSize_x) {
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for(int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if(data & 1) result ^= c_Table[bit+localID_y*QRNG_RESOLUTION];

        d_Output[mul24(localID_y,N) + pos] = (float)(result + 1) * INT_SCALE;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
float MoroInvCNDgpu(unsigned int x)
{
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;

    float z;

    bool negate = false;
    
    // Ensure the conversion to floating point will give a value in the
    // range (0,0.5] by restricting the input to the bottom half of the
    // input domain. We will later reflect the result if the input was
    // originally in the top half of the input domain
    if (x >= 0x80000000UL)
    {
        x = 0xffffffffUL - x;
        negate = true;
    }

    // x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
    // Convert to floating point in (0,0.5]
    const float x1 = 1.0f / (float)0xffffffffUL;
    const float x2 = x1 / 2.0f;
    float p1 = x * x1 + x2;
    // Convert to floating point in (-0.5,0]
    float p2 = p1 - 0.5f;

    // The input to the Moro inversion is p2 which is in the range
    // (-0.5,0]. This means that our output will be the negative side
    // of the bell curve (which we will reflect if "negate" is true).

    // Main body of the bell curve for |p| < 0.42
    if (p2 > -0.42f)
    {
        z = p2 * p2;
        z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }
    // Special case (Chebychev) for tail
    else
    {
        z = log(-log(p1));
        z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
    }

    // If the original input (x) was in the top half of the range, reflect
    // to get the positive side of the bell curve
    return negate ? -z : z;
}

__kernel void InverseCND(__global float *d_Output,
	                     const unsigned int pathN,
                         const unsigned int iDevice,
                         const unsigned int nDevice)
{
    const unsigned int distance = ((unsigned int)-1) / (pathN * nDevice + 1);
    const unsigned int globalID   = get_global_id(0);
    const unsigned int globalSize = get_global_size(0);

    for(unsigned int pos = globalID; pos < pathN; pos += globalSize){
        unsigned int d = (iDevice*pathN + pos + 1) * distance;
        d_Output[pos] = MoroInvCNDgpu(d);
    }
}
