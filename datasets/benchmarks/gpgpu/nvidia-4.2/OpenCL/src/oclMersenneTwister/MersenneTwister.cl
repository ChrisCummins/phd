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

typedef struct{
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
} mt_struct_stripped;

#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18
#define PI 3.14159265358979f

////////////////////////////////////////////////////////////////////////////////
// OpenCL Kernel for Mersenne Twister RNG
////////////////////////////////////////////////////////////////////////////////
__kernel void MersenneTwister(__global float* d_Rand, 
			      __global mt_struct_stripped* d_MT,
			      int nPerRng)
{
    int globalID = get_global_id(0);

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN], matrix_a, mask_b, mask_c; 

    //Load bit-vector Mersenne Twister parameters
    matrix_a = d_MT[globalID].matrix_a;
    mask_b   = d_MT[globalID].mask_b;
    mask_c   = d_MT[globalID].mask_c;
        
    //Initialize current state
    mt[0] = d_MT[globalID].seed;
    for (iState = 1; iState < MT_NN; iState++)
        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

    iState = 0;
    mti1 = mt[0];
    for (iOut = 0; iOut < nPerRng; iOut++) {
        iState1 = iState + 1;
        iStateM = iState + MT_MM;
        if(iState1 >= MT_NN) iState1 -= MT_NN;
        if(iStateM >= MT_NN) iStateM -= MT_NN;
        mti  = mti1;
        mti1 = mt[iState1];
        mtiM = mt[iStateM];

	    // MT recurrence
        x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
	    x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

        mt[iState] = x;
        iState = iState1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & mask_b;
        x ^= (x << MT_SHIFTC) & mask_c;
        x ^= (x >> MT_SHIFT1);

        //Convert to (0, 1] float and write to global memory
        d_Rand[globalID + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of nPerRng uniformly distributed
// random samples, produced by MersenneTwister(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// nPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
void BoxMullerTrans(__global float *u1, __global float *u2)
{
    float   r = native_sqrt(-2.0f * log(*u1));
    float phi = 2 * PI * (*u2);
    *u1 = r * native_cos(phi);
    *u2 = r * native_sin(phi);
}

__kernel void BoxMuller(__global float *d_Rand, int nPerRng) 
{
    int globalID = get_global_id(0);

    for (int iOut = 0; iOut < nPerRng; iOut += 2)
        BoxMullerTrans(&d_Rand[globalID + (iOut + 0) * MT_RNG_COUNT],
		       &d_Rand[globalID + (iOut + 1) * MT_RNG_COUNT]);
}
