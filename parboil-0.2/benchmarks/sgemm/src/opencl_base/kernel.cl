/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 */

__kernel void mysgemmNT( __global const float *A, int lda, __global const float *B, int ldb, __global float* C, int ldc, int k, float alpha, float beta )
{
    float c = 0.0f;
    int m = get_global_id(0);
    int n = get_global_id(1);

    for (int i = 0; i < k; ++i) {
	float a = A[m + i * lda]; 
	float b = B[n + i * ldb];
	c += a * b;
    }
    C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
}
