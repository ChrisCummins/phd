/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

// CML x RML = CML, baseline version, 510FLOP/s on Fermi
/* Pseudo code
for i < M ; i += 64   // thread block.x
 for j < N; j += 16   // thread block.y
  for tx = 0; tx < 16; tx++ // thread index x; tile of M loop
  for ty = 0; ty < 4 ; ty++ // thread index y; tile of M loop

  for m < 16; m += 1;
     c[m] = 0.0f

  for k < K; k += 4   // seq

   b[ty][tx] = B[k+ty][j+tx]

   for l < 4; l +=1   // seq
    for m < 16; m +=1 // seq
      c[m] += A[i+ty*16+tx][k+l]+b[l][m]

*/

__kernel void mysgemmNT(__global const float *A, int lda, __global const float *B, int ldb, __global float* C, int ldc, int k, float alpha, float beta)
{
    // Partial results 
    float c[TILE_N];
    for (int i=0; i < TILE_N; i++)
	c[i] = 0.0f;
   
    int mid = get_local_id(1)*get_local_size(0)+get_local_id(0);
    int m = get_group_id(0) * TILE_M + mid;
    int n = get_group_id(1) * TILE_N + get_local_id(0);

    __local float b_s[TILE_TB_HEIGHT][TILE_N];

    for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
	float a; 
	b_s[get_local_id(1)][get_local_id(0)] = B[n+(i+get_local_id(1))*ldb];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = 0; j < TILE_TB_HEIGHT; j++) {
	    a = A[m + (i+j)*lda];
	    for (int kk = 0; kk < TILE_N; kk++)
		c[kk] += a * b_s[j][kk];

	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    int t = ldc * get_group_id(1) * TILE_N + m;
    for (int i = 0; i < TILE_N; i++) {
	C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
    }
}

