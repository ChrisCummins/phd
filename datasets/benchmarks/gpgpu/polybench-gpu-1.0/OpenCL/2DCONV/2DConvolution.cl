/**
 * 2DConvolution.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

__kernel void Convolution2D_kernel(__global DATA_TYPE *A, __global DATA_TYPE *B, int ni, int nj) 
{
	int j = get_global_id(0);
	int i = get_global_id(1);

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
	if ((i < (ni-1)) && (j < (nj - 1)) && (i > 0) && (j > 0))
	{
		B[i*nj + j] =  c11 * A[(i - 1) * nj + (j - 1)]  + c21 * A[(i - 1) * nj + (j + 0)] + c31 * A[(i - 1) * nj + (j + 1)] 
		      + c12 * A[(i + 0) * nj + (j - 1)]  + c22 * A[(i + 0) * nj + (j + 0)] + c32 * A[(i + 0) * nj + (j + 1)]
		      + c13 * A[(i + 1) * nj + (j - 1)]  + c23 * A[(i + 1) * nj + (j + 0)] + c33 * A[(i + 1) * nj + (j + 1)];
	}
}
