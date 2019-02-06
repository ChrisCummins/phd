/**
 * 3DConvolution.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

typedef float DATA_TYPE;

__kernel void Convolution3D_kernel(__global DATA_TYPE *A, __global DATA_TYPE *B, int ni, int nj, int nk, int i)
{
    
	int k = get_global_id(0);
	int j = get_global_id(1);
	
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;
	
    	// Do the operation
    	if ((i < (ni - 1)) && (j < (nj - 1)) &&  (k < (nk - 1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(nk * nj) + j*nk + k] = c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c21 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c23 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c31 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c33 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c12 * A[(i + 0)*(nk * nj) + (j - 1)*nk + (k + 0)]  +  c22 * A[(i + 0)*(nk * nj) + (j + 0)*nk + (k + 0)]   
					     +   c32 * A[(i + 0)*(nk * nj) + (j + 1)*nk + (k + 0)]  +  c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  
					     +   c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  +  c21 * A[(i - 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  
					     +   c23 * A[(i + 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  +  c31 * A[(i - 1)*(nk * nj) + (j + 1)*nk + (k + 1)]  
					     +   c33 * A[(i + 1)*(nk * nj) + (j + 1)*nk + (k + 1)];
	}
	else 
	{
		B[i*(nk * nj) + j*nk + k] = 0;
	}
}
