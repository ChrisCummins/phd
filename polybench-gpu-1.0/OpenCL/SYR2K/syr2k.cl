/**
 * syr2k.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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


__kernel void syr2k_kernel(__global DATA_TYPE *a, __global DATA_TYPE *b, __global DATA_TYPE *c, DATA_TYPE alpha, DATA_TYPE beta, int m, int n) 
{    
   	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < n) && (j < n))
	{
		c[i * n + j] *= beta;
		
		int k;
		for(k = 0; k < m; k++)
		{
			c[i * n + j] += alpha * a[i * m + k] * b[j * m + k] + alpha * b[i * m + k] * a[j * m + k];
		}
	}
}


