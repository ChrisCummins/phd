/**
 * mvt.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

typedef double DATA_TYPE;



__kernel void mvt_kernel1(__global DATA_TYPE *a, __global DATA_TYPE *x1, __global DATA_TYPE *y1, int n) 
{    
	int i = get_global_id(0);

	if (i < n)
	{
		int j;	
		for (j=0; j < n; j++)
		{
			x1[i] += a[i * n + j] * y1[j];
		}
	}
}

__kernel void mvt_kernel2(__global DATA_TYPE *a, __global DATA_TYPE *x2, __global DATA_TYPE *y2, int n) 
{    
	int i = get_global_id(0);

	if (i < n)
	{
		int j;	
		for (j=0; j < n; j++)
		{
			x2[i] += a[j * n + i] * y2[j];	
		}
	}
}