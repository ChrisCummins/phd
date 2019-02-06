/**
 * gramschmidt.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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


__kernel void gramschmidt_kernel1(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int tid = get_global_id(0);
	
	if (tid == 0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < m; i++)
		{
			nrm += a[i * n + k] * a[i * n + k];
		}
      		r[k * n + k] = sqrt(nrm);
	}
}


__kernel void gramschmidt_kernel2(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int i = get_global_id(0);

        if (i < m)
	{	
		q[i * n + k] = a[i * n + k] / r[k * n + k];
	}
}


__kernel void gramschmidt_kernel3(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int j = get_global_id(0);

	if ((j > k) && (j < n))
	{
		r[k*n + j] = 0.0;

		int i;
		for (i = 0; i < m; i++)
		{
			r[k*n + j] += q[i*n + k] * a[i*n + j];
		}
		
		for (i = 0; i < m; i++)
		{
			a[i*n + j] -= q[i*n + k] * r[k*n + j];
		}
	}
}

