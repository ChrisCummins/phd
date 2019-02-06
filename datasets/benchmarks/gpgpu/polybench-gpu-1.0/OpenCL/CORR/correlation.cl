/**
 * correlation.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

__kernel void mean_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *data, DATA_TYPE float_n, int m, int n) 
{    
	int j = get_global_id(0) + 1;
	
	if ((j >= 1) && (j < (m+1)))
	{
		mean[j] = 0.0;

		int i;
		for(i=1; i < (n+1); i++)
		{
			mean[j] += data[i*(m+1) + j];
		}
		
		mean[j] /= (DATA_TYPE)float_n;
	}
}


__kernel void std_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *std, __global DATA_TYPE *data, DATA_TYPE float_n, DATA_TYPE eps, int m, int n) 
{
	int j = get_global_id(0) + 1;

	if ((j >= 1) && (j < (m+1)))
	{
		std[j] = 0.0;

		int i;
		for (i = 1; i < (n+1); i++)
		{
			std[j] += (data[i*(m+1) + j] - mean[j]) * (data[i*(m+1) + j] - mean[j]);
		}
		std[j] /= float_n;
		std[j] =  sqrt(std[j]);
		if(std[j] <= eps) 
		{
			std[j] = 1.0;
		}
	}
}


__kernel void reduce_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *std, __global DATA_TYPE *data, DATA_TYPE float_n, int m, int n) 
{
	int j = get_global_id(0)+1;
	int i = get_global_id(1)+1;
	
	if ((i >= 1) && (i < (n+1)) && (j >= 1) && (j < (m+1)))
	{
		data[i*(m+1) + j] -= mean[j];
		data[i*(m+1) + j] /= (sqrt(float_n) * std[j]);
	}
}


__kernel void corr_kernel(__global DATA_TYPE *symmat, __global DATA_TYPE *data, int m, int n) 
{
	int j1 = get_global_id(0) + 1;
	
	int i, j2;
	if ((j1 >= 1) && (j1 < m))
	{
		symmat[j1*(m+1) + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < (m+1); j2++)
		{
			for(i = 1; i < (n+1); i++)
			{
				symmat[j1*(m+1) + j2] += data[i*(m+1) + j1] * data[i*(m+1) + j2];
			}
			symmat[j2*(m+1) + j1] = symmat[j1*(m+1) + j2];
		}
	}
}



