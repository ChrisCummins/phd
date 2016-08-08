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
 
 /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Device code for parallel cyclic reduction (PCR).
 *
 * Original CUDA kernels: UC Davis, Yao Zhang & John Owens, 2009
 * 
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#define NATIVE_DIVIDE

__kernel void pcr_small_systems_kernel(__global float *a_d, __global float *b_d, __global float *c_d, __global float *d_d, __global float *x_d, 
									   __local float *shared, int system_size, int num_systems, int iterations)
{
    int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;

	__local float* a = shared;
	__local float* b = &a[system_size+1];
	__local float* c = &b[system_size+1];
	__local float* d = &c[system_size+1];
	__local float* x = &d[system_size+1];

	a[thid] = a_d[thid + blid * system_size];
	b[thid] = b_d[thid + blid * system_size];
	c[thid] = c_d[thid + blid * system_size];
	d[thid] = d_d[thid + blid * system_size];
  
	float aNew, bNew, cNew, dNew;
  
	barrier(CLK_LOCAL_MEM_FENCE);

	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++)
	{
		int i = thid;

		if(i < delta)
		{
#ifndef NATIVE_DIVIDE
			float tmp2 = c[i] / b[i+delta];
#else
			float tmp2 = native_divide(c[i], b[i+delta]);
#endif
			bNew = b[i] - a[i+delta] * tmp2;
 			dNew = d[i] - d[i+delta] * tmp2;
			aNew = 0;
			cNew = -c[i+delta] * tmp2;	
		}
		else if((system_size-i-1) < delta)
		{
#ifndef NATIVE_DIVIDE
			float tmp = a[i] / b[i-delta];
#else
			float tmp = native_divide(a[i], b[i-delta]);
#endif
			bNew = b[i] - c[i-delta] * tmp;
			dNew = d[i] - d[i-delta] * tmp;
			aNew = -a[i-delta] * tmp;
			cNew = 0;			
		}
		else		    
		{
#ifndef NATIVE_DIVIDE
			float tmp1 = a[i] / b[i-delta];
			float tmp2 = c[i] / b[i+delta];
#else
			float tmp1 = native_divide(a[i], b[i-delta]);
			float tmp2 = native_divide(c[i], b[i+delta]);
#endif
   			bNew = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
 			dNew = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
			aNew = -a[i-delta] * tmp1;
			cNew = -c[i+delta] * tmp2;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;	
    
		delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (thid < delta)
	{
		int addr1 = thid;
		int addr2 = thid + delta;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
#ifndef NATIVE_DIVIDE
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
#else
		x[addr1] = native_divide((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);
		x[addr2] = native_divide((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);
#endif
	}
    
	barrier(CLK_LOCAL_MEM_FENCE);
    
    x_d[thid + blid * system_size] = x[thid];
}

__kernel void pcr_branch_free_kernel(__global float *a_d, __global float *b_d, __global float *c_d, __global float *d_d, __global float *x_d, 
									 __local float *shared, int system_size, int num_systems, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;

	__local float* a = shared;
	__local float* b = &a[system_size+1];
	__local float* c = &b[system_size+1];
	__local float* d = &c[system_size+1];
	__local float* x = &d[system_size+1];

	a[thid] = a_d[thid + blid * system_size];
	b[thid] = b_d[thid + blid * system_size];
	c[thid] = c_d[thid + blid * system_size];
	d[thid] = d_d[thid + blid * system_size];
  
	float aNew, bNew, cNew, dNew;
  
	barrier(CLK_LOCAL_MEM_FENCE);

	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++)
	{
		int i = thid;

		int iRight = i+delta;
		iRight = iRight & (system_size-1);

		int iLeft = i-delta;
		iLeft = iLeft & (system_size-1);

#ifndef NATIVE_DIVIDE
		float tmp1 = a[i] / b[iLeft];
		float tmp2 = c[i] / b[iRight];
#else
		float tmp1 = native_divide(a[i], b[iLeft]);
		float tmp2 = native_divide(c[i], b[iRight]);
#endif

		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew = -a[iLeft] * tmp1;
		cNew = -c[iRight] * tmp2;

		barrier(CLK_LOCAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;	
    
	    delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (thid < delta)
	{
		int addr1 = thid;
		int addr2 = thid + delta;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
#ifndef NATIVE_DIVIDE
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
#else
		x[addr1] = native_divide((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);
		x[addr2] = native_divide((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);
#endif
	}
    
	barrier(CLK_LOCAL_MEM_FENCE);
    
    x_d[thid + blid * system_size] = x[thid];
}