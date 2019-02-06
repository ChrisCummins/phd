/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
 * ScanLargeArrays : Scan is done for each block and the sum of each
 * block is stored in separate array (sumBuffer). SumBuffer is scanned
 * and results are added to every value of next corresponding block to
 * compute the scan of a large array.(not limited to 2*MAX_GROUP_SIZE)
 * Scan uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
 * and Their Applications"
 * @param output output data 
 * @param input  input data
 * @param block  local memory used in the kernel
 * @param sumBuffer  sum of blocks
 * @param length length of the input data
 */

__kernel
void blockAddition(__global float* input, __global float* output)
{	
	int globalId = get_global_id(0);
	int groupId = get_group_id(0);
	int localId = get_local_id(0);

	__local float value[1];

	/* Only 1 thread of a group will read from global buffer */
	if(localId == 0)
	{
		value[0] = input[groupId];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	output[globalId] += value[0];
}


__kernel 
void ScanLargeArrays(__global float *output,
               		__global float *input,
              		 __local  float *block,	 // Size : block_size
					const uint block_size,	 // size of block				
					 __global float *sumBuffer)  // sum of blocks
			
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);
	int bid = get_group_id(0);
	
    /* Cache the computational window in shared memory */
	block[2*tid]     = input[2*gid];
	block[2*tid + 1] = input[2*gid + 1];
	barrier(CLK_LOCAL_MEM_FENCE);

	float cache0 = block[0];
	float cache1 = cache0 + block[1];

    /* build the sum in place up the tree */
	for(int stride = 1; stride < block_size; stride *=2)
	{
		
		if(2*tid>=stride)
		{
			cache0 = block[2*tid-stride]+block[2*tid];
			cache1 = block[2*tid+1-stride]+block[2*tid+1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		block[2*tid] = cache0;
		block[2*tid+1] = cache1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
		
    /* store the value in sum buffer before making it to 0 */ 	
	sumBuffer[bid] = block[block_size-1];

    /*write the results back to global memory */
	if(tid==0)
	{
		output[2*gid]     = 0;
		output[2*gid+1]   = block[2*tid];
	}
	else
	{
		output[2*gid]     = block[2*tid-1];
		output[2*gid + 1] = block[2*tid];
	}
	
}

__kernel 
void prefixSum(__global float *output, 
                  __global float *input,
              		 __local  float *block,	 // Size : block_size
					const uint block_size)
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);
	int bid = get_group_id(0);
	
    /* Cache the computational window in shared memory */
	block[2*tid]     = input[2*gid];
	block[2*tid + 1] = input[2*gid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);

	float cache0 = block[0];
	float cache1 = cache0 + block[1];

    /* build the sum in place up the tree */
	for(int stride = 1; stride < block_size; stride *=2)
	{
		
		if(2*tid>=stride)
		{
			cache0 = block[2*tid-stride]+block[2*tid];
			cache1 = block[2*tid+1-stride]+block[2*tid+1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		block[2*tid] = cache0;
		block[2*tid+1] = cache1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

    /*write the results back to global memory */
	if(tid==0)
	{
		output[2*gid]     = 0;
		output[2*gid+1]   = block[2*tid];
	}
	else
	{
		output[2*gid]     = block[2*tid-1];
		output[2*gid + 1] = block[2*tid];
	}
}