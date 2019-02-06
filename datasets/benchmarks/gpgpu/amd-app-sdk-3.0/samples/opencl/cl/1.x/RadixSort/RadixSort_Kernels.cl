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
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable 
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define RADIX 8
#define RADICES (1 << RADIX)

/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   unsortedData    array of unsorted elements
 * @param   buckets         histogram buckets    
 * @param   shiftCount      shift count
 * @param   sharedArray     shared array for thread-histogram bins
  */
__kernel
void histogram(__global const uint* unsortedData,
               __global uint* buckets,
               uint shiftCount,
               __local uint* sharedArray)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);
    
    uint numGroups = get_global_size(0) / get_local_size(0);
   
    /* Initialize shared array to zero */
    
        sharedArray[localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Calculate thread-histograms */
      uint value = unsortedData[globalId];
        value = (value >> shiftCount) & 0xFFU;
        atomic_inc(sharedArray+value);
    
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Copy calculated histogram bin to global memory */
    
        uint bucketPos = groupId  * groupSize + localId ;
        //uint bucketPos = localId * numGroups + groupId ;
        buckets[bucketPos] = sharedArray[localId];
    
}

/**
 * @brief   Permutes the element to appropriate places based on
 *          prescaned buckets values
 * @param   unsortedData        array of unsorted elments
 * @param   scanedBuckets       prescaned buckets for permuations
 * @param   shiftCount          shift count
 * @param   sharedBuckets       shared array for scaned buckets
 * @param   sortedData          array for sorted elements
 */


__kernel
void permute(__global const uint* unsortedData,
             __global const uint* scanedBuckets,
             uint shiftCount,
             __local ushort* sharedBuckets,
             __global uint* sortedData)
{

    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupSize = get_local_size(0);
    
    
    /* Copy prescaned thread histograms to corresponding thread shared block */
    for(int i = 0; i < RADICES; ++i)
    {
        uint bucketPos = groupId * RADICES * groupSize + localId * RADICES + i;
        sharedBuckets[localId * RADICES + i] = scanedBuckets[bucketPos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Premute elements to appropriate location */
    for(int i = 0; i < RADICES; ++i)
    {
        uint value = unsortedData[globalId * RADICES + i];
        value = (value >> shiftCount) & 0xFFU;
        uint index = sharedBuckets[localId * RADICES + value];
        sortedData[index] = unsortedData[globalId * RADICES + i];
        sharedBuckets[localId * RADICES + value] = index + 1;
	barrier(CLK_LOCAL_MEM_FENCE);

    }
}


__kernel void ScanArraysdim2(__global uint *output,
                         __global uint *input,
                         __local uint* block,
                         const uint block_size,
                         const uint stride,
                         __global uint* sumBuffer)
{

      int tidx = get_local_id(0);
      int tidy = get_local_id(1);
	  int gidx = get_global_id(0);
	  int gidy = get_global_id(1);
	  int bidx = get_group_id(0);
	  int bidy = get_group_id(1);
	  
	  int lIndex = tidy * block_size + tidx;
	  int gpos = (gidx << RADIX) + gidy;
	  int groupIndex = bidy * (get_global_size(0)/block_size) + bidx;
	 
	  
	  /* Cache the computational window in shared memory */
	  block[tidx] = input[gpos];
	  barrier(CLK_LOCAL_MEM_FENCE);
	  
	  uint cache = block[0];

    /* build the sum in place up the tree */
	for(int dis = 1; dis < block_size; dis *=2)
	{
	
		
		if(tidx>=dis)
		{
			cache = block[tidx-dis]+block[tidx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		block[tidx] = cache;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

		
    /* store the value in sum buffer before making it to 0 */ 	
	sumBuffer[groupIndex] = block[block_size-1];


    /*write the results back to global memory */

	if(tidx == 0)
	{	
		
		output[gpos] = 0;
	}
	else
	{
		
		output[gpos] = block[tidx-1];
	}
		
	
}   

 __kernel void ScanArraysdim1(__global uint *output,
                         __global uint *input,
                         __local uint* block,
                         const uint block_size
                         ) 
  {
   int tid = get_local_id(0);
 	int gid = get_global_id(0);
	int bid = get_group_id(0);
	
         /* Cache the computational window in shared memory */
	block[tid]     = input[gid];

	uint cache = block[0];

    /* build the sum in place up the tree */
	for(int stride = 1; stride < block_size; stride *=2)
	{
		
		if(tid>=stride)
		{
			cache = block[tid-stride]+block[tid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		block[tid] = cache;
		barrier(CLK_LOCAL_MEM_FENCE);
		
	}
		
	

    /*write the results back to global memory */

	if(tid == 0)
	{	
		output[gid]     = 0;
	}
	else
	{
		output[gid]     = block[tid-1];
	}
  } 
  
  
  __kernel void prefixSum(__global uint* output,__global uint* input,__global uint* summary,int stride) 
  {
     int gidx = get_global_id(0);
     int gidy = get_global_id(1);
     int Index = gidy * stride +gidx;
     output[Index] = 0;
     
     if(gidx > 0)
       
        {
            for(int i =0;i<gidx;i++)
               output[Index] += input[gidy * stride +i];
        }
        
        if(gidx == stride -1)
          summary[gidy] = output[Index] + input[gidy * stride + (stride -1)];
  } 
  
  
  __kernel void blockAddition  (__global uint* input,__global uint* output,uint stride)
  {
    
	  int gidx = get_global_id(0);
	  int gidy = get_global_id(1);
	  int bidx = get_group_id(0);
	  int bidy = get_group_id(1);
	
	  
	  int gpos = gidy + (gidx << RADIX);
	 
	  int groupIndex = bidy * stride + bidx;
	  
	  uint temp;
	  temp = input[groupIndex];
	  
	  output[gpos] += temp;
  }
   
   
   __kernel void FixOffset(__global uint* input,__global uint* output)
   {
    
	  int gidx = get_global_id(0);
	  int gidy = get_global_id(1);
	  int gpos = gidy + (gidx << RADIX );
	  
	  
	  
	  
	  output[gpos] += input[gidy];
	  
   }           