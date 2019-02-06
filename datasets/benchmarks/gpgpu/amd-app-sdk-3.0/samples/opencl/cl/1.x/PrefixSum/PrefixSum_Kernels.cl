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
 * Work-efficient compute implementation of scan, one thread per 2 elements
 * O(log(n)) stepas and O(n) adds using shared memory
 * Uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
 * and Their Applications"
 * @param output	output data 
 * @param input		input data
 * @param block		local memory used in the kernel
 * @param length	lenght of the input data
 * @param idxOffset	offset between two consecutive index.
 */
__kernel 
void group_prefixSum(__global float * output,
					 __global float * input,
					 __local  float * block,
					 const uint length,
					 const uint idxOffset) {
	int localId = get_local_id(0);
	int localSize = get_local_size(0);
	int globalIdx = get_group_id(0);

	// Cache the computational window in shared memory
	globalIdx = (idxOffset *(2 *(globalIdx*localSize + localId) +1)) - 1;
	if(globalIdx < length)             { block[2*localId]     = input[globalIdx];				}
    if(globalIdx + idxOffset < length) { block[2*localId + 1] = input[globalIdx + idxOffset];	}

	// Build up tree 
	int offset = 1;
	for(int l = length>>1; l > 0; l >>= 1)
	{
	  barrier(CLK_LOCAL_MEM_FENCE);
	  if(localId < l) {
            int ai = offset*(2*localId + 1) - 1;
            int bi = offset*(2*localId + 2) - 1;
            block[bi] += block[ai];
         }
         offset <<= 1;
	}
		 
	if (length > 2)
	{
		if(offset < length) { offset <<= 1; }

		// Build down tree
		int maxThread = offset>>1;
		for(int d = 0; d < maxThread; d<<=1)
		{
			d += 1;
			offset >>=1;
			barrier(CLK_LOCAL_MEM_FENCE);

			if(localId < d) {
				int ai = offset*(localId + 1) - 1;
				int bi = ai + (offset>>1);
				block[bi] += block[ai];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
    // write the results back to global memory
    if(globalIdx < length)           { output[globalIdx]             = block[2*localId];		}
    if(globalIdx+idxOffset < length) { output[globalIdx + idxOffset] = block[2*localId + 1];	}
}

/*
 * Work-efficient compute implementation of scan, one thread per 2 elements
 * O(log(n)) stepas and O(n) adds using shared memory
 * Uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
 * and Their Applications"
 * @param buffer	input/output data 
 * @param offset	Multiple of Offset positions are already updated by group_prefixSum kernel
 * @param length	lenght of the input data
 */
__kernel
void global_prefixSum(__global float * buffer,
                      const uint offset,
					  const uint length) {
	int localSize = get_local_size(0);
    int groupIdx  = get_group_id(0);

	int sortedLocalBlocks = offset / localSize;		// sorted groups per block
	// Map the gids to unsorted local blocks.
	int gidToUnsortedBlocks = groupIdx + (groupIdx / ((offset<<1) - sortedLocalBlocks) +1) * sortedLocalBlocks;

	// Get the corresponding global index
    int globalIdx = (gidToUnsortedBlocks*localSize + get_local_id(0));
	if(((globalIdx+1) % offset != 0) && (globalIdx < length))
		buffer[globalIdx] += buffer[globalIdx - (globalIdx%offset + 1)];
}


