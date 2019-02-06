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

/**
 * Counts number of occurrences of value in input array using 
 * atomic counters
 */

#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable        

__kernel 
void atomicCounters(
		volatile __global uint *input,
		uint value,
		counter32_t counter)                          
{
	
	size_t globalId = get_global_id(0);
	
	if(value == input[globalId])
		atomic_inc(counter);
		
}                                                                         



/**
 * Counts number of occurrences of value in input array using 
 * global atomics
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel 
void globalAtomics(
		volatile __global uint *input,
		uint value,
		__global uint* counter)                         
{                                                                         
	size_t globalId = get_global_id(0);
	
	if(value == input[globalId])
		atomic_inc(&counter[0]);
}                                                                         

