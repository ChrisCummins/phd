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
 * \File MemoryModel_Kernels.cl
 * \briefly show that:
 * - how to transfer data between host and device
 * - the characteristics of different memory regions: global, constant, local, private
 * - basic synchronization
 */

#define GROUP_SIZE 64

__constant int mask[] = 
{
	1, -1, 2, -2
};
__kernel void MemoryModel(__global int *outputbuffer,__global int *inputbuffer)
{  	
    __local int localBuffer[GROUP_SIZE];
	__private int result=0;
	__private size_t group_id=get_group_id(0);
    __private size_t item_id=get_local_id(0);
    __private size_t gid = get_global_id(0);

    // Each workitem within a work group initialize one element of the local buffer
    localBuffer[item_id]=inputbuffer[gid];
    // Synchronize the local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // add 4 elements from the local buffer
    // and store the result into a private variable
    for (int i = 0; i < 4; i++) {
      result += localBuffer[(item_id+i)%GROUP_SIZE];
    }
    // multiply the partial result with a value from the constant memory
    result *= mask[group_id%4];

    // store the result into a buffer
	outputbuffer[gid]= result;
}


