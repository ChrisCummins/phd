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
__kernel void printfKernel(__global float *inputbuffer)
{
	uint globalID = get_global_id(0);
	uint groupID = get_group_id(0);
	uint localID = get_local_id(0);
	__local int data[WGSIZE];
	int idx = WGSIZE - 1;

	if(idx == globalID)
	{
		float4 f = (float4)(inputbuffer[0], inputbuffer[1], inputbuffer[2], inputbuffer[3]);
		printf("Output vector data: f4 = %2.2v4hlf\n", f); 
	}
	
	data[localID] = localID;
	barrier(CLK_LOCAL_MEM_FENCE);

	if(idx == localID)
	{
		printf("\tThis is group %d\n",groupID);
		printf("\tOutput LDS data:  %d\n",data[idx]);
	}
	printf("the global ID of this thread is : %d\n",globalID);
}
