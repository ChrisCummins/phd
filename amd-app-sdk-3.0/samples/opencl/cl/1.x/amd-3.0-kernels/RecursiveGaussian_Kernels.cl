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
 * Transpose Kernel 
 * input image is transposed by reading the data into a block
 * and writing it to output image
 */
__kernel 
void transpose_kernel(__global uchar4 *output,
                      __global uchar4  *input,
                      __local  uchar4 *block,
                      const    uint    width,
                      const    uint    height,
                      const    uint blockSize)
{
	uint globalIdx = get_global_id(0);
	uint globalIdy = get_global_id(1);
	
	uint localIdx = get_local_id(0);
	uint localIdy = get_local_id(1);
	
    /* copy from input to local memory */
	block[localIdy * blockSize + localIdx] = input[globalIdy*width + globalIdx];

    /* wait until the whole block is filled */
	barrier(CLK_LOCAL_MEM_FENCE);

    /* calculate the corresponding raster indices of source and target */
	uint sourceIndex = localIdy * blockSize + localIdx;
	uint targetIndex = globalIdy + globalIdx * height; 
	
	output[targetIndex] = block[sourceIndex];
}




/*  Recursive Gaussian filter
 *  parameters:	
 *      input - pointer to input data 
 *      output - pointer to output data 
 *      width  - image width
 *      iheight  - image height
 *      a0-a3, b1, b2, coefp, coefn - gaussian parameters
 */
__kernel void RecursiveGaussian_kernel(__global const uchar4* input, __global uchar4* output, 
				       const int width, const int height, 
				       const float a0, const float a1, 
				       const float a2, const float a3, 
				       const float b1, const float b2, 
				       const float coefp, const float coefn)
{
    // compute x : current column ( kernel executes on 1 column )
    unsigned int x = get_global_id(0);

    if (x >= width) 
	return;

    // start forward filter pass
    float4 xp = (float4)0.0f;  // previous input
    float4 yp = (float4)0.0f;  // previous output
    float4 yb = (float4)0.0f;  // previous output by 2

    for (int y = 0; y < height; y++) 
    {
	  int pos = x + y * width;
        float4 xc = (float4)(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
        float4 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
	  output[pos] = (uchar4)(yc.x, yc.y, yc.z, yc.w);
        xp = xc; 
        yb = yp; 
        yp = yc; 

    }

     barrier(CLK_GLOBAL_MEM_FENCE);


    // start reverse filter pass: ensures response is symmetrical
    float4 xn = (float4)(0.0f);
    float4 xa = (float4)(0.0f);
    float4 yn = (float4)(0.0f);
    float4 ya = (float4)(0.0f);


    for (int y = height - 1; y > -1; y--) 
    {
        int pos = x + y * width;
        float4 xc =  (float4)(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
        float4 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
	  float4 temp = (float4)(output[pos].x, output[pos].y, output[pos].z, output[pos].w) + yc;
	  output[pos] = (uchar4)(temp.x, temp.y, temp.z, temp.w);

    }
}






	 






	

	




	

	

	
	