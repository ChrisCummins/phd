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


#define IA 16807    			// a
#define IM 2147483647 			// m
#define AM (1.0f/IM) 			// 1/m - To calculate floating point result
#define IQ 127773 
#define IR 2836
#define NTAB 16
#define NDIV (1 + (IM - 1)/ NTAB)
#define EPS 1.2e-7
#define RMAX (1.0f - EPS)
#define GROUP_SIZE 64



/* Generate uniform random deviation */
/* Park-Miller with Bays-Durham shuffle and added safeguards
   Returns a uniform random deviate between (-FACTOR/2, FACTOR/2)
   input seed should be negative */ 
float ran1(int idum, __local int *iv)
{
    int j;
    int k;
    int iy = 0;
    int tid = get_local_id(0) + get_local_id(1) * get_local_size(0);

    for(j = NTAB; j >=0; j--)			//Load the shuffle
    {
        k = idum / IQ;
        idum = IA * (idum - k * IQ) - IR * k;

        if(idum < 0)
            idum += IM;

        if(j < NTAB)
            iv[NTAB* tid + j] = idum;
    }
    iy = iv[NTAB* tid];

    k = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;

    if(idum < 0)
        idum += IM;

    j = iy / NDIV;
    iy = iv[NTAB * tid + j];
    return (AM * iy);	//AM *iy will be between 0.0 and 1.0
}



__kernel void noise_uniform(__global uchar4* inputImage, __global uchar4* outputImage, int factor)
{
	int pos = get_global_id(0) + get_global_id(1) * get_global_size(0);

	float4 temp = convert_float4(inputImage[pos]);

	/* compute average value of a pixel from its compoments */
	float avg = (temp.x + temp.y + temp.z + temp.y) / 4;

	/* Each thread has NTAB private values */
	/* Local memory is used as indexed arrays use global memory instead of registers */
	__local int iv[NTAB * GROUP_SIZE];  

	/* Calculate deviation from the avg value of a pixel */
	float dev = ran1(-avg, iv);
	dev = (dev - 0.55f) * factor;

	/* Saturate(clamp) the values */
	outputImage[pos] = convert_uchar4_sat(temp + (float4)(dev));

	
}
