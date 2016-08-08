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

#define NBINS        256
#define BITS_PER_PIX 8


#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

#define NBANKS 16
#define SCALAR_NBANKS 4

__kernel
void histogramKernel_Vector(__global uint4 *Image,
                            __global uint  *Histogram,
                            uint  n4VectorsPerThread)
{
    __local uint subhists[NBANKS * NBINS];
    
    uint tid     = get_global_id(0);
    uint ltid    = get_local_id(0);
    uint Stride  = get_global_size(0);
    uint groupSize = get_local_size(0);

    uint i;
    uint4 temp, temp2;
    
    const uint shft = (uint) BITS_PER_PIX;
    const uint msk =  (uint) (NBINS-1);
    uint offset = (uint) ltid % (uint) (NBANKS);
    uint lmem_items = (NBANKS * NBINS);

    // now, clear LDS
    __local uint4 *p = (__local uint4 *) subhists;
    for(i=ltid; i<(lmem_items/4); i+=groupSize)
    {
        p[i] = 0;
    }

    barrier( CLK_LOCAL_MEM_FENCE );
    
    // read & scatter phase
    for( i=0; i<n4VectorsPerThread; i++)
    {
        temp = Image[tid + i * Stride];
        temp2 = (temp & msk) * (uint4) NBANKS + offset;

        atom_inc( subhists + temp2.x );
        atom_inc( subhists + temp2.y );
        atom_inc( subhists + temp2.z );
        atom_inc( subhists + temp2.w );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint4) NBANKS + offset;

        atom_inc( subhists + temp2.x );
        atom_inc( subhists + temp2.y );
        atom_inc( subhists + temp2.z );
        atom_inc( subhists + temp2.w );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint4) NBANKS + offset;
       
        atom_inc( subhists + temp2.x );
        atom_inc( subhists + temp2.y );
        atom_inc( subhists + temp2.z );
        atom_inc( subhists + temp2.w );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint4) NBANKS + offset;
       
        atom_inc( subhists + temp2.x );
        atom_inc( subhists + temp2.y );
        atom_inc( subhists + temp2.z );
        atom_inc( subhists + temp2.w );
    }

    barrier( CLK_LOCAL_MEM_FENCE );

    // reduce __local banks to single histogram per work-group
    for( i=ltid; i<NBINS; i+=groupSize)
    {
		uint bin = 0, off = offset;
		//off helps in generating LDS fetch request from different depths of LDS Banks
    	for(int j=0; j<NBANKS; j++, off++)
		{
			bin += subhists[i * NBANKS + (off % NBANKS)];
		}
	
		Histogram[ (get_group_id(0) * NBINS) + i] = bin;
	}
}

__kernel
void histogramKernel_Scalar(__global uint *Image,
                            __global uint *Histogram,
                            uint  nVectorsPerThread)
{
    __local uint subhists[SCALAR_NBANKS * NBINS];
    
    uint tid     = get_global_id(0);
    uint ltid    = get_local_id(0);
    uint Stride  = get_global_size(0);
    uint groupSize = get_local_size(0);

    uint i, j, idx;
    uint temp, temp2;
    
    const uint shft = (uint) BITS_PER_PIX;
    const uint msk =  (uint) (NBINS-1);
    uint offset = (uint) ltid % (uint) (SCALAR_NBANKS);
    uint lmem_items = SCALAR_NBANKS * NBINS;

    // now, clear LDS
    __local uint *p = (__local uint *) subhists;
    for(i=ltid; i<lmem_items; i+=groupSize)
    {
        p[i] = 0;
    }

    barrier( CLK_LOCAL_MEM_FENCE );
    
    // read & scatter phase
    for( i=0; i<nVectorsPerThread; i++)
    {
        temp = Image[tid + i * Stride];
        temp2 = (temp & msk) * (uint) SCALAR_NBANKS + offset;
        atom_inc( subhists + temp2 );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint) SCALAR_NBANKS + offset;
        atom_inc( subhists + temp2 );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint) SCALAR_NBANKS + offset;
        atom_inc( subhists + temp2 );

        temp = temp >> shft;
        temp2 = (temp & msk) * (uint) SCALAR_NBANKS + offset;
        atom_inc( subhists + temp2 );
    }

    barrier( CLK_LOCAL_MEM_FENCE );

    // reduce __local banks to single histogram per work-group
    for( i= ltid; i<NBINS; i+=groupSize)
    {
        uint bin = 0, off = offset;
		//off helps in generating LDS fetch request from different depths of LDS Banks
		for(int j=0; j<SCALAR_NBANKS; j++, off++)
		{
			bin += subhists[i * SCALAR_NBANKS + (off % SCALAR_NBANKS)];
		}
		Histogram[ (get_group_id(0) * NBINS) + i] = bin;
    }
}

__kernel void reduceKernel( __global uint *Histogram, uint nSubHists )
{
     uint tid = get_global_id(0);
    uint bin = 0;
      
    // Reduce work-group histograms into single histogram,
    // one thread for each bin.
 

    for( int i=0; i < nSubHists; i = i + 4)
        bin += Histogram[ i * NBINS + tid ] + Histogram[ i * NBINS + tid + NBINS ] + 
        Histogram[ i * NBINS + tid + 2 * NBINS ] + Histogram[ i * NBINS + tid + 3 * NBINS];

    Histogram[ tid ] = bin;
}
