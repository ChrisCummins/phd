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


#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void read_kernel( __read_only image2d_t in,
                           __global    uint     *out,
                                       uint      np,
                                       uint      val,
                                       uint      nk )
{
   if( nk == 0 ) return;
   
   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                       CLK_ADDRESS_CLAMP_TO_EDGE |
                       CLK_FILTER_NEAREST;

           uint  pcount = 0;
   __local uint  lcount; 
           uint  i, idx;
           uint4 pix;
           int2  coord = ( int2 ) ( get_global_id(0), 0 );

   if( get_local_id(0) == 0 && get_local_id(1) == 0 )
      lcount=0;

   barrier( CLK_LOCAL_MEM_FENCE );

   for(int n=0; n<nk; n++)
      for( i=0, idx=get_global_id(1); i<np; i++, idx+=get_global_size(1) )
      {
         coord.y = idx;
         pix = read_imageui( in, sampler, coord );

         if( pix.x == val ) pcount++;
         if( pix.y == val ) pcount++;
         if( pix.z == val ) pcount++;
         if( pix.w == val ) pcount++;
      }

   (void) atomic_add( &lcount, pcount );

   barrier( CLK_LOCAL_MEM_FENCE );
   
   uint gid1D = get_group_id(1) * get_num_groups(0) + get_group_id(0);

   if( get_local_id(0) == 0 && get_local_id(1) == 0 )
      out[ gid1D ] = lcount / nk;
}

__kernel void write_kernel ( __global     uint     *in,
                             __write_only image2d_t out,
                                          uint      np,
                                          uint      val,
                                          uint      nk )
{
   if( nk == 0 ) return;
   
   uint4 pval;
   uint  i, idx;
   int2  coord = ( int2 ) ( get_global_id(0), 0 );

   pval = (uint4) ( val, val, val, val );

   for(int n=0; n<nk; n++)
      for( i=0, idx=get_global_id(1); i<np; i++, idx+=get_global_size(1) )
      {
         coord.y = idx;
         write_imageui( out, coord, pval );
      }
}
