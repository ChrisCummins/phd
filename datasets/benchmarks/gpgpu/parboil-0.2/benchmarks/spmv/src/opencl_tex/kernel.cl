/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

__kernel void spmv_jds_texture(__global float *dst_vector, __global float *d_data,
			       __global int *d_index, __global int *d_perm,
			       __read_only image2d_t x_vec, const int dim,
			       __constant int *jds_ptr_int,
			       __constant int *sh_zcnt_int)
{
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	int imgWidth = get_image_width(x_vec);

	int ix = get_global_id(0);

  	if (ix < dim) {
    		float sum = 0.0f;
    		// 32 is warp size
    		int bound=sh_zcnt_int[ix/32];

    		for(int k=0;k<bound;k++)
    		{  
      			int j = jds_ptr_int[k] + ix;    
      			float d = d_data[j];

  			int2 i;
			i.x = d_index[j]%imgWidth;
			i.y = d_index[j]/imgWidth;
			float4 t = read_imagef(x_vec,sampler,i);

			sum += d*t.x;
    		}  
  
   	 	dst_vector[d_perm[ix]] = sum; 
  	}
}
