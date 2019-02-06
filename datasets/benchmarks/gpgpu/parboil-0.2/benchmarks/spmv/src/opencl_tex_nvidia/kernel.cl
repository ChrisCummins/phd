/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#define WARP_BITS 5

__kernel void spmv_jds_texture(__global float *dst_vector, __global float *d_data,
			       __global int *d_index, __global int *d_perm,
			       __read_only image2d_t x_vec, const int dim,
			       __constant int *jds_ptr_int,
			       __constant int *sh_zcnt_int)
{
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	int imgWidth = get_image_width(x_vec);
	
	int ix = get_global_id(0);
	int warp_id= ix>>WARP_BITS;

	if(ix<dim)
	{
		float sum=0.0f;	
		int pt =d_perm[ix];
		int bound=sh_zcnt_int[warp_id];

		//prefetch 0
		int j=jds_ptr_int[0]+ix;   
		float d = d_data[j]; 
		int2 i;
		i.x=d_index[j]%imgWidth;
		i.y=d_index[j]/imgWidth;

		float4 t = read_imagef(x_vec,sampler,i);

		if (bound>1)  //bound >=2
		{
			//prefetch 1
			j=jds_ptr_int[1]+ix;    
			i.x=d_index[j]%imgWidth;
                	i.y=d_index[j]/imgWidth;

			int in;
			float dn;
			float4 tn;
	
			for(int k=2;k<bound;k++)
			{	
				//prefetch k-1
				dn = d_data[j]; 
				//prefetch k
				j=jds_ptr_int[k]+ix;     
				//prefetch k-1
				tn = read_imagef(x_vec,sampler,i);

				//compute k-2 data
				sum += d*t.x;
				//sweep to k
				i.x=d_index[j]%imgWidth;
                		i.y=d_index[j]/imgWidth;
				//sweep to k-1
				d = dn;  
				t = tn;
			}	

			//fetch last
			dn = d_data[j];   
			//fetch last 
			tn = read_imagef(x_vec,sampler,i);
			//compute last -1
			sum += d*t.x;
			//sweep to last
			d=dn;
			t=tn;
		}
		//compute last one
		sum += d*t.x;

		//write out data 
		dst_vector[pt]=sum; 
	}
}
