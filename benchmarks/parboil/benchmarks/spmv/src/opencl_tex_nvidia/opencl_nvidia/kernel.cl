/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#define WARP_BITS 5

__kernel void spmv_jds(__global float *dst_vector, __global float *d_data,
		       __global int *d_index, __global int *d_perm,
		       __global float *x_vec, const int dim, 
		       __constant int *jds_ptr_int,
		       __constant int *sh_zcnt_int)
{
	int ix = get_global_id(0);
	int warp_id=ix>>WARP_BITS;

	if(ix<dim)
	{
		float sum=0.0f;
		int bound=sh_zcnt_int[warp_id];
		//prefetch 0
		int j=jds_ptr_int[0]+ix;  
		float d = d_data[j]; 
		int i = d_index[j];  
		float t = x_vec[i];
		
		if (bound>1)  //bound >=2
		{
			//prefetch 1
			j=jds_ptr_int[1]+ix;    
			i =  d_index[j];  
			int in;
			float dn;
			float tn;
			for(int k=2;k<bound;k++ )
			{	
				//prefetch k-1
				dn = d_data[j]; 
				//prefetch k
				j=jds_ptr_int[k]+ix;    
				in = d_index[j]; 
				//prefetch k-1
				tn = x_vec[i];
				
				//compute k-2
				sum += d*t; 
				//sweep to k
				i = in;  
				//sweep to k-1
				d = dn;
				t =tn; 
			}	
		
			//fetch last
			dn = d_data[j];
			tn = x_vec[i];
	
			//compute last-1
			sum += d*t; 
			//sweep to last
			d=dn;
			t=tn;
		}
		//compute last
		sum += d*t;  // 3 3
		
		//write out data
		dst_vector[d_perm[ix]]=sum; 
	}
}

