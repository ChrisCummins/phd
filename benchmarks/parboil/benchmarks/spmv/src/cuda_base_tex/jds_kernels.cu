#define WARP_BITS 5
__global__ void spmv_jds_texture(float *dst_vector,
            const float *d_data,const int *d_index, const int *d_perm,
            const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;


  if (ix < dim) {
    float sum = 0.0f;
    // 32 is warp size
    int  bound=sh_zcnt_int[ix / 32];

    for(int k=0;k<bound;k++ )
    {  
      int j = jds_ptr_int[k] + ix;    
      int in = d_index[j]; 
  
      float d = d_data[j];
      float t = tex1Dfetch(tex_x_float,in);
  
      sum += d*t; 
    }  
  
    dst_vector[d_perm[ix]] = sum; 
    // dst_vector[ix] = ix; 
  }
}


