/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include "model.h"

#define WARP_SIZE 32
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)
#define HISTS_PER_WARP 16
#define NUM_HISTOGRAMS  (NUM_WARPS*HISTS_PER_WARP)
#define THREADS_PER_HIST (WARP_SIZE/HISTS_PER_WARP)

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel 
void gen_hists(__global hist_t* histograms, __global float* all_x_data,
	       __constant float* dev_binb, int NUM_SETS, int NUM_ELEMENTS)
{
  __global float* all_y_data = all_x_data + NUM_ELEMENTS*(NUM_SETS+1); 
  __global float* all_z_data = all_y_data + NUM_ELEMENTS*(NUM_SETS+1);

  unsigned int bx = get_group_id(0);
  unsigned int tid = get_local_id(0);

  bool do_self = (bx < (NUM_SETS + 1));

  __global float* data_x;
  __global float* data_y;
  __global float* data_z;
  __global float* random_x;
  __global float* random_y;
  __global float* random_z;
  
  __local struct cartesian data_s[BLOCK_SIZE];
  
  __local unsigned int
    warp_hists[NUM_BINS][NUM_HISTOGRAMS]; // 640B <1k  
    
  for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
    {
      if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
	{
	  warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
	}
    }
    
  // Get stuff into shared memory to kick off the loop.
  if( !do_self)
    {
      data_x = all_x_data;
      data_y = all_y_data;
      data_z = all_z_data;
      random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
      random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
      random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
    }
  else
    {
      random_x = all_x_data + NUM_ELEMENTS * (bx);
      random_y = all_y_data + NUM_ELEMENTS * (bx);
      random_z = all_z_data + NUM_ELEMENTS * (bx);
      
      data_x = random_x;
      data_y = random_y;
      data_z = random_z;
    }
    
  // Iterate over all data points
  for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
    {
      // load current set of data into shared memory
      // (total of BLOCK_SIZE points loaded)
      if( tid + i < NUM_ELEMENTS )
	{ // reading outside of bounds is a-okay
	  data_s[tid] = (struct cartesian)
            {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
	}
      
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      // Iterate over all random points
      for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; 
	  j += BLOCK_SIZE)
	{
	  // load current random point values
	  float random_x_s;
	  float random_y_s;
	  float random_z_s;
	  
	  if(tid + j < NUM_ELEMENTS)
	    {
	      random_x_s = random_x[tid + j];
	      random_y_s = random_y[tid + j];
	      random_z_s = random_z[tid + j];
	    }

	  // Iterate for all elements of current set of data points 
	  // (BLOCK_SIZE iterations per thread)
	  // Each thread calcs against 1 random point within cur set of random
	  // (so BLOCK_SIZE threads covers all random points within cur set)
	  for(unsigned int k = 0; 
	      (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
	      k += 1)
	    {
	      // do actual calculations on the values:
	      float distance = data_s[k].x * random_x_s +
		data_s[k].y * random_y_s +
		data_s[k].z * random_z_s;

	      unsigned int bin_index;

	      // run binary search to find bin_index
	      unsigned int min = 0;
	      unsigned int max = NUM_BINS;
	      {
		unsigned int k2;
	      
		while (max > min+1)
		  {
		    k2 = (min + max) / 2;
		    if (distance >= dev_binb[k2]) 
		      max = k2;
		    else 
		      min = k2;
		  }
		bin_index = max - 1;
	      }

	      unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
	      if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
		 (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
		{
	      	  atom_inc(&(warp_hists[bin_index][warpnum]));
		}
	    }
	}
    }
    
  // coalesce the histograms in a block
  unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
  unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
  for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
      offset >>= 1)
    {
      for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
	  bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
	{
  	  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	  if(warp_index < offset && bin_base+bin_index < NUM_BINS )
	    {
	      unsigned long sum =
		warp_hists[bin_base + bin_index][warp_index] + 
		warp_hists[bin_base + bin_index][warp_index+offset];
	      warp_hists[bin_base + bin_index][warp_index] = sum;
	    }
	}
    }
    
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
  // Put the results back in the real histogram
  // warp_hists[x][0] holds sum of all locations of bin x
  __global hist_t* hist_base = histograms + NUM_BINS * bx;
  if(tid < NUM_BINS)
    {
      hist_base[tid] = warp_hists[tid][0];
    }
}

// **===-----------------------------------------------------------===**

#endif // _PRESCAN_CU_
