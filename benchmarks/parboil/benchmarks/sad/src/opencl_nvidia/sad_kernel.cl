/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* The compute kernel. */
/* The macros THREADS_W and THREADS_H specify the width and height of the
 * area to be processed by one thread, measured in 4-by-4 pixel blocks.
 * Larger numbers mean more computation per thread block.
 *
 * The macro POS_PER_THREAD specifies the number of search positions for which
 * an SAD is computed.  A larger value indicates more computation per thread,
 * and fewer threads per thread block.  It must be a multiple of 3 and also
 * must be at most 33 because the loop to copy from shared memory uses
 * 32 threads per 4-by-4 pixel block.
 *
 */
 
#define NV_OPENCL 0
 
#define TIMES_DIM_POS(x) (((x) << 5) + (x)) 
 
/* Macros to access temporary frame storage in shared memory */
#define FRAME_GET(n, x, y) \
  (frame_loc[((n) << 4) + ((y) << 2) + (x)])
#define FRAME_PUT_1(n, x, value) \
  (frame_loc[((n) << 4) + (x)] = value)

/* Macros to access temporary SAD storage in shared memory */
#define SAD_LOC_GET(blocknum, pos) \
  (sad_loc[(blocknum) * MAX_POS_PADDED + (pos)])
#define SAD_LOC_PUT(blocknum, pos, value) \
  (sad_loc[(blocknum) * MAX_POS_PADDED + (pos)] = (value))

/* When reading from this array, we use an "index" rather than a
   search position.  Also, the number of array elements is divided by
   four relative to SAD_LOC_GET() since this is an array of 8byte
   data, while SAD_LOC_GET() sees an array of 2byte data. */
#define SAD_LOC_8B_GET(blocknum, ix) \
  (sad_loc_8b[(blocknum) * (MAX_POS_PADDED/4) + (ix)])

/* The size of one row of sad_loc_8b.  This is the group of elements
 * holding SADs for all search positions for one 4x4 block. */
#define SAD_LOC_8B_ROW_SIZE (MAX_POS_PADDED/4)

/* The presence of this preprocessor variable controls which
 * of two means of computing the current search position is used. */
#define SEARCHPOS_RECURRENCE

__kernel void mb_sad_calc(__global unsigned short *blk_sad,
			    __global unsigned short *frame,
			    int mb_width,
			    int mb_height,
			    __read_only image2d_t img_ref)
{

	const sampler_t texSampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

// A local copy of the current 4x4 block 
    __local unsigned short frame_loc[THREADS_W * THREADS_H * 16];
    // The local SAD array on the device.  This is an array of short ints.  It is
    // interpreted as an array of 8-byte data for global data transfers.
    __local unsigned short sad_loc[SAD_LOC_SIZE_BYTES];
    __local uint2 *sad_loc_8b = sad_loc;

  int txy_tmp = get_local_id(0) / CEIL_POS;
  int ty = txy_tmp / THREADS_W;
  int tx = txy_tmp - mul24(ty, THREADS_W);
  int bx = get_global_id(0) / get_local_size(0);
  int by = get_global_id(1) / get_local_size(1);

  // Macroblock and sub-block coordinates 
  int mb_x = (tx + mul24(bx, THREADS_W)) >> 2;
  int mb_y = (ty + mul24(by, THREADS_H)) >> 2;
  int block_x = (tx + mul24(bx, THREADS_W)) & 0x03;
  int block_y = (ty + mul24(by, THREADS_H)) & 0x03;

  // Block-copy data into shared memory.
  // Threads are grouped into sets of 16, leaving some threads idle.
  if ((get_local_id(0) >> 4) < (THREADS_W * THREADS_H))
  {
    int ty = (get_local_id(0) >> 4) / THREADS_W;
    int tx = (get_local_id(0) >> 4) - mul24(ty, THREADS_W);
    int tgroup = get_local_id(0) & 15;

    // Width of the image in pixels
    int img_width = mb_width*16;

    // Pixel offset of the origin of the current 4x4 block 
    int frame_x = (tx + mul24(bx, THREADS_W)) << 2;
    int frame_y = (ty + mul24(by, THREADS_H)) << 2;

    // Origin in the current frame for this 4x4 block
    int cur_o = frame_y * img_width + frame_x;

    // If this is an invalid 4x4 block, do nothing
    if (((frame_x >> 4) < mb_width) && ((frame_y >> 4) < mb_height))
      {
	// Copy one pixel into 'frame'
	FRAME_PUT_1(mul24(ty, THREADS_W) + tx, tgroup,
		    frame[cur_o + (tgroup >> 2) * img_width + (tgroup & 3)]); 
      }
  }

    barrier(CLK_LOCAL_MEM_FENCE);
  //__syncthreads();

  // If this thread is assigned to an invalid 4x4 block, do nothing 
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      // Pixel offset of the origin of the current 4x4 block 
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      // Origin of the search area for this 4x4 block 
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      // Origin in the current frame for this 4x4 block 
      int cur_o = ty * THREADS_W + tx;

      int search_pos;
      int search_pos_base =
	(get_local_id(0) % CEIL_POS) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      int sotmp = search_pos_base / SEARCH_DIMENSION;
      int local_search_off_x = search_pos_base - TIMES_DIM_POS(sotmp);
      int search_off_y = ref_y + sotmp;

      // Don't go past bounds 
      if (search_pos_end > MAX_POS)
	search_pos_end = MAX_POS;

      // For each search position, within the range allocated to this thread 
      for (search_pos = search_pos_base;
	   search_pos < search_pos_end;
	   search_pos += 3) {
	// It is also beneficial to fuse (jam) the enclosed loops if this loop
	// is unrolled. 
	unsigned short sad1 = 0, sad2 = 0, sad3 = 0;
	int search_off_x = ref_x + local_search_off_x;

	// 4x4 SAD computation 
	for(int y=0; y<4; y++) {
	  int t; // signed int or unsigned short works, but not unsigned int
	  
	  t = (read_imageui(img_ref, texSampler, (int2)(search_off_x, search_off_y + y) )).x;
	  sad1 += abs(t - FRAME_GET(cur_o, 0, y));
    
      t = (read_imageui(img_ref, texSampler, (int2)(search_off_x + 1, search_off_y + y) )).x;
	  sad1 += abs(t - FRAME_GET(cur_o, 1, y));
	  sad2 += abs(t - FRAME_GET(cur_o, 0, y));

      t = (read_imageui(img_ref, texSampler, (int2)(search_off_x + 2, search_off_y + y) )).x;
	  sad1 += abs(t - FRAME_GET(cur_o, 2, y));
	  sad2 += abs(t - FRAME_GET(cur_o, 1, y));
	  sad3 += abs(t - FRAME_GET(cur_o, 0, y));

      t = (read_imageui(img_ref, texSampler, (int2)(search_off_x + 3, search_off_y + y) )).x;
	  sad1 += abs(t - FRAME_GET(cur_o, 3, y));
	  sad2 += abs(t - FRAME_GET(cur_o, 2, y));
	  sad3 += abs(t - FRAME_GET(cur_o, 1, y));

      t = (read_imageui(img_ref, texSampler, (int2)(search_off_x + 4, search_off_y + y) )).x;
	  sad2 += abs(t - FRAME_GET(cur_o, 3, y));
	  sad3 += abs(t - FRAME_GET(cur_o, 2, y));

      t = (read_imageui(img_ref, texSampler, (int2)(search_off_x + 5, search_off_y + y) )).x;
	  sad3 += abs(t - FRAME_GET(cur_o, 3, y));
	}

	// Save this value into the local SAD array 
	SAD_LOC_PUT(mul24(ty, THREADS_W) + tx, search_pos, sad1);
	SAD_LOC_PUT(mul24(ty, THREADS_W) + tx, search_pos+1, sad2);
	SAD_LOC_PUT(mul24(ty, THREADS_W) + tx, search_pos+2, sad3);

	local_search_off_x += 3;
	if (local_search_off_x >= SEARCH_DIMENSION)
	  {
	    local_search_off_x -= SEARCH_DIMENSION;
	    search_off_y++;
	  }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
//  __syncthreads();

  // Block-copy data into global memory.
  // Threads are grouped into sets of 32, leaving some threads idle. 
  if ((get_local_id(0) >> 5) < (THREADS_W * THREADS_H))
  {
    int tgroup = get_local_id(0) & 31;
    int ty = (get_local_id(0) >> 5) / THREADS_W;
    int tx = (get_local_id(0) >> 5) - mul24(ty, THREADS_W);
    int index;

    // Macroblock and sub-block coordinates 
    int mb_x = (tx + mul24(bx, THREADS_W)) >> 2;
    int mb_y = (ty + mul24(by, THREADS_H)) >> 2;
    int block_x = (tx + mul24(bx, THREADS_W)) & 0x03;
    int block_y = (ty + mul24(by, THREADS_H)) & 0x03;

    if ((mb_x < mb_width) && (mb_y < mb_height))
      {
	// All SADs from this thread are stored in a contiguous chunk
	// of memory starting at this offset 
	blk_sad += (mul24(mul24(mb_width, mb_height), 25) +
		    (mul24(mb_y, mb_width) + mb_x) * 16 +
		    (4 * block_y + block_x)) *
	  MAX_POS_PADDED;

	// Block copy, 32 threads at a time 
	for (index = tgroup; index < SAD_LOC_8B_ROW_SIZE; index += 32)
	  ((__global uint2 *)blk_sad)[index] 
	    = SAD_LOC_8B_GET(mul24(ty, THREADS_W) + tx, index);
      }
  }
  
}

//typedef unsigned int uint;

__kernel void larger_sad_calc_8(__global unsigned short *blk_sad,
				  int mb_width,
				  int mb_height)
{
  int tx = get_local_id(1) & 1;
  int ty = get_local_id(1) >> 1;

  // Macroblock and sub-block coordinates
  int mb_x = get_global_id(0) / get_local_size(0);
  int mb_y = get_global_id(1) / get_local_size(1);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height);
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  int search_pos;

  __global unsigned short *bi;
  __global unsigned short *bo_6, *bo_5, *bo_4;

  bi = blk_sad    
    + (mul24(macroblocks, 25) + (ty * 8 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 16;

  // Block type 6: 4x8
  bo_6 = blk_sad
    + ((macroblocks << 4) + macroblocks + (ty * 4 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 8;

  if (ty < 100) // always true, but improves register allocation
    {
      // Block type 5: 8x4
      bo_5 = blk_sad
	+ ((macroblocks << 3) + macroblocks + (ty * 4 + tx)) * MAX_POS_PADDED
	+ macroblock_index * 8;

      // Block type 4: 8x8
      bo_4 = blk_sad
	+ ((macroblocks << 2) + macroblocks + (ty * 2 + tx)) * MAX_POS_PADDED
	+ macroblock_index * 4;
    }

  for (search_pos = get_local_id(0); search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
      // Each uint is actually two 2-byte integers packed together.
      // Only addition is used and there is no chance of integer overflow
      // so this can be done to reduce computation time.
      
      #if NV_OPENCL
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 4*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 5*MAX_POS_PADDED/2];

      ((__global uint *)bo_6)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_6)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_5)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_4)[search_pos]                  = (i00 + i01) + (i10 + i11);
      
      #else
      // AMD OpenCL will not correctly compile casting to unsigned int
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 4*MAX_POS_PADDED/2)*2], bi[(search_pos + 4*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 5*MAX_POS_PADDED/2)*2], bi[(search_pos + 5*MAX_POS_PADDED/2)*2+1]);
      ((__global ushort2 *)bo_6)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_6)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_5)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_4)[search_pos]                  = (s00 + s01) + (s10 + s11);
      #endif
    }
    
}



__kernel void larger_sad_calc_16(__global unsigned short *blk_sad,
				   int mb_width,
				   int mb_height)
{
  // Macroblock coordinates 
  int mb_x = get_global_id(0) / get_local_size(0);
  int mb_y = get_global_id(1) / get_local_size(1);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height) * MAX_POS_PADDED;
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  int search_pos;

  __global unsigned short *bi;
  __global unsigned short *bo_3, *bo_2, *bo_1;

  //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  // Block type 3: 8x16
  //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  // Block type 5: 8x4
  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  // Block type 4: 8x8
  bo_1 = blk_sad + macroblock_index;

  for (search_pos = get_local_id(0); search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
      // Each uint is actually two 2-byte integers packed together.
      // Only addition is used and there is no chance of integer overflow
      // so this can be done to reduce computation time.
      
      #if NV_OPENCL
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];

      ((__global uint *)bo_3)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_2)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
      #else
      // AMD OpenCL will not correctly compile casting to unsigned int
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 2*MAX_POS_PADDED/2)*2], bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 3*MAX_POS_PADDED/2)*2], bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1]); 
      ((__global ushort2 *)bo_3)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_3)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_2)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_2)[search_pos+MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_1)[search_pos]                  = (s00 + s01) + (s10 + s11);
      #endif
      
    }
}


