/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "sad.h"
#include "largerBlocks.h"
#include <stdio.h>


typedef struct {
  unsigned short x;
  unsigned short y;
} __align__(4) uhvec;

typedef unsigned int uint;

__global__ void larger_sad_calc_8(unsigned short *blk_sad,
				  int mb_width,
				  int mb_height)
{
  int tx = threadIdx.y & 1;
  int ty = threadIdx.y >> 1;

  /* Macroblock and sub-block coordinates */
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  /* Number of macroblocks in a frame */
  int macroblocks = __mul24(mb_width, mb_height);
  int macroblock_index = (__mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  int search_pos;

  unsigned short *bi;
  unsigned short *bo_6, *bo_5, *bo_4;

  bi = blk_sad
    + (__mul24(macroblocks, 25) + (ty * 8 + tx * 2)) * MAX_POS_PADDED
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

  for (search_pos = threadIdx.x; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
      /* Each uint is actually two 2-byte integers packed together.
       * Only addition is used and there is no chance of integer overflow
       * so this can be done to reduce computation time. */
      uint i00 = ((uint *)bi)[search_pos];
      uint i01 = ((uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((uint *)bi)[search_pos + 4*MAX_POS_PADDED/2];
      uint i11 = ((uint *)bi)[search_pos + 5*MAX_POS_PADDED/2];

      ((uint *)bo_6)[search_pos]                  = i00 + i10;
      ((uint *)bo_6)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((uint *)bo_5)[search_pos]                  = i00 + i01;
      ((uint *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = i10 + i11;
      ((uint *)bo_4)[search_pos]                  = (i00 + i01) + (i10 + i11);
    }
}

__global__ void larger_sad_calc_16(unsigned short *blk_sad,
				   int mb_width,
				   int mb_height)
{
  /* Macroblock coordinates */
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  /* Number of macroblocks in a frame */
  int macroblocks = __mul24(mb_width, mb_height) * MAX_POS_PADDED;
  int macroblock_index = (__mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  int search_pos;

  unsigned short *bi;
  unsigned short *bo_3, *bo_2, *bo_1;

  //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  // Block type 3: 8x16
  //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  // Block type 5: 8x4
  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  // Block type 4: 8x8
  bo_1 = blk_sad + macroblock_index;

  for (search_pos = threadIdx.x; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
      /* Each uint is actually two 2-byte integers packed together.
       * Only addition is used and there is no chance of integer overflow
       * so this can be done to reduce computation time. */
      uint i00 = ((uint *)bi)[search_pos];
      uint i01 = ((uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
      uint i11 = ((uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];
      
      ((uint *)bo_3)[search_pos]                  = i00 + i10;
      ((uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((uint *)bo_2)[search_pos]                  = i00 + i01;
      ((uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
      ((uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
 /*
      ushort2 s00 = { bi[search_pos*2], bi[search_pos*2+1] };
      ushort2 s01 = { bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1] };
      ushort2 s10 = { bi[(search_pos + 2*MAX_POS_PADDED/2)*2], bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1] };
      ushort2 s11 = { bi[(search_pos + 3*MAX_POS_PADDED/2)*2], bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1] };

      ((ushort2 *)bo_3)[search_pos]                  = make_ushort2(s00.x + s10.x, s00.y + s10.y);
      ((ushort2 *)bo_3)[search_pos+MAX_POS_PADDED/2] = make_ushort2(s01.x + s11.x, s01.y + s11.y);
      ((ushort2 *)bo_2)[search_pos]                  = make_ushort2(s00.x + s01.x, s00.y + s01.y);
      ((ushort2 *)bo_2)[search_pos+MAX_POS_PADDED/2] = make_ushort2(s10.x + s11.x, s10.y + s11.y);
      ((ushort2 *)bo_1)[search_pos]                  = make_ushort2((s00.x + s01.x)+(s10.x + s11.x), (s00.y + s01.y)+(s10.y + s11.y));
      */
    }
}
