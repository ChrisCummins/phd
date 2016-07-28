//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB IS code. This OpenCL    //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenCL version to                //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Gangwon Jo, Jungwon Kim, Jun Lee, Jeongho Nah,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

#include "is_kernel.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void create_seq(__global INT_TYPE *key_array, double seed, double a)
{
  double x, s;
  INT_TYPE i, k;

  INT_TYPE k1, k2;
  double an = a;
  int myid, num_procs;
  INT_TYPE mq;

  // initialization for randlc
  double R23, R46, T23, T46;
  R23 = 1.0;
  R46 = 1.0;
  T23 = 1.0;
  T46 = 1.0;

  for (i=1; i<=23; i++)
  {
    R23 = 0.50 * R23;
    T23 = 2.0 * T23;
  }
  for (i=1; i<=46; i++)
  {
    R46 = 0.50 * R46;
    T46 = 2.0 * T46;
  }

  // create_seq routine
  myid = get_global_id(0);
  num_procs = get_global_size(0);

  mq = (NUM_KEYS + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if ( k2 > NUM_KEYS ) k2 = NUM_KEYS;

  s = find_my_seed( myid, num_procs, (long)4*NUM_KEYS, seed, an,
                    R23, R46, T23, T46 );

  k = MAX_KEY/4;

  for (i=k1; i<k2; i++)
  {
    x = randlc(&s, &an, R23, R46, T23, T46);
    x += randlc(&s, &an, R23, R46, T23, T46);
    x += randlc(&s, &an, R23, R46, T23, T46);
    x += randlc(&s, &an, R23, R46, T23, T46);  

    key_array[i] = k*x;
  }
}


__kernel void rank0(__global INT_TYPE *key_array,
                    __global INT_TYPE *partial_verify_vals,
                    __global const INT_TYPE *test_index_array,
                    int iteration)
{
  int i;

  key_array[iteration] = iteration;
  key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;

  /*  Determine where the partial verify test keys are, load into  */
  /*  top of array bucket_size                                     */
  for (i = 0; i < TEST_ARRAY_SIZE; i++)
    partial_verify_vals[i] = key_array[test_index_array[i]];
}


/*  Clear the work array */
__kernel void rank1(__global INT_TYPE *key_buff1)
{
  int i = get_global_id(0);
  key_buff1[i] = 0;
}


/*  Ranking of all keys occurs in this section:                 */
__kernel void rank2(__global INT_TYPE *key_buff_ptr,
                    __global INT_TYPE *key_buff_ptr2)
{
  int i = get_global_id(0);

  /*  In this section, the keys themselves are used as their 
      own indexes to determine how many of each there are: their
      individual population                                       */
  INT_TYPE key = key_buff_ptr2[i];
  atom_inc(&key_buff_ptr[key]);
}


//---------------------------------------------------------------------------
// rank3_0, rank3_1, and rank3_2 implement the following loop.
//---------------------------------------------------------------------------
// int i;
// for( i=0; i<MAX_KEY-1; i++ )   
//   key_buff_ptr[i+1] += key_buff_ptr[i];  
//---------------------------------------------------------------------------

/* This kernel implements inclusive scan operation. */
__kernel void rank3_0(__global INT_TYPE *src,
                      __global INT_TYPE *dst,
                      __global INT_TYPE *sum,
                      __local INT_TYPE *ldata)
{
  INT_TYPE lid = get_local_id(0);
  INT_TYPE lsize = get_local_size(0);

  ldata[lid] = 0;
  int pos = lsize + lid;

  INT_TYPE factor = MAX_KEY / get_num_groups(0);
  INT_TYPE start = factor * get_group_id(0);
  INT_TYPE end = start + factor;

  for (INT_TYPE i = start; i < end; i += lsize) {
    ldata[pos] = src[i + lid];

    for (uint offset = 1; offset < lsize; offset <<= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);

      INT_TYPE t = ldata[pos] + ldata[pos - offset];

      barrier(CLK_LOCAL_MEM_FENCE);

      ldata[pos] = t;
    }

    INT_TYPE prv_val = (i == start) ? 0 : dst[i - 1];
    dst[i + lid] = ldata[pos] + prv_val;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) sum[get_group_id(0)] = dst[end - 1];
}

/* This kernel (exclusive scan) works for only a single work-group. */
__kernel void rank3_1(__global INT_TYPE *src,
                      __global INT_TYPE *dst,
                      __local INT_TYPE *ldata)
{
  INT_TYPE lid = get_local_id(0);
  INT_TYPE lsize = get_local_size(0);

  ldata[lid] = 0;
  int pos = lsize + lid;
  ldata[pos] = src[lid];
  
  for (uint offset = 1; offset < lsize; offset <<= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);

    INT_TYPE t = ldata[pos] + ldata[pos - offset];

    barrier(CLK_LOCAL_MEM_FENCE);

    ldata[pos] = t;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  dst[lid] = ldata[pos - 1];  //exclusive scan
}

__kernel void rank3_2(__global INT_TYPE *src,
                      __global INT_TYPE *dst,
                      __global INT_TYPE *offset)
{
  INT_TYPE lid = get_local_id(0);
  INT_TYPE lsize = get_local_size(0);

  INT_TYPE factor = MAX_KEY / get_num_groups(0);
  INT_TYPE start = factor * get_group_id(0);
  INT_TYPE end = start + factor;

  INT_TYPE sum = offset[get_group_id(0)];
  for (INT_TYPE i = start; i < end; i += lsize) {
    dst[i + lid] = src[i + lid] + sum;
  }
}


/* This is the partial verify test section */
/* Observe that test_rank_array vals are   */
/* shifted differently for different cases */
__kernel void rank4(__global INT_TYPE *partial_verify_vals,
                    __global INT_TYPE *key_buff_ptr,
                    __global const INT_TYPE *test_rank_array,
                    __global int *g_passed_verification,
                    int iteration)
{
  int i, k;
  int passed_verification = 0;

  /* This is the partial verify test section */
  /* Observe that test_rank_array vals are   */
  /* shifted differently for different cases */
  for( i=0; i<TEST_ARRAY_SIZE; i++ )
  {                                             
    k = partial_verify_vals[i];          /* test vals were put here */
    if( 0 < k  &&  k <= NUM_KEYS-1 )
    {
      INT_TYPE key_rank = key_buff_ptr[k-1];
      int failed = 0;

      switch( CLASS )
      {
        case 'S':
          {
          if( i <= 2 )
          {
            if( key_rank != test_rank_array[i]+iteration )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
          }
        case 'W':
          if( i < 2 )
          {
            if( key_rank != test_rank_array[i]+(iteration-2) )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'A':
          if( i <= 2 )
          {
            if( key_rank != test_rank_array[i]+(iteration-1) )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-(iteration-1) )
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'B':
          if( i == 1 || i == 2 || i == 4 )
          {
            if( key_rank != test_rank_array[i]+iteration )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'C':
          if( i <= 2 )
          {
            if( key_rank != test_rank_array[i]+iteration )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'D':
          if( i < 2 )
          {
            if( key_rank != test_rank_array[i]+iteration )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
      }
      if( failed == 1 ) {
#if defined(cl_amd_printf)
        printf( "Failed partial verification: "
                "iteration %d, test key %d\n", 
                iteration, (int)i );
#endif
      }
    }
  }

  *g_passed_verification += passed_verification;
}


__kernel void full_verify0(__global INT_TYPE *key_array,
                           __global INT_TYPE *key_buff2)
{
  INT_TYPE i = get_global_id(0);
  key_buff2[i] = key_array[i];
}

__kernel void full_verify1(__global INT_TYPE *key_buff2,
                           __global INT_TYPE *key_buff_ptr_global,
                           __global INT_TYPE *key_array)
{
  INT_TYPE i = get_global_id(0);
  
  INT_TYPE val = key_buff2[i];
  INT_TYPE idx = atom_dec(&key_buff_ptr_global[val]) - 1;
  key_array[idx] = val;
}

__kernel void full_verify2(__global INT_TYPE *key_array,
                           __global INT_TYPE *gj,
                           __local INT_TYPE *lj)
{
  INT_TYPE i = get_global_id(0) + 1;
  int lid = get_local_id(0);

  if (i < NUM_KEYS) {
    if (key_array[i-1] > key_array[i])
      lj[lid] = 1;
    else
      lj[lid] = 0;
  } else {
    lj[lid] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      lj[lid] += lj[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) gj[get_group_id(0)] = lj[0];
}

