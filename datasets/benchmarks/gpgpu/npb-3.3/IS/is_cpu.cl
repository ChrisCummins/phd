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


__kernel void rank1(__global INT_TYPE *key_array,
                    __global INT_TYPE *bucket_size)
{
  int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;

  INT_TYPE i, k1, k2, mq;

  int myid = get_global_id(0);
  int num_procs = get_global_size(0);

  /*  Bucket sort is known to improve cache performance on some   */
  /*  cache based systems.  But the actual performance may depend */
  /*  on cache size, problem size. */
  __global INT_TYPE *work_buff = &bucket_size[myid*NUM_BUCKETS];

  /*  Initialize */
  for (i = 0; i < NUM_BUCKETS; i++)
    work_buff[i] = 0;

  /*  Determine the number of keys in each bucket */
  mq = (NUM_KEYS + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if (k2 > NUM_KEYS) k2 = NUM_KEYS;
  for (i = k1; i < k2; i++)
    work_buff[key_array[i] >> shift]++;
}


__kernel void rank2(__global INT_TYPE *key_array,
                    __global INT_TYPE *bucket_size,
                    __global INT_TYPE *g_bucket_ptrs,
                    __global INT_TYPE *key_buff2)
{
  int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;

  INT_TYPE i, k, k1, k2, mq;

  int myid = get_global_id(0);
  int num_procs = get_global_size(0);

  __global INT_TYPE *bucket_ptrs = &g_bucket_ptrs[myid*NUM_BUCKETS];

  /*  Accumulative bucket sizes are the bucket pointers.
      These are global sizes accumulated upon to each bucket */
  bucket_ptrs[0] = 0;
  for (k = 0; k < myid; k++)
    bucket_ptrs[0] += bucket_size[k*NUM_BUCKETS + 0];

  for (i = 1; i < NUM_BUCKETS; i++) { 
    bucket_ptrs[i] = bucket_ptrs[i-1];
    for (k = 0; k < myid; k++)
      bucket_ptrs[i] += bucket_size[k*NUM_BUCKETS + i];
    for (k = myid; k < num_procs; k++)
      bucket_ptrs[i] += bucket_size[k*NUM_BUCKETS + i-1];
  }

  /*  Sort into appropriate bucket */
  mq = (NUM_KEYS + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if (k2 > NUM_KEYS) k2 = NUM_KEYS;
  for (i = k1; i < k2; i++) {
    k = key_array[i];
    key_buff2[bucket_ptrs[k >> shift]++] = k;
  }
}


__kernel void rank3(__global INT_TYPE *bucket_size,
                    __global INT_TYPE *g_bucket_ptrs,
                    __global INT_TYPE *key_buff_ptr,
                    __global const INT_TYPE *key_buff_ptr2)
{
  int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
  INT_TYPE num_bucket_keys = (1L << shift);

  INT_TYPE i, i1, i2, k, k1, k2, m, mq;

  int myid = get_global_id(0);
  int num_procs = get_global_size(0);

  __global INT_TYPE *bucket_ptrs = &g_bucket_ptrs[myid*NUM_BUCKETS];

  /*  The bucket pointers now point to the final accumulated sizes */
  if (myid < num_procs-1) {
    for (i = 0; i < NUM_BUCKETS; i++)
      for (k = myid+1; k < num_procs; k++)
        bucket_ptrs[i] += bucket_size[k*NUM_BUCKETS + i];
  }

  /*  Now, buckets are sorted.  We only need to sort keys inside
      each bucket, which can be done in parallel.  Because the distribution
      of the number of keys in the buckets is Gaussian, the use of
      a dynamic schedule should improve load balance, thus, performance */
  mq = (NUM_BUCKETS + num_procs - 1) / num_procs;
  i1 = mq * myid;
  i2 = i1 + mq;
  if (i2 > NUM_BUCKETS) i2 = NUM_BUCKETS;
  for (i = i1; i < i2; i++) {
    /*  Clear the work array section associated with each bucket    */
    k1 = i * num_bucket_keys;
    k2 = k1 + num_bucket_keys;
    for (k = k1; k < k2; k++)
      key_buff_ptr[k] = 0;

    /*  Ranking of all keys occurs in this section:                 */

    /*  In this section, the keys themselves are used as their 
        own indexes to determine how many of each there are: their
        individual population                                       */
    m = (i > 0) ? bucket_ptrs[i-1] : 0;
    for (k = m; k < bucket_ptrs[i]; k++)
      key_buff_ptr[key_buff_ptr2[k]]++;  /* Now they have individual key */
                                         /* population                   */

    /*  To obtain ranks of each key, successively add the individual key
        population, not forgetting to add m, the total of lesser keys,
        to the first key population                                      */
    key_buff_ptr[k1] += m;
    for (k = k1+1; k < k2; k++)
      key_buff_ptr[k] += key_buff_ptr[k-1];
  }
}


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


__kernel void full_verify1(__global INT_TYPE *g_bucket_ptrs,
                           __global INT_TYPE *key_buff2,
                           __global INT_TYPE *key_buff_ptr_global,
                           __global INT_TYPE *key_array)
{
  INT_TYPE i, j, j1, j2, mq;
  INT_TYPE k, k1;

  int myid = get_global_id(0);
  int num_procs = get_global_size(0);

  __global INT_TYPE *bucket_ptrs = &g_bucket_ptrs[myid*NUM_BUCKETS];

  mq = (NUM_BUCKETS + num_procs - 1) / num_procs;
  j1 = mq * myid;
  j2 = j1 + mq;
  if (j2 > NUM_BUCKETS) j2 = NUM_BUCKETS;

  for (j = j1; j < j2; j++) {
    k1 = (j > 0) ? bucket_ptrs[j-1] : 0;
    for (i = k1; i < bucket_ptrs[j]; i++) {
      k = --key_buff_ptr_global[key_buff2[i]];
      key_array[k] = key_buff2[i];
    }
  }
}


__kernel void full_verify2(__global INT_TYPE *key_array,
                           __global INT_TYPE *gj,
                           __local INT_TYPE *lj)
{
  INT_TYPE i, j, k1, k2, mq;

  int myid = get_global_id(0);
  int num_procs = get_global_size(0);

  mq = (NUM_KEYS - 1 + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if (k1 == 0) k1 = 1;
  if (k2 > NUM_KEYS) k2 = NUM_KEYS;

  j = 0;
  for (i = k1; i < k2; i++)
    if (key_array[i-1] > key_array[i])
      j++;

  int lid = get_local_id(0);
  lj[lid] = j;

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  if (lid == 0) {
    int wgid = get_group_id(0);

    gj[wgid] = lj[0];
    for (i = 1; i < get_local_size(0); i++) {
      gj[wgid] += lj[i];
    }
  }
}

