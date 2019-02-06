//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB EP code. This OpenCL    //
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

#include "ep_kernel.h"

#define CHUNK_SIZE  128

__kernel void embar(
    __local double *q, __local double *sx, __local double *sy,
    __global double *gq, __global double *gsx, __global double *gsy,
    int k_offset, double an)
{
  int    i, j, ii, ik, kk, l;
  double t1, t2, t3, t4, x1, x2, temp_t1;

  double x[2*CHUNK_SIZE];

  int k = get_global_id(0);
  int lid = get_local_id(0);

  int lsize = get_local_size(0);

  double my_sx = 0.0;
  double my_sy = 0.0;
  for(j = 0; j < NQ; j++) {
    q[j*lsize + lid] = 0.0;
  }

  kk = k_offset + k + 1; 
  t1 = S;
  t2 = an;

  // Find starting seed t1 for this kk.

  for (i = 1; i <= 100; i++) {
    ik = kk / 2;
    if ((2 * ik) != kk)	t3 = randlc(&t1, t2);
    if (ik == 0) break;
    t3 = randlc(&t2, t2);
    kk = ik;
  }

  //--------------------------------------------------------------------
  // Compute Gaussian deviates by acceptance-rejection method and 
  // tally counts in concentri//square annuli.  This loop is not 
  // vectorizable. 
  //--------------------------------------------------------------------

  temp_t1 = t1;

  for (ii = 0; ii < NK; ii = ii + CHUNK_SIZE) {
    // Compute uniform pseudorandom numbers.
    vranlc(2*CHUNK_SIZE, &temp_t1, A, x);

    for (i = 0; i < CHUNK_SIZE; i++) {
      x1 = 2.0 * x[2*i] - 1.0;
      x2 = 2.0 * x[2*i+1] - 1.0;
      t1 = x1 * x1 + x2 * x2;
      if (t1 <= 1.0) {
        t2 = sqrt(-2.0 * log(t1) / t1);
        t3 = (x1 * t2);
        t4 = (x2 * t2);
        l  = MAX(fabs(t3), fabs(t4));
        q[l*lsize + lid] += 1.0;
        my_sx += t3;
        my_sy += t4;
      }
    }
  }
  sx[lid] = my_sx;
  sy[lid] = my_sy;

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (j = 0; j < NQ; j++) {
    for (i = get_local_size(0) / 2; i > 0; i >>= 1) {
      if (lid < i) {
        q[j*lsize + lid] += q[j*lsize + lid + i];	        
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  for (i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      sx[lid] += sx[lid + i];
      sy[lid] += sy[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    int wgid = get_group_id(0);
    for (j = 0; j < NQ; j++) {
      gq[wgid*NQ + j] = q[j*lsize];
    }
    gsx[wgid] = sx[0];
    gsy[wgid] = sy[0];
  }
}

