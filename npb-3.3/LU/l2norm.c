//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB LU code. This OpenCL    //
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

#include <math.h>
#include "applu.incl"

//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void l2norm (int ldx, int ldy, int ldz, int nx0, int ny0, int nz0,
     int ist, int iend, int jst, int jend,
     cl_mem *m_v, double sum[5])
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, m;
  size_t wg_num;
  double (*g_sum)[5];
  cl_int ecode;

  for (m = 0; m < 5; m++) {
    sum[m] = 0.0;
  }

  ecode  = clSetKernelArg(k_l2norm, 0, sizeof(cl_mem), m_v);
  ecode |= clSetKernelArg(k_l2norm, 3, sizeof(int), &nz0);
  ecode |= clSetKernelArg(k_l2norm, 4, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_l2norm, 5, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_l2norm, 6, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_l2norm, 7, sizeof(int), &jend);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_l2norm,
                                 1, NULL,
                                 l2norm_gws,
                                 l2norm_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  wg_num = l2norm_gws[0] / l2norm_lws[0];
  g_sum = (double (*)[5])malloc(sum_size);

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_sum,
                              CL_TRUE,
                              0, sum_size,
                              g_sum,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction
  for (i = 0; i < wg_num; i++) {
    for (m = 0; m < 5; m++) {
      sum[m] += g_sum[i][m];
    }
  }

  for (m = 0; m < 5; m++) {
    sum[m] = sqrt ( sum[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }

  free(g_sum);
}

