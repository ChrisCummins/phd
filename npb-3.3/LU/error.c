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

#include <stdio.h>
#include <math.h>
#include "applu.incl"

//---------------------------------------------------------------------
// compute the solution error
//---------------------------------------------------------------------
void error()
{
  DTIMER_START(t_error);

  int i, m;

  cl_kernel k_error;
  cl_mem m_errnm;
  double (*g_errnm)[5];
  size_t local_ws, global_ws, temp, wg_num, buf_size;
  cl_int ecode;

  for (m = 0; m < 5; m++) {
    errnm[m] = 0.0;
  }

  temp = (nz-2) / max_compute_units;
  local_ws  = 1; //temp == 0 ? 1 : temp;
  global_ws = clu_RoundWorkSize((size_t)(nz-2), local_ws);
  wg_num = global_ws / local_ws;

  buf_size = sizeof(double) * 5 * wg_num;
  m_errnm = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           buf_size, 
                           NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  k_error = clCreateKernel(p_post, "error", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ecode  = clSetKernelArg(k_error, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_error, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_error, 2, sizeof(cl_mem), &m_errnm);
  ecode |= clSetKernelArg(k_error, 3, sizeof(double)*5*local_ws, NULL);
  ecode |= clSetKernelArg(k_error, 4, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_error, 5, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_error, 6, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_error,
                                 1, NULL,
                                 &global_ws,
                                 &local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  g_errnm = (double (*)[5])malloc(buf_size);

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_errnm,
                              CL_TRUE,
                              0, buf_size,
                              g_errnm,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction
  for (i = 0; i < wg_num; i++) {
    for (m = 0; m < 5; m++) {
      errnm[m] += g_errnm[i][m];
    }
  }

  for (m = 0; m < 5; m++) {
    errnm[m] = sqrt ( errnm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }

  free(g_errnm);
  clReleaseMemObject(m_errnm);
  clReleaseKernel(k_error);

  DTIMER_STOP(t_error);
}

