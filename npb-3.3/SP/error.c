//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB SP code. This OpenCL    //
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

#include "header.h"
#include <stdlib.h>
#include <math.h>

//---------------------------------------------------------------------
// this function computes the norm of the difference between the
// computed solution and the exact solution
//---------------------------------------------------------------------
void error_norm(double rms[5])
{
  int i, m, d;

  cl_kernel k_error_norm;
  cl_mem m_rms;
  double (*g_rms)[5];
  size_t local_ws, global_ws, temp, wg_num, buf_size;
  cl_int ecode;

  int d0 = grid_points[0];
  int d1 = grid_points[1];
  int d2 = grid_points[2];

  for (m = 0; m < 5; m++) {
    rms[m] = 0.0;
  }

  temp = d2 / max_compute_units;
  local_ws  = temp == 0 ? 1 : temp;
  global_ws = clu_RoundWorkSize((size_t)d2, local_ws);
  wg_num = global_ws / local_ws;

  buf_size = sizeof(double) * 5 * wg_num;
  m_rms = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         buf_size, 
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  k_error_norm = clCreateKernel(p_error, "error_norm", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ecode  = clSetKernelArg(k_error_norm, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_error_norm, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_error_norm, 2, sizeof(cl_mem), &m_rms);
  ecode |= clSetKernelArg(k_error_norm, 3, sizeof(double)*5*local_ws, NULL);
  ecode |= clSetKernelArg(k_error_norm, 4, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_error_norm, 5, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_error_norm, 6, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_error_norm,
                                 1, NULL,
                                 &global_ws,
                                 &local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  g_rms = (double (*)[5])malloc(buf_size);

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_rms,
                              CL_TRUE,
                              0, buf_size,
                              g_rms,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction
  for (i = 0; i < wg_num; i++) {
    for (m = 0; m < 5; m++) {
      rms[m] += g_rms[i][m];
    }
  }
  
  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    }
    rms[m] = sqrt(rms[m]);
  }

  free(g_rms);
  clReleaseMemObject(m_rms);
  clReleaseKernel(k_error_norm);
}


void rhs_norm(double rms[5])
{
  int i, m, d;

  cl_kernel k_rhs_norm;
  cl_mem m_rms;
  double (*g_rms)[5];
  size_t local_ws, global_ws, temp, wg_num, buf_size;
  cl_int ecode;

  for (m = 0; m < 5; m++) {
    rms[m] = 0.0;
  }

  temp = nz2 / max_compute_units;
  local_ws  = temp == 0 ? 1 : temp;
  global_ws = clu_RoundWorkSize((size_t)nz2, local_ws);
  wg_num = global_ws / local_ws;

  buf_size = sizeof(double) * 5 * wg_num;
  m_rms = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         buf_size, 
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  k_rhs_norm = clCreateKernel(p_error, "rhs_norm", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ecode  = clSetKernelArg(k_rhs_norm, 0, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_rhs_norm, 1, sizeof(cl_mem), &m_rms);
  ecode |= clSetKernelArg(k_rhs_norm, 2, sizeof(double)*5*local_ws, NULL);
  ecode |= clSetKernelArg(k_rhs_norm, 3, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_rhs_norm, 4, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_rhs_norm, 5, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rhs_norm,
                                 1, NULL,
                                 &global_ws,
                                 &local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  g_rms = (double (*)[5])malloc(buf_size);

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_rms,
                              CL_TRUE,
                              0, buf_size,
                              g_rms,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction
  for (i = 0; i < wg_num; i++) {
    for (m = 0; m < 5; m++) {
      rms[m] += g_rms[i][m];
    }
  }

  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    }
    rms[m] = sqrt(rms[m]);
  }

  free(g_rms);
  clReleaseMemObject(m_rms);
  clReleaseKernel(k_rhs_norm);
}

