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
#include "applu.incl"

void pintgr()
{
  DTIMER_START(t_pintgr);

  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i;
  int ibeg, ifin, ifin1;
  int jbeg, jfin, jfin1;
  double frc1, frc2, frc3;

  cl_kernel k_pintgr1, k_pintgr2, k_pintgr3, k_pintgr_reduce;
  cl_mem m_phi1, m_phi2, m_frc;
  double *g_frc;
  size_t local_ws[2], global_ws[2], temp; 
  size_t frc_lws, frc_gws, wg_num, buf_size;
  cl_int ecode;

  // Create buffers
  m_phi1 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(double)*(ISIZ3+2)*(ISIZ2+2),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_phi1");

  m_phi2 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(double)*(ISIZ3+2)*(ISIZ2+2),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_phi2");

  temp = (nz0-2) / max_compute_units;
  frc_lws  = temp == 0 ? 1 : temp;
  frc_gws = clu_RoundWorkSize((size_t)(nz0-2), frc_lws);
  wg_num = frc_gws / frc_lws;

  buf_size = sizeof(double) * wg_num;
  m_frc = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         buf_size, 
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_frc");

  //---------------------------------------------------------------------
  // set up the sub-domains for integeration in each processor
  //---------------------------------------------------------------------
  ibeg = ii1;
  ifin = ii2;
  jbeg = ji1;
  jfin = ji2;
  ifin1 = ifin - 1;
  jfin1 = jfin - 1;

  //---------------------------------------------------------------------
  k_pintgr1 = clCreateKernel(p_post, "pintgr1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for pintgr1");
  ecode  = clSetKernelArg(k_pintgr1, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_pintgr1, 1, sizeof(cl_mem), &m_phi1);
  ecode |= clSetKernelArg(k_pintgr1, 2, sizeof(cl_mem), &m_phi2);
  ecode |= clSetKernelArg(k_pintgr1, 3, sizeof(int), &ibeg);
  ecode |= clSetKernelArg(k_pintgr1, 4, sizeof(int), &ifin);
  ecode |= clSetKernelArg(k_pintgr1, 5, sizeof(int), &jbeg);
  ecode |= clSetKernelArg(k_pintgr1, 6, sizeof(int), &jfin);
  ecode |= clSetKernelArg(k_pintgr1, 7, sizeof(int), &ki1);
  ecode |= clSetKernelArg(k_pintgr1, 8, sizeof(int), &ki2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (PINTGR1_DIM == 2) {
    local_ws[0] = (ifin-ibeg) < work_item_sizes[0] ? (ifin-ibeg) : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = (jfin-jbeg) < temp ? (jfin-jbeg) : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(ifin-ibeg), local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)(jfin-jbeg), local_ws[1]);
  } else {
    //temp = (jfin-jbeg) / max_compute_units;
    temp = 1;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(jfin-jbeg), local_ws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr1,
                                 PINTGR1_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");


  //---------------------------------------------------------------------
  // k_pintgr_reduce: frc1
  k_pintgr_reduce = clCreateKernel(p_post, "pintgr_reduce", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  ecode  = clSetKernelArg(k_pintgr_reduce, 0, sizeof(cl_mem), &m_phi1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 1, sizeof(cl_mem), &m_phi2);
  ecode |= clSetKernelArg(k_pintgr_reduce, 2, sizeof(cl_mem), &m_frc);
  ecode |= clSetKernelArg(k_pintgr_reduce, 3, sizeof(double)*frc_lws, NULL);
  ecode |= clSetKernelArg(k_pintgr_reduce, 4, sizeof(int), &ibeg);
  ecode |= clSetKernelArg(k_pintgr_reduce, 5, sizeof(int), &ifin1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 6, sizeof(int), &jbeg);
  ecode |= clSetKernelArg(k_pintgr_reduce, 7, sizeof(int), &jfin1);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr_reduce,
                                 1, NULL,
                                 &frc_gws,
                                 &frc_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  g_frc = (double (*))malloc(buf_size);

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_frc,
                              CL_TRUE,
                              0, buf_size,
                              g_frc,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction(+:frc1)
  frc1 = 0.0;
  for (i = 0; i < wg_num; i++) {
    frc1 += g_frc[i];
  }
  frc1 = dxi * deta * frc1;


  //---------------------------------------------------------------------
  k_pintgr2 = clCreateKernel(p_post, "pintgr2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for pintgr2");
  ecode  = clSetKernelArg(k_pintgr2, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_pintgr2, 1, sizeof(cl_mem), &m_phi1);
  ecode |= clSetKernelArg(k_pintgr2, 2, sizeof(cl_mem), &m_phi2);
  ecode |= clSetKernelArg(k_pintgr2, 3, sizeof(int), &ibeg);
  ecode |= clSetKernelArg(k_pintgr2, 4, sizeof(int), &ifin);
  ecode |= clSetKernelArg(k_pintgr2, 5, sizeof(int), &jbeg);
  ecode |= clSetKernelArg(k_pintgr2, 6, sizeof(int), &jfin);
  ecode |= clSetKernelArg(k_pintgr2, 7, sizeof(int), &ki1);
  ecode |= clSetKernelArg(k_pintgr2, 8, sizeof(int), &ki2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (PINTGR2_DIM == 2) {
    local_ws[0] = (ifin-ibeg) < work_item_sizes[0] ? (ifin-ibeg) : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = (ki2-ki1) < temp ? (ki2-ki1) : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(ifin-ibeg), local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)(ki2-ki1), local_ws[1]);
  } else {
    //temp = (ki2-ki1) / max_compute_units;
    temp = 1;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(ki2-ki1), local_ws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr2,
                                 PINTGR2_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // k_pintgr_reduce: frc2
  int ki2m1 = ki2-1;
  ecode  = clSetKernelArg(k_pintgr_reduce, 4, sizeof(int), &ibeg);
  ecode |= clSetKernelArg(k_pintgr_reduce, 5, sizeof(int), &ifin1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 6, sizeof(int), &ki1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 7, sizeof(int), &ki2m1);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr_reduce,
                                 1, NULL,
                                 &frc_gws,
                                 &frc_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_frc,
                              CL_TRUE,
                              0, buf_size,
                              g_frc,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction(+:frc2)
  frc2 = 0.0;
  for (i = 0; i < wg_num; i++) {
    frc2 += g_frc[i];
  }
  frc2 = dxi * dzeta * frc2;


  //---------------------------------------------------------------------
  k_pintgr3 = clCreateKernel(p_post, "pintgr3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for pintgr3");
  ecode  = clSetKernelArg(k_pintgr3, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_pintgr3, 1, sizeof(cl_mem), &m_phi1);
  ecode |= clSetKernelArg(k_pintgr3, 2, sizeof(cl_mem), &m_phi2);
  ecode |= clSetKernelArg(k_pintgr3, 3, sizeof(int), &ibeg);
  ecode |= clSetKernelArg(k_pintgr3, 4, sizeof(int), &ifin);
  ecode |= clSetKernelArg(k_pintgr3, 5, sizeof(int), &jbeg);
  ecode |= clSetKernelArg(k_pintgr3, 6, sizeof(int), &jfin);
  ecode |= clSetKernelArg(k_pintgr3, 7, sizeof(int), &ki1);
  ecode |= clSetKernelArg(k_pintgr3, 8, sizeof(int), &ki2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (PINTGR3_DIM == 2) {
    local_ws[0] = (jfin-jbeg) < work_item_sizes[0] ? (jfin-jbeg) : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = (ki2-ki1) < temp ? (ki2-ki1) : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(jfin-jbeg), local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)(ki2-ki1), local_ws[1]);
  } else {
    //temp = (ki2-ki1) / max_compute_units;
    temp = 1;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(ki2-ki1), local_ws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr3,
                                 PINTGR3_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // k_pintgr_reduce: frc3
  ecode  = clSetKernelArg(k_pintgr_reduce, 4, sizeof(int), &jbeg);
  ecode |= clSetKernelArg(k_pintgr_reduce, 5, sizeof(int), &jfin1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 6, sizeof(int), &ki1);
  ecode |= clSetKernelArg(k_pintgr_reduce, 7, sizeof(int), &ki2m1);
  clu_CheckError(ecode, "clSetKernelArg()");
  
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_pintgr_reduce,
                                 1, NULL,
                                 &frc_gws,
                                 &frc_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_frc,
                              CL_TRUE,
                              0, buf_size,
                              g_frc,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction(+:frc3)
  frc3 = 0.0;
  for (i = 0; i < wg_num; i++) {
    frc3 += g_frc[i];
  }
  frc3 = deta * dzeta * frc3;

  frc = 0.25 * ( frc1 + frc2 + frc3 );
  //printf("\n\n     surface integral = %12.5E\n\n\n", frc);

  // Release OpenCL objects
  free(g_frc);
  clReleaseKernel(k_pintgr1);
  clReleaseKernel(k_pintgr2);
  clReleaseKernel(k_pintgr3);
  clReleaseKernel(k_pintgr_reduce);
  clReleaseMemObject(m_phi1);
  clReleaseMemObject(m_phi2);
  clReleaseMemObject(m_frc);

  DTIMER_STOP(t_pintgr);
}

