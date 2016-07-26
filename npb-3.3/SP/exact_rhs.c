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

//---------------------------------------------------------------------
// compute the right hand side based on exact solution
//---------------------------------------------------------------------
void exact_rhs()
{
  cl_kernel k_exact_rhs1, k_exact_rhs2, k_exact_rhs3,
            k_exact_rhs4, k_exact_rhs5;
  cl_mem m_cuf, m_q, m_ue, m_buf;
  size_t local_ws[3], global_ws[3], temp;
  cl_int ecode;

  int d0 = grid_points[0];
  int d1 = grid_points[1];
  int d2 = grid_points[2];

  size_t max_work_items = PROBLEM_SIZE * PROBLEM_SIZE;
  size_t buf_size1 = sizeof(double)*PROBLEM_SIZE * max_work_items;
  size_t buf_size2 = sizeof(double)*PROBLEM_SIZE*5 * max_work_items;
  m_cuf = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_cuf");
  
  m_q = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_q");
  
  m_ue = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size2,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_ue");
  
  m_buf = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size2,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_buf");

  //-----------------------------------------------------------------------
  k_exact_rhs1 = clCreateKernel(p_exact_rhs, "exact_rhs1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for exact_rhs1");
  
  ecode  = clSetKernelArg(k_exact_rhs1, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_exact_rhs1, 1, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_exact_rhs1, 2, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_exact_rhs1, 3, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (EXACT_RHS1_DIM == 3) {
    local_ws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d1 < temp ? d1 : temp;
    temp = temp / local_ws[1];
    local_ws[2] = d2 < temp ? d2 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d0, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d1, local_ws[1]);
    global_ws[2] = clu_RoundWorkSize((size_t)d2, local_ws[2]);
  } else if (EXACT_RHS1_DIM == 2) {
    local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d2 < temp ? d2 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);
  } else {
    temp = d2 / max_compute_units;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)d2, local_ws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_exact_rhs1,
                                 EXACT_RHS1_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  k_exact_rhs2 = clCreateKernel(p_exact_rhs, "exact_rhs2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for exact_rhs2");

  ecode  = clSetKernelArg(k_exact_rhs2, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_exact_rhs2, 1, sizeof(cl_mem), &m_ue);
  ecode |= clSetKernelArg(k_exact_rhs2, 2, sizeof(cl_mem), &m_buf);
  ecode |= clSetKernelArg(k_exact_rhs2, 3, sizeof(cl_mem), &m_cuf);
  ecode |= clSetKernelArg(k_exact_rhs2, 4, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_exact_rhs2, 5, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_exact_rhs2, 6, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_exact_rhs2, 7, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_exact_rhs2, 8, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");
 
  local_ws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = (d2-2) < temp ? (d2-1) : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)(d1-2), local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)(d2-2), local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_exact_rhs2,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  k_exact_rhs3 = clCreateKernel(p_exact_rhs, "exact_rhs3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for exact_rhs3");

  ecode  = clSetKernelArg(k_exact_rhs3, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_exact_rhs3, 1, sizeof(cl_mem), &m_ue);
  ecode |= clSetKernelArg(k_exact_rhs3, 2, sizeof(cl_mem), &m_buf);
  ecode |= clSetKernelArg(k_exact_rhs3, 3, sizeof(cl_mem), &m_cuf);
  ecode |= clSetKernelArg(k_exact_rhs3, 4, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_exact_rhs3, 5, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_exact_rhs3, 6, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_exact_rhs3, 7, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_exact_rhs3, 8, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");
 
  local_ws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = (d2-2) < temp ? (d2-2) : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)(d0-2), local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)(d2-2), local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_exact_rhs3,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  k_exact_rhs4 = clCreateKernel(p_exact_rhs, "exact_rhs4", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for exact_rhs4");

  ecode  = clSetKernelArg(k_exact_rhs4, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_exact_rhs4, 1, sizeof(cl_mem), &m_ue);
  ecode |= clSetKernelArg(k_exact_rhs4, 2, sizeof(cl_mem), &m_buf);
  ecode |= clSetKernelArg(k_exact_rhs4, 3, sizeof(cl_mem), &m_cuf);
  ecode |= clSetKernelArg(k_exact_rhs4, 4, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_exact_rhs4, 5, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_exact_rhs4, 6, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_exact_rhs4, 7, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_exact_rhs4, 8, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");
 
  local_ws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = (d1-2) < temp ? (d1-2) : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)(d0-2), local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)(d1-2), local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_exact_rhs4,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  //-----------------------------------------------------------------------
  k_exact_rhs5 = clCreateKernel(p_exact_rhs, "exact_rhs5", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for exact_rhs5");
  
  ecode  = clSetKernelArg(k_exact_rhs5, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_exact_rhs5, 1, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_exact_rhs5, 2, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_exact_rhs5, 3, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (EXACT_RHS5_DIM == 3) {
    local_ws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / local_ws[1];
    local_ws[2] = (d2-2) < temp ? (d2-2) : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)(d0-2), local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)(d1-2), local_ws[1]);
    global_ws[2] = clu_RoundWorkSize((size_t)(d2-2), local_ws[2]);
  } else if (EXACT_RHS5_DIM == 2) {
    local_ws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = (d2-2) < temp ? (d2-2) : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)(d1-2), local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)(d2-2), local_ws[1]);
  } else {
    temp = (d2-2) / max_compute_units;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)(d2-2), local_ws[0]);
  }

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_exact_rhs5,
                                 EXACT_RHS5_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  clReleaseMemObject(m_cuf);
  clReleaseMemObject(m_q);
  clReleaseMemObject(m_ue);
  clReleaseMemObject(m_buf);

  clReleaseKernel(k_exact_rhs1);
  clReleaseKernel(k_exact_rhs2);
  clReleaseKernel(k_exact_rhs3);
  clReleaseKernel(k_exact_rhs4);
  CHECK_FINISH();
  clReleaseKernel(k_exact_rhs5);
}

