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
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
void initialize()
{
  cl_kernel k_initialize1;
  cl_kernel k_initialize2;
  cl_kernel k_initialize3;
  cl_kernel k_initialize4;
  cl_kernel k_initialize5;

  size_t local_ws[3], global_ws[3], temp;
  cl_int ecode;

  int d0 = grid_points[0];
  int d1 = grid_points[1];
  int d2 = grid_points[2];

  //-----------------------------------------------------------------------
  k_initialize1 = clCreateKernel(p_initialize, "initialize1", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  
  ecode  = clSetKernelArg(k_initialize1, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_initialize1, 1, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_initialize1, 2, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_initialize1, 3, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = d2 < temp ? d2 : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_initialize1,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //---------------------------------------------------------------------
  // first store the "interpolated" values everywhere on the grid    
  //---------------------------------------------------------------------
  k_initialize2 = clCreateKernel(p_initialize, "initialize2", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  
  ecode  = clSetKernelArg(k_initialize2, 0, sizeof(cl_mem), &m_u);
  ecode  = clSetKernelArg(k_initialize2, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_initialize2, 2, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_initialize2, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_initialize2, 4, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (INITIALIZE2_DIM == 3) {
    local_ws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d1 < temp ? d1 : temp;
    temp = temp / local_ws[1];
    local_ws[2] = d2 < temp ? d2 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d0, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d1, local_ws[1]);
    global_ws[2] = clu_RoundWorkSize((size_t)d2, local_ws[2]);
  } else if (INITIALIZE2_DIM == 2) {
    local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d2 < temp ? d2 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);
  }
  
  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_initialize2,
                                 INITIALIZE2_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  //---------------------------------------------------------------------
  // now store the exact values on the boundaries        
  //---------------------------------------------------------------------
  k_initialize3 = clCreateKernel(p_initialize, "initialize3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  
  ecode  = clSetKernelArg(k_initialize3, 0, sizeof(cl_mem), &m_u);
  ecode  = clSetKernelArg(k_initialize3, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_initialize3, 2, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_initialize3, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_initialize3, 4, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = d2 < temp ? d2 : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_initialize3,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  k_initialize4 = clCreateKernel(p_initialize, "initialize4", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  
  ecode  = clSetKernelArg(k_initialize4, 0, sizeof(cl_mem), &m_u);
  ecode  = clSetKernelArg(k_initialize4, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_initialize4, 2, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_initialize4, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_initialize4, 4, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  local_ws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = d2 < temp ? d2 : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)d0, local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_initialize4,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  k_initialize5 = clCreateKernel(p_initialize, "initialize5", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  
  ecode  = clSetKernelArg(k_initialize5, 0, sizeof(cl_mem), &m_u);
  ecode  = clSetKernelArg(k_initialize5, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_initialize5, 2, sizeof(int), &d0);
  ecode |= clSetKernelArg(k_initialize5, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_initialize5, 4, sizeof(int), &d2);
  clu_CheckError(ecode, "clSetKernelArg()");

  local_ws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
  temp = max_work_group_size / local_ws[0];
  local_ws[1] = d1 < temp ? d1 : temp;

  global_ws[0] = clu_RoundWorkSize((size_t)d0, local_ws[0]);
  global_ws[1] = clu_RoundWorkSize((size_t)d1, local_ws[1]);

  CHECK_FINISH();
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_initialize5,
                                 2, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  //-----------------------------------------------------------------------

  clReleaseKernel(k_initialize1);
  clReleaseKernel(k_initialize2);
  clReleaseKernel(k_initialize3);
  clReleaseKernel(k_initialize4);
  CHECK_FINISH();
  clReleaseKernel(k_initialize5);
}

