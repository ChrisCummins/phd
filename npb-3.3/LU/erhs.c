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

#include "applu.incl"

//---------------------------------------------------------------------
// compute the right hand side based on exact solution
//---------------------------------------------------------------------
void erhs()
{
  DTIMER_START(t_erhs);

  cl_kernel k_erhs1, k_erhs2, k_erhs3, k_erhs4;
  size_t erhs1_lws[3], erhs1_gws[3];
  size_t erhs2_lws[3], erhs2_gws[3];
  size_t erhs3_lws[3], erhs3_gws[3];
  size_t erhs4_lws[3], erhs4_gws[3];
  size_t temp;
  cl_int ecode;

  //------------------------------------------------------------------------
  k_erhs1 = clCreateKernel(p_pre, "erhs1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for erhs1");
  ecode  = clSetKernelArg(k_erhs1, 0, sizeof(cl_mem), &m_frct);
  ecode |= clSetKernelArg(k_erhs1, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_erhs1, 2, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_erhs1, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_erhs1, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_erhs1, 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (ERHS1_DIM == 3) {
    erhs1_lws[0] = nx < work_item_sizes[0] ? nx : work_item_sizes[0];
    temp = max_work_group_size / erhs1_lws[0];
    erhs1_lws[1] = ny < temp ? ny : temp;
    temp = temp / erhs1_lws[1];
    erhs1_lws[2] = nz < temp ? nz : temp;
    erhs1_gws[0] = clu_RoundWorkSize((size_t)nx, erhs1_lws[0]);
    erhs1_gws[1] = clu_RoundWorkSize((size_t)ny, erhs1_lws[1]);
    erhs1_gws[2] = clu_RoundWorkSize((size_t)nz, erhs1_lws[2]);
  } else if (ERHS1_DIM == 2) {
    erhs1_lws[0] = ny < work_item_sizes[0] ? ny : work_item_sizes[0];
    temp = max_work_group_size / erhs1_lws[0];
    erhs1_lws[1] = nz < temp ? nz : temp;
    erhs1_gws[0] = clu_RoundWorkSize((size_t)ny, erhs1_lws[0]);
    erhs1_gws[1] = clu_RoundWorkSize((size_t)nz, erhs1_lws[1]);
  } else {
    temp = nz / max_compute_units;
    erhs1_lws[0] = temp == 0 ? 1 : temp;
    erhs1_gws[0] = clu_RoundWorkSize((size_t)nz, erhs1_lws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_erhs1,
                                 ERHS1_DIM, NULL,
                                 erhs1_gws,
                                 erhs1_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  //------------------------------------------------------------------------
  k_erhs2 = clCreateKernel(p_pre, "erhs2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for erhs2");
  ecode  = clSetKernelArg(k_erhs2, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_erhs2, 1, sizeof(cl_mem), &m_frct);
  ecode |= clSetKernelArg(k_erhs2, 2, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_erhs2, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_erhs2, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_erhs2, 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (ERHS2_DIM == 2) {
    erhs2_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
    temp = max_work_group_size / erhs2_lws[0];
    erhs2_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    erhs2_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), erhs2_lws[0]);
    erhs2_gws[1] = clu_RoundWorkSize((size_t)(nz-2), erhs2_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    erhs2_lws[0] = temp == 0 ? 1 : temp;
    erhs2_gws[0] = clu_RoundWorkSize((size_t)(nz-2), erhs2_lws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_erhs2,
                                 ERHS2_DIM, NULL,
                                 erhs2_gws,
                                 erhs2_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  //------------------------------------------------------------------------
  k_erhs3 = clCreateKernel(p_pre, "erhs3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for erhs3");
  ecode  = clSetKernelArg(k_erhs3, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_erhs3, 1, sizeof(cl_mem), &m_frct);
  ecode |= clSetKernelArg(k_erhs3, 2, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_erhs3, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_erhs3, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_erhs3, 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (ERHS3_DIM == 2) {
    erhs3_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / erhs3_lws[0];
    erhs3_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    erhs3_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), erhs3_lws[0]);
    erhs3_gws[1] = clu_RoundWorkSize((size_t)(nz-2), erhs3_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    erhs3_lws[0] = temp == 0 ? 1 : temp;
    erhs3_gws[0] = clu_RoundWorkSize((size_t)(nz-2), erhs3_lws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_erhs3,
                                 ERHS3_DIM, NULL,
                                 erhs3_gws,
                                 erhs3_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  //------------------------------------------------------------------------
  k_erhs4 = clCreateKernel(p_pre, "erhs4", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for erhs4");
  ecode  = clSetKernelArg(k_erhs4, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_erhs4, 1, sizeof(cl_mem), &m_frct);
  ecode |= clSetKernelArg(k_erhs4, 2, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_erhs4, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_erhs4, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_erhs4, 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (ERHS4_DIM == 2) {
    erhs4_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / erhs4_lws[0];
    erhs4_lws[1] = (jend-jst) < temp ? (jend-jst) : temp;
    erhs4_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), erhs4_lws[0]);
    erhs4_gws[1] = clu_RoundWorkSize((size_t)(jend-jst), erhs4_lws[1]);
  } else {
    //temp = (jend-jst) / max_compute_units;
    temp = 1;
    erhs4_lws[0] = temp == 0 ? 1 : temp;
    erhs4_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), erhs4_lws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_erhs4,
                                 ERHS4_DIM, NULL,
                                 erhs4_gws,
                                 erhs4_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();

  clReleaseKernel(k_erhs1);
  clReleaseKernel(k_erhs2);
  clReleaseKernel(k_erhs3);
  clReleaseKernel(k_erhs4);

  DTIMER_STOP(t_erhs);
}

