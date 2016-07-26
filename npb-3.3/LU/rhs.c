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
#include "timers.h"

//---------------------------------------------------------------------
// compute the right hand sides
//---------------------------------------------------------------------
void rhs()
{
  cl_int ecode;

  if (timeron) timer_start(t_rhs);

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rhs,
                                 RHS_DIM, NULL,
                                 rhs_gws,
                                 rhs_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();

  if (timeron) timer_start(t_rhsx);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rhsx,
                                 RHSX_DIM, NULL,
                                 rhsx_gws,
                                 rhsx_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  if (timeron) timer_stop(t_rhsx);

  if (timeron) timer_start(t_rhsy);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rhsy,
                                 RHSY_DIM, NULL,
                                 rhsy_gws,
                                 rhsy_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  if (timeron) timer_stop(t_rhsy);

  if (timeron) timer_start(t_rhsz);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rhsz,
                                 RHSZ_DIM, NULL,
                                 rhsz_gws,
                                 rhsz_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  if (timeron) timer_stop(t_rhsz);

  if (timeron) timer_stop(t_rhs);
}

