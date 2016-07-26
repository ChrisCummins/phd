//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB BT code. This OpenCL    //
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
#include "timers.h"

/* CEC profiling. */
#include <stdio.h>
#include <stdlib.h>
static cl_event cec_event;
static void cec_profile(cl_event event, const char* name) {
  clWaitForEvents(1, &event);
  cl_int err;
  cl_ulong start_time, end_time;

  err = clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_QUEUED,
                                sizeof(start_time), &start_time,
                                NULL);
  if (err != CL_SUCCESS) {
    printf("[CEC] fatal: Kernel timer 1!");
    exit(104);
  }

  err = clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(end_time), &end_time,
                                NULL);
  if (err != CL_SUCCESS) {
    printf("[CEC] fatal: Kernel timer 2!");
    exit(105);
  }

  float elapsed_ms = (float)(end_time - start_time) / 1000;
  printf("\n[CEC] %s %.3f\n", name, elapsed_ms);
}
/* END CEC profiling. */

//---------------------------------------------------------------------
// Performs line solves in X direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix,
// and then performing back substitution to solve for the unknow
// vectors of each line.
//
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------
void x_solve()
{
  cl_int ecode;

  if (timeron) timer_start(t_xsolve);

  if (X_SOLVE_DIM == 3) {
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve1,
                                   X_SOLVE_DIM, NULL,
                                   x_solve1_gws,
                                   x_solve1_lws,
                                   0, NULL, &cec_event);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    cec_profile(cec_event, "x_solve1");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve2,
                                   X_SOLVE_DIM, NULL,
                                   x_solve2_gws,
                                   x_solve2_lws,
                                   0, NULL, &cec_event);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    cec_profile(cec_event, "x_solve2");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve3,
                                   X_SOLVE_DIM, NULL,
                                   x_solve3_gws,
                                   x_solve3_lws,
                                   0, NULL, &cec_event);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    cec_profile(cec_event, "x_solve3");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve,
                                   2, NULL,
                                   x_solve_gws,
                                   x_solve_lws,
                                   0, NULL, &cec_event);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    cec_profile(cec_event, "x_solve");
    CHECK_FINISH();
  } else {
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve,
                                   X_SOLVE_DIM, NULL,
                                   x_solve_gws,
                                   x_solve_lws,
                                   0, NULL, &cec_event);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    cec_profile(cec_event, "x_solve");
    CHECK_FINISH();
  }

  if (timeron) timer_stop(t_xsolve);
}
