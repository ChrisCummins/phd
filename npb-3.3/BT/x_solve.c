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
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve2,
                                   X_SOLVE_DIM, NULL,
                                   x_solve2_gws,
                                   x_solve2_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve3,
                                   X_SOLVE_DIM, NULL,
                                   x_solve3_gws,
                                   x_solve3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve,
                                   2, NULL,
                                   x_solve_gws,
                                   x_solve_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();
  } else {
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_x_solve,
                                   X_SOLVE_DIM, NULL,
                                   x_solve_gws,
                                   x_solve_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();
  }

  if (timeron) timer_stop(t_xsolve);
}
