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
#include "timers.h"

//---------------------------------------------------------------------
// to perform pseudo-time stepping SSOR iterations
// for five nonlinear pde's.
//---------------------------------------------------------------------
void ssor(int niter)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, k;
  int istep;
  double tmp, tmp2;
  double delunm[5];
  int lbk, ubk, lbj, ubj;
  int temp;
  cl_int ecode;

  //---------------------------------------------------------------------
  // begin pseudo-time stepping iterations
  //---------------------------------------------------------------------
  tmp = 1.0 / ( omega * ( 2.0 - omega ) );

  //---------------------------------------------------------------------
  // initialize a,b,c,d to zero (guarantees that page tables have been
  // formed, if applicable on given architecture, before timestepping).
  //---------------------------------------------------------------------
  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  //---------------------------------------------------------------------
  // compute the steady-state residuals
  //---------------------------------------------------------------------
  rhs();

  //---------------------------------------------------------------------
  // compute the L2 norms of newton iteration residuals
  //---------------------------------------------------------------------
  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
          ist, iend, jst, jend, &m_rsd, rsdnm );

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_start(1);

  //---------------------------------------------------------------------
  // the timestep loop
  //---------------------------------------------------------------------
  for (istep = 1; istep <= niter; istep++) {
    if ((istep % 20) == 0 || istep == itmax || istep == 1) {
      if (niter > 1) printf(" Time step %4d\n", istep);
    }

    //---------------------------------------------------------------------
    // perform SSOR iteration
    //---------------------------------------------------------------------
    if (timeron) timer_start(t_rhs);
    tmp2 = dt;

    ecode = clSetKernelArg(k_ssor2, 1, sizeof(double), &tmp2);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_ssor2,
                                   SSOR2_DIM, NULL,
                                   ssor2_gws,
                                   ssor2_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();

    if (timeron) timer_stop(t_rhs);

    if (timeron) timer_start(t_blts);

    for (k = 0; k <= (nz-3)+(iend-ist-1)+(jend-jst-1); k++) {
      lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
      ubk = k < (nz-3) ? k : (nz-3);
      lbj = (k-(iend-ist-1)-(nz-3)) >= 0 ? (k-(iend-ist-1)-(nz-3)) : 0;
      ubj = k < (jend-jst-1) ? k : (jend-jst-1);

      ecode  = clSetKernelArg(k_blts, 7, sizeof(int), &k);
      ecode |= clSetKernelArg(k_blts, 8, sizeof(int), &lbk);
      ecode |= clSetKernelArg(k_blts, 9, sizeof(int), &lbj);
      clu_CheckError(ecode, "clSetKernelArg()");
      blts_lws[0] = (ubj-lbj+1) < work_item_sizes[0] ? (ubj-lbj+1) : work_item_sizes[0];
      temp = max_work_group_size / blts_lws[0];
      blts_lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
      blts_gws[0] = clu_RoundWorkSize((size_t)(ubj-lbj+1), blts_lws[0]);
      blts_gws[1] = clu_RoundWorkSize((size_t)(ubk-lbk+1), blts_lws[1]);
      ecode = clEnqueueNDRangeKernel(cmd_queue, k_blts, 2, NULL,
                                     blts_gws, blts_lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    }

    if (timeron) timer_stop(t_blts);

    if (timeron) timer_start(t_buts);

    for (k = (nz-3)+(iend-ist-1)+(jend-jst-1); k >= 0; k--) {
      lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
      ubk = k < (nz-3) ? k : (nz-3);
      lbj = (k-(iend-ist-1)-(nz-3)) >= 0 ? (k-(iend-ist-1)-(nz-3)) : 0;
      ubj = k < (jend-jst-1) ? k : (jend-jst-1);

      ecode  = clSetKernelArg(k_buts, 7, sizeof(int), &k);
      ecode |= clSetKernelArg(k_buts, 8, sizeof(int), &lbk);
      ecode |= clSetKernelArg(k_buts, 9, sizeof(int), &lbj);
      clu_CheckError(ecode, "clSetKernelArg()");
      buts_lws[0] = (ubj-lbj+1) < work_item_sizes[0] ? (ubj-lbj+1) : work_item_sizes[0];
      temp = max_work_group_size / buts_lws[0];
      buts_lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
      buts_gws[0] = clu_RoundWorkSize((size_t)(ubj-lbj+1), buts_lws[0]);
      buts_gws[1] = clu_RoundWorkSize((size_t)(ubk-lbk+1), buts_lws[1]);
      ecode = clEnqueueNDRangeKernel(cmd_queue, k_buts, 2, NULL,
                                     buts_gws, buts_lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    }

    if (timeron) timer_stop(t_buts);

    //---------------------------------------------------------------------
    // update the variables
    //---------------------------------------------------------------------
    if (timeron) timer_start(t_add);
    tmp2 = tmp;

    ecode = clSetKernelArg(k_ssor3, 2, sizeof(double), &tmp2);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_ssor3,
                                   SSOR3_DIM, NULL,
                                   ssor3_gws,
                                   ssor3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();
    if (timeron) timer_stop(t_add);

    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration corrections
    //---------------------------------------------------------------------
    if ( (istep % inorm) == 0 ) {
      if (timeron) timer_start(t_l2norm);
      l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
              ist, iend, jst, jend,
              &m_rsd, delunm );
      if (timeron) timer_stop(t_l2norm);
      /*
      if ( ipr == 1 ) {
        printf(" \n RMS-norm of SSOR-iteration correction "
               "for first pde  = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for second pde = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for third pde  = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for fourth pde = %12.5E\n",
               " RMS-norm of SSOR-iteration correction "
               "for fifth pde  = %12.5E\n", 
               delunm[0], delunm[1], delunm[2], delunm[3], delunm[4]); 
      } else if ( ipr == 2 ) {
        printf("(%5d,%15.6f)\n", istep, delunm[4]);
      }
      */
    }
 
    //---------------------------------------------------------------------
    // compute the steady-state residuals
    //---------------------------------------------------------------------
    rhs();
 
    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration residuals
    //---------------------------------------------------------------------
    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      if (timeron) timer_start(t_l2norm);
      l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
              ist, iend, jst, jend, &m_rsd, rsdnm );
      if (timeron) timer_stop(t_l2norm);
      /*
      if ( ipr == 1 ) {
        printf(" \n RMS-norm of steady-state residual for "
               "first pde  = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "second pde = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "third pde  = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "fourth pde = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "fifth pde  = %12.5E\n", 
               rsdnm[0], rsdnm[1], rsdnm[2], rsdnm[3], rsdnm[4]);
      }
      */
    }

    //---------------------------------------------------------------------
    // check the newton-iteration residuals against the tolerance levels
    //---------------------------------------------------------------------
    if ( ( rsdnm[0] < tolrsd[0] ) && ( rsdnm[1] < tolrsd[1] ) &&
         ( rsdnm[2] < tolrsd[2] ) && ( rsdnm[3] < tolrsd[3] ) &&
         ( rsdnm[4] < tolrsd[4] ) ) {
      //if (ipr == 1 ) {
      printf(" \n convergence was achieved after %4d pseudo-time steps\n",
          istep);
      //}
      break;
    }
  }

  timer_stop(1);
  maxtime = timer_read(1);
}

