//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB IS code. This OpenCL    //
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

#ifndef __IS_KERNEL_H__
#define __IS_KERNEL_H__

#include "is.h"

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#else
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
#endif

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif

/*
 *    FUNCTION RANDLC (X, A)
 *
 *  This routine returns a uniform pseudorandom double precision number in the
 *  range (0, 1) by using the linear congruential generator
 *
 *  x_{k+1} = a x_k  (mod 2^46)
 *
 *  where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
 *  before repeating.  The argument A is the same as 'a' in the above formula,
 *  and X is the same as x_0.  A and X must be odd double precision integers
 *  in the range (1, 2^46).  The returned value RANDLC is normalized to be
 *  between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
 *  the new seed x_1, so that subsequent calls to RANDLC using the same
 *  arguments will generate a continuous sequence.
 *
 *  This routine should produce the same results on any computer with at least
 *  48 mantissa bits in double precision floating point data.  On Cray systems,
 *  double precision should be disabled.
 *
 *  David H. Bailey     October 26, 1990
 *
 *     IMPLICIT DOUBLE PRECISION (A-H, O-Z)
 *     SAVE KS, R23, R46, T23, T46
 *     DATA KS/0/
 *
 *  If this is the first call to RANDLC, compute R23 = 2 ^ -23, R46 = 2 ^ -46,
 *  T23 = 2 ^ 23, and T46 = 2 ^ 46.  These are computed in loops, rather than
 *  by merely using the ** operator, in order to insure that the results are
 *  exact on all systems.  This code assumes that 0.5D0 is represented exactly.
 */


/*****************************************************************/
/*************           R  A  N  D  L  C             ************/
/*************                                        ************/
/*************    portable random number generator    ************/
/*****************************************************************/
double randlc( double *X, double *A,
               double R23, double R46, double T23, double T46 )
{
  double T1, T2, T3, T4;
  double A1;
  double A2;
  double X1;
  double X2;
  double Z;
  int j;

  /*  Break A into two parts such that A = 2^23 * A1 + A2 and set X = N.  */

  T1 = R23 * *A;
  j  = T1;
  A1 = j;
  A2 = *A - T23 * A1;

  /*  Break X into two parts such that X = 2^23 * X1 + X2, compute
      Z = A1 * X2 + A2 * X1  (mod 2^23), and then
      X = 2^23 * Z + A2 * X2  (mod 2^46).                            */

  T1 = R23 * *X;
  j  = T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j  = R23 * T1;
  T2 = j;
  Z = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j  = R46 * T3;
  T4 = j;
  *X = T3 - T46 * T4;
  return(R46 * *X);
} 


/*****************************************************************/
/************   F  I  N  D  _  M  Y  _  S  E  E  D    ************/
/************                                         ************/
/************ returns parallel random number seq seed ************/
/*****************************************************************/

/*
 * Create a random number sequence of total length nn residing
 * on np number of processors.  Each processor will therefore have a
 * subsequence of length nn/np.  This routine returns that random
 * number which is the first random number for the subsequence belonging
 * to processor rank kn, and which is used as seed for proc kn ran # gen.
 */

double find_my_seed( int kn,        /* my processor rank, 0<=kn<=num procs */
                     int np,        /* np = num procs                      */
                     long nn,       /* total num of ran numbers, all procs */
                     double s,      /* Ran num seed, for ex.: 314159265.00 */
                     double a,      /* Ran num gen mult, try 1220703125.00 */
                     double R23,
                     double R46,
                     double T23,
                     double T46 )
{
  double t1,t2;
  long   mq,nq,kk,ik;

  if ( kn == 0 ) return s;

  mq = (nn/4 + np - 1) / np;
  nq = mq * 4 * kn;               /* number of rans to be skipped */

  t1 = s;
  t2 = a;
  kk = nq;
  while ( kk > 1 ) {
    ik = kk / 2;
    if( 2 * ik ==  kk ) {
      (void)randlc( &t2, &t2, R23, R46, T23, T46 );
      kk = ik;
    }
    else {
      (void)randlc( &t1, &t2, R23, R46, T23, T46 );
      kk = kk - 1;
    }
  }
  (void)randlc( &t1, &t2, R23, R46, T23, T46 );

  return( t1 );
}

#endif //__IS_KERNEL_H__
