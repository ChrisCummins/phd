//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB FT code. This OpenCL    //
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

#include "ft.h"

#define CFFTS_BARRIER(m)    barrier(m)


void cfftz(int is, int m, int n, 
           __global dcomplex *u, __local dcomplex *x, __local dcomplex *y);
void fftz2(int is, int l, int m, int n,
           __global dcomplex *u, 
           __local dcomplex *x,
           __local dcomplex *y);
inline void vranlc(int n, double *x, double a, __global double y[]);


__kernel void init_ui(__global dcomplex *u0,
                      __global dcomplex *u1,
                      __global double *twiddle,
                      int n)
{
  int i = get_global_id(0);
  if (i >= n) return;

  u0[i] = dcmplx(0.0, 0.0);
  u1[i] = dcmplx(0.0, 0.0);
  twiddle[i] = 0.0;
}


__kernel void compute_indexmap(__global double *twiddle,
                               int d1,
                               int d2,
                               int d3,
                               double ap)
{
#if COMPUTE_IMAP_DIM == 3
  int kk, kk2, jj, kj2, ii;

  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);
  if (k >= d3 || j >= d2 || i >= d1) return;

  //---------------------------------------------------------------------
  // basically we want to convert the fortran indices 
  //   1 2 3 4 5 6 7 8 
  // to 
  //   0 1 2 3 -4 -3 -2 -1
  // The following magic formula does the trick:
  // mod(i-1+n/2, n) - n/2
  //---------------------------------------------------------------------

  kk = ((k + NZ/2) % NZ) - NZ/2;
  kk2 = kk*kk;
  jj = ((j + NY/2) % NY) - NY/2;
  kj2 = jj*jj + kk2;
  ii = ((i + NX/2) % NX) - NX/2;
  twiddle[k*d2*(d1+1) + j*(d1+1) + i] = exp(ap * (double)(ii*ii+kj2));

#elif COMPUTE_IMAP_DIM == 2
  int kk, kk2, jj, kj2, ii;

  int k = get_global_id(1);
  int j = get_global_id(0);
  if (k >= d3 || j >= d2) return;

  //---------------------------------------------------------------------
  // basically we want to convert the fortran indices 
  //   1 2 3 4 5 6 7 8 
  // to 
  //   0 1 2 3 -4 -3 -2 -1
  // The following magic formula does the trick:
  // mod(i-1+n/2, n) - n/2
  //---------------------------------------------------------------------

  int i;
  int kj_idx = k*d2*(d1+1) + j*(d1+1);

  kk = ((k + NZ/2) % NZ) - NZ/2;
  kk2 = kk*kk;
  jj = ((j + NY/2) % NY) - NY/2;
  kj2 = jj*jj + kk2;
  for (i = 0; i < d1; i++) {
    ii = ((i + NX/2) % NX) - NX/2;
    twiddle[kj_idx + i] = exp(ap * (double)(ii*ii+kj2));
  }

#else
  int kk, kk2, jj, kj2, ii;

  int k = get_global_id(0);
  if (k >= d3) return;

  //---------------------------------------------------------------------
  // basically we want to convert the fortran indices 
  //   1 2 3 4 5 6 7 8 
  // to 
  //   0 1 2 3 -4 -3 -2 -1
  // The following magic formula does the trick:
  // mod(i-1+n/2, n) - n/2
  //---------------------------------------------------------------------

  int i, j;
  int k_idx = k*d2*(d1+1);

  kk = ((k + NZ/2) % NZ) - NZ/2;
  kk2 = kk*kk;
  for (j = 0; j < d2; j++) {
    int kj_idx = k_idx + j*(d1+1);
    jj = ((j + NY/2) % NY) - NY/2;
    kj2 = jj*jj + kk2;
    for (i = 0; i < d1; i++) {
      ii = ((i + NX/2) % NX) - NX/2;
      twiddle[kj_idx + i] = exp(ap * (double)(ii*ii+kj2));
    }
  }
#endif
}


//---------------------------------------------------------------------
// Go through by z planes filling in one square at a time.
//---------------------------------------------------------------------
__kernel void compute_initial_conditions(__global dcomplex *u0,
                                         __global const double *starts,
                                         int d1,
                                         int d2,
                                         int d3)
{
  double x0;
  int j;

  int k = get_global_id(0);
  if (k >= d3) return;

  x0 = starts[k];
  int kidx = k * d2 * (d1+1);
  for (j = 0; j < d2; j++) {
    vranlc(2*NX, &x0, A, (__global double *)&u0[kidx + j*(d1+1) + 0]);
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
__kernel void evolve(__global dcomplex *u0,
                     __global dcomplex *u1,
                     __global const double *twiddle,
                     int d1,
                     int d2,
                     int d3)
{
#if EVOLVE_DIM == 3
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);
  if (k >= d3 || j >= d2 || i >= d1) return;

  int idx = k*d2*(d1+1) + j*(d1+1) + i;
  u0[idx] = dcmplx_mul2(u0[idx], twiddle[idx]);
  u1[idx].real = u0[idx].real;
  u1[idx].imag = u0[idx].imag;

#elif EVOLVE_DIM == 2
  int k = get_global_id(1);
  int j = get_global_id(0);
  if (k >= d3 || j >= d2) return;

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = 0; i < d1; i++) {
    int idx = kj_idx + i;
    u0[idx] = dcmplx_mul2(u0[idx], twiddle[idx]);
    u1[idx].real = u0[idx].real;
    u1[idx].imag = u0[idx].imag;
  }

#else
  int k = get_global_id(0);
  if (k >= d3) return;

  int k_idx = k*d2*(d1+1);
  for (int j = 0; j < d2; j++) {
    int kj_idx = k_idx + j*(d1+1);
    for (int i = 0; i < d1; i++) {
      int idx = kj_idx + i;
      u0[idx] = dcmplx_mul2(u0[idx], twiddle[idx]);
      u1[idx].real = u0[idx].real;
      u1[idx].imag = u0[idx].imag;
    }
  }
#endif
}


__kernel void checksum(__global dcomplex *u1,
                       __global dcomplex *g_chk,
                       __local dcomplex *l_chk,
                       int d1,
                       int d2)
{
  int q, r, s;
  int j = get_global_id(0) + 1;
  int lid = get_local_id(0);

  if (j <= 1024) {
    q = j % NX;
    r = 3*j % NY;
    s = 5*j % NZ;
    int u1_idx = s*d2*(d1+1) + r*(d1+1) + q;
    l_chk[lid].real = u1[u1_idx].real;
    l_chk[lid].imag = u1[u1_idx].imag;
  } else {
    l_chk[lid] = dcmplx(0.0, 0.0);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      l_chk[lid] = dcmplx_add(l_chk[lid], l_chk[lid+i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    g_chk[get_group_id(0)].real = l_chk[0].real;
    g_chk[get_group_id(0)].imag = l_chk[0].imag;
  }
}


__kernel void cffts1(__global dcomplex *x,
                     __global dcomplex *xout,
                     __global dcomplex *u,
                     int is,
                     int d1,
                     int d2,
                     int d3,
                     int logd1)
{
#if CFFTS_DIM == 2
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int k = get_global_id(1);
  int j = get_group_id(0);
  int lid = get_local_id(0);
  
  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += LSIZE) {
    int x_idx = kj_idx + i;
    ty1[i].real = x[x_idx].real;
    ty1[i].imag = x[x_idx].imag;
  }

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  cfftz(is, logd1, d1, u, ty1, ty2);

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  for (int i = lid; i < d1; i += LSIZE) {
    int xout_idx = kj_idx + i;
    xout[xout_idx].real = ty1[i].real;
    xout[xout_idx].imag = ty1[i].imag;
  }

#elif CFFTS_DIM == 1
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int k = get_group_id(0);
  int lid = get_local_id(0);

  int k_idx = k*d2*(d1+1);
  for (int j = 0; j < d2; j++) {
    int kj_idx = k_idx + j*(d1+1);
    for (int i = lid; i < d1; i += LSIZE) {
      int x_idx = kj_idx + i;
      ty1[i].real = x[x_idx].real;
      ty1[i].imag = x[x_idx].imag;
    }

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    cfftz(is, logd1, d1, u, ty1, ty2);

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < d1; i += LSIZE) {
      int xout_idx = kj_idx + i;
      xout[xout_idx].real = ty1[i].real;
      xout[xout_idx].imag = ty1[i].imag;
    }
  }

#else 
#error "ERROR: CFFTS_DIM"
#endif
}


__kernel void cffts2(__global dcomplex *x,
                     __global dcomplex *xout,
                     __global dcomplex *u,
                     int is,
                     int d1,
                     int d2,
                     int d3,
                     int logd2)
{
#if CFFTS_DIM == 2
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int k = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += LSIZE) {
    int x_idx = ki_idx + j*(d1+1);
    ty1[j].real = x[x_idx].real;
    ty1[j].imag = x[x_idx].imag;
  }

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  cfftz(is, logd2, d2, u, ty1, ty2);

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  for (int j = lid; j < d2; j += LSIZE) {
    int xout_idx = ki_idx + j*(d1+1);
    xout[xout_idx].real = ty1[j].real;
    xout[xout_idx].imag = ty1[j].imag;
  }

#elif CFFTS_DIM == 1
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int LSIZE = get_local_size(0);
  int k = get_group_id(0);
  int lid = get_local_id(0);

  int k_idx = k*d2*(d1+1);
  for (int i = 0; i < d1; i++) {
    int ki_idx = k_idx + i;
    for (int j = lid; j < d2; j += LSIZE) {
      int x_idx = ki_idx + j*(d1+1);
      ty1[j].real = x[x_idx].real;
      ty1[j].imag = x[x_idx].imag;
    }

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    cfftz(is, logd2, d2, u, ty1, ty2);

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    for (int j = lid; j < d2; j += LSIZE) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx].real = ty1[j].real;
      xout[xout_idx].imag = ty1[j].imag;
    }
  }
#endif
}


__kernel void cffts3(__global dcomplex *x,
                     __global dcomplex *xout,
                     __global dcomplex *u,
                     int is,
                     int d1,
                     int d2,
                     int d3,
                     int logd3)
{
#if CFFTS_DIM == 2
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int j = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += LSIZE) {
    int x_idx = k*d2*(d1+1) + ji_idx;
    ty1[k].real = x[x_idx].real;
    ty1[k].imag = x[x_idx].imag;
  }

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  cfftz(is, logd3, d3, u, ty1, ty2);

  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

  for (int k = lid; k < d3; k += LSIZE) {
    int xout_idx = k*d2*(d1+1) + ji_idx;
    xout[xout_idx].real = ty1[k].real;
    xout[xout_idx].imag = ty1[k].imag;
  }

#elif CFFTS_DIM == 1
  __local dcomplex ty1[MAXDIM];
  __local dcomplex ty2[MAXDIM];

  int LSIZE = get_local_size(0);
  int j = get_group_id(0);
  int lid = get_local_id(0);

  int j_idx = j*(d1+1);
  for (int i = 0; i < d1; i++) {
    int ji_idx = j_idx + i;
    for (int k = lid; k < d3; k += LSIZE) {
      int x_idx = k*d2*(d1+1) + ji_idx;
      ty1[k].real = x[x_idx].real;
      ty1[k].imag = x[x_idx].imag;
    }

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    cfftz(is, logd3, d3, u, ty1, ty2);

    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);

    for (int k = lid; k < d3; k += LSIZE) {
      int xout_idx = k*d2*(d1+1) + ji_idx;
      xout[xout_idx].real = ty1[k].real;
      xout[xout_idx].imag = ty1[k].imag;
    }
  }

#endif
}


//---------------------------------------------------------------------
// Computes NY N-point complex-to-complex FFTs of X using an algorithm due
// to Swarztrauber.  X is both the input and the output array, while Y is a 
// scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to 
// perform FFTs, the array U must be initialized by calling CFFTZ with IS 
// set to 0 and M set to MX, where MX is the maximum value of M for any 
// subsequent call.
//---------------------------------------------------------------------
void cfftz(int is, int m, int n, 
           __global dcomplex *u, __local dcomplex *x, __local dcomplex *y)
{
//  int i, j, l, mx;
  int j, l;
  int lid = get_local_id(0);

  //---------------------------------------------------------------------
  // Check if input parameters are invalid.
  //---------------------------------------------------------------------
//  mx = (int)(u[0].real);
//  if ((is != 1 && is != -1) || m < 1 || m > mx) {
//    printf("CFFTZ: Either U has not been initialized, or else\n"    
//           "one of the input parameters is invalid%5d%5d%5d\n", is, m, mx);
//    exit(EXIT_FAILURE); 
//  }

  //---------------------------------------------------------------------
  // Perform one variant of the Stockham FFT.
  //---------------------------------------------------------------------
  for (l = 1; l <= m; l += 2) {
    fftz2(is, l, m, n, u, x, y);
    if (l == m) {
      //-----------------------------------------------------------------
      // Copy Y to X.
      //-----------------------------------------------------------------
      for (j = lid; j < n; j += LSIZE) {
        x[j].real = y[j].real;
        x[j].imag = y[j].imag;
      }
      return;
    }
    fftz2(is, l + 1, m, n, u, y, x);
  }
}


//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
void fftz2(int is, int l, int m, int n,
           __global dcomplex *u, __local dcomplex *x, __local dcomplex *y)
{
  int k, n1, li, lj, lk, ku, i, i11, i12, i21, i22;
  dcomplex u1, x11, x21, tmp;
  int lid = get_local_id(0);

  //---------------------------------------------------------------------
  // Set initial parameters.
  //---------------------------------------------------------------------
  n1 = n / 2;
  lk = 1 << (l - 1);
  li = 1 << (m - l);
  lj = 2 * lk;
  ku = li;

  for (i = 0; i <= li - 1; i++) {
    i11 = i * lk;
    i12 = i11 + n1;
    i21 = i * lj;
    i22 = i21 + lk;
    if (is >= 1) {
      u1.real = u[ku+i].real;
      u1.imag = u[ku+i].imag;
    } else {
      u1 = dconjg(u[ku+i]);
    }

    //---------------------------------------------------------------------
    // This loop is vectorizable.
    //---------------------------------------------------------------------
    for (k = lid; k <= lk - 1; k += LSIZE) {
      x11.real = x[i11+k].real;
      x11.imag = x[i11+k].imag;
      x21.real = x[i12+k].real;
      x21.imag = x[i12+k].imag;
      y[i21+k] = dcmplx_add(x11, x21);
      tmp = dcmplx_sub(x11, x21);
      y[i22+k] = dcmplx_mul(u1, tmp);
    }
    CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);
  }
  CFFTS_BARRIER(CLK_LOCAL_MEM_FENCE);
}


inline void vranlc(int n, double *x, double a, __global double y[])
{
  /*--------------------------------------------------------------------
   This routine generates N uniform pseudorandom double precision numbers in
   the range (0, 1) by using the linear congruential generator
  
   x_{k+1} = a x_k  (mod 2^46)
  
   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
   before repeating.  The argument A is the same as 'a' in the above formula,
   and X is the same as x_0.  A and X must be odd double precision integers
   in the range (1, 2^46).  The N results are placed in Y and are normalized
   to be between 0 and 1.  X is updated to contain the new seed, so that
   subsequent calls to VRANLC using the same arguments will generate a
   continuous sequence.  If N is zero, only initialization is performed, and
   the variables X, A and Y are ignored.
  
   This routine is the standard version designed for scalar or RISC systems.
   However, it should produce the same results on any single processor
   computer with at least 48 mantissa bits in double precision floating point
   data.  On 64 bit systems, double precision should be disabled.
  --------------------------------------------------------------------*/

  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;

  int i;

  //--------------------------------------------------------------------
  //  Break A into two parts such that A = 2^23 * A1 + A2.
  //--------------------------------------------------------------------
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  //--------------------------------------------------------------------
  //  Generate N results.   This loop is not vectorizable.
  //--------------------------------------------------------------------
  for ( i = 0; i < n; i++ ) {
    //--------------------------------------------------------------------
    //  Break X into two parts such that X = 2^23 * X1 + X2, compute
    //  Z = A1 * X2 + A2 * X1  (mod 2^23), and then
    //  X = 2^23 * Z + A2 * X2  (mod 2^46).
    //--------------------------------------------------------------------
    t1 = r23 * (*x);
    x1 = (int) t1;
    x2 = *x - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int) (r23 * t1);
    z = t1 - t23 * t2;
    t3 = t23 * z + a2 * x2;
    t4 = (int) (r46 * t3) ;
    *x = t3 - t46 * t4;
    y[i] = r46 * (*x);
  }
}


