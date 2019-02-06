//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB CG code. This OpenCL    //
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

#include "cg.h"

#ifndef LSIZE
#error "LSIZE is not defined!"
#endif

//////////////////////////////////////////////////////////////////////////
// Kernels for main()
//////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(LSIZE,1,1)))
void main_0(__global int *colidx,
            __global int *rowstr,
            int firstcol,
            int n)
{
  int j = (int)(get_global_id(0) / LSIZE);
  int lid = get_local_id(0);

  int row_start = rowstr[j];
  int row_end = rowstr[j+1];
  for (int k = row_start+lid; k < row_end; k += LSIZE) {
    colidx[k] = colidx[k] - firstcol;
  }
}

__kernel void main_1(__global double *x)
{
  int i = get_global_id(0);
  if (i >= (NA+1)) return;

  x[i] = 1.0;
}

__kernel void main_2(__global double *q,
                     __global double *z,
                     __global double *r,
                     __global double *p,
                     int n)
{
  int j = get_global_id(0);
  if (j >= n) return;

  q[j] = 0.0;
  z[j] = 0.0;
  r[j] = 0;
  p[j] = 0;
}

__kernel void main_3(__global double *x,
                     __global double *z,
                     __global double *g_norm_temp1,
                     __global double *g_norm_temp2,
                     __local double *l_norm_temp1,
                     __local double *l_norm_temp2,
                     int n)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);

  if (j < n) {
    double z_val = z[j];
    l_norm_temp1[lid] = x[j] * z_val;
    l_norm_temp2[lid] = z_val * z_val;
  } else {
    l_norm_temp1[lid] = 0.0;
    l_norm_temp2[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      l_norm_temp1[lid] += l_norm_temp1[lid + i];
      l_norm_temp2[lid] += l_norm_temp2[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    int wgid = get_group_id(0);
    g_norm_temp1[wgid] = l_norm_temp1[0];
    g_norm_temp2[wgid] = l_norm_temp2[0];
  }
}

__kernel void main_4(__global double *x,
                     __global double *z,
                     double norm_temp2,
                     int n)
{
  int j = get_global_id(0);
  if (j >= n) return;

  x[j] = norm_temp2 * z[j];
}


//////////////////////////////////////////////////////////////////////////
// Kernels for conj_grad()
//////////////////////////////////////////////////////////////////////////
//---------------------------------------------------------------------
// Initialize the CG algorithm:
//---------------------------------------------------------------------
__kernel void conj_grad_0(__global double *q,
                          __global double *z,
                          __global double *r,
                          __global double *x,
                          __global double *p)
{
  int j = get_global_id(0);
  if (j >= (NA+1)) return;    // naa = NA
  
  q[j] = 0.0;
  z[j] = 0.0;
  double x_val = x[j];
  r[j] = x_val;
  p[j] = x_val;
}


//---------------------------------------------------------------------
// rho = r.r
// Now, obtain the norm of r: First, sum squares of r elements locally...
//---------------------------------------------------------------------
__kernel void conj_grad_1(__global double *r,
                          __global double *g_rho,
                          int n,
                          __local double *l_rho)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);

  if (j < n) {
    double r_val = r[j];
    l_rho[lid] = r_val * r_val;
  } else {
    l_rho[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) l_rho[lid] += l_rho[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) g_rho[get_group_id(0)] = l_rho[0];
}


//---------------------------------------------------------------------
// q = A.p
// The partition submatrix-vector multiply: use workspace w
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(LSIZE,1,1)))
void conj_grad_2(__global int *rowstr,
                 __global double *a,
                 __global double *p,
                 __global int *colidx,
                 __global double *q,
                 int n)
{
  __local double l_sum[LSIZE];

  int j = (int)(get_global_id(0) / LSIZE);
  int lid = get_local_id(0);

  double sum = 0.0;
  int row_start = rowstr[j];
  int row_end = rowstr[j+1];
  for (int k = row_start+lid; k < row_end; k += LSIZE) {
    sum = sum + a[k]*p[colidx[k]];
  }
  l_sum[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = LSIZE / 2; i > 0; i >>= 1) {
    if (lid < i) l_sum[lid] += l_sum[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) q[j] = l_sum[0];
}


//---------------------------------------------------------------------
// Obtain p.q
//---------------------------------------------------------------------
__kernel void conj_grad_3(__global double *p,
                          __global double *q,
                          __global double *g_d,
                          int n,
                          __local double *l_d)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);

  if (j < n) {
    l_d[lid] = p[j] * q[j];
  } else {
    l_d[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) l_d[lid] += l_d[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) g_d[get_group_id(0)] = l_d[0];
}


//---------------------------------------------------------------------
// Obtain z = z + alpha*p
// and    r = r - alpha*q
//---------------------------------------------------------------------
__kernel void conj_grad_4(__global double *p,
                          __global double *q,
                          __global double *r,
                          __global double *z,
                          __global double *g_rho,
                          double alpha,
                          int n,
                          __local double *l_rho)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);

  if (j < n) {
    double r_val;
    z[j] = z[j] + alpha*p[j];
    r_val = r[j] - alpha*q[j];
    r[j] = r_val;

    //---------------------------------------------------------------------
    // rho = r.r
    // Now, obtain the norm of r: First, sum squares of r elements locally..
    //---------------------------------------------------------------------
    l_rho[lid] = r_val * r_val;
  } else {
    l_rho[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) l_rho[lid] += l_rho[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) g_rho[get_group_id(0)] = l_rho[0];
}


//---------------------------------------------------------------------
// p = r + beta*p
//---------------------------------------------------------------------
__kernel void conj_grad_5(__global double *p,
                          __global double *r,
                          const double beta,
                          int n)
{
  int j = get_global_id(0);
  if (j >= n) return;

  p[j] = r[j] + beta*p[j];
}


//---------------------------------------------------------------------
// Compute residual norm explicitly:  ||r|| = ||x - A.z||
// First, form A.z
// The partition submatrix-vector multiply
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(LSIZE,1,1)))
void conj_grad_6(__global int *rowstr,
                 __global double *a,
                 __global double *z,
                 __global int *colidx,
                 __global double *r,
                 int n)
{
  __local double l_sum[LSIZE];

  int j = (int)(get_global_id(0) / LSIZE);
  int lid = get_local_id(0);

  double suml = 0.0;
  int row_start = rowstr[j];
  int row_end = rowstr[j+1];
  for (int k = row_start+lid; k < row_end; k += LSIZE) {
    suml = suml + a[k]*z[colidx[k]];
  }
  l_sum[lid] = suml;
  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = LSIZE / 2; i > 0; i >>= 1) {
    if (lid < i) l_sum[lid] += l_sum[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) r[j] = l_sum[0];
}


//---------------------------------------------------------------------
// At this point, r contains A.z
//---------------------------------------------------------------------
__kernel void conj_grad_7(__global double *x,
                          __global double *r,
                          __global double *g_sum,
                          int n,
                          __local double *l_sum)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);

  if (j < n) {
    double suml = x[j] - r[j];
    l_sum[lid] = suml*suml;
  } else {
    l_sum[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) l_sum[lid] += l_sum[lid + i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) g_sum[get_group_id(0)] = l_sum[0];
}

