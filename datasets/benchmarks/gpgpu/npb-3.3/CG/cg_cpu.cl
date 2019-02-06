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

//////////////////////////////////////////////////////////////////////////
// Kernels for main()
//////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void main_0(__global int *colidx,
            __global int *rowstr,
            int firstcol,
            int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    for (int k = rowstr[j]; k < rowstr[j+1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }
}

__kernel __attribute__((reqd_work_group_size(1,1,1)))
void main_1(__global double *x)
{
  int n = NA+1;
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    x[j] = 1.0;
  }
}

__kernel __attribute__((reqd_work_group_size(1,1,1)))
void main_2(__global double *q,
            __global double *z,
            __global double *r,
            __global double *p,
            int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0;
    p[j] = 0;
  }
}

__kernel __attribute__((reqd_work_group_size(1,1,1)))
void main_3(__global double *x,
            __global double *z,
            __global double *g_norm_temp1,
            __global double *g_norm_temp2,
            __local double *l_norm_temp1,
            __local double *l_norm_temp2,
            int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  double norm_temp1 = 0.0;
  double norm_temp2 = 0.0;
  for (int j = j_start; j < j_end; j++) {
    norm_temp1 = norm_temp1 + x[j] * z[j];
    norm_temp2 = norm_temp2 + z[j] * z[j];
  }
  g_norm_temp1[id] = norm_temp1;
  g_norm_temp2[id] = norm_temp2;
}

__kernel __attribute__((reqd_work_group_size(1,1,1)))
void main_4(__global double *x,
            __global double *z,
            double norm_temp2,
            int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    x[j] = norm_temp2 * z[j];
  }
}


//////////////////////////////////////////////////////////////////////////
// Kernels for conj_grad()
//////////////////////////////////////////////////////////////////////////
//---------------------------------------------------------------------
// Initialize the CG algorithm:
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_0(__global double *q,
                 __global double *z,
                 __global double *r,
                 __global double *x,
                 __global double *p)
{
  int n = NA+1;
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    double x_val = x[j];
    r[j] = x_val;
    p[j] = x_val;
  }
}


//---------------------------------------------------------------------
// rho = r.r
// Now, obtain the norm of r: First, sum squares of r elements locally...
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_1(__global double *r,
                 __global double *g_rho,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  double rho = 0.0;
  for (int j = j_start; j < j_end; j++) {
    rho = rho + r[j]*r[j];
  }
  g_rho[id] = rho;
}


//---------------------------------------------------------------------
// q = A.p
// The partition submatrix-vector multiply: use workspace w
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_2(__global int *rowstr,
                 __global double *a,
                 __global double *p,
                 __global int *colidx,
                 __global double *q,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    double suml = 0.0;
    for (int k = rowstr[j]; k < rowstr[j+1]; k++) {
      suml = suml + a[k]*p[colidx[k]];
    }
    q[j] = suml;
  }
}


//---------------------------------------------------------------------
// Obtain p.q
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_3(__global double *p,
                 __global double *q,
                 __global double *g_d,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  double d = 0.0;
  for (int j = j_start; j < j_end; j++) {
    d = d + p[j]*q[j];
  }
  g_d[id] = d;
}


//---------------------------------------------------------------------
// Obtain z = z + alpha*p
// and    r = r - alpha*q
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_4(__global double *p,
                 __global double *q,
                 __global double *r,
                 __global double *z,
                 __global double *g_rho,
                 double alpha,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  double rho = 0.0;
  for (int j = j_start; j < j_end; j++) {
    z[j] = z[j] + alpha*p[j];
    r[j] = r[j] - alpha*q[j];

    //---------------------------------------------------------------------
    // rho = r.r
    // Now, obtain the norm of r: First, sum squares of r elements locally..
    //---------------------------------------------------------------------
    rho = rho + r[j]*r[j];
  }
  g_rho[id] = rho;
}


//---------------------------------------------------------------------
// p = r + beta*p
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_5(__global double *p,
                 __global double *r,
                 const double beta,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    p[j] = r[j] + beta*p[j];
  }
}


//---------------------------------------------------------------------
// Compute residual norm explicitly:  ||r|| = ||x - A.z||
// First, form A.z
// The partition submatrix-vector multiply
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_6(__global int *rowstr,
                 __global double *a,
                 __global double *z,
                 __global int *colidx,
                 __global double *r,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  for (int j = j_start; j < j_end; j++) {
    double suml = 0.0;
    for (int k = rowstr[j]; k < rowstr[j+1]; k++) {
      suml = suml + a[k]*z[colidx[k]];
    }
    r[j] = suml;
  }
}


//---------------------------------------------------------------------
// At this point, r contains A.z
//---------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void conj_grad_7(__global double *x,
                 __global double *r,
                 __global double *g_sum,
                 int n)
{
  int id = get_global_id(0);
  int gsize = get_global_size(0);
  int chunk = (n + gsize - 1) / gsize;
  int j_start = id * chunk;
  int j_end = j_start + chunk;
  if (j_end > n) j_end = n;

  double sum = 0.0;
  for (int j = j_start; j < j_end; j++) {
    double suml = x[j] - r[j];
    sum  = sum + suml*suml;
  }
  g_sum[id] = sum;
}
