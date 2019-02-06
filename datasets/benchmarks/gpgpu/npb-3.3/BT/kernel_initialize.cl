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

#include "bt.h"

//---------------------------------------------------------------------
// this function returns the exact solution at point xi, eta, zeta  
//---------------------------------------------------------------------
void exact_solution(double xi, double eta, double zeta, double dtemp[5],
                    __global double *g_ce)
{
  int m;
  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  for (m = 0; m < 5; m++) {
    dtemp[m] =  ce[m][0] +
      xi*(ce[m][1] + xi*(ce[m][4] + xi*(ce[m][7] + xi*ce[m][10]))) +
      eta*(ce[m][2] + eta*(ce[m][5] + eta*(ce[m][8] + eta*ce[m][11])))+
      zeta*(ce[m][3] + zeta*(ce[m][6] + zeta*(ce[m][9] + 
      zeta*ce[m][12])));
  }
}


//--------------------------------------------------------------------------
// initialize()
//--------------------------------------------------------------------------
//---------------------------------------------------------------------
// Later (in compute_rhs) we compute 1/u for every element. A few of 
// the corner elements are not used, but it convenient (and faster) 
// to compute the whole thing with a simple loop. Make sure those 
// values are nonzero by initializing the whole thing here. 
//---------------------------------------------------------------------
__kernel void initialize1(__global double *g_u,
                          int gp0,
                          int gp1,
                          int gp2)
{
  int i, j, k, m;

  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= gp2 || j >= gp1) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  for (i = 0; i < gp0; i++) {
    for (m = 0; m < 5; m++) {
      u[k][j][i][m] = 1.0;
    }
  }
}


//---------------------------------------------------------------------
// first store the "interpolated" values everywhere on the grid    
//---------------------------------------------------------------------
__kernel void initialize2(__global double *g_u,
                          __global double *g_ce,
                          int gp0,
                          int gp1,
                          int gp2)
{
  int i, j, k, m, ix, iy, iz;
  double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta;

#if INITIALIZE2_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k >= gp2 || j >= gp1 || i >= gp0) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  zeta = (double)(k) * dnzm1;
  eta = (double)(j) * dnym1;
  xi = (double)(i) * dnxm1;

  for (ix = 0; ix < 2; ix++) {
    exact_solution((double)ix, eta, zeta, &Pface[ix][0][0], g_ce);
  }

  for (iy = 0; iy < 2; iy++) {
    exact_solution(xi, (double)iy , zeta, &Pface[iy][1][0], g_ce);
  }

  for (iz = 0; iz < 2; iz++) {
    exact_solution(xi, eta, (double)iz, &Pface[iz][2][0], g_ce);
  }

  for (m = 0; m < 5; m++) {
    Pxi   = xi   * Pface[1][0][m] + (1.0-xi)   * Pface[0][0][m];
    Peta  = eta  * Pface[1][1][m] + (1.0-eta)  * Pface[0][1][m];
    Pzeta = zeta * Pface[1][2][m] + (1.0-zeta) * Pface[0][2][m];

    u[k][j][i][m] = Pxi + Peta + Pzeta - 
                    Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + 
                    Pxi*Peta*Pzeta;
  }

#elif INITIALIZE2_DIM == 2
  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= gp2 || j >= gp1) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  zeta = (double)(k) * dnzm1;
  eta = (double)(j) * dnym1;
  for (i = 0; i < gp0; i++) {
    xi = (double)(i) * dnxm1;

    for (ix = 0; ix < 2; ix++) {
      exact_solution((double)ix, eta, zeta, &Pface[ix][0][0], g_ce);
    }

    for (iy = 0; iy < 2; iy++) {
      exact_solution(xi, (double)iy , zeta, &Pface[iy][1][0], g_ce);
    }

    for (iz = 0; iz < 2; iz++) {
      exact_solution(xi, eta, (double)iz, &Pface[iz][2][0], g_ce);
    }

    for (m = 0; m < 5; m++) {
      Pxi   = xi   * Pface[1][0][m] + (1.0-xi)   * Pface[0][0][m];
      Peta  = eta  * Pface[1][1][m] + (1.0-eta)  * Pface[0][1][m];
      Pzeta = zeta * Pface[1][2][m] + (1.0-zeta) * Pface[0][2][m];

      u[k][j][i][m] = Pxi + Peta + Pzeta - 
                      Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + 
                      Pxi*Peta*Pzeta;
    }
  }
#endif
}


//---------------------------------------------------------------------
// west face and east face                                                  
//---------------------------------------------------------------------
__kernel void initialize3(__global double *g_u,
                          __global double *g_ce,
                          int gp0,
                          int gp1,
                          int gp2)
{
  int i, j, k, m;
  double xi, eta, zeta, temp[5];

  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= gp2 || j >= gp1) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  zeta = (double)k * dnzm1;
  eta = (double)j * dnym1;

  i  = 0;
  xi = 0.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }

  i  = gp0-1;
  xi = 1.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }
}


//---------------------------------------------------------------------
// south face and north face                                                 
//---------------------------------------------------------------------
__kernel void initialize4(__global double *g_u,
                          __global double *g_ce,
                          int gp0,
                          int gp1,
                          int gp2)
{
  int i, j, k, m;
  double xi, eta, zeta, temp[5];

  k = get_global_id(1);
  i = get_global_id(0);
  if (k >= gp2 || i >= gp0) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  zeta = (double)k * dnzm1;
  xi = (double)i * dnxm1;

  j   = 0;
  eta = 0.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }

  j   = gp1-1;
  eta = 1.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }
}


//---------------------------------------------------------------------
// bottom face and top face
//---------------------------------------------------------------------
__kernel void initialize5(__global double *g_u,
                          __global double *g_ce,
                          int gp0,
                          int gp1,
                          int gp2)
{
  int i, j, k, m;
  double xi, eta, zeta, temp[5];

  j = get_global_id(1);
  i = get_global_id(0);
  if (j >= gp1 || i >= gp0) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  eta = (double)j * dnym1;
  xi = (double)i * dnxm1;

  k    = 0;
  zeta = 0.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }

  k    = gp2-1;
  zeta = 1.0;
  exact_solution(xi, eta, zeta, temp, g_ce);
  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = temp[m];
  }
}

