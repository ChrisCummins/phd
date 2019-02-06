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

#include "lu.h"

//---------------------------------------------------------------------
//
//   compute the exact solution at (i,j,k)
//
//---------------------------------------------------------------------
double exact_scalar(int i, int j, int k, int m, __global double *g_ce)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  double xi, eta, zeta;
  double sol;

  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  xi   = ( (double)i ) / ( nx0 - 1 );
  eta  = ( (double)j ) / ( ny0 - 1 );
  zeta = ( (double)k ) / ( nz0 - 1 );

  sol =  ce[m][0]
      + (ce[m][1]
      + (ce[m][4]
      + (ce[m][7]
      +  ce[m][10] * xi) * xi) * xi) * xi
      + (ce[m][2]
      + (ce[m][5]
      + (ce[m][8]
      +  ce[m][11] * eta) * eta) * eta) * eta
      + (ce[m][3]
      + (ce[m][6]
      + (ce[m][9]
      +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
  return sol;
}

void exact(int i, int j, int k, double u000ijk[], __global double *g_ce)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int m;
  double xi, eta, zeta;

  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  xi   = ( (double)i ) / ( nx0 - 1 );
  eta  = ( (double)j ) / ( ny0 - 1 );
  zeta = ( (double)k ) / ( nz0 - 1 );

  for (m = 0; m < 5; m++) {
    u000ijk[m] =  ce[m][0]
      + (ce[m][1]
      + (ce[m][4]
      + (ce[m][7]
      +  ce[m][10] * xi) * xi) * xi) * xi
      + (ce[m][2]
      + (ce[m][5]
      + (ce[m][8]
      +  ce[m][11] * eta) * eta) * eta) * eta
      + (ce[m][3]
      + (ce[m][6]
      + (ce[m][9]
      +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
  }
}


__kernel void setbv1(__global double *g_u,
                     __global double *g_ce,
                     int nx,
                     int ny,
                     int nz)
{
#if SETBV1_DIM == 3
  int i, j, m;

  j = get_global_id(2);
  i = get_global_id(1);
  m = get_global_id(0);  // should be [0,5)
  if (j >= ny || i >= nx) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  u[0][j][i][m] = exact_scalar( i, j, 0, m, g_ce );
  u[nz-1][j][i][m] = exact_scalar( i, j, nz-1, m, g_ce );

#elif SETBV1_DIM == 2
  int i, j, m;
  double temp1[5], temp2[5];

  j = get_global_id(1);
  i = get_global_id(0);
  if (j >= ny || i >= nx) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  exact( i, j, 0, temp1, g_ce );
  exact( i, j, nz-1, temp2, g_ce );
  for (m = 0; m < 5; m++) {
    u[0][j][i][m] = temp1[m];
    u[nz-1][j][i][m] = temp2[m];
  }

#else //SETBV1_DIM == 1
  int i, j, m;
  double temp1[5], temp2[5];

  j = get_global_id(0);
  if (j >= ny) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  for (i = 0; i < nx; i++) {
    exact( i, j, 0, temp1, g_ce );
    exact( i, j, nz-1, temp2, g_ce );
    for (m = 0; m < 5; m++) {
      u[0][j][i][m] = temp1[m];
      u[nz-1][j][i][m] = temp2[m];
    }
  }
#endif
}


__kernel void setbv2(__global double *g_u,
                     __global double *g_ce,
                     int nx,
                     int ny,
                     int nz)
{
#if SETBV2_DIM == 3
  int i, k, m;

  k = get_global_id(2);
  i = get_global_id(1);
  m = get_global_id(0);  // should be [0,5)
  if (k >= nz || i >= nx) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  u[k][0][i][m] = exact_scalar( i, 0, k, m, g_ce );
  u[k][ny-1][i][m] = exact_scalar( i, ny-1, k, m, g_ce );

#elif SETBV2_DIM == 2
  int i, k, m;
  double temp1[5], temp2[5];

  k = get_global_id(1);
  i = get_global_id(0);
  if (k >= nz || i >= nx) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  exact( i, 0, k, temp1, g_ce );
  exact( i, ny-1, k, temp2, g_ce );
  for (m = 0; m < 5; m++) {
    u[k][0][i][m] = temp1[m];
    u[k][ny-1][i][m] = temp2[m];
  }

#else //SETBV2_DIM == 1
  int i, k, m;
  double temp1[5], temp2[5];

  k = get_global_id(0);
  if (k >= nz) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  for (i = 0; i < nx; i++) {
    exact( i, 0, k, temp1, g_ce );
    exact( i, ny-1, k, temp2, g_ce );
    for (m = 0; m < 5; m++) {
      u[k][0][i][m] = temp1[m];
      u[k][ny-1][i][m] = temp2[m];
    }
  }
#endif
}


__kernel void setbv3(__global double *g_u,
                     __global double *g_ce,
                     int nx,
                     int ny,
                     int nz)
{
#if SETBV3_DIM == 3
  int j, k, m;

  k = get_global_id(2);
  j = get_global_id(1);
  m = get_global_id(0);  // should be [0,5)
  if (k >= nz || j >= ny) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  u[k][j][0][m] = exact_scalar( 0, j, k, m, g_ce );
  u[k][j][nx-1][m] = exact_scalar( nx-1, j, k, m, g_ce );

#elif SETBV3_DIM == 2
  int j, k, m;
  double temp1[5], temp2[5];

  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= nz || j >= ny) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  exact( 0, j, k, temp1, g_ce );
  exact( nx-1, j, k, temp2, g_ce );
  for (m = 0; m < 5; m++) {
    u[k][j][0][m] = temp1[m];
    u[k][j][nx-1][m] = temp2[m];
  }

#else //SETBV3_DIM == 1
  int j, k, m;
  double temp1[5], temp2[5];

  k = get_global_id(0);
  if (k >= nz) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  for (j = 0; j < ny; j++) {
    exact( 0, j, k, temp1, g_ce );
    exact( nx-1, j, k, temp2, g_ce );
    for (m = 0; m < 5; m++) {
      u[k][j][0][m] = temp1[m];
      u[k][j][nx-1][m] = temp2[m];
    }
  }
#endif
}


__kernel void setiv(__global double *g_u,
                    __global double *g_ce,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  double xi, eta, zeta;
  double pxi, peta, pzeta;
  double ue_1jk[5], ue_nx0jk[5], ue_i1k[5];
  double ue_iny0k[5], ue_ij1[5], ue_ijnz[5];

#if SETIV_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k >= (nz-1) || j >= (ny-1) || i >= (nx-1)) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  zeta = ( (double)k ) / (nz-1);
  eta = ( (double)j ) / (ny0-1);
  xi = ( (double)i ) / (nx0-1);
  exact(0, j, k, ue_1jk, g_ce);
  exact(nx0-1, j, k, ue_nx0jk, g_ce);
  exact(i, 0, k, ue_i1k, g_ce);
  exact(i, ny0-1, k, ue_iny0k, g_ce);
  exact(i, j, 0, ue_ij1, g_ce);
  exact(i, j, nz-1, ue_ijnz, g_ce);

  for (m = 0; m < 5; m++) {
    pxi =   ( 1.0 - xi ) * ue_1jk[m]
                  + xi   * ue_nx0jk[m];
    peta =  ( 1.0 - eta ) * ue_i1k[m]
                  + eta   * ue_iny0k[m];
    pzeta = ( 1.0 - zeta ) * ue_ij1[m]
                  + zeta   * ue_ijnz[m];

    u[k][j][i][m] = pxi + peta + pzeta
      - pxi * peta - peta * pzeta - pzeta * pxi
      + pxi * peta * pzeta;
  }

#elif SETIV_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k >= (nz-1) || j >= (ny-1)) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  zeta = ( (double)k ) / (nz-1);
  eta = ( (double)j ) / (ny0-1);
  for (i = 1; i < nx - 1; i++) {
    xi = ( (double)i ) / (nx0-1);
    exact(0, j, k, ue_1jk, g_ce);
    exact(nx0-1, j, k, ue_nx0jk, g_ce);
    exact(i, 0, k, ue_i1k, g_ce);
    exact(i, ny0-1, k, ue_iny0k, g_ce);
    exact(i, j, 0, ue_ij1, g_ce);
    exact(i, j, nz-1, ue_ijnz, g_ce);

    for (m = 0; m < 5; m++) {
      pxi =   ( 1.0 - xi ) * ue_1jk[m]
                    + xi   * ue_nx0jk[m];
      peta =  ( 1.0 - eta ) * ue_i1k[m]
                    + eta   * ue_iny0k[m];
      pzeta = ( 1.0 - zeta ) * ue_ij1[m]
                    + zeta   * ue_ijnz[m];

      u[k][j][i][m] = pxi + peta + pzeta
        - pxi * peta - peta * pzeta - pzeta * pxi
        + pxi * peta * pzeta;
    }
  }

#else //SETIV_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

  zeta = ( (double)k ) / (nz-1);
  for (j = 1; j < ny - 1; j++) {
    eta = ( (double)j ) / (ny0-1);
    for (i = 1; i < nx - 1; i++) {
      xi = ( (double)i ) / (nx0-1);
      exact(0, j, k, ue_1jk, g_ce);
      exact(nx0-1, j, k, ue_nx0jk, g_ce);
      exact(i, 0, k, ue_i1k, g_ce);
      exact(i, ny0-1, k, ue_iny0k, g_ce);
      exact(i, j, 0, ue_ij1, g_ce);
      exact(i, j, nz-1, ue_ijnz, g_ce);

      for (m = 0; m < 5; m++) {
        pxi =   ( 1.0 - xi ) * ue_1jk[m]
                      + xi   * ue_nx0jk[m];
        peta =  ( 1.0 - eta ) * ue_i1k[m]
                      + eta   * ue_iny0k[m];
        pzeta = ( 1.0 - zeta ) * ue_ij1[m]
                      + zeta   * ue_ijnz[m];

        u[k][j][i][m] = pxi + peta + pzeta
          - pxi * peta - peta * pzeta - pzeta * pxi
          + pxi * peta * pzeta;
      }
    }
  }
#endif
}


__kernel void erhs1(__global double *g_frct,
                    __global double *g_rsd,
                    __global double *g_ce,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  double xi, eta, zeta;

#if ERHS1_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k >= nz || j >= ny || i >= nx) return;

  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  zeta = ( (double)k ) / ( nz - 1 );
  eta = ( (double)j ) / ( ny0 - 1 );
  xi = ( (double)i ) / ( nx0 - 1 );
  for (m = 0; m < 5; m++) {
    frct[k][j][i][m] = 0.0;

    rsd[k][j][i][m] =  ce[m][0]
      + (ce[m][1]
      + (ce[m][4]
      + (ce[m][7]
      +  ce[m][10] * xi) * xi) * xi) * xi
      + (ce[m][2]
      + (ce[m][5]
      + (ce[m][8]
      +  ce[m][11] * eta) * eta) * eta) * eta
      + (ce[m][3]
      + (ce[m][6]
      + (ce[m][9]
      +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
  }

#elif ERHS1_DIM == 2
  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= nz || j >= ny) return;

  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  zeta = ( (double)k ) / ( nz - 1 );
  eta = ( (double)j ) / ( ny0 - 1 );
  for (i = 0; i < nx; i++) {
    xi = ( (double)i ) / ( nx0 - 1 );
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] = 0.0;

      rsd[k][j][i][m] =  ce[m][0]
        + (ce[m][1]
        + (ce[m][4]
        + (ce[m][7]
        +  ce[m][10] * xi) * xi) * xi) * xi
        + (ce[m][2]
        + (ce[m][5]
        + (ce[m][8]
        +  ce[m][11] * eta) * eta) * eta) * eta
        + (ce[m][3]
        + (ce[m][6]
        + (ce[m][9]
        +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
    }
  }

#else //EHRS1_DIM == 1
  k = get_global_id(0);
  if (k >= nz) return;

  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*ce)[13] = (__global double (*)[13])g_ce;

  zeta = ( (double)k ) / ( nz - 1 );
  for (j = 0; j < ny; j++) {
    eta = ( (double)j ) / ( ny0 - 1 );
    for (i = 0; i < nx; i++) {
      xi = ( (double)i ) / ( nx0 - 1 );
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] = 0.0;

        rsd[k][j][i][m] =  ce[m][0]
          + (ce[m][1]
          + (ce[m][4]
          + (ce[m][7]
          +  ce[m][10] * xi) * xi) * xi) * xi
          + (ce[m][2]
          + (ce[m][5]
          + (ce[m][8]
          +  ce[m][11] * eta) * eta) * eta) * eta
          + (ce[m][3]
          + (ce[m][6]
          + (ce[m][9]
          +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
      }
    }
  }
#endif
}


//---------------------------------------------------------------------
// xi-direction flux differences
//---------------------------------------------------------------------
__kernel void erhs2(__global double *g_rsd,
                    __global double *g_frct,
                    __global double *g_flux,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  double q;
  double u21;
  double tmp;
  double u21i, u31i, u41i, u51i;
  double u21im1, u31im1, u41im1, u51im1;

#if ERHS2_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + jst;
  if (k >= (nz-1) || j >= jend) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = (k-1)*ISIZ2 + (j-jst);
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (i = 0; i < nx; i++) {
    flux[i][0] = rsd[k][j][i][1];
    u21 = rsd[k][j][i][1] / rsd[k][j][i][0];
    q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                + rsd[k][j][i][2] * rsd[k][j][i][2]
                + rsd[k][j][i][3] * rsd[k][j][i][3] )
             / rsd[k][j][i][0];
    flux[i][1] = rsd[k][j][i][1] * u21 + C2 * ( rsd[k][j][i][4] - q );
    flux[i][2] = rsd[k][j][i][2] * u21;
    flux[i][3] = rsd[k][j][i][3] * u21;
    flux[i][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u21;
  }

  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] =  frct[k][j][i][m]
                - tx2 * ( flux[i+1][m] - flux[i-1][m] );
    }
  }
  for (i = ist; i < nx; i++) {
    tmp = 1.0 / rsd[k][j][i][0];

    u21i = tmp * rsd[k][j][i][1];
    u31i = tmp * rsd[k][j][i][2];
    u41i = tmp * rsd[k][j][i][3];
    u51i = tmp * rsd[k][j][i][4];

    tmp = 1.0 / rsd[k][j][i-1][0];

    u21im1 = tmp * rsd[k][j][i-1][1];
    u31im1 = tmp * rsd[k][j][i-1][2];
    u41im1 = tmp * rsd[k][j][i-1][3];
    u51im1 = tmp * rsd[k][j][i-1][4];

    flux[i][1] = (4.0/3.0) * tx3 * ( u21i - u21im1 );
    flux[i][2] = tx3 * ( u31i - u31im1 );
    flux[i][3] = tx3 * ( u41i - u41im1 );
    flux[i][4] = 0.50 * ( 1.0 - C1*C5 )
      * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
              - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
      + (1.0/6.0)
      * tx3 * ( u21i*u21i - u21im1*u21im1 )
      + C1 * C5 * tx3 * ( u51i - u51im1 );
  }

  for (i = ist; i < iend; i++) {
    frct[k][j][i][0] = frct[k][j][i][0]
      + dx1 * tx1 * (        rsd[k][j][i-1][0]
                     - 2.0 * rsd[k][j][i][0]
                     +       rsd[k][j][i+1][0] );
    frct[k][j][i][1] = frct[k][j][i][1]
      + tx3 * C3 * C4 * ( flux[i+1][1] - flux[i][1] )
      + dx2 * tx1 * (        rsd[k][j][i-1][1]
                     - 2.0 * rsd[k][j][i][1]
                     +       rsd[k][j][i+1][1] );
    frct[k][j][i][2] = frct[k][j][i][2]
      + tx3 * C3 * C4 * ( flux[i+1][2] - flux[i][2] )
      + dx3 * tx1 * (        rsd[k][j][i-1][2]
                     - 2.0 * rsd[k][j][i][2]
                     +       rsd[k][j][i+1][2] );
    frct[k][j][i][3] = frct[k][j][i][3]
      + tx3 * C3 * C4 * ( flux[i+1][3] - flux[i][3] )
      + dx4 * tx1 * (        rsd[k][j][i-1][3]
                     - 2.0 * rsd[k][j][i][3]
                     +       rsd[k][j][i+1][3] );
    frct[k][j][i][4] = frct[k][j][i][4]
      + tx3 * C3 * C4 * ( flux[i+1][4] - flux[i][4] )
      + dx5 * tx1 * (        rsd[k][j][i-1][4]
                     - 2.0 * rsd[k][j][i][4]
                     +       rsd[k][j][i+1][4] );
  }

  //---------------------------------------------------------------------
  // Fourth-order dissipation
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    frct[k][j][1][m] = frct[k][j][1][m]
      - dssp * ( + 5.0 * rsd[k][j][1][m]
                 - 4.0 * rsd[k][j][2][m]
                 +       rsd[k][j][3][m] );
    frct[k][j][2][m] = frct[k][j][2][m]
      - dssp * ( - 4.0 * rsd[k][j][1][m]
                 + 6.0 * rsd[k][j][2][m]
                 - 4.0 * rsd[k][j][3][m]
                 +       rsd[k][j][4][m] );
  }

  for (i = 3; i < nx - 3; i++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] = frct[k][j][i][m]
        - dssp * (        rsd[k][j][i-2][m]
                  - 4.0 * rsd[k][j][i-1][m]
                  + 6.0 * rsd[k][j][i][m]
                  - 4.0 * rsd[k][j][i+1][m]
                  +       rsd[k][j][i+2][m] );
    }
  }

  for (m = 0; m < 5; m++) {
    frct[k][j][nx-3][m] = frct[k][j][nx-3][m]
      - dssp * (        rsd[k][j][nx-5][m]
                - 4.0 * rsd[k][j][nx-4][m]
                + 6.0 * rsd[k][j][nx-3][m]
                - 4.0 * rsd[k][j][nx-2][m] );
    frct[k][j][nx-2][m] = frct[k][j][nx-2][m]
      - dssp * (        rsd[k][j][nx-4][m]
                - 4.0 * rsd[k][j][nx-3][m]
                + 5.0 * rsd[k][j][nx-2][m] );
  }

#else //ERHS2_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = k - 1;
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (j = jst; j < jend; j++) {
    for (i = 0; i < nx; i++) {
      flux[i][0] = rsd[k][j][i][1];
      u21 = rsd[k][j][i][1] / rsd[k][j][i][0];
      q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                  + rsd[k][j][i][2] * rsd[k][j][i][2]
                  + rsd[k][j][i][3] * rsd[k][j][i][3] )
               / rsd[k][j][i][0];
      flux[i][1] = rsd[k][j][i][1] * u21 + C2 * ( rsd[k][j][i][4] - q );
      flux[i][2] = rsd[k][j][i][2] * u21;
      flux[i][3] = rsd[k][j][i][3] * u21;
      flux[i][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u21;
    }

    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] =  frct[k][j][i][m]
                  - tx2 * ( flux[i+1][m] - flux[i-1][m] );
      }
    }
    for (i = ist; i < nx; i++) {
      tmp = 1.0 / rsd[k][j][i][0];

      u21i = tmp * rsd[k][j][i][1];
      u31i = tmp * rsd[k][j][i][2];
      u41i = tmp * rsd[k][j][i][3];
      u51i = tmp * rsd[k][j][i][4];

      tmp = 1.0 / rsd[k][j][i-1][0];

      u21im1 = tmp * rsd[k][j][i-1][1];
      u31im1 = tmp * rsd[k][j][i-1][2];
      u41im1 = tmp * rsd[k][j][i-1][3];
      u51im1 = tmp * rsd[k][j][i-1][4];

      flux[i][1] = (4.0/3.0) * tx3 * ( u21i - u21im1 );
      flux[i][2] = tx3 * ( u31i - u31im1 );
      flux[i][3] = tx3 * ( u41i - u41im1 );
      flux[i][4] = 0.50 * ( 1.0 - C1*C5 )
        * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
                - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
        + (1.0/6.0)
        * tx3 * ( u21i*u21i - u21im1*u21im1 )
        + C1 * C5 * tx3 * ( u51i - u51im1 );
    }

    for (i = ist; i < iend; i++) {
      frct[k][j][i][0] = frct[k][j][i][0]
        + dx1 * tx1 * (        rsd[k][j][i-1][0]
                       - 2.0 * rsd[k][j][i][0]
                       +       rsd[k][j][i+1][0] );
      frct[k][j][i][1] = frct[k][j][i][1]
        + tx3 * C3 * C4 * ( flux[i+1][1] - flux[i][1] )
        + dx2 * tx1 * (        rsd[k][j][i-1][1]
                       - 2.0 * rsd[k][j][i][1]
                       +       rsd[k][j][i+1][1] );
      frct[k][j][i][2] = frct[k][j][i][2]
        + tx3 * C3 * C4 * ( flux[i+1][2] - flux[i][2] )
        + dx3 * tx1 * (        rsd[k][j][i-1][2]
                       - 2.0 * rsd[k][j][i][2]
                       +       rsd[k][j][i+1][2] );
      frct[k][j][i][3] = frct[k][j][i][3]
        + tx3 * C3 * C4 * ( flux[i+1][3] - flux[i][3] )
        + dx4 * tx1 * (        rsd[k][j][i-1][3]
                       - 2.0 * rsd[k][j][i][3]
                       +       rsd[k][j][i+1][3] );
      frct[k][j][i][4] = frct[k][j][i][4]
        + tx3 * C3 * C4 * ( flux[i+1][4] - flux[i][4] )
        + dx5 * tx1 * (        rsd[k][j][i-1][4]
                       - 2.0 * rsd[k][j][i][4]
                       +       rsd[k][j][i+1][4] );
    }

    //---------------------------------------------------------------------
    // Fourth-order dissipation
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      frct[k][j][1][m] = frct[k][j][1][m]
        - dssp * ( + 5.0 * rsd[k][j][1][m]
                   - 4.0 * rsd[k][j][2][m]
                   +       rsd[k][j][3][m] );
      frct[k][j][2][m] = frct[k][j][2][m]
        - dssp * ( - 4.0 * rsd[k][j][1][m]
                   + 6.0 * rsd[k][j][2][m]
                   - 4.0 * rsd[k][j][3][m]
                   +       rsd[k][j][4][m] );
    }

    for (i = 3; i < nx - 3; i++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] = frct[k][j][i][m]
          - dssp * (        rsd[k][j][i-2][m]
                    - 4.0 * rsd[k][j][i-1][m]
                    + 6.0 * rsd[k][j][i][m]
                    - 4.0 * rsd[k][j][i+1][m]
                    +       rsd[k][j][i+2][m] );
      }
    }

    for (m = 0; m < 5; m++) {
      frct[k][j][nx-3][m] = frct[k][j][nx-3][m]
        - dssp * (        rsd[k][j][nx-5][m]
                  - 4.0 * rsd[k][j][nx-4][m]
                  + 6.0 * rsd[k][j][nx-3][m]
                  - 4.0 * rsd[k][j][nx-2][m] );
      frct[k][j][nx-2][m] = frct[k][j][nx-2][m]
        - dssp * (        rsd[k][j][nx-4][m]
                  - 4.0 * rsd[k][j][nx-3][m]
                  + 5.0 * rsd[k][j][nx-2][m] );
    }
  }
#endif
}


//---------------------------------------------------------------------
// eta-direction flux differences
//---------------------------------------------------------------------
__kernel void erhs3(__global double *g_rsd,
                    __global double *g_frct,
                    __global double *g_flux,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  double q;
  double u31;
  double tmp;
  double u21j, u31j, u41j, u51j;
  double u21jm1, u31jm1, u41jm1, u51jm1;

#if ERHS3_DIM == 2
  k = get_global_id(1) + 1;
  i = get_global_id(0) + ist;
  if (k >= (nz-1) || i >= iend) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = (k-1)*ISIZ2 + (i-ist);
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (j = 0; j < ny; j++) {
    flux[j][0] = rsd[k][j][i][2];
    u31 = rsd[k][j][i][2] / rsd[k][j][i][0];
    q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                + rsd[k][j][i][2] * rsd[k][j][i][2]
                + rsd[k][j][i][3] * rsd[k][j][i][3] )
             / rsd[k][j][i][0];
    flux[j][1] = rsd[k][j][i][1] * u31;
    flux[j][2] = rsd[k][j][i][2] * u31 + C2 * ( rsd[k][j][i][4] - q );
    flux[j][3] = rsd[k][j][i][3] * u31;
    flux[j][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u31;
  }

  for (j = jst; j < jend; j++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] =  frct[k][j][i][m]
        - ty2 * ( flux[j+1][m] - flux[j-1][m] );
    }
  }

  for (j = jst; j < ny; j++) {
    tmp = 1.0 / rsd[k][j][i][0];

    u21j = tmp * rsd[k][j][i][1];
    u31j = tmp * rsd[k][j][i][2];
    u41j = tmp * rsd[k][j][i][3];
    u51j = tmp * rsd[k][j][i][4];

    tmp = 1.0 / rsd[k][j-1][i][0];

    u21jm1 = tmp * rsd[k][j-1][i][1];
    u31jm1 = tmp * rsd[k][j-1][i][2];
    u41jm1 = tmp * rsd[k][j-1][i][3];
    u51jm1 = tmp * rsd[k][j-1][i][4];

    flux[j][1] = ty3 * ( u21j - u21jm1 );
    flux[j][2] = (4.0/3.0) * ty3 * ( u31j - u31jm1 );
    flux[j][3] = ty3 * ( u41j - u41jm1 );
    flux[j][4] = 0.50 * ( 1.0 - C1*C5 )
      * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
              - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
      + (1.0/6.0)
      * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
      + C1 * C5 * ty3 * ( u51j - u51jm1 );
  }

  for (j = jst; j < jend; j++) {
    frct[k][j][i][0] = frct[k][j][i][0]
      + dy1 * ty1 * (        rsd[k][j-1][i][0]
                     - 2.0 * rsd[k][j][i][0]
                     +       rsd[k][j+1][i][0] );
    frct[k][j][i][1] = frct[k][j][i][1]
      + ty3 * C3 * C4 * ( flux[j+1][1] - flux[j][1] )
      + dy2 * ty1 * (        rsd[k][j-1][i][1]
                     - 2.0 * rsd[k][j][i][1]
                     +       rsd[k][j+1][i][1] );
    frct[k][j][i][2] = frct[k][j][i][2]
      + ty3 * C3 * C4 * ( flux[j+1][2] - flux[j][2] )
      + dy3 * ty1 * (        rsd[k][j-1][i][2]
                     - 2.0 * rsd[k][j][i][2]
                     +       rsd[k][j+1][i][2] );
    frct[k][j][i][3] = frct[k][j][i][3]
      + ty3 * C3 * C4 * ( flux[j+1][3] - flux[j][3] )
      + dy4 * ty1 * (        rsd[k][j-1][i][3]
                     - 2.0 * rsd[k][j][i][3]
                     +       rsd[k][j+1][i][3] );
    frct[k][j][i][4] = frct[k][j][i][4]
      + ty3 * C3 * C4 * ( flux[j+1][4] - flux[j][4] )
      + dy5 * ty1 * (        rsd[k][j-1][i][4]
                     - 2.0 * rsd[k][j][i][4]
                     +       rsd[k][j+1][i][4] );
  }

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    frct[k][1][i][m] = frct[k][1][i][m]
      - dssp * ( + 5.0 * rsd[k][1][i][m]
                 - 4.0 * rsd[k][2][i][m]
                 +       rsd[k][3][i][m] );
    frct[k][2][i][m] = frct[k][2][i][m]
      - dssp * ( - 4.0 * rsd[k][1][i][m]
                 + 6.0 * rsd[k][2][i][m]
                 - 4.0 * rsd[k][3][i][m]
                 +       rsd[k][4][i][m] );
  }

  for (j = 3; j < ny - 3; j++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] = frct[k][j][i][m]
        - dssp * (        rsd[k][j-2][i][m]
                  - 4.0 * rsd[k][j-1][i][m]
                  + 6.0 * rsd[k][j][i][m]
                  - 4.0 * rsd[k][j+1][i][m]
                  +       rsd[k][j+2][i][m] );
    }
  }

  for (m = 0; m < 5; m++) {
    frct[k][ny-3][i][m] = frct[k][ny-3][i][m]
      - dssp * (        rsd[k][ny-5][i][m]
                - 4.0 * rsd[k][ny-4][i][m]
                + 6.0 * rsd[k][ny-3][i][m]
                - 4.0 * rsd[k][ny-2][i][m] );
    frct[k][ny-2][i][m] = frct[k][ny-2][i][m]
      - dssp * (        rsd[k][ny-4][i][m]
                - 4.0 * rsd[k][ny-3][i][m]
                + 5.0 * rsd[k][ny-2][i][m] );
  }

#else //ERHS3_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = k - 1;
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (i = ist; i < iend; i++) {
    for (j = 0; j < ny; j++) {
      flux[j][0] = rsd[k][j][i][2];
      u31 = rsd[k][j][i][2] / rsd[k][j][i][0];
      q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                  + rsd[k][j][i][2] * rsd[k][j][i][2]
                  + rsd[k][j][i][3] * rsd[k][j][i][3] )
               / rsd[k][j][i][0];
      flux[j][1] = rsd[k][j][i][1] * u31;
      flux[j][2] = rsd[k][j][i][2] * u31 + C2 * ( rsd[k][j][i][4] - q );
      flux[j][3] = rsd[k][j][i][3] * u31;
      flux[j][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u31;
    }

    for (j = jst; j < jend; j++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] =  frct[k][j][i][m]
          - ty2 * ( flux[j+1][m] - flux[j-1][m] );
      }
    }

    for (j = jst; j < ny; j++) {
      tmp = 1.0 / rsd[k][j][i][0];

      u21j = tmp * rsd[k][j][i][1];
      u31j = tmp * rsd[k][j][i][2];
      u41j = tmp * rsd[k][j][i][3];
      u51j = tmp * rsd[k][j][i][4];

      tmp = 1.0 / rsd[k][j-1][i][0];

      u21jm1 = tmp * rsd[k][j-1][i][1];
      u31jm1 = tmp * rsd[k][j-1][i][2];
      u41jm1 = tmp * rsd[k][j-1][i][3];
      u51jm1 = tmp * rsd[k][j-1][i][4];

      flux[j][1] = ty3 * ( u21j - u21jm1 );
      flux[j][2] = (4.0/3.0) * ty3 * ( u31j - u31jm1 );
      flux[j][3] = ty3 * ( u41j - u41jm1 );
      flux[j][4] = 0.50 * ( 1.0 - C1*C5 )
        * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
                - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
        + (1.0/6.0)
        * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
        + C1 * C5 * ty3 * ( u51j - u51jm1 );
    }

    for (j = jst; j < jend; j++) {
      frct[k][j][i][0] = frct[k][j][i][0]
        + dy1 * ty1 * (        rsd[k][j-1][i][0]
                       - 2.0 * rsd[k][j][i][0]
                       +       rsd[k][j+1][i][0] );
      frct[k][j][i][1] = frct[k][j][i][1]
        + ty3 * C3 * C4 * ( flux[j+1][1] - flux[j][1] )
        + dy2 * ty1 * (        rsd[k][j-1][i][1]
                       - 2.0 * rsd[k][j][i][1]
                       +       rsd[k][j+1][i][1] );
      frct[k][j][i][2] = frct[k][j][i][2]
        + ty3 * C3 * C4 * ( flux[j+1][2] - flux[j][2] )
        + dy3 * ty1 * (        rsd[k][j-1][i][2]
                       - 2.0 * rsd[k][j][i][2]
                       +       rsd[k][j+1][i][2] );
      frct[k][j][i][3] = frct[k][j][i][3]
        + ty3 * C3 * C4 * ( flux[j+1][3] - flux[j][3] )
        + dy4 * ty1 * (        rsd[k][j-1][i][3]
                       - 2.0 * rsd[k][j][i][3]
                       +       rsd[k][j+1][i][3] );
      frct[k][j][i][4] = frct[k][j][i][4]
        + ty3 * C3 * C4 * ( flux[j+1][4] - flux[j][4] )
        + dy5 * ty1 * (        rsd[k][j-1][i][4]
                       - 2.0 * rsd[k][j][i][4]
                       +       rsd[k][j+1][i][4] );
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      frct[k][1][i][m] = frct[k][1][i][m]
        - dssp * ( + 5.0 * rsd[k][1][i][m]
                   - 4.0 * rsd[k][2][i][m]
                   +       rsd[k][3][i][m] );
      frct[k][2][i][m] = frct[k][2][i][m]
        - dssp * ( - 4.0 * rsd[k][1][i][m]
                   + 6.0 * rsd[k][2][i][m]
                   - 4.0 * rsd[k][3][i][m]
                   +       rsd[k][4][i][m] );
    }

    for (j = 3; j < ny - 3; j++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] = frct[k][j][i][m]
          - dssp * (        rsd[k][j-2][i][m]
                    - 4.0 * rsd[k][j-1][i][m]
                    + 6.0 * rsd[k][j][i][m]
                    - 4.0 * rsd[k][j+1][i][m]
                    +       rsd[k][j+2][i][m] );
      }
    }

    for (m = 0; m < 5; m++) {
      frct[k][ny-3][i][m] = frct[k][ny-3][i][m]
        - dssp * (        rsd[k][ny-5][i][m]
                  - 4.0 * rsd[k][ny-4][i][m]
                  + 6.0 * rsd[k][ny-3][i][m]
                  - 4.0 * rsd[k][ny-2][i][m] );
      frct[k][ny-2][i][m] = frct[k][ny-2][i][m]
        - dssp * (        rsd[k][ny-4][i][m]
                  - 4.0 * rsd[k][ny-3][i][m]
                  + 5.0 * rsd[k][ny-2][i][m] );
    }
  }
#endif
}


//---------------------------------------------------------------------
// zeta-direction flux differences
//---------------------------------------------------------------------
__kernel void erhs4(__global double *g_rsd,
                    __global double *g_frct,
                    __global double *g_flux,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  double q;
  double u41;
  double tmp;
  double u21k, u31k, u41k, u51k;
  double u21km1, u31km1, u41km1, u51km1;

#if ERHS4_DIM == 2
  j = get_global_id(1) + jst;
  i = get_global_id(0) + ist;
  if (j >= jend || i >= iend) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = (j-jst)*ISIZ2 + (i-ist);
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (k = 0; k < nz; k++) {
    flux[k][0] = rsd[k][j][i][3];
    u41 = rsd[k][j][i][3] / rsd[k][j][i][0];
    q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                + rsd[k][j][i][2] * rsd[k][j][i][2]
                + rsd[k][j][i][3] * rsd[k][j][i][3] )
             / rsd[k][j][i][0];
    flux[k][1] = rsd[k][j][i][1] * u41;
    flux[k][2] = rsd[k][j][i][2] * u41; 
    flux[k][3] = rsd[k][j][i][3] * u41 + C2 * ( rsd[k][j][i][4] - q );
    flux[k][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u41;
  }

  for (k = 1; k < nz - 1; k++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] =  frct[k][j][i][m]
        - tz2 * ( flux[k+1][m] - flux[k-1][m] );
    }
  }

  for (k = 1; k < nz; k++) {
    tmp = 1.0 / rsd[k][j][i][0];

    u21k = tmp * rsd[k][j][i][1];
    u31k = tmp * rsd[k][j][i][2];
    u41k = tmp * rsd[k][j][i][3];
    u51k = tmp * rsd[k][j][i][4];

    tmp = 1.0 / rsd[k-1][j][i][0];

    u21km1 = tmp * rsd[k-1][j][i][1];
    u31km1 = tmp * rsd[k-1][j][i][2];
    u41km1 = tmp * rsd[k-1][j][i][3];
    u51km1 = tmp * rsd[k-1][j][i][4];

    flux[k][1] = tz3 * ( u21k - u21km1 );
    flux[k][2] = tz3 * ( u31k - u31km1 );
    flux[k][3] = (4.0/3.0) * tz3 * ( u41k - u41km1 );
    flux[k][4] = 0.50 * ( 1.0 - C1*C5 )
      * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
              - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
      + (1.0/6.0)
      * tz3 * ( u41k*u41k - u41km1*u41km1 )
      + C1 * C5 * tz3 * ( u51k - u51km1 );
  }

  for (k = 1; k < nz - 1; k++) {
    frct[k][j][i][0] = frct[k][j][i][0]
      + dz1 * tz1 * (        rsd[k+1][j][i][0]
                     - 2.0 * rsd[k][j][i][0]
                     +       rsd[k-1][j][i][0] );
    frct[k][j][i][1] = frct[k][j][i][1]
      + tz3 * C3 * C4 * ( flux[k+1][1] - flux[k][1] )
      + dz2 * tz1 * (        rsd[k+1][j][i][1]
                     - 2.0 * rsd[k][j][i][1]
                     +       rsd[k-1][j][i][1] );
    frct[k][j][i][2] = frct[k][j][i][2]
      + tz3 * C3 * C4 * ( flux[k+1][2] - flux[k][2] )
      + dz3 * tz1 * (        rsd[k+1][j][i][2]
                     - 2.0 * rsd[k][j][i][2]
                     +       rsd[k-1][j][i][2] );
    frct[k][j][i][3] = frct[k][j][i][3]
      + tz3 * C3 * C4 * ( flux[k+1][3] - flux[k][3] )
      + dz4 * tz1 * (        rsd[k+1][j][i][3]
                     - 2.0 * rsd[k][j][i][3]
                     +       rsd[k-1][j][i][3] );
    frct[k][j][i][4] = frct[k][j][i][4]
      + tz3 * C3 * C4 * ( flux[k+1][4] - flux[k][4] )
      + dz5 * tz1 * (        rsd[k+1][j][i][4]
                     - 2.0 * rsd[k][j][i][4]
                     +       rsd[k-1][j][i][4] );
  }

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    frct[1][j][i][m] = frct[1][j][i][m]
      - dssp * ( + 5.0 * rsd[1][j][i][m]
                 - 4.0 * rsd[2][j][i][m]
                 +       rsd[3][j][i][m] );
    frct[2][j][i][m] = frct[2][j][i][m]
      - dssp * ( - 4.0 * rsd[1][j][i][m]
                 + 6.0 * rsd[2][j][i][m]
                 - 4.0 * rsd[3][j][i][m]
                 +       rsd[4][j][i][m] );
  }

  for (k = 3; k < nz - 3; k++) {
    for (m = 0; m < 5; m++) {
      frct[k][j][i][m] = frct[k][j][i][m]
        - dssp * (        rsd[k-2][j][i][m]
                  - 4.0 * rsd[k-1][j][i][m]
                  + 6.0 * rsd[k][j][i][m]
                  - 4.0 * rsd[k+1][j][i][m]
                  +       rsd[k+2][j][i][m] );
    }
  }

  for (m = 0; m < 5; m++) {
    frct[nz-3][j][i][m] = frct[nz-3][j][i][m]
      - dssp * (        rsd[nz-5][j][i][m]
                - 4.0 * rsd[nz-4][j][i][m]
                + 6.0 * rsd[nz-3][j][i][m]
                - 4.0 * rsd[nz-2][j][i][m] );
    frct[nz-2][j][i][m] = frct[nz-2][j][i][m]
      - dssp * (        rsd[nz-4][j][i][m]
                - 4.0 * rsd[nz-3][j][i][m]
                + 5.0 * rsd[nz-2][j][i][m] );
  }

#else //ERHS4_DIM == 1
  j = get_global_id(0) + jst;
  if (j >= jend) return;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;

  int my_id = j - jst;
  int my_offset = my_id * ISIZ1*5;
  __global double (*flux)[5] = (__global double (*)[5])&g_flux[my_offset];

  for (i = ist; i < iend; i++) {
    for (k = 0; k < nz; k++) {
      flux[k][0] = rsd[k][j][i][3];
      u41 = rsd[k][j][i][3] / rsd[k][j][i][0];
      q = 0.50 * (  rsd[k][j][i][1] * rsd[k][j][i][1]
                  + rsd[k][j][i][2] * rsd[k][j][i][2]
                  + rsd[k][j][i][3] * rsd[k][j][i][3] )
               / rsd[k][j][i][0];
      flux[k][1] = rsd[k][j][i][1] * u41;
      flux[k][2] = rsd[k][j][i][2] * u41; 
      flux[k][3] = rsd[k][j][i][3] * u41 + C2 * ( rsd[k][j][i][4] - q );
      flux[k][4] = ( C1 * rsd[k][j][i][4] - C2 * q ) * u41;
    }

    for (k = 1; k < nz - 1; k++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] =  frct[k][j][i][m]
          - tz2 * ( flux[k+1][m] - flux[k-1][m] );
      }
    }

    for (k = 1; k < nz; k++) {
      tmp = 1.0 / rsd[k][j][i][0];

      u21k = tmp * rsd[k][j][i][1];
      u31k = tmp * rsd[k][j][i][2];
      u41k = tmp * rsd[k][j][i][3];
      u51k = tmp * rsd[k][j][i][4];

      tmp = 1.0 / rsd[k-1][j][i][0];

      u21km1 = tmp * rsd[k-1][j][i][1];
      u31km1 = tmp * rsd[k-1][j][i][2];
      u41km1 = tmp * rsd[k-1][j][i][3];
      u51km1 = tmp * rsd[k-1][j][i][4];

      flux[k][1] = tz3 * ( u21k - u21km1 );
      flux[k][2] = tz3 * ( u31k - u31km1 );
      flux[k][3] = (4.0/3.0) * tz3 * ( u41k - u41km1 );
      flux[k][4] = 0.50 * ( 1.0 - C1*C5 )
        * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
                - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
        + (1.0/6.0)
        * tz3 * ( u41k*u41k - u41km1*u41km1 )
        + C1 * C5 * tz3 * ( u51k - u51km1 );
    }

    for (k = 1; k < nz - 1; k++) {
      frct[k][j][i][0] = frct[k][j][i][0]
        + dz1 * tz1 * (        rsd[k+1][j][i][0]
                       - 2.0 * rsd[k][j][i][0]
                       +       rsd[k-1][j][i][0] );
      frct[k][j][i][1] = frct[k][j][i][1]
        + tz3 * C3 * C4 * ( flux[k+1][1] - flux[k][1] )
        + dz2 * tz1 * (        rsd[k+1][j][i][1]
                       - 2.0 * rsd[k][j][i][1]
                       +       rsd[k-1][j][i][1] );
      frct[k][j][i][2] = frct[k][j][i][2]
        + tz3 * C3 * C4 * ( flux[k+1][2] - flux[k][2] )
        + dz3 * tz1 * (        rsd[k+1][j][i][2]
                       - 2.0 * rsd[k][j][i][2]
                       +       rsd[k-1][j][i][2] );
      frct[k][j][i][3] = frct[k][j][i][3]
        + tz3 * C3 * C4 * ( flux[k+1][3] - flux[k][3] )
        + dz4 * tz1 * (        rsd[k+1][j][i][3]
                       - 2.0 * rsd[k][j][i][3]
                       +       rsd[k-1][j][i][3] );
      frct[k][j][i][4] = frct[k][j][i][4]
        + tz3 * C3 * C4 * ( flux[k+1][4] - flux[k][4] )
        + dz5 * tz1 * (        rsd[k+1][j][i][4]
                       - 2.0 * rsd[k][j][i][4]
                       +       rsd[k-1][j][i][4] );
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      frct[1][j][i][m] = frct[1][j][i][m]
        - dssp * ( + 5.0 * rsd[1][j][i][m]
                   - 4.0 * rsd[2][j][i][m]
                   +       rsd[3][j][i][m] );
      frct[2][j][i][m] = frct[2][j][i][m]
        - dssp * ( - 4.0 * rsd[1][j][i][m]
                   + 6.0 * rsd[2][j][i][m]
                   - 4.0 * rsd[3][j][i][m]
                   +       rsd[4][j][i][m] );
    }

    for (k = 3; k < nz - 3; k++) {
      for (m = 0; m < 5; m++) {
        frct[k][j][i][m] = frct[k][j][i][m]
          - dssp * (        rsd[k-2][j][i][m]
                    - 4.0 * rsd[k-1][j][i][m]
                    + 6.0 * rsd[k][j][i][m]
                    - 4.0 * rsd[k+1][j][i][m]
                    +       rsd[k+2][j][i][m] );
      }
    }

    for (m = 0; m < 5; m++) {
      frct[nz-3][j][i][m] = frct[nz-3][j][i][m]
        - dssp * (        rsd[nz-5][j][i][m]
                  - 4.0 * rsd[nz-4][j][i][m]
                  + 6.0 * rsd[nz-3][j][i][m]
                  - 4.0 * rsd[nz-2][j][i][m] );
      frct[nz-2][j][i][m] = frct[nz-2][j][i][m]
        - dssp * (        rsd[nz-4][j][i][m]
                  - 4.0 * rsd[nz-3][j][i][m]
                  + 5.0 * rsd[nz-2][j][i][m] );
    }
  }
#endif
}

