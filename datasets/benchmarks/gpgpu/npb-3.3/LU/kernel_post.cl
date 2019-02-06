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


__kernel void error(__global double *g_u,
                    __global double *g_ce,
                    __global double *g_errnm,
                    __local double *l_errnm,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m, lid;
  double tmp;
  double u000ijk[5];
  __local double *errnm_local;

  k = get_global_id(0) + 1;
  lid = get_local_id(0);
  errnm_local = &l_errnm[lid * 5];

  for (m = 0; m < 5; m++) {
    errnm_local[m] = 0.0;
  }

  if (k < nz-1) {
    __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
      (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;

    for (j = jst; j < jend; j++) {
      for (i = ist; i < iend; i++) {
        exact( i, j, k, u000ijk, g_ce );
        for (m = 0; m < 5; m++) {
          tmp = ( u000ijk[m] - u[k][j][i][m] );
          errnm_local[m] = errnm_local[m] + tmp * tmp;
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < get_local_size(0); i++) {
      __local double *errnm_other = &l_errnm[i * 5];
      for (m = 0; m < 5; m++) {
        errnm_local[m] += errnm_other[m];
      }
    }

    __global double *errnm = &g_errnm[get_group_id(0) * 5];
    for (m = 0; m < 5; m++) {
      errnm[m] = errnm_local[m];
    }
  }
}


__kernel void pintgr1(__global double *g_u,
                      __global double *g_phi1,
                      __global double *g_phi2,
                      int ibeg,
                      int ifin,
                      int jbeg,
                      int jfin,
                      int ki1,
                      int ki2)
{
  int i, j, k;

#if PINTGR1_DIM == 2
  j = get_global_id(1) + jbeg;
  i = get_global_id(0) + ibeg;
  if (j >= jfin || i >= ifin) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  k = ki1;

  phi1[j][i] = C2*(  u[k][j][i][4]
      - 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                + u[k][j][i][2] * u[k][j][i][2]
                + u[k][j][i][3] * u[k][j][i][3] )
               / u[k][j][i][0] );

  k = ki2 - 1;

  phi2[j][i] = C2*(  u[k][j][i][4]
      - 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                + u[k][j][i][2] * u[k][j][i][2]
                + u[k][j][i][3] * u[k][j][i][3] )
               / u[k][j][i][0] );

#else //PINTGR1_DIM == 1
  j = get_global_id(0) + jbeg;
  if (j >= jfin) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  for (i = ibeg; i < ifin; i++) {
    k = ki1;

    phi1[j][i] = C2*(  u[k][j][i][4]
        - 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                  + u[k][j][i][2] * u[k][j][i][2]
                  + u[k][j][i][3] * u[k][j][i][3] )
                 / u[k][j][i][0] );

    k = ki2 - 1;

    phi2[j][i] = C2*(  u[k][j][i][4]
        - 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                  + u[k][j][i][2] * u[k][j][i][2]
                  + u[k][j][i][3] * u[k][j][i][3] )
                 / u[k][j][i][0] );
  }
#endif
}


__kernel void pintgr2(__global double *g_u,
                      __global double *g_phi1,
                      __global double *g_phi2,
                      int ibeg,
                      int ifin,
                      int jbeg,
                      int jfin,
                      int ki1,
                      int ki2)
{
  int i, k;

#if PINTGR2_DIM == 2
  k = get_global_id(1) + ki1;
  i = get_global_id(0) + ibeg;
  if (k >= ki2 || i >= ifin) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  phi1[k][i] = C2*(  u[k][jbeg][i][4]
      - 0.50 * (  u[k][jbeg][i][1] * u[k][jbeg][i][1]
                + u[k][jbeg][i][2] * u[k][jbeg][i][2]
                + u[k][jbeg][i][3] * u[k][jbeg][i][3] )
               / u[k][jbeg][i][0] );

  phi2[k][i] = C2*(  u[k][jfin-1][i][4]
      - 0.50 * (  u[k][jfin-1][i][1] * u[k][jfin-1][i][1]
                + u[k][jfin-1][i][2] * u[k][jfin-1][i][2]
                + u[k][jfin-1][i][3] * u[k][jfin-1][i][3] )
               / u[k][jfin-1][i][0] );

#else //PINTGR2_DIM == 1
  k = get_global_id(0) + ki1;
  if (k >= ki2) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  for (i = ibeg; i < ifin; i++) {
    phi1[k][i] = C2*(  u[k][jbeg][i][4]
        - 0.50 * (  u[k][jbeg][i][1] * u[k][jbeg][i][1]
                  + u[k][jbeg][i][2] * u[k][jbeg][i][2]
                  + u[k][jbeg][i][3] * u[k][jbeg][i][3] )
                 / u[k][jbeg][i][0] );

    phi2[k][i] = C2*(  u[k][jfin-1][i][4]
        - 0.50 * (  u[k][jfin-1][i][1] * u[k][jfin-1][i][1]
                  + u[k][jfin-1][i][2] * u[k][jfin-1][i][2]
                  + u[k][jfin-1][i][3] * u[k][jfin-1][i][3] )
                 / u[k][jfin-1][i][0] );
  }
#endif
}


__kernel void pintgr3(__global double *g_u,
                      __global double *g_phi1,
                      __global double *g_phi2,
                      int ibeg,
                      int ifin,
                      int jbeg,
                      int jfin,
                      int ki1,
                      int ki2)
{
  int j, k;

#if PINTGR3_DIM == 2
  k = get_global_id(1) + ki1;
  j = get_global_id(0) + jbeg;
  if (k >= ki2 || j >= jfin) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  phi1[k][j] = C2*(  u[k][j][ibeg][4]
      - 0.50 * (  u[k][j][ibeg][1] * u[k][j][ibeg][1]
                + u[k][j][ibeg][2] * u[k][j][ibeg][2]
                + u[k][j][ibeg][3] * u[k][j][ibeg][3] )
               / u[k][j][ibeg][0] );

  phi2[k][j] = C2*(  u[k][j][ifin-1][4]
      - 0.50 * (  u[k][j][ifin-1][1] * u[k][j][ifin-1][1]
                + u[k][j][ifin-1][2] * u[k][j][ifin-1][2]
                + u[k][j][ifin-1][3] * u[k][j][ifin-1][3] )
               / u[k][j][ifin-1][0] );

#else //PINTGR3_DIM == 1
  k = get_global_id(0) + ki1;
  if (k >= ki2) return;

  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] = 
    (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
  __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;

  for (j = jbeg; j < jfin; j++) {
    phi1[k][j] = C2*(  u[k][j][ibeg][4]
        - 0.50 * (  u[k][j][ibeg][1] * u[k][j][ibeg][1]
                  + u[k][j][ibeg][2] * u[k][j][ibeg][2]
                  + u[k][j][ibeg][3] * u[k][j][ibeg][3] )
                 / u[k][j][ibeg][0] );

    phi2[k][j] = C2*(  u[k][j][ifin-1][4]
        - 0.50 * (  u[k][j][ifin-1][1] * u[k][j][ifin-1][1]
                  + u[k][j][ifin-1][2] * u[k][j][ifin-1][2]
                  + u[k][j][ifin-1][3] * u[k][j][ifin-1][3] )
                 / u[k][j][ifin-1][0] );
  }
#endif
}


__kernel void pintgr_reduce(__global double *g_phi1,
                            __global double *g_phi2,
                            __global double *g_frc,
                            __local double *l_frc,
                            int ibeg,
                            int ifin1,
                            int jbeg,
                            int jfin1)
{
  int i, j, lid;
  double my_frc = 0.0;

  j = get_global_id(0) + jbeg;
  lid = get_local_id(0);

  if (j < jfin1) {
    __global double (*phi1)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi1;
    __global double (*phi2)[ISIZ2+2] = (__global double (*)[ISIZ2+2])g_phi2;
    
    for (i = ibeg; i < ifin1; i++) {
      my_frc = my_frc + (  phi1[j][i]
                         + phi1[j][i+1]
                         + phi1[j+1][i]
                         + phi1[j+1][i+1]
                         + phi2[j][i]
                         + phi2[j][i+1]
                         + phi2[j+1][i]
                         + phi2[j+1][i+1] );
    }
  }
  l_frc[lid] = my_frc;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < get_local_size(0); i++) {
      my_frc += l_frc[i];
    }

    g_frc[get_group_id(0)] = my_frc;
  }
}


