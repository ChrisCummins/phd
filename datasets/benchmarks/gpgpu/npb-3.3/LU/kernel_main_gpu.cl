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

__kernel void l2norm(__global double *g_v,
                     __global double *g_sum,
                     __local double *l_sum,
                     int l_nz0,
                     int l_ist,
                     int l_iend,
                     int l_jst,
                     int l_jend)
{
  int i, j, k, m, lid;
  __global double (*v)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __local double *sum_local;

  k = get_global_id(0) + 1;
  lid = get_local_id(0);
  sum_local = &l_sum[lid * 5];

  for (m = 0; m < 5; m++) {
    sum_local[m] = 0.0;
  }

  if (k < nz0-1) {
    v = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_v;

    for (j = l_jst; j < l_jend; j++) {
      for (i = l_ist; i < l_iend; i++) {
        for (m = 0; m < 5; m++) {
          sum_local[m] = sum_local[m] + v[k][j][i][m] * v[k][j][i][m];
        }
      }
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < get_local_size(0); i++) {
      __local double *sum_other = &l_sum[i * 5];
      for (m = 0; m < 5; m++) {
        sum_local[m] += sum_other[m];
      }
    }

    __global double *sum = &g_sum[get_group_id(0) * 5];
    for (m = 0; m < 5; m++) {
      sum[m] = sum_local[m];
    }
  }
}


__kernel void rhs(__global double *g_u,
                  __global double *g_rsd,
                  __global double *g_frct,
                  __global double *g_qs,
                  __global double *g_rho_i,
                  int nx,
                  int ny,
                  int nz)
{
  int i, j, k, m;
  double tmp;
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

#if RHS_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k >= nz || j >= ny || i >= nx) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  frct  = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  for (m = 0; m < 5; m++) {
    rsd[k][j][i][m] = - frct[k][j][i][m];
  }
  tmp = 1.0 / u[k][j][i][0];
  rho_i[k][j][i] = tmp;
  qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                        + u[k][j][i][2] * u[k][j][i][2]
                        + u[k][j][i][3] * u[k][j][i][3] )
                     * tmp;

#elif RHS_DIM == 2
  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= nz || j >= ny) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  frct  = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  for (i = 0; i < nx; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = - frct[k][j][i][m];
    }
    tmp = 1.0 / u[k][j][i][0];
    rho_i[k][j][i] = tmp;
    qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                          + u[k][j][i][2] * u[k][j][i][2]
                          + u[k][j][i][3] * u[k][j][i][3] )
                       * tmp;
  }

#else //RHS_DIM == 1
  k = get_global_id(0);
  if (k >= nz) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  frct  = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_frct;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = - frct[k][j][i][m];
      }
      tmp = 1.0 / u[k][j][i][0];
      rho_i[k][j][i] = tmp;
      qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
                            + u[k][j][i][2] * u[k][j][i][2]
                            + u[k][j][i][3] * u[k][j][i][3] )
                         * tmp;
    }
  }
#endif
}


__kernel void rhsx(__global double *g_u,
                   __global double *g_rsd,
                   __global double *g_qs,
                   __global double *g_rho_i,
                   int nx,
                   int ny,
                   int nz)
{
  int i, j, k, i1, i2, m;
  double q;
  double tmp;
  double u21;
  double u21i, u31i, u41i, u51i;
  double u21im1, u31im1, u41im1, u51im1;
  double flux1[3][5], flux2[2][5];
  double p_u[5][5], p_rsd[5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

#if RHSX_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + jst;
  if (k >= (nz-1) || j >= jend) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

#define LOAD_P_U(p_idx, g_idx) \
  p_u[p_idx][0] = u[k][j][g_idx][0]; \
  p_u[p_idx][1] = u[k][j][g_idx][1]; \
  p_u[p_idx][2] = u[k][j][g_idx][2]; \
  p_u[p_idx][3] = u[k][j][g_idx][3]; \
  p_u[p_idx][4] = u[k][j][g_idx][4];

#define COMPUTE_FLUX_1(idx) \
  flux1[idx][0] = p_u[3][1]; \
  u21 = p_u[3][1] * rho_i[k][j][i1]; \
  q = qs[k][j][i1]; \
  flux1[idx][1] = p_u[3][1] * u21 + C2 * ( p_u[3][4] - q ); \
  flux1[idx][2] = p_u[3][2] * u21; \
  flux1[idx][3] = p_u[3][3] * u21; \
  flux1[idx][4] = ( C1 * p_u[3][4] - C2 * q ) * u21;

#define COMPUTE_FLUX_2(idx) \
  tmp = rho_i[k][j][i1]; \
  u21i = tmp * p_u[3][1]; \
  u31i = tmp * p_u[3][2]; \
  u41i = tmp * p_u[3][3]; \
  u51i = tmp * p_u[3][4]; \
  tmp = rho_i[k][j][i]; \
  u21im1 = tmp * p_u[2][1]; \
  u31im1 = tmp * p_u[2][2]; \
  u41im1 = tmp * p_u[2][3]; \
  u51im1 = tmp * p_u[2][4]; \
  flux2[idx][1] = (4.0/3.0) * tx3 * (u21i-u21im1); \
  flux2[idx][2] = tx3 * ( u31i - u31im1 ); \
  flux2[idx][3] = tx3 * ( u41i - u41im1 ); \
  flux2[idx][4] = 0.50 * ( 1.0 - C1*C5 ) \
    * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i ) \
            - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) ) \
    + (1.0/6.0) \
    * tx3 * ( u21i*u21i - u21im1*u21im1 ) \
    + C1 * C5 * tx3 * ( u51i - u51im1 );

#define LOOP_PROLOGUE_FULL \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  LOAD_P_U(4, i2) \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_PROLOGUE_HALF \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_BODY \
  for (m = 0; m < 5; m++) { \
    p_rsd[m] =  rsd[k][j][i][m] \
      - tx2 * ( flux1[2][m] - flux1[0][m] ); \
  } \
  p_rsd[0] = p_rsd[0] \
    + dx1 * tx1 * (        p_u[1][0] \
                   - 2.0 * p_u[2][0] \
                   +       p_u[3][0] ); \
  p_rsd[1] = p_rsd[1] \
    + tx3 * C3 * C4 * ( flux2[1][1] - flux2[0][1] ) \
    + dx2 * tx1 * (        p_u[1][1] \
                   - 2.0 * p_u[2][1] \
                   +       p_u[3][1] ); \
  p_rsd[2] = p_rsd[2] \
    + tx3 * C3 * C4 * ( flux2[1][2] - flux2[0][2] ) \
    + dx3 * tx1 * (        p_u[1][2] \
                   - 2.0 * p_u[2][2] \
                   +       p_u[3][2] ); \
  p_rsd[3] = p_rsd[3] \
    + tx3 * C3 * C4 * ( flux2[1][3] - flux2[0][3] ) \
    + dx4 * tx1 * (        p_u[1][3] \
                   - 2.0 * p_u[2][3] \
                   +       p_u[3][3] ); \
  p_rsd[4] = p_rsd[4] \
    + tx3 * C3 * C4 * ( flux2[1][4] - flux2[0][4] ) \
    + dx5 * tx1 * (        p_u[1][4] \
                   - 2.0 * p_u[2][4] \
                   +       p_u[3][4] );

  i1 = 0;
  LOAD_P_U(3, 0)
  COMPUTE_FLUX_1(1)

  i = 0; i1 = ist; // ist == 1
  for (m = 0; m < 5; m++) p_u[2][m] = p_u[3][m]; // LOAD_P_U(2, 0)
  LOAD_P_U(3, 1)
  COMPUTE_FLUX_1(2)
  COMPUTE_FLUX_2(1)
  LOAD_P_U(4, 2)

  //---------------------------------------------------------------------
  // Fourth-order dissipation
  //---------------------------------------------------------------------

  i = 1; i1 = 2; i2 = 3;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][1][m] = p_rsd[m]
        - dssp * ( + 5.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }

  i = 2; i1 = 3; i2 = 4;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][2][m] = p_rsd[m]
        - dssp * ( - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }

  for (i = 3; i < nx - 3; i++) {
    i1 = i + 1; i2 = i + 2;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }
  }

  i = nx-3; i1 = nx-2; i2 = nx-1;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][nx-3][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m] );
    }

  i = nx-2; i1 = nx-1;
    LOOP_PROLOGUE_HALF
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][nx-2][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 5.0 * p_u[2][m] );
    }

#undef LOAD_P_U
#undef COMPUTE_FLUX_1
#undef COMPUTE_FLUX_2
#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY

#else //RHSX_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  int my_id = k - 1;
  int my_offset = my_id * ISIZ1*5;
  flux = (__global double (*)[5])&g_flux[my_offset];

  for (j = jst; j < jend; j++) {
    for (i = 0; i < nx; i++) {
      flux[i][0] = u[k][j][i][1];
      u21 = u[k][j][i][1] * rho_i[k][j][i];

      q = qs[k][j][i];

      flux[i][1] = u[k][j][i][1] * u21 + C2 * ( u[k][j][i][4] - q );
      flux[i][2] = u[k][j][i][2] * u21;
      flux[i][3] = u[k][j][i][3] * u21;
      flux[i][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u21;
    }

    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] =  rsd[k][j][i][m]
          - tx2 * ( flux[i+1][m] - flux[i-1][m] );
      }
    }

    for (i = ist; i < nx; i++) {
      tmp = rho_i[k][j][i];

      u21i = tmp * u[k][j][i][1];
      u31i = tmp * u[k][j][i][2];
      u41i = tmp * u[k][j][i][3];
      u51i = tmp * u[k][j][i][4];

      tmp = rho_i[k][j][i-1];

      u21im1 = tmp * u[k][j][i-1][1];
      u31im1 = tmp * u[k][j][i-1][2];
      u41im1 = tmp * u[k][j][i-1][3];
      u51im1 = tmp * u[k][j][i-1][4];

      flux[i][1] = (4.0/3.0) * tx3 * (u21i-u21im1);
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
      rsd[k][j][i][0] = rsd[k][j][i][0]
        + dx1 * tx1 * (        u[k][j][i-1][0]
                       - 2.0 * u[k][j][i][0]
                       +       u[k][j][i+1][0] );
      rsd[k][j][i][1] = rsd[k][j][i][1]
        + tx3 * C3 * C4 * ( flux[i+1][1] - flux[i][1] )
        + dx2 * tx1 * (        u[k][j][i-1][1]
                       - 2.0 * u[k][j][i][1]
                       +       u[k][j][i+1][1] );
      rsd[k][j][i][2] = rsd[k][j][i][2]
        + tx3 * C3 * C4 * ( flux[i+1][2] - flux[i][2] )
        + dx3 * tx1 * (        u[k][j][i-1][2]
                       - 2.0 * u[k][j][i][2]
                       +       u[k][j][i+1][2] );
      rsd[k][j][i][3] = rsd[k][j][i][3]
        + tx3 * C3 * C4 * ( flux[i+1][3] - flux[i][3] )
        + dx4 * tx1 * (        u[k][j][i-1][3]
                       - 2.0 * u[k][j][i][3]
                       +       u[k][j][i+1][3] );
      rsd[k][j][i][4] = rsd[k][j][i][4]
        + tx3 * C3 * C4 * ( flux[i+1][4] - flux[i][4] )
        + dx5 * tx1 * (        u[k][j][i-1][4]
                       - 2.0 * u[k][j][i][4]
                       +       u[k][j][i+1][4] );
    }

    //---------------------------------------------------------------------
    // Fourth-order dissipation
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      rsd[k][j][1][m] = rsd[k][j][1][m]
        - dssp * ( + 5.0 * u[k][j][1][m]
                   - 4.0 * u[k][j][2][m]
                   +       u[k][j][3][m] );
      rsd[k][j][2][m] = rsd[k][j][2][m]
        - dssp * ( - 4.0 * u[k][j][1][m]
                   + 6.0 * u[k][j][2][m]
                   - 4.0 * u[k][j][3][m]
                   +       u[k][j][4][m] );
    }

    for (i = 3; i < nx - 3; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rsd[k][j][i][m]
          - dssp * (         u[k][j][i-2][m]
                     - 4.0 * u[k][j][i-1][m]
                     + 6.0 * u[k][j][i][m]
                     - 4.0 * u[k][j][i+1][m]
                     +       u[k][j][i+2][m] );
      }
    }

    for (m = 0; m < 5; m++) {
      rsd[k][j][nx-3][m] = rsd[k][j][nx-3][m]
        - dssp * (         u[k][j][nx-5][m]
                   - 4.0 * u[k][j][nx-4][m]
                   + 6.0 * u[k][j][nx-3][m]
                   - 4.0 * u[k][j][nx-2][m] );
      rsd[k][j][nx-2][m] = rsd[k][j][nx-2][m]
        - dssp * (         u[k][j][nx-4][m]
                   - 4.0 * u[k][j][nx-3][m]
                   + 5.0 * u[k][j][nx-2][m] );
    }
  }
#endif
}


__kernel void rhsy(__global double *g_u,
                   __global double *g_rsd,
                   __global double *g_qs,
                   __global double *g_rho_i,
                   int nx,
                   int ny,
                   int nz)
{
  int i, j, k, j1, j2, m;
  double q;
  double tmp;
  double u31;
  double u21j, u31j, u41j, u51j;
  double u21jm1, u31jm1, u41jm1, u51jm1;
  double flux1[3][5], flux2[2][5];
  double p_u[5][5], p_rsd[5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

#if RHSY_DIM == 2
  k = get_global_id(1) + 1;
  i = get_global_id(0) + ist;
  if (k >= (nz-1) || i >= iend) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

#define LOAD_P_U(p_idx, g_idx) \
  p_u[p_idx][0] = u[k][g_idx][i][0]; \
  p_u[p_idx][1] = u[k][g_idx][i][1]; \
  p_u[p_idx][2] = u[k][g_idx][i][2]; \
  p_u[p_idx][3] = u[k][g_idx][i][3]; \
  p_u[p_idx][4] = u[k][g_idx][i][4];

#define COMPUTE_FLUX_1(idx) \
  flux1[idx][0] = p_u[3][2]; \
  u31 = p_u[3][2] * rho_i[k][j1][i]; \
  q = qs[k][j1][i]; \
  flux1[idx][1] = p_u[3][1] * u31; \
  flux1[idx][2] = p_u[3][2] * u31 + C2 * (p_u[3][4]-q); \
  flux1[idx][3] = p_u[3][3] * u31; \
  flux1[idx][4] = ( C1 * p_u[3][4] - C2 * q ) * u31;

#define COMPUTE_FLUX_2(idx) \
  tmp = rho_i[k][j1][i]; \
  u21j = tmp * p_u[3][1]; \
  u31j = tmp * p_u[3][2]; \
  u41j = tmp * p_u[3][3]; \
  u51j = tmp * p_u[3][4]; \
  tmp = rho_i[k][j][i]; \
  u21jm1 = tmp * p_u[2][1]; \
  u31jm1 = tmp * p_u[2][2]; \
  u41jm1 = tmp * p_u[2][3]; \
  u51jm1 = tmp * p_u[2][4]; \
  flux2[idx][1] = ty3 * ( u21j - u21jm1 ); \
  flux2[idx][2] = (4.0/3.0) * ty3 * (u31j-u31jm1); \
  flux2[idx][3] = ty3 * ( u41j - u41jm1 ); \
  flux2[idx][4] = 0.50 * ( 1.0 - C1*C5 ) \
    * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j ) \
            - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) ) \
    + (1.0/6.0) \
    * ty3 * ( u31j*u31j - u31jm1*u31jm1 ) \
    + C1 * C5 * ty3 * ( u51j - u51jm1 );

#define LOOP_PROLOGUE_FULL \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  LOAD_P_U(4, j2) \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_PROLOGUE_HALF \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_BODY \
  for (m = 0; m < 5; m++) { \
    p_rsd[m] =  rsd[k][j][i][m] \
      - ty2 * ( flux1[2][m] - flux1[0][m] ); \
  } \
  p_rsd[0] = p_rsd[0] \
    + dy1 * ty1 * (         p_u[1][0] \
                    - 2.0 * p_u[2][0] \
                    +       p_u[3][0] ); \
  p_rsd[1] = p_rsd[1] \
    + ty3 * C3 * C4 * ( flux2[1][1] - flux2[0][1] ) \
    + dy2 * ty1 * (         p_u[1][1] \
                    - 2.0 * p_u[2][1] \
                    +       p_u[3][1] ); \
  p_rsd[2] = p_rsd[2] \
    + ty3 * C3 * C4 * ( flux2[1][2] - flux2[0][2] ) \
    + dy3 * ty1 * (         p_u[1][2] \
                    - 2.0 * p_u[2][2] \
                    +       p_u[3][2] ); \
  p_rsd[3] = p_rsd[3] \
    + ty3 * C3 * C4 * ( flux2[1][3] - flux2[0][3] ) \
    + dy4 * ty1 * (         p_u[1][3] \
                    - 2.0 * p_u[2][3] \
                    +       p_u[3][3] ); \
  p_rsd[4] = p_rsd[4] \
    + ty3 * C3 * C4 * ( flux2[1][4] - flux2[0][4] ) \
    + dy5 * ty1 * (         p_u[1][4] \
                    - 2.0 * p_u[2][4] \
                    +       p_u[3][4] );

  j1 = 0;
  LOAD_P_U(3, 0)
  COMPUTE_FLUX_1(1)

  j = 0; j1 = jst; // jst == 1
  for (m = 0; m < 5; m++) p_u[2][m] = p_u[3][m]; // LOAD_P_U(2, 0)
  LOAD_P_U(3, 1)
  COMPUTE_FLUX_1(2)
  COMPUTE_FLUX_2(1)
  LOAD_P_U(4, 2)

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  j = 1; j1 = 2; j2 = 3;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][1][i][m] = p_rsd[m]
        - dssp * ( + 5.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }

  j = 2; j1 = 3; j2 = 4;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][2][i][m] = p_rsd[m]
        - dssp * ( - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }

  for (j = 3; j < ny - 3; j++) {
    j1 = j + 1; j2 = j + 2;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }
  }

  j = ny-3; j1 = ny-2; j2 = ny-1;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][ny-3][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m] );
    }

  j = ny-2; j1 = ny-1;
    LOOP_PROLOGUE_HALF
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][ny-2][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 5.0 * p_u[2][m] );
    }

#undef LOAD_P_U
#undef COMPUTE_FLUX_1
#undef COMPUTE_FLUX_2
#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY

#else //RHSY_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  int my_id = k - 1;
  int my_offset = my_id * ISIZ1*5;
  flux = (__global double (*)[5])&g_flux[my_offset];

  for (i = ist; i < iend; i++) {
    for (j = 0; j < ny; j++) {
      flux[j][0] = u[k][j][i][2];
      u31 = u[k][j][i][2] * rho_i[k][j][i];

      q = qs[k][j][i];

      flux[j][1] = u[k][j][i][1] * u31;
      flux[j][2] = u[k][j][i][2] * u31 + C2 * (u[k][j][i][4]-q);
      flux[j][3] = u[k][j][i][3] * u31;
      flux[j][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u31;
    }

    for (j = jst; j < jend; j++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] =  rsd[k][j][i][m]
          - ty2 * ( flux[j+1][m] - flux[j-1][m] );
      }
    }

    for (j = jst; j < ny; j++) {
      tmp = rho_i[k][j][i];

      u21j = tmp * u[k][j][i][1];
      u31j = tmp * u[k][j][i][2];
      u41j = tmp * u[k][j][i][3];
      u51j = tmp * u[k][j][i][4];

      tmp = rho_i[k][j-1][i];
      u21jm1 = tmp * u[k][j-1][i][1];
      u31jm1 = tmp * u[k][j-1][i][2];
      u41jm1 = tmp * u[k][j-1][i][3];
      u51jm1 = tmp * u[k][j-1][i][4];

      flux[j][1] = ty3 * ( u21j - u21jm1 );
      flux[j][2] = (4.0/3.0) * ty3 * (u31j-u31jm1);
      flux[j][3] = ty3 * ( u41j - u41jm1 );
      flux[j][4] = 0.50 * ( 1.0 - C1*C5 )
        * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
                - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
        + (1.0/6.0)
        * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
        + C1 * C5 * ty3 * ( u51j - u51jm1 );
    }

    for (j = jst; j < jend; j++) {
      rsd[k][j][i][0] = rsd[k][j][i][0]
        + dy1 * ty1 * (         u[k][j-1][i][0]
                        - 2.0 * u[k][j][i][0]
                        +       u[k][j+1][i][0] );

      rsd[k][j][i][1] = rsd[k][j][i][1]
        + ty3 * C3 * C4 * ( flux[j+1][1] - flux[j][1] )
        + dy2 * ty1 * (         u[k][j-1][i][1]
                        - 2.0 * u[k][j][i][1]
                        +       u[k][j+1][i][1] );

      rsd[k][j][i][2] = rsd[k][j][i][2]
        + ty3 * C3 * C4 * ( flux[j+1][2] - flux[j][2] )
        + dy3 * ty1 * (         u[k][j-1][i][2]
                        - 2.0 * u[k][j][i][2]
                        +       u[k][j+1][i][2] );

      rsd[k][j][i][3] = rsd[k][j][i][3]
        + ty3 * C3 * C4 * ( flux[j+1][3] - flux[j][3] )
        + dy4 * ty1 * (         u[k][j-1][i][3]
                        - 2.0 * u[k][j][i][3]
                        +       u[k][j+1][i][3] );

      rsd[k][j][i][4] = rsd[k][j][i][4]
        + ty3 * C3 * C4 * ( flux[j+1][4] - flux[j][4] )
        + dy5 * ty1 * (         u[k][j-1][i][4]
                        - 2.0 * u[k][j][i][4]
                        +       u[k][j+1][i][4] );
    }
  }

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][1][i][m] = rsd[k][1][i][m]
        - dssp * ( + 5.0 * u[k][1][i][m]
                   - 4.0 * u[k][2][i][m]
                   +       u[k][3][i][m] );
      rsd[k][2][i][m] = rsd[k][2][i][m]
        - dssp * ( - 4.0 * u[k][1][i][m]
                   + 6.0 * u[k][2][i][m]
                   - 4.0 * u[k][3][i][m]
                   +       u[k][4][i][m] );
    }
  }

  for (j = 3; j < ny - 3; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rsd[k][j][i][m]
          - dssp * (         u[k][j-2][i][m]
                     - 4.0 * u[k][j-1][i][m]
                     + 6.0 * u[k][j][i][m]
                     - 4.0 * u[k][j+1][i][m]
                     +       u[k][j+2][i][m] );
      }
    }
  }

  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][ny-3][i][m] = rsd[k][ny-3][i][m]
        - dssp * (         u[k][ny-5][i][m]
                   - 4.0 * u[k][ny-4][i][m]
                   + 6.0 * u[k][ny-3][i][m]
                   - 4.0 * u[k][ny-2][i][m] );
      rsd[k][ny-2][i][m] = rsd[k][ny-2][i][m]
        - dssp * (         u[k][ny-4][i][m]
                   - 4.0 * u[k][ny-3][i][m]
                   + 5.0 * u[k][ny-2][i][m] );
    }
  }
#endif
}


__kernel void rhsz(__global double *g_u,
                   __global double *g_rsd,
                   __global double *g_qs,
                   __global double *g_rho_i,
                   int nx,
                   int ny,
                   int nz)
{
  int i, j, k, k1, k2, m;
  double q;
  double tmp;
  double u41;
  double u21k, u31k, u41k, u51k;
  double u21km1, u31km1, u41km1, u51km1;
  double flux1[3][5], flux2[2][5];
  double p_u[5][5], p_rsd[5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

#if RHSZ_DIM == 2
  j = get_global_id(1) + jst;
  i = get_global_id(0) + ist;
  if (j >= jend || i >= iend) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

#define LOAD_P_U(p_idx, g_idx) \
  p_u[p_idx][0] = u[g_idx][j][i][0]; \
  p_u[p_idx][1] = u[g_idx][j][i][1]; \
  p_u[p_idx][2] = u[g_idx][j][i][2]; \
  p_u[p_idx][3] = u[g_idx][j][i][3]; \
  p_u[p_idx][4] = u[g_idx][j][i][4];

#define COMPUTE_FLUX_1(idx) \
  flux1[idx][0] = p_u[3][3]; \
  u41 = p_u[3][3] * rho_i[k1][j][i]; \
  q = qs[k1][j][i]; \
  flux1[idx][1] = p_u[3][1] * u41; \
  flux1[idx][2] = p_u[3][2] * u41; \
  flux1[idx][3] = p_u[3][3] * u41 + C2 * (p_u[3][4]-q); \
  flux1[idx][4] = ( C1 * p_u[3][4] - C2 * q ) * u41; \

#define COMPUTE_FLUX_2(idx) \
  tmp = rho_i[k1][j][i]; \
  u21k = tmp * p_u[3][1]; \
  u31k = tmp * p_u[3][2]; \
  u41k = tmp * p_u[3][3]; \
  u51k = tmp * p_u[3][4]; \
  tmp = rho_i[k][j][i]; \
  u21km1 = tmp * p_u[2][1]; \
  u31km1 = tmp * p_u[2][2]; \
  u41km1 = tmp * p_u[2][3]; \
  u51km1 = tmp * p_u[2][4]; \
  flux2[idx][1] = tz3 * ( u21k - u21km1 ); \
  flux2[idx][2] = tz3 * ( u31k - u31km1 ); \
  flux2[idx][3] = (4.0/3.0) * tz3 * (u41k-u41km1); \
  flux2[idx][4] = 0.50 * ( 1.0 - C1*C5 ) \
    * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k ) \
            - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) ) \
    + (1.0/6.0) \
    * tz3 * ( u41k*u41k - u41km1*u41km1 ) \
    + C1 * C5 * tz3 * ( u51k - u51km1 );

#define LOOP_PROLOGUE_FULL \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  LOAD_P_U(4, k2) \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_PROLOGUE_HALF \
  for (m = 0; m < 5; m++) { \
    p_u[0][m] = p_u[1][m]; \
    p_u[1][m] = p_u[2][m]; \
    p_u[2][m] = p_u[3][m]; \
    p_u[3][m] = p_u[4][m]; \
    flux1[0][m] = flux1[1][m]; \
    flux1[1][m] = flux1[2][m]; \
    flux2[0][m] = flux2[1][m]; \
  } \
  COMPUTE_FLUX_1(2) \
  COMPUTE_FLUX_2(1)

#define LOOP_BODY \
    for (m = 0; m < 5; m++) { \
      p_rsd[m] =  rsd[k][j][i][m] \
        - tz2 * ( flux1[2][m] - flux1[0][m] ); \
    } \
    p_rsd[0] = p_rsd[0] \
      + dz1 * tz1 * (         p_u[1][0] \
                      - 2.0 * p_u[2][0] \
                      +       p_u[3][0] ); \
    p_rsd[1] = p_rsd[1] \
      + tz3 * C3 * C4 * ( flux2[1][1] - flux2[0][1] ) \
      + dz2 * tz1 * (         p_u[1][1] \
                      - 2.0 * p_u[2][1] \
                      +       p_u[3][1] ); \
    p_rsd[2] = p_rsd[2] \
      + tz3 * C3 * C4 * ( flux2[1][2] - flux2[0][2] ) \
      + dz3 * tz1 * (         p_u[1][2] \
                      - 2.0 * p_u[2][2] \
                      +       p_u[3][2] ); \
    p_rsd[3] = p_rsd[3] \
      + tz3 * C3 * C4 * ( flux2[1][3] - flux2[0][3] ) \
      + dz4 * tz1 * (         p_u[1][3] \
                      - 2.0 * p_u[2][3] \
                      +       p_u[3][3] ); \
    p_rsd[4] = p_rsd[4] \
      + tz3 * C3 * C4 * ( flux2[1][4] - flux2[0][4] ) \
      + dz5 * tz1 * (         p_u[1][4] \
                      - 2.0 * p_u[2][4] \
                      +       p_u[3][4] );

  k1 = 0;
  LOAD_P_U(3, 0)
  COMPUTE_FLUX_1(1)

  k = 0; k1 = 1;
  for (m = 0; m < 5; m++) p_u[2][m] = p_u[3][m]; // LOAD_P_U(2, 0)
  LOAD_P_U(3, 1)
  COMPUTE_FLUX_1(2)
  COMPUTE_FLUX_2(1)
  LOAD_P_U(4, 2)

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------

  k = 1; k1 = 2; k2 = 3;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[1][j][i][m] = p_rsd[m]
        - dssp * ( + 5.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }

  k = 2; k1 = 3; k2 = 4;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[2][j][i][m] = p_rsd[m]
        - dssp * ( - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }
  
  for (k = 3; k < nz - 3; k++) {
    k1 = k + 1; k2 = k + 2;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m]
                   +       p_u[4][m] );
    }
  }
 
  k = nz-3; k1 = nz-2; k2 = nz-1;
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[nz-3][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 6.0 * p_u[2][m]
                   - 4.0 * p_u[3][m] );
    }

  k = nz-2; k1 = nz-1;
    LOOP_PROLOGUE_HALF
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rsd[nz-2][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
                   - 4.0 * p_u[1][m]
                   + 5.0 * p_u[2][m] );
    }
  
#undef LOAD_P_U
#undef COMPUTE_FLUX_1
#undef COMPUTE_FLUX_2
#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY

#else //RHSZ_DIM == 1
  j = get_global_id(0) + jst;
  if (j >= jend) return;

  u     = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  qs    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  int my_id = j - jst;
  int my_offset = my_id * ISIZ1*5;
  flux = (__global double (*)[5])&g_flux[my_offset];

  for (i = ist; i < iend; i++) {
    for (k = 0; k < nz; k++) {
      flux[k][0] = u[k][j][i][3];
      u41 = u[k][j][i][3] * rho_i[k][j][i];

      q = qs[k][j][i];

      flux[k][1] = u[k][j][i][1] * u41;
      flux[k][2] = u[k][j][i][2] * u41;
      flux[k][3] = u[k][j][i][3] * u41 + C2 * (u[k][j][i][4]-q);
      flux[k][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u41;
    }

    for (k = 1; k < nz - 1; k++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] =  rsd[k][j][i][m]
          - tz2 * ( flux[k+1][m] - flux[k-1][m] );
      }
    }

    for (k = 1; k < nz; k++) {
      tmp = rho_i[k][j][i];

      u21k = tmp * u[k][j][i][1];
      u31k = tmp * u[k][j][i][2];
      u41k = tmp * u[k][j][i][3];
      u51k = tmp * u[k][j][i][4];

      tmp = rho_i[k-1][j][i];

      u21km1 = tmp * u[k-1][j][i][1];
      u31km1 = tmp * u[k-1][j][i][2];
      u41km1 = tmp * u[k-1][j][i][3];
      u51km1 = tmp * u[k-1][j][i][4];

      flux[k][1] = tz3 * ( u21k - u21km1 );
      flux[k][2] = tz3 * ( u31k - u31km1 );
      flux[k][3] = (4.0/3.0) * tz3 * (u41k-u41km1);
      flux[k][4] = 0.50 * ( 1.0 - C1*C5 )
        * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
                - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
        + (1.0/6.0)
        * tz3 * ( u41k*u41k - u41km1*u41km1 )
        + C1 * C5 * tz3 * ( u51k - u51km1 );
    }

    for (k = 1; k < nz - 1; k++) {
      rsd[k][j][i][0] = rsd[k][j][i][0]
        + dz1 * tz1 * (         u[k-1][j][i][0]
                        - 2.0 * u[k][j][i][0]
                        +       u[k+1][j][i][0] );
      rsd[k][j][i][1] = rsd[k][j][i][1]
        + tz3 * C3 * C4 * ( flux[k+1][1] - flux[k][1] )
        + dz2 * tz1 * (         u[k-1][j][i][1]
                        - 2.0 * u[k][j][i][1]
                        +       u[k+1][j][i][1] );
      rsd[k][j][i][2] = rsd[k][j][i][2]
        + tz3 * C3 * C4 * ( flux[k+1][2] - flux[k][2] )
        + dz3 * tz1 * (         u[k-1][j][i][2]
                        - 2.0 * u[k][j][i][2]
                        +       u[k+1][j][i][2] );
      rsd[k][j][i][3] = rsd[k][j][i][3]
        + tz3 * C3 * C4 * ( flux[k+1][3] - flux[k][3] )
        + dz4 * tz1 * (         u[k-1][j][i][3]
                        - 2.0 * u[k][j][i][3]
                        +       u[k+1][j][i][3] );
      rsd[k][j][i][4] = rsd[k][j][i][4]
        + tz3 * C3 * C4 * ( flux[k+1][4] - flux[k][4] )
        + dz5 * tz1 * (         u[k-1][j][i][4]
                        - 2.0 * u[k][j][i][4]
                        +       u[k+1][j][i][4] );
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      rsd[1][j][i][m] = rsd[1][j][i][m]
        - dssp * ( + 5.0 * u[1][j][i][m]
                   - 4.0 * u[2][j][i][m]
                   +       u[3][j][i][m] );
      rsd[2][j][i][m] = rsd[2][j][i][m]
        - dssp * ( - 4.0 * u[1][j][i][m]
                   + 6.0 * u[2][j][i][m]
                   - 4.0 * u[3][j][i][m]
                   +       u[4][j][i][m] );
    }

    for (k = 3; k < nz - 3; k++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rsd[k][j][i][m]
          - dssp * (         u[k-2][j][i][m]
                     - 4.0 * u[k-1][j][i][m]
                     + 6.0 * u[k][j][i][m]
                     - 4.0 * u[k+1][j][i][m]
                     +       u[k+2][j][i][m] );
      }
    }

    for (m = 0; m < 5; m++) {
      rsd[nz-3][j][i][m] = rsd[nz-3][j][i][m]
        - dssp * (         u[nz-5][j][i][m]
                   - 4.0 * u[nz-4][j][i][m]
                   + 6.0 * u[nz-3][j][i][m]
                   - 4.0 * u[nz-2][j][i][m] );
      rsd[nz-2][j][i][m] = rsd[nz-2][j][i][m]
        - dssp * (         u[nz-4][j][i][m]
                   - 4.0 * u[nz-3][j][i][m]
                   + 5.0 * u[nz-2][j][i][m] );
    }
  }
#endif
}


__kernel void ssor1(__global double *g_a,
                    __global double *g_b,
                    __global double *g_c,
                    __global double *g_d,
                    __global double *g_au,
                    __global double *g_bu,
                    __global double *g_cu,
                    __global double *g_du,
                    int nx,
                    int ny)
{
  int i, j, m, n;
  __global double (*a)[ISIZ1/2*2+1][5][5];
  __global double (*b)[ISIZ1/2*2+1][5][5];
  __global double (*c)[ISIZ1/2*2+1][5][5];
  __global double (*d)[ISIZ1/2*2+1][5][5];
  __global double (*au)[ISIZ1/2*2+1][5][5];
  __global double (*bu)[ISIZ1/2*2+1][5][5];
  __global double (*cu)[ISIZ1/2*2+1][5][5];
  __global double (*du)[ISIZ1/2*2+1][5][5];

#if SSOR1_DIM == 3
  j = get_global_id(2) + jst;
  i = get_global_id(1) + ist;
  n = get_global_id(0);
  if (j >= jend || i >= iend) return;

  a = (__global double (*)[ISIZ1/2*2+1][5][5])g_a;
  b = (__global double (*)[ISIZ1/2*2+1][5][5])g_b;
  c = (__global double (*)[ISIZ1/2*2+1][5][5])g_c;
  d = (__global double (*)[ISIZ1/2*2+1][5][5])g_d;
  au = (__global double (*)[ISIZ1/2*2+1][5][5])g_au;
  bu = (__global double (*)[ISIZ1/2*2+1][5][5])g_bu;
  cu = (__global double (*)[ISIZ1/2*2+1][5][5])g_cu;
  du = (__global double (*)[ISIZ1/2*2+1][5][5])g_du;

  for (m = 0; m < 5; m++) {
    a[j][i][n][m] = 0.0;
    b[j][i][n][m] = 0.0;
    c[j][i][n][m] = 0.0;
    d[j][i][n][m] = 0.0;
    au[j][i][n][m] = 0.0;
    bu[j][i][n][m] = 0.0;
    cu[j][i][n][m] = 0.0;
    du[j][i][n][m] = 0.0;
  }

#elif SSOR1_DIM == 2
  j = get_global_id(1) + jst;
  i = get_global_id(0) + ist;
  if (j >= jend || i >= iend) return;

  a = (__global double (*)[ISIZ1/2*2+1][5][5])g_a;
  b = (__global double (*)[ISIZ1/2*2+1][5][5])g_b;
  c = (__global double (*)[ISIZ1/2*2+1][5][5])g_c;
  d = (__global double (*)[ISIZ1/2*2+1][5][5])g_d;
  au = (__global double (*)[ISIZ1/2*2+1][5][5])g_au;
  bu = (__global double (*)[ISIZ1/2*2+1][5][5])g_bu;
  cu = (__global double (*)[ISIZ1/2*2+1][5][5])g_cu;
  du = (__global double (*)[ISIZ1/2*2+1][5][5])g_du;

  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      a[j][i][n][m] = 0.0;
      b[j][i][n][m] = 0.0;
      c[j][i][n][m] = 0.0;
      d[j][i][n][m] = 0.0;
      au[j][i][n][m] = 0.0;
      bu[j][i][n][m] = 0.0;
      cu[j][i][n][m] = 0.0;
      du[j][i][n][m] = 0.0;
    }
  }

#else //SSOR1_DIM == 1
  j = get_global_id(0) + jst;
  if (j >= jend) return;

  a = (__global double (*)[ISIZ1/2*2+1][5][5])g_a;
  b = (__global double (*)[ISIZ1/2*2+1][5][5])g_b;
  c = (__global double (*)[ISIZ1/2*2+1][5][5])g_c;
  d = (__global double (*)[ISIZ1/2*2+1][5][5])g_d;
  au = (__global double (*)[ISIZ1/2*2+1][5][5])g_au;
  bu = (__global double (*)[ISIZ1/2*2+1][5][5])g_bu;
  cu = (__global double (*)[ISIZ1/2*2+1][5][5])g_cu;
  du = (__global double (*)[ISIZ1/2*2+1][5][5])g_du;

  for (i = ist; i < iend; i++) {
    for (n = 0; n < 5; n++) {
      for (m = 0; m < 5; m++) {
        a[j][i][n][m] = 0.0;
        b[j][i][n][m] = 0.0;
        c[j][i][n][m] = 0.0;
        d[j][i][n][m] = 0.0;
        au[j][i][n][m] = 0.0;
        bu[j][i][n][m] = 0.0;
        cu[j][i][n][m] = 0.0;
        du[j][i][n][m] = 0.0;
      }
    }
  }
#endif
}


__kernel void ssor2(__global double *g_rsd,
                    double tmp2,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];

#if SSOR2_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + jst;
  i = get_global_id(0) + ist;
  if (k >= (nz-1) || j >= jend || i >= iend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (m = 0; m < 5; m++) {
    rsd[k][j][i][m] = tmp2 * rsd[k][j][i][m];
  }

#elif SSOR2_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + jst;
  if (k >= (nz-1) || j >= jend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = tmp2 * rsd[k][j][i][m];
    }
  }

#else //SSOR2_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = tmp2 * rsd[k][j][i][m];
      }
    }
  }
#endif
}


__kernel void ssor3(__global double *g_u,
                    __global double *g_rsd,
                    double tmp2,
                    int nx,
                    int ny,
                    int nz)
{
  int i, j, k, m;
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];

#if SSOR3_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + jst;
  i = get_global_id(0) + ist;
  if (k >= (nz-1) || j >= jend || i >= iend) return;

  u   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = u[k][j][i][m] + tmp2 * rsd[k][j][i][m];
  }

#elif SSOR3_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + jst;
  if (k >= (nz-1) || j >= jend) return;

  u   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      u[k][j][i][m] = u[k][j][i][m] + tmp2 * rsd[k][j][i][m];
    }
  }

#else //SSOR3_DIM == 1
  k = get_global_id(0) + 1;
  if (k >= (nz-1)) return;

  u   = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;

  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        u[k][j][i][m] = u[k][j][i][m] + tmp2 * rsd[k][j][i][m];
      }
    }
  }
#endif
}


__kernel void blts(__global double *g_rsd,
                   __global double *g_u,
                   __global double *g_qs,
                   __global double *g_rho_i,
                   int nz, int ny, int nx,
                   int wf_sum, int wf_base_k, int wf_base_j)
{
  int k, j, i, m;
  double a[5][5], b[5][5], c[5][5], d[5][5];
  double r43, c1345, c34;
  double tmp, tmp1, tmp2, tmp3;
  double tmat[5][5], tv[5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  k = get_global_id(1) + 1 + wf_base_k;
  j = get_global_id(0) + jst + wf_base_j;
  i = wf_sum - get_global_id(1) - get_global_id(0) - wf_base_k - wf_base_j + ist;
  if (k >= nz - 1 || j >= jend || i < ist || i >= iend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  u = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  qs = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //---------------------------------------------------------------------
  // form the block daigonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  d[0][0] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
  d[1][0] =  0.0;
  d[2][0] =  0.0;
  d[3][0] =  0.0;
  d[4][0] =  0.0;

  d[0][1] = -dt * 2.0
    * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u[k][j][i][1];
  d[1][1] =  1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
    + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
  d[2][1] = 0.0;
  d[3][1] = 0.0;
  d[4][1] = 0.0;

  d[0][2] = -dt * 2.0 
    * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u[k][j][i][2];
  d[1][2] = 0.0;
  d[2][2] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
    + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
  d[3][2] = 0.0;
  d[4][2] = 0.0;

  d[0][3] = -dt * 2.0
    * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u[k][j][i][3];
  d[1][3] = 0.0;
  d[2][3] = 0.0;
  d[3][3] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
    + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
  d[4][3] = 0.0;

  d[0][4] = -dt * 2.0
    * ( ( ( tx1 * ( r43*c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][1]*u[k][j][i][1] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( r43*c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][2]*u[k][j][i][2] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( r43*c34 - c1345 ) ) * (u[k][j][i][3]*u[k][j][i][3])
        ) * tmp3
        + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[k][j][i][4] );

  d[1][4] = dt * 2.0 * tmp2 * u[k][j][i][1]
    * ( tx1 * ( r43*c34 - c1345 )
      + ty1 * (     c34 - c1345 )
      + tz1 * (     c34 - c1345 ) );
  d[2][4] = dt * 2.0 * tmp2 * u[k][j][i][2]
    * ( tx1 * ( c34 - c1345 )
      + ty1 * ( r43*c34 -c1345 )
      + tz1 * ( c34 - c1345 ) );
  d[3][4] = dt * 2.0 * tmp2 * u[k][j][i][3]
    * ( tx1 * ( c34 - c1345 )
      + ty1 * ( c34 - c1345 )
      + tz1 * ( r43*c34 - c1345 ) );
  d[4][4] = 1.0
    + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
    + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

  //---------------------------------------------------------------------
  // form the first block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k-1][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  a[0][0] = - dt * tz1 * dz1;
  a[1][0] =   0.0;
  a[2][0] =   0.0;
  a[3][0] = - dt * tz2;
  a[4][0] =   0.0;

  a[0][1] = - dt * tz2
    * ( - ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][1] );
  a[1][1] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * c34 * tmp1
    - dt * tz1 * dz2;
  a[2][1] = 0.0;
  a[3][1] = - dt * tz2 * ( u[k-1][j][i][1] * tmp1 );
  a[4][1] = 0.0;

  a[0][2] = - dt * tz2
    * ( - ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][2] );
  a[1][2] = 0.0;
  a[2][2] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( c34 * tmp1 )
    - dt * tz1 * dz3;
  a[3][2] = - dt * tz2 * ( u[k-1][j][i][2] * tmp1 );
  a[4][2] = 0.0;

  a[0][3] = - dt * tz2
    * ( - ( u[k-1][j][i][3] * tmp1 ) * ( u[k-1][j][i][3] * tmp1 )
        + C2 * qs[k-1][j][i] * tmp1 )
    - dt * tz1 * ( - r43 * c34 * tmp2 * u[k-1][j][i][3] );
  a[1][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1] * tmp1 ) );
  a[2][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2] * tmp1 ) );
  a[3][3] = - dt * tz2 * ( 2.0 - C2 )
    * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( r43 * c34 * tmp1 )
    - dt * tz1 * dz4;
  a[4][3] = - dt * tz2 * C2;

  a[0][4] = - dt * tz2
    * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[k-1][j][i][4] )
        * u[k-1][j][i][3] * tmp2 )
    - dt * tz1
    * ( - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][1]*u[k-1][j][i][1])
        - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][2]*u[k-1][j][i][2])
        - ( r43*c34 - c1345 )* tmp3 * (u[k-1][j][i][3]*u[k-1][j][i][3])
        - c1345 * tmp2 * u[k-1][j][i][4] );
  a[1][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][1];
  a[2][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][2];
  a[3][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][4] * tmp1 )
      - C2 * ( qs[k-1][j][i] * tmp1
             + u[k-1][j][i][3]*u[k-1][j][i][3] * tmp2 ) )
    - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[k-1][j][i][3];
  a[4][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][3] * tmp1 ) )
    - dt * tz1 * c1345 * tmp1
    - dt * tz1 * dz5;

  //---------------------------------------------------------------------
  // form the second block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j-1][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  b[0][0] = - dt * ty1 * dy1;
  b[1][0] =   0.0;
  b[2][0] = - dt * ty2;
  b[3][0] =   0.0;
  b[4][0] =   0.0;

  b[0][1] = - dt * ty2
    * ( - ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][1] );
  b[1][1] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy2;
  b[2][1] = - dt * ty2 * ( u[k][j-1][i][1] * tmp1 );
  b[3][1] = 0.0;
  b[4][1] = 0.0;

  b[0][2] = - dt * ty2
    * ( - ( u[k][j-1][i][2] * tmp1 ) * ( u[k][j-1][i][2] * tmp1 )
        + C2 * ( qs[k][j-1][i] * tmp1 ) )
    - dt * ty1 * ( - r43 * c34 * tmp2 * u[k][j-1][i][2] );
  b[1][2] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1] * tmp1 ) );
  b[2][2] = - dt * ty2 * ( (2.0 - C2) * (u[k][j-1][i][2] * tmp1) )
    - dt * ty1 * ( r43 * c34 * tmp1 )
    - dt * ty1 * dy3;
  b[3][2] = - dt * ty2 * ( - C2 * ( u[k][j-1][i][3] * tmp1 ) );
  b[4][2] = - dt * ty2 * C2;

  b[0][3] = - dt * ty2
    * ( - ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][3] );
  b[1][3] = 0.0;
  b[2][3] = - dt * ty2 * ( u[k][j-1][i][3] * tmp1 );
  b[3][3] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy4;
  b[4][3] = 0.0;

  b[0][4] = - dt * ty2
    * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[k][j-1][i][4] )
        * ( u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1
    * ( - (     c34 - c1345 )*tmp3*(u[k][j-1][i][1]*u[k][j-1][i][1])
        - ( r43*c34 - c1345 )*tmp3*(u[k][j-1][i][2]*u[k][j-1][i][2])
        - (     c34 - c1345 )*tmp3*(u[k][j-1][i][3]*u[k][j-1][i][3])
        - c1345*tmp2*u[k][j-1][i][4] );
  b[1][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][1];
  b[2][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][4] * tmp1 )
      - C2 * ( qs[k][j-1][i] * tmp1
             + u[k][j-1][i][2]*u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j-1][i][2];
  b[3][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][3];
  b[4][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][2] * tmp1 ) )
    - dt * ty1 * c1345 * tmp1
    - dt * ty1 * dy5;

  //---------------------------------------------------------------------
  // form the third block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i-1];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  c[0][0] = - dt * tx1 * dx1;
  c[1][0] = - dt * tx2;
  c[2][0] =   0.0;
  c[3][0] =   0.0;
  c[4][0] =   0.0;

  c[0][1] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * tmp1 ) * ( u[k][j][i-1][1] * tmp1 )
        + C2 * qs[k][j][i-1] * tmp1 )
    - dt * tx1 * ( - r43 * c34 * tmp2 * u[k][j][i-1][1] );
  c[1][1] = - dt * tx2
    * ( ( 2.0 - C2 ) * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * ( r43 * c34 * tmp1 )
    - dt * tx1 * dx2;
  c[2][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2] * tmp1 ) );
  c[3][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3] * tmp1 ) );
  c[4][1] = - dt * tx2 * C2;

  c[0][2] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * u[k][j][i-1][2] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][2] );
  c[1][2] = - dt * tx2 * ( u[k][j][i-1][2] * tmp1 );
  c[2][2] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx3;
  c[3][2] = 0.0;
  c[4][2] = 0.0;

  c[0][3] = - dt * tx2
    * ( - ( u[k][j][i-1][1]*u[k][j][i-1][3] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][3] );
  c[1][3] = - dt * tx2 * ( u[k][j][i-1][3] * tmp1 );
  c[2][3] = 0.0;
  c[3][3] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
  c[4][3] = 0.0;

  c[0][4] = - dt * tx2
    * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[k][j][i-1][4] )
        * u[k][j][i-1][1] * tmp2 )
    - dt * tx1
    * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i-1][1]*u[k][j][i-1][1] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][2]*u[k][j][i-1][2] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][3]*u[k][j][i-1][3] )
        - c1345 * tmp2 * u[k][j][i-1][4] );
  c[1][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][4] * tmp1 )
      - C2 * ( u[k][j][i-1][1]*u[k][j][i-1][1] * tmp2
             + qs[k][j][i-1] * tmp1 ) )
    - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j][i-1][1];
  c[2][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][2];
  c[3][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][3];
  c[4][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * c1345 * tmp1
    - dt * tx1 * dx5;

  for (m = 0; m < 5; m++) {
    tv[m] =  rsd[k][j][i][m]
      - omega * (  a[0][m] * rsd[k-1][j][i][0]
                 + a[1][m] * rsd[k-1][j][i][1]
                 + a[2][m] * rsd[k-1][j][i][2]
                 + a[3][m] * rsd[k-1][j][i][3]
                 + a[4][m] * rsd[k-1][j][i][4] );
  }

  for (m = 0; m < 5; m++) {
    tv[m] =  tv[m]
      - omega * ( b[0][m] * rsd[k][j-1][i][0]
                + c[0][m] * rsd[k][j][i-1][0]
                + b[1][m] * rsd[k][j-1][i][1]
                + c[1][m] * rsd[k][j][i-1][1]
                + b[2][m] * rsd[k][j-1][i][2]
                + c[2][m] * rsd[k][j][i-1][2]
                + b[3][m] * rsd[k][j-1][i][3]
                + c[3][m] * rsd[k][j][i-1][3]
                + b[4][m] * rsd[k][j-1][i][4]
                + c[4][m] * rsd[k][j][i-1][4] );
  }

  //---------------------------------------------------------------------
  // diagonal block inversion
  // 
  // forward elimination
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    tmat[m][0] = d[0][m];
    tmat[m][1] = d[1][m];
    tmat[m][2] = d[2][m];
    tmat[m][3] = d[3][m];
    tmat[m][4] = d[4][m];
  }

  tmp1 = 1.0 / tmat[0][0];
  tmp = tmp1 * tmat[1][0];
  tmat[1][1] =  tmat[1][1] - tmp * tmat[0][1];
  tmat[1][2] =  tmat[1][2] - tmp * tmat[0][2];
  tmat[1][3] =  tmat[1][3] - tmp * tmat[0][3];
  tmat[1][4] =  tmat[1][4] - tmp * tmat[0][4];
  tv[1] = tv[1] - tv[0] * tmp;

  tmp = tmp1 * tmat[2][0];
  tmat[2][1] =  tmat[2][1] - tmp * tmat[0][1];
  tmat[2][2] =  tmat[2][2] - tmp * tmat[0][2];
  tmat[2][3] =  tmat[2][3] - tmp * tmat[0][3];
  tmat[2][4] =  tmat[2][4] - tmp * tmat[0][4];
  tv[2] = tv[2] - tv[0] * tmp;

  tmp = tmp1 * tmat[3][0];
  tmat[3][1] =  tmat[3][1] - tmp * tmat[0][1];
  tmat[3][2] =  tmat[3][2] - tmp * tmat[0][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[0][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[0][4];
  tv[3] = tv[3] - tv[0] * tmp;

  tmp = tmp1 * tmat[4][0];
  tmat[4][1] =  tmat[4][1] - tmp * tmat[0][1];
  tmat[4][2] =  tmat[4][2] - tmp * tmat[0][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[0][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[0][4];
  tv[4] = tv[4] - tv[0] * tmp;

  tmp1 = 1.0 / tmat[1][1];
  tmp = tmp1 * tmat[2][1];
  tmat[2][2] =  tmat[2][2] - tmp * tmat[1][2];
  tmat[2][3] =  tmat[2][3] - tmp * tmat[1][3];
  tmat[2][4] =  tmat[2][4] - tmp * tmat[1][4];
  tv[2] = tv[2] - tv[1] * tmp;

  tmp = tmp1 * tmat[3][1];
  tmat[3][2] =  tmat[3][2] - tmp * tmat[1][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[1][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[1][4];
  tv[3] = tv[3] - tv[1] * tmp;

  tmp = tmp1 * tmat[4][1];
  tmat[4][2] =  tmat[4][2] - tmp * tmat[1][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[1][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[1][4];
  tv[4] = tv[4] - tv[1] * tmp;

  tmp1 = 1.0 / tmat[2][2];
  tmp = tmp1 * tmat[3][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[2][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[2][4];
  tv[3] = tv[3] - tv[2] * tmp;

  tmp = tmp1 * tmat[4][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[2][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[2][4];
  tv[4] = tv[4] - tv[2] * tmp;

  tmp1 = 1.0 / tmat[3][3];
  tmp = tmp1 * tmat[4][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[3][4];
  tv[4] = tv[4] - tv[3] * tmp;

  //---------------------------------------------------------------------
  // back substitution
  //---------------------------------------------------------------------
  rsd[k][j][i][4] = tv[4] / tmat[4][4];

  tv[3] = tv[3] 
    - tmat[3][4] * rsd[k][j][i][4];
  rsd[k][j][i][3] = tv[3] / tmat[3][3];

  tv[2] = tv[2]
    - tmat[2][3] * rsd[k][j][i][3]
    - tmat[2][4] * rsd[k][j][i][4];
  rsd[k][j][i][2] = tv[2] / tmat[2][2];

  tv[1] = tv[1]
    - tmat[1][2] * rsd[k][j][i][2]
    - tmat[1][3] * rsd[k][j][i][3]
    - tmat[1][4] * rsd[k][j][i][4];
  rsd[k][j][i][1] = tv[1] / tmat[1][1];

  tv[0] = tv[0]
    - tmat[0][1] * rsd[k][j][i][1]
    - tmat[0][2] * rsd[k][j][i][2]
    - tmat[0][3] * rsd[k][j][i][3]
    - tmat[0][4] * rsd[k][j][i][4];
  rsd[k][j][i][0] = tv[0] / tmat[0][0];
}


__kernel void buts(__global double *g_rsd,
                   __global double *g_u,
                   __global double *g_qs,
                   __global double *g_rho_i,
                   int nz, int ny, int nx,
                   int wf_sum, int wf_base_k, int wf_base_j)
{
  int k, j, i, m;
  double au[5][5], bu[5][5], cu[5][5], du[5][5];
  double r43, c1345, c34;
  double tmp, tmp1, tmp2, tmp3;
  double tmat[5][5], tv[5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  k = get_global_id(1) + 1 + wf_base_k;
  j = get_global_id(0) + jst + wf_base_j;
  i = wf_sum - get_global_id(1) - get_global_id(0) - wf_base_k - wf_base_j + ist;
  if (k >= nz - 1 || j >= jend || i < ist || i >= iend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_rsd;
  u = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])g_u;
  qs = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])g_rho_i;
  
  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //---------------------------------------------------------------------
  // form the block daigonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  du[0][0] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
  du[1][0] = 0.0;
  du[2][0] = 0.0;
  du[3][0] = 0.0;
  du[4][0] = 0.0;

  du[0][1] =  dt * 2.0
    * ( - tx1 * r43 - ty1 - tz1 )
    * ( c34 * tmp2 * u[k][j][i][1] );
  du[1][1] =  1.0
    + dt * 2.0 * c34 * tmp1 
    * (  tx1 * r43 + ty1 + tz1 )
    + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
  du[2][1] = 0.0;
  du[3][1] = 0.0;
  du[4][1] = 0.0;

  du[0][2] = dt * 2.0
    * ( - tx1 - ty1 * r43 - tz1 )
    * ( c34 * tmp2 * u[k][j][i][2] );
  du[1][2] = 0.0;
  du[2][2] = 1.0
    + dt * 2.0 * c34 * tmp1
    * (  tx1 + ty1 * r43 + tz1 )
    + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
  du[3][2] = 0.0;
  du[4][2] = 0.0;

  du[0][3] = dt * 2.0
    * ( - tx1 - ty1 - tz1 * r43 )
    * ( c34 * tmp2 * u[k][j][i][3] );
  du[1][3] = 0.0;
  du[2][3] = 0.0;
  du[3][3] = 1.0
    + dt * 2.0 * c34 * tmp1
    * (  tx1 + ty1 + tz1 * r43 )
    + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
  du[4][3] = 0.0;

  du[0][4] = -dt * 2.0
    * ( ( ( tx1 * ( r43*c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][1]*u[k][j][i][1] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( r43*c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][2]*u[k][j][i][2] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( r43*c34 - c1345 ) ) * (u[k][j][i][3]*u[k][j][i][3])
        ) * tmp3
        + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[k][j][i][4] );

  du[1][4] = dt * 2.0
    * ( tx1 * ( r43*c34 - c1345 )
      + ty1 * (     c34 - c1345 )
      + tz1 * (     c34 - c1345 ) ) * tmp2 * u[k][j][i][1];
  du[2][4] = dt * 2.0
    * ( tx1 * ( c34 - c1345 )
      + ty1 * ( r43*c34 -c1345 )
      + tz1 * ( c34 - c1345 ) ) * tmp2 * u[k][j][i][2];
  du[3][4] = dt * 2.0
    * ( tx1 * ( c34 - c1345 )
      + ty1 * ( c34 - c1345 )
      + tz1 * ( r43*c34 - c1345 ) ) * tmp2 * u[k][j][i][3];
  du[4][4] = 1.0
    + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1
    + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );

  //---------------------------------------------------------------------
  // form the first block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i+1];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  au[0][0] = - dt * tx1 * dx1;
  au[1][0] =   dt * tx2;
  au[2][0] =   0.0;
  au[3][0] =   0.0;
  au[4][0] =   0.0;

  au[0][1] =  dt * tx2
    * ( - ( u[k][j][i+1][1] * tmp1 ) * ( u[k][j][i+1][1] * tmp1 )
        + C2 * qs[k][j][i+1] * tmp1 )
    - dt * tx1 * ( - r43 * c34 * tmp2 * u[k][j][i+1][1] );
  au[1][1] =  dt * tx2
    * ( ( 2.0 - C2 ) * ( u[k][j][i+1][1] * tmp1 ) )
    - dt * tx1 * ( r43 * c34 * tmp1 )
    - dt * tx1 * dx2;
  au[2][1] =  dt * tx2
    * ( - C2 * ( u[k][j][i+1][2] * tmp1 ) );
  au[3][1] =  dt * tx2
    * ( - C2 * ( u[k][j][i+1][3] * tmp1 ) );
  au[4][1] =  dt * tx2 * C2 ;

  au[0][2] =  dt * tx2
    * ( - ( u[k][j][i+1][1] * u[k][j][i+1][2] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i+1][2] );
  au[1][2] =  dt * tx2 * ( u[k][j][i+1][2] * tmp1 );
  au[2][2] =  dt * tx2 * ( u[k][j][i+1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx3;
  au[3][2] = 0.0;
  au[4][2] = 0.0;

  au[0][3] = dt * tx2
    * ( - ( u[k][j][i+1][1]*u[k][j][i+1][3] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i+1][3] );
  au[1][3] = dt * tx2 * ( u[k][j][i+1][3] * tmp1 );
  au[2][3] = 0.0;
  au[3][3] = dt * tx2 * ( u[k][j][i+1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx4;
  au[4][3] = 0.0;

  au[0][4] = dt * tx2
    * ( ( C2 * 2.0 * qs[k][j][i+1]
        - C1 * u[k][j][i+1][4] )
    * ( u[k][j][i+1][1] * tmp2 ) )
    - dt * tx1
    * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i+1][1]*u[k][j][i+1][1] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i+1][2]*u[k][j][i+1][2] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i+1][3]*u[k][j][i+1][3] )
        - c1345 * tmp2 * u[k][j][i+1][4] );
  au[1][4] = dt * tx2
    * ( C1 * ( u[k][j][i+1][4] * tmp1 )
        - C2
        * ( u[k][j][i+1][1]*u[k][j][i+1][1] * tmp2
          + qs[k][j][i+1] * tmp1 ) )
    - dt * tx1
    * ( r43*c34 - c1345 ) * tmp2 * u[k][j][i+1][1];
  au[2][4] = dt * tx2
    * ( - C2 * ( u[k][j][i+1][2]*u[k][j][i+1][1] ) * tmp2 )
    - dt * tx1
    * (  c34 - c1345 ) * tmp2 * u[k][j][i+1][2];
  au[3][4] = dt * tx2
    * ( - C2 * ( u[k][j][i+1][3]*u[k][j][i+1][1] ) * tmp2 )
    - dt * tx1
    * (  c34 - c1345 ) * tmp2 * u[k][j][i+1][3];
  au[4][4] = dt * tx2
    * ( C1 * ( u[k][j][i+1][1] * tmp1 ) )
    - dt * tx1 * c1345 * tmp1
    - dt * tx1 * dx5;

  //---------------------------------------------------------------------
  // form the second block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j+1][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  bu[0][0] = - dt * ty1 * dy1;
  bu[1][0] =   0.0;
  bu[2][0] =  dt * ty2;
  bu[3][0] =   0.0;
  bu[4][0] =   0.0;

  bu[0][1] =  dt * ty2
    * ( - ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j+1][i][1] );
  bu[1][1] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy2;
  bu[2][1] =  dt * ty2 * ( u[k][j+1][i][1] * tmp1 );
  bu[3][1] = 0.0;
  bu[4][1] = 0.0;

  bu[0][2] =  dt * ty2
    * ( - ( u[k][j+1][i][2] * tmp1 ) * ( u[k][j+1][i][2] * tmp1 )
        + C2 * ( qs[k][j+1][i] * tmp1 ) )
    - dt * ty1 * ( - r43 * c34 * tmp2 * u[k][j+1][i][2] );
  bu[1][2] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][1] * tmp1 ) );
  bu[2][2] =  dt * ty2 * ( ( 2.0 - C2 )
      * ( u[k][j+1][i][2] * tmp1 ) )
    - dt * ty1 * ( r43 * c34 * tmp1 )
    - dt * ty1 * dy3;
  bu[3][2] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][3] * tmp1 ) );
  bu[4][2] =  dt * ty2 * C2;

  bu[0][3] =  dt * ty2
    * ( - ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j+1][i][3] );
  bu[1][3] = 0.0;
  bu[2][3] =  dt * ty2 * ( u[k][j+1][i][3] * tmp1 );
  bu[3][3] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy4;
  bu[4][3] = 0.0;

  bu[0][4] =  dt * ty2
    * ( ( C2 * 2.0 * qs[k][j+1][i]
        - C1 * u[k][j+1][i][4] )
    * ( u[k][j+1][i][2] * tmp2 ) )
    - dt * ty1
    * ( - (     c34 - c1345 )*tmp3*(u[k][j+1][i][1]*u[k][j+1][i][1])
        - ( r43*c34 - c1345 )*tmp3*(u[k][j+1][i][2]*u[k][j+1][i][2])
        - (     c34 - c1345 )*tmp3*(u[k][j+1][i][3]*u[k][j+1][i][3])
        - c1345*tmp2*u[k][j+1][i][4] );
  bu[1][4] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2 )
    - dt * ty1
    * ( c34 - c1345 ) * tmp2 * u[k][j+1][i][1];
  bu[2][4] =  dt * ty2
    * ( C1 * ( u[k][j+1][i][4] * tmp1 )
        - C2 
        * ( qs[k][j+1][i] * tmp1
          + u[k][j+1][i][2]*u[k][j+1][i][2] * tmp2 ) )
    - dt * ty1
    * ( r43*c34 - c1345 ) * tmp2 * u[k][j+1][i][2];
  bu[3][4] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j+1][i][3];
  bu[4][4] =  dt * ty2
    * ( C1 * ( u[k][j+1][i][2] * tmp1 ) )
    - dt * ty1 * c1345 * tmp1
    - dt * ty1 * dy5;

  //---------------------------------------------------------------------
  // form the third block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k+1][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  cu[0][0] = - dt * tz1 * dz1;
  cu[1][0] =   0.0;
  cu[2][0] =   0.0;
  cu[3][0] = dt * tz2;
  cu[4][0] =   0.0;

  cu[0][1] = dt * tz2
    * ( - ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k+1][j][i][1] );
  cu[1][1] = dt * tz2 * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * c34 * tmp1
    - dt * tz1 * dz2;
  cu[2][1] = 0.0;
  cu[3][1] = dt * tz2 * ( u[k+1][j][i][1] * tmp1 );
  cu[4][1] = 0.0;

  cu[0][2] = dt * tz2
    * ( - ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k+1][j][i][2] );
  cu[1][2] = 0.0;
  cu[2][2] = dt * tz2 * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * ( c34 * tmp1 )
    - dt * tz1 * dz3;
  cu[3][2] = dt * tz2 * ( u[k+1][j][i][2] * tmp1 );
  cu[4][2] = 0.0;

  cu[0][3] = dt * tz2
    * ( - ( u[k+1][j][i][3] * tmp1 ) * ( u[k+1][j][i][3] * tmp1 )
        + C2 * ( qs[k+1][j][i] * tmp1 ) )
    - dt * tz1 * ( - r43 * c34 * tmp2 * u[k+1][j][i][3] );
  cu[1][3] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][1] * tmp1 ) );
  cu[2][3] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][2] * tmp1 ) );
  cu[3][3] = dt * tz2 * ( 2.0 - C2 )
    * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * ( r43 * c34 * tmp1 )
    - dt * tz1 * dz4;
  cu[4][3] = dt * tz2 * C2;

  cu[0][4] = dt * tz2
    * ( ( C2 * 2.0 * qs[k+1][j][i]
        - C1 * u[k+1][j][i][4] )
             * ( u[k+1][j][i][3] * tmp2 ) )
    - dt * tz1
    * ( - ( c34 - c1345 ) * tmp3 * (u[k+1][j][i][1]*u[k+1][j][i][1])
        - ( c34 - c1345 ) * tmp3 * (u[k+1][j][i][2]*u[k+1][j][i][2])
        - ( r43*c34 - c1345 )* tmp3 * (u[k+1][j][i][3]*u[k+1][j][i][3])
        - c1345 * tmp2 * u[k+1][j][i][4] );
  cu[1][4] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k+1][j][i][1];
  cu[2][4] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k+1][j][i][2];
  cu[3][4] = dt * tz2
    * ( C1 * ( u[k+1][j][i][4] * tmp1 )
        - C2
        * ( qs[k+1][j][i] * tmp1
          + u[k+1][j][i][3]*u[k+1][j][i][3] * tmp2 ) )
    - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[k+1][j][i][3];
  cu[4][4] = dt * tz2
    * ( C1 * ( u[k+1][j][i][3] * tmp1 ) )
    - dt * tz1 * c1345 * tmp1
    - dt * tz1 * dz5;

  for (m = 0; m < 5; m++) {
    tv[m] = 
      omega * (  cu[0][m] * rsd[k+1][j][i][0]
               + cu[1][m] * rsd[k+1][j][i][1]
               + cu[2][m] * rsd[k+1][j][i][2]
               + cu[3][m] * rsd[k+1][j][i][3]
               + cu[4][m] * rsd[k+1][j][i][4] );
  }
    for (m = 0; m < 5; m++) {
      tv[m] = tv[m]
        + omega * ( bu[0][m] * rsd[k][j+1][i][0]
                  + au[0][m] * rsd[k][j][i+1][0]
                  + bu[1][m] * rsd[k][j+1][i][1]
                  + au[1][m] * rsd[k][j][i+1][1]
                  + bu[2][m] * rsd[k][j+1][i][2]
                  + au[2][m] * rsd[k][j][i+1][2]
                  + bu[3][m] * rsd[k][j+1][i][3]
                  + au[3][m] * rsd[k][j][i+1][3]
                  + bu[4][m] * rsd[k][j+1][i][4]
                  + au[4][m] * rsd[k][j][i+1][4] );
    }

    //---------------------------------------------------------------------
    // diagonal block inversion
    //---------------------------------------------------------------------
    for (m = 0; m < 5; m++) {
      tmat[m][0] = du[0][m];
      tmat[m][1] = du[1][m];
      tmat[m][2] = du[2][m];
      tmat[m][3] = du[3][m];
      tmat[m][4] = du[4][m];
    }

    tmp1 = 1.0 / tmat[0][0];
    tmp = tmp1 * tmat[1][0];
    tmat[1][1] =  tmat[1][1] - tmp * tmat[0][1];
    tmat[1][2] =  tmat[1][2] - tmp * tmat[0][2];
    tmat[1][3] =  tmat[1][3] - tmp * tmat[0][3];
    tmat[1][4] =  tmat[1][4] - tmp * tmat[0][4];
    tv[1] = tv[1] - tv[0] * tmp;

    tmp = tmp1 * tmat[2][0];
    tmat[2][1] =  tmat[2][1] - tmp * tmat[0][1];
    tmat[2][2] =  tmat[2][2] - tmp * tmat[0][2];
    tmat[2][3] =  tmat[2][3] - tmp * tmat[0][3];
    tmat[2][4] =  tmat[2][4] - tmp * tmat[0][4];
    tv[2] = tv[2] - tv[0] * tmp;

    tmp = tmp1 * tmat[3][0];
    tmat[3][1] =  tmat[3][1] - tmp * tmat[0][1];
    tmat[3][2] =  tmat[3][2] - tmp * tmat[0][2];
    tmat[3][3] =  tmat[3][3] - tmp * tmat[0][3];
    tmat[3][4] =  tmat[3][4] - tmp * tmat[0][4];
    tv[3] = tv[3] - tv[0] * tmp;

    tmp = tmp1 * tmat[4][0];
    tmat[4][1] =  tmat[4][1] - tmp * tmat[0][1];
    tmat[4][2] =  tmat[4][2] - tmp * tmat[0][2];
    tmat[4][3] =  tmat[4][3] - tmp * tmat[0][3];
    tmat[4][4] =  tmat[4][4] - tmp * tmat[0][4];
    tv[4] = tv[4] - tv[0] * tmp;

    tmp1 = 1.0 / tmat[1][1];
    tmp = tmp1 * tmat[2][1];
    tmat[2][2] =  tmat[2][2] - tmp * tmat[1][2];
    tmat[2][3] =  tmat[2][3] - tmp * tmat[1][3];
    tmat[2][4] =  tmat[2][4] - tmp * tmat[1][4];
    tv[2] = tv[2] - tv[1] * tmp;

    tmp = tmp1 * tmat[3][1];
    tmat[3][2] =  tmat[3][2] - tmp * tmat[1][2];
    tmat[3][3] =  tmat[3][3] - tmp * tmat[1][3];
    tmat[3][4] =  tmat[3][4] - tmp * tmat[1][4];
    tv[3] = tv[3] - tv[1] * tmp;

    tmp = tmp1 * tmat[4][1];
    tmat[4][2] =  tmat[4][2] - tmp * tmat[1][2];
    tmat[4][3] =  tmat[4][3] - tmp * tmat[1][3];
    tmat[4][4] =  tmat[4][4] - tmp * tmat[1][4];
    tv[4] = tv[4] - tv[1] * tmp;

    tmp1 = 1.0 / tmat[2][2];
    tmp = tmp1 * tmat[3][2];
    tmat[3][3] =  tmat[3][3] - tmp * tmat[2][3];
    tmat[3][4] =  tmat[3][4] - tmp * tmat[2][4];
    tv[3] = tv[3] - tv[2] * tmp;

    tmp = tmp1 * tmat[4][2];
    tmat[4][3] =  tmat[4][3] - tmp * tmat[2][3];
    tmat[4][4] =  tmat[4][4] - tmp * tmat[2][4];
    tv[4] = tv[4] - tv[2] * tmp;

    tmp1 = 1.0 / tmat[3][3];
    tmp = tmp1 * tmat[4][3];
    tmat[4][4] =  tmat[4][4] - tmp * tmat[3][4];
    tv[4] = tv[4] - tv[3] * tmp;

    //---------------------------------------------------------------------
    // back substitution
    //---------------------------------------------------------------------
    tv[4] = tv[4] / tmat[4][4];

    tv[3] = tv[3] - tmat[3][4] * tv[4];
    tv[3] = tv[3] / tmat[3][3];

    tv[2] = tv[2]
      - tmat[2][3] * tv[3]
      - tmat[2][4] * tv[4];
    tv[2] = tv[2] / tmat[2][2];

    tv[1] = tv[1]
      - tmat[1][2] * tv[2]
      - tmat[1][3] * tv[3]
      - tmat[1][4] * tv[4];
    tv[1] = tv[1] / tmat[1][1];

    tv[0] = tv[0]
      - tmat[0][1] * tv[1]
      - tmat[0][2] * tv[2]
      - tmat[0][3] * tv[3]
      - tmat[0][4] * tv[4];
    tv[0] = tv[0] / tmat[0][0];

    rsd[k][j][i][0] = rsd[k][j][i][0] - tv[0];
    rsd[k][j][i][1] = rsd[k][j][i][1] - tv[1];
    rsd[k][j][i][2] = rsd[k][j][i][2] - tv[2];
    rsd[k][j][i][3] = rsd[k][j][i][3] - tv[3];
    rsd[k][j][i][4] = rsd[k][j][i][4] - tv[4];
}
