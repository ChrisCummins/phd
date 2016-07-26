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
// compute the reciprocal of density, and the kinetic energy, 
// and the speed of sound.
//---------------------------------------------------------------------
__kernel void compute_rhs1(__global const double *g_u,
                           __global double *g_us,
                           __global double *g_vs,
                           __global double *g_ws,
                           __global double *g_qs,
                           __global double *g_rho_i,
                           __global double *g_square,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k;
  double rho_inv;
  double p_u[4];
  double p_square;

#if COMPUTE_RHS1_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k >= gp2 || j >= gp1 || i >= gp0) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  p_u[0] = u[k][j][i][0];
  p_u[1] = u[k][j][i][1];
  p_u[2] = u[k][j][i][2];
  p_u[3] = u[k][j][i][3];

  rho_inv = 1.0/p_u[0];
  rho_i[k][j][i] = rho_inv;
  us[k][j][i] = p_u[1] * rho_inv;
  vs[k][j][i] = p_u[2] * rho_inv;
  ws[k][j][i] = p_u[3] * rho_inv;
  p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
  square[k][j][i] = p_square;
  qs[k][j][i] = p_square * rho_inv;

#elif COMPUTE_RHS1_DIM == 2
  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= gp2 || j >= gp1) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  for (i = 0; i < gp0; i++) {
    p_u[0] = u[k][j][i][0];
    p_u[1] = u[k][j][i][1];
    p_u[2] = u[k][j][i][2];
    p_u[3] = u[k][j][i][3];

    rho_inv = 1.0/p_u[0];
    rho_i[k][j][i] = rho_inv;
    us[k][j][i] = p_u[1] * rho_inv;
    vs[k][j][i] = p_u[2] * rho_inv;
    ws[k][j][i] = p_u[3] * rho_inv;
    p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
    square[k][j][i] = p_square;
    qs[k][j][i] = p_square * rho_inv;
  }

#else
  k = get_global_id(0);
  if (k >= gp2) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  for (j = 0; j < gp1; j++) {
    for (i = 0; i < gp0; i++) {
      p_u[0] = u[k][j][i][0];
      p_u[1] = u[k][j][i][1];
      p_u[2] = u[k][j][i][2];
      p_u[3] = u[k][j][i][3];

      rho_inv = 1.0/p_u[0];
      rho_i[k][j][i] = rho_inv;
      us[k][j][i] = p_u[1] * rho_inv;
      vs[k][j][i] = p_u[2] * rho_inv;
      ws[k][j][i] = p_u[3] * rho_inv;
      p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
      square[k][j][i] = p_square;
      qs[k][j][i] = p_square * rho_inv;
    }
  }
#endif
}


//---------------------------------------------------------------------
// copy the exact forcing term to the right hand side;  because 
// this forcing term is known, we can store it on the whole grid
// including the boundary                   
//---------------------------------------------------------------------
__kernel void compute_rhs2(__global const double *g_forcing,
                           __global double *g_rhs,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k, m;

#if COMPUTE_RHS2_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k >= gp2 || j >= gp1 || i >= gp0) return;

  __global double (*forcing)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_forcing;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = forcing[k][j][i][m];
  }

#elif COMPUTE_RHS2_DIM == 2
  k = get_global_id(1);
  j = get_global_id(0);
  if (k >= gp2 || j >= gp1) return;

  __global double (*forcing)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_forcing;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 0; i < gp0; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = forcing[k][j][i][m];
    }
  }

#else //COMPUTE_RHS2_DIM == 1
  k = get_global_id(0);
  if (k >= gp2) return;

  __global double (*forcing)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_forcing;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 0; j < gp1; j++) {
    for (i = 0; i < gp0; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = forcing[k][j][i][m];
      }
    }
  }
#endif
}


//---------------------------------------------------------------------
// compute xi-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs3(__global const double *g_u,
                           __global const double *g_us,
                           __global const double *g_vs,
                           __global const double *g_ws,
                           __global const double *g_qs,
                           __global const double *g_rho_i,
                           __global const double *g_square,
                           __global double *g_rhs,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k, m;
  double p_rhs[5];
  double p_u[5], p_up1[5], p_up2[5], p_um2[5];
  double p_um1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  p_u[0]   = u[k][j][0][0];
  p_u[1]   = u[k][j][0][1];
  p_u[2]   = u[k][j][0][2];
  p_u[3]   = u[k][j][0][3];
  p_u[4]   = u[k][j][0][4];
  p_up1[0] = u[k][j][1][0];
  p_up1[1] = u[k][j][1][1];
  p_up1[2] = u[k][j][1][2];
  p_up1[3] = u[k][j][1][3];
  p_up1[4] = u[k][j][1][4];
  p_up2[0] = u[k][j][2][0];
  p_up2[1] = u[k][j][2][1];
  p_up2[2] = u[k][j][2][2];
  p_up2[3] = u[k][j][2][3];
  p_up2[4] = u[k][j][2][4];

  p_us   = us[k][j][0];
  p_usp1 = us[k][j][1];
  p_vs   = vs[k][j][0];
  p_vsp1 = vs[k][j][1];
  p_ws   = ws[k][j][0];
  p_wsp1 = ws[k][j][1];
  p_qs   = qs[k][j][0];
  p_qsp1 = qs[k][j][1];
  p_rho_i   = rho_i[k][j][0];
  p_rho_ip1 = rho_i[k][j][1];
  p_square   = square[k][j][0];
  p_squarep1 = square[k][j][1];

#define LOOP_PROLOGUE_FULL         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
  p_up2[0] = u[k][j][i+2][0];      \
  p_up2[1] = u[k][j][i+2][1];      \
  p_up2[2] = u[k][j][i+2][2];      \
  p_up2[3] = u[k][j][i+2][3];      \
  p_up2[4] = u[k][j][i+2][4];      \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k][j][i+1];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k][j][i+1];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k][j][i+1];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k][j][i+1];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k][j][i+1];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k][j][i+1];

#define LOOP_PROLOGUE_HALF         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k][j][i+1];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k][j][i+1];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k][j][i+1];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k][j][i+1];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k][j][i+1];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k][j][i+1];

#define LOOP_BODY                                                     \
  p_rhs[0] = rhs[k][j][i][0] + dx1tx1 *                               \
    (p_up1[0] - 2.0*p_u[0] + p_um1[0]) -                              \
    tx2 * (p_up1[1] - p_um1[1]);                                      \
                                                                      \
  p_rhs[1] = rhs[k][j][i][1] + dx2tx1 *                               \
    (p_up1[1] - 2.0*p_u[1] + p_um1[1]) +                              \
    xxcon2*con43 * (p_usp1 - 2.0*p_us + p_usm1) -                     \
    tx2 * (p_up1[1]*p_usp1 - p_um1[1]*p_usm1 +                        \
          (p_up1[4] - p_squarep1 -                                    \
           p_um1[4] + p_squarem1) * c2);                              \
                                                                      \
  p_rhs[2] = rhs[k][j][i][2] + dx3tx1 *                               \
    (p_up1[2] - 2.0*p_u[2] + p_um1[2]) +                              \
    xxcon2 * (p_vsp1 - 2.0*p_vs + p_vsm1) -                           \
    tx2 * (p_up1[2]*p_usp1 - p_um1[2]*p_usm1);                        \
                                                                      \
  p_rhs[3] = rhs[k][j][i][3] + dx4tx1 *                               \
    (p_up1[3] - 2.0*p_u[3] + p_um1[3]) +                              \
    xxcon2 * (p_wsp1 - 2.0*p_ws + p_wsm1) -                           \
    tx2 * (p_up1[3]*p_usp1 - p_um1[3]*p_usm1);                        \
                                                                      \
  p_rhs[4] = rhs[k][j][i][4] + dx5tx1 *                               \
    (p_up1[4] - 2.0*p_u[4] + p_um1[4]) +                              \
    xxcon3 * (p_qsp1 - 2.0*p_qs + p_qsm1) +                           \
    xxcon4 * (p_usp1*p_usp1 -       2.0*p_us*p_us + p_usm1*p_usm1) +  \
    xxcon5 * (p_up1[4]*p_rho_ip1 -                                    \
          2.0*p_u[4]*p_rho_i +                                        \
              p_um1[4]*p_rho_im1) -                                   \
    tx2 * ( (c1*p_up1[4] - c2*p_squarep1)*p_usp1 -                    \
            (c1*p_um1[4] - c2*p_squarem1)*p_usm1 );

  i = 1;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m]- dssp * 
      (5.0*p_u[m] - 4.0*p_up1[m] + p_up2[m]);
  }

  i = 2;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp * 
      (-4.0*p_um1[m] + 6.0*p_u[m] -
        4.0*p_up1[m] + p_up2[m]);
  }

  for (i = 3; i <= gp0-4; i++) {
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = p_rhs[m] - dssp * 
        ( p_um2[m] - 4.0*p_um1[m] + 
        6.0*p_u[m] - 4.0*p_up1[m] + 
          p_up2[m] );
    }
  }

  i = gp0-3;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 
      6.0*p_u[m] - 4.0*p_up1[m] );
  }

  i = gp0-2;
  LOOP_PROLOGUE_HALF
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 5.0*p_u[m] );
  }

#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY
}


//---------------------------------------------------------------------
// compute eta-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs4(__global const double *g_u,
                           __global const double *g_us,
                           __global const double *g_vs,
                           __global const double *g_ws,
                           __global const double *g_qs,
                           __global const double *g_rho_i,
                           __global const double *g_square,
                           __global double *g_rhs,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k, m;
  double p_rhs[5];
  double p_u[5], p_up1[5], p_up2[5], p_um2[5];
  double p_um1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

  k = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || i > (gp0-2)) return;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  p_u[0]   = u[k][0][i][0];
  p_u[1]   = u[k][0][i][1];
  p_u[2]   = u[k][0][i][2];
  p_u[3]   = u[k][0][i][3];
  p_u[4]   = u[k][0][i][4];
  p_up1[0] = u[k][1][i][0];
  p_up1[1] = u[k][1][i][1];
  p_up1[2] = u[k][1][i][2];
  p_up1[3] = u[k][1][i][3];
  p_up1[4] = u[k][1][i][4];
  p_up2[0] = u[k][2][i][0];
  p_up2[1] = u[k][2][i][1];
  p_up2[2] = u[k][2][i][2];
  p_up2[3] = u[k][2][i][3];
  p_up2[4] = u[k][2][i][4];

  p_us   = us[k][0][i];
  p_usp1 = us[k][1][i];
  p_vs   = vs[k][0][i];
  p_vsp1 = vs[k][1][i];
  p_ws   = ws[k][0][i];
  p_wsp1 = ws[k][1][i];
  p_qs   = qs[k][0][i];
  p_qsp1 = qs[k][1][i];
  p_rho_i   = rho_i[k][0][i];
  p_rho_ip1 = rho_i[k][1][i];
  p_square   = square[k][0][i];
  p_squarep1 = square[k][1][i];

#define LOOP_PROLOGUE_FULL         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
  p_up2[0] = u[k][j+2][i][0];      \
  p_up2[1] = u[k][j+2][i][1];      \
  p_up2[2] = u[k][j+2][i][2];      \
  p_up2[3] = u[k][j+2][i][3];      \
  p_up2[4] = u[k][j+2][i][4];      \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k][j+1][i];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k][j+1][i];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k][j+1][i];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k][j+1][i];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k][j+1][i];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k][j+1][i];

#define LOOP_PROLOGUE_HALF         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k][j+1][i];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k][j+1][i];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k][j+1][i];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k][j+1][i];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k][j+1][i];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k][j+1][i];

#define LOOP_BODY                                                      \
   p_rhs[0] = rhs[k][j][i][0] + dy1ty1 *                               \
     (p_up1[0] - 2.0*p_u[0] + p_um1[0]) -                              \
     ty2 * (p_up1[2] - p_um1[2]);                                      \
                                                                       \
   p_rhs[1] = rhs[k][j][i][1] + dy2ty1 *                               \
     (p_up1[1] - 2.0*p_u[1] + p_um1[1]) +                              \
     yycon2 * (p_usp1 - 2.0*p_us + p_usm1) -                           \
     ty2 * (p_up1[1]*p_vsp1 - p_um1[1]*p_vsm1);                        \
                                                                       \
   p_rhs[2] = rhs[k][j][i][2] + dy3ty1 *                               \
     (p_up1[2] - 2.0*p_u[2] + p_um1[2]) +                              \
     yycon2*con43 * (p_vsp1 - 2.0*p_vs + p_vsm1) -                     \
     ty2 * (p_up1[2]*p_vsp1 - p_um1[2]*p_vsm1 +                        \
           (p_up1[4] - p_squarep1 -                                    \
            p_um1[4] + p_squarem1) * c2);                              \
                                                                       \
   p_rhs[3] = rhs[k][j][i][3] + dy4ty1 *                               \
     (p_up1[3] - 2.0*p_u[3] + p_um1[3]) +                              \
     yycon2 * (p_wsp1 - 2.0*p_ws + p_wsm1) -                           \
     ty2 * (p_up1[3]*p_vsp1 - p_um1[3]*p_vsm1);                        \
                                                                       \
   p_rhs[4] = rhs[k][j][i][4] + dy5ty1 *                               \
     (p_up1[4] - 2.0*p_u[4] + p_um1[4]) +                              \
     yycon3 * (p_qsp1 - 2.0*p_qs + p_qsm1) +                           \
     yycon4 * (p_vsp1*p_vsp1       - 2.0*p_vs*p_vs + p_vsm1*p_vsm1) +  \
     yycon5 * (p_up1[4]*p_rho_ip1 -                                    \
             2.0*p_u[4]*p_rho_i +                                      \
               p_um1[4]*p_rho_im1) -                                   \
     ty2 * ((c1*p_up1[4] - c2*p_squarep1) * p_vsp1 -                   \
            (c1*p_um1[4] - c2*p_squarem1) * p_vsm1);

  //---------------------------------------------------------------------
  // add fourth order eta-direction dissipation         
  //---------------------------------------------------------------------
  j = 1;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m]- dssp * 
      ( 5.0*p_u[m] - 4.0*p_up1[m] + p_up2[m]);
  }

  j = 2;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp * 
      (-4.0*p_um1[m] + 6.0*p_u[m] -
        4.0*p_up1[m] + p_up2[m]);
  }

  for (j = 3; j <= gp1-4; j++) {
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = p_rhs[m] - dssp * 
        ( p_um2[m] - 4.0*p_um1[m] + 
        6.0*p_u[m] - 4.0*p_up1[m] + 
          p_up2[m] );
    }
  }

  j = gp1-3;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 
      6.0*p_u[m] - 4.0*p_up1[m] );
  }

  j = gp1-2;
  LOOP_PROLOGUE_HALF
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 5.0*p_u[m] );
  }

#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY
}


//---------------------------------------------------------------------
// compute zeta-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs5(__global const double *g_u,
                           __global const double *g_us,
                           __global const double *g_vs,
                           __global const double *g_ws,
                           __global const double *g_qs,
                           __global const double *g_rho_i,
                           __global const double *g_square,
                           __global double *g_rhs,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k, m;
  double p_rhs[5];
  double p_u[5], p_up1[5], p_up2[5], p_um2[5];
  double p_um1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (j > (gp1-2) || i > (gp0-2)) return;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  p_u[0]   = u[0][j][i][0];
  p_u[1]   = u[0][j][i][1];
  p_u[2]   = u[0][j][i][2];
  p_u[3]   = u[0][j][i][3];
  p_u[4]   = u[0][j][i][4];
  p_up1[0] = u[1][j][i][0];
  p_up1[1] = u[1][j][i][1];
  p_up1[2] = u[1][j][i][2];
  p_up1[3] = u[1][j][i][3];
  p_up1[4] = u[1][j][i][4];
  p_up2[0] = u[2][j][i][0];
  p_up2[1] = u[2][j][i][1];
  p_up2[2] = u[2][j][i][2];
  p_up2[3] = u[2][j][i][3];
  p_up2[4] = u[2][j][i][4];

  p_us   = us[0][j][i];
  p_usp1 = us[1][j][i];
  p_vs   = vs[0][j][i];
  p_vsp1 = vs[1][j][i];
  p_ws   = ws[0][j][i];
  p_wsp1 = ws[1][j][i];
  p_qs   = qs[0][j][i];
  p_qsp1 = qs[1][j][i];
  p_rho_i   = rho_i[0][j][i];
  p_rho_ip1 = rho_i[1][j][i];
  p_square   = square[0][j][i];
  p_squarep1 = square[1][j][i];

#define LOOP_PROLOGUE_FULL         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
  p_up2[0] = u[k+2][j][i][0];      \
  p_up2[1] = u[k+2][j][i][1];      \
  p_up2[2] = u[k+2][j][i][2];      \
  p_up2[3] = u[k+2][j][i][3];      \
  p_up2[4] = u[k+2][j][i][4];      \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k+1][j][i];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k+1][j][i];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k+1][j][i];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k+1][j][i];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k+1][j][i];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k+1][j][i];

#define LOOP_PROLOGUE_HALF         \
  p_um2[0] = p_um1[0];             \
  p_um2[1] = p_um1[1];             \
  p_um2[2] = p_um1[2];             \
  p_um2[3] = p_um1[3];             \
  p_um2[4] = p_um1[4];             \
  p_um1[0] = p_u[0];               \
  p_um1[1] = p_u[1];               \
  p_um1[2] = p_u[2];               \
  p_um1[3] = p_u[3];               \
  p_um1[4] = p_u[4];               \
  p_u[0]   = p_up1[0];             \
  p_u[1]   = p_up1[1];             \
  p_u[2]   = p_up1[2];             \
  p_u[3]   = p_up1[3];             \
  p_u[4]   = p_up1[4];             \
  p_up1[0] = p_up2[0];             \
  p_up1[1] = p_up2[1];             \
  p_up1[2] = p_up2[2];             \
  p_up1[3] = p_up2[3];             \
  p_up1[4] = p_up2[4];             \
                                   \
  p_usm1 = p_us;                   \
  p_us   = p_usp1;                 \
  p_usp1 = us[k+1][j][i];          \
  p_vsm1 = p_vs;                   \
  p_vs   = p_vsp1;                 \
  p_vsp1 = vs[k+1][j][i];          \
  p_wsm1 = p_ws;                   \
  p_ws   = p_wsp1;                 \
  p_wsp1 = ws[k+1][j][i];          \
  p_qsm1 = p_qs;                   \
  p_qs   = p_qsp1;                 \
  p_qsp1 = qs[k+1][j][i];          \
  p_rho_im1 = p_rho_i;             \
  p_rho_i   = p_rho_ip1;           \
  p_rho_ip1 = rho_i[k+1][j][i];    \
  p_squarem1 = p_square;           \
  p_square   = p_squarep1;         \
  p_squarep1 = square[k+1][j][i];

#define LOOP_BODY                                               \
  p_rhs[0] = rhs[k][j][i][0] + dz1tz1 *                         \
    (p_up1[0] - 2.0*p_u[0] + p_um1[0]) -                        \
    tz2 * (p_up1[3] - p_um1[3]);                                \
                                                                \
  p_rhs[1] = rhs[k][j][i][1] + dz2tz1 *                         \
    (p_up1[1] - 2.0*p_u[1] + p_um1[1]) +                        \
    zzcon2 * (p_usp1 - 2.0*p_us + p_usm1) -                     \
    tz2 * (p_up1[1]*p_wsp1 - p_um1[1]*p_wsm1);                  \
                                                                \
  p_rhs[2] = rhs[k][j][i][2] + dz3tz1 *                         \
    (p_up1[2] - 2.0*p_u[2] + p_um1[2]) +                        \
    zzcon2 * (p_vsp1 - 2.0*p_vs + p_vsm1) -                     \
    tz2 * (p_up1[2]*p_wsp1 - p_um1[2]*p_wsm1);                  \
                                                                \
  p_rhs[3] = rhs[k][j][i][3] + dz4tz1 *                         \
    (p_up1[3] - 2.0*p_u[3] + p_um1[3]) +                        \
    zzcon2*con43 * (p_wsp1 - 2.0*p_ws + p_wsm1) -               \
    tz2 * (p_up1[3]*p_wsp1 - p_um1[3]*p_wsm1 +                  \
          (p_up1[4] - p_squarep1 -                              \
           p_um1[4] + p_squarem1) * c2);                        \
                                                                \
  p_rhs[4] = rhs[k][j][i][4] + dz5tz1 *                         \
    (p_up1[4] - 2.0*p_u[4] + p_um1[4]) +                        \
    zzcon3 * (p_qsp1 - 2.0*p_qs + p_qsm1) +                     \
    zzcon4 * (p_wsp1*p_wsp1 - 2.0*p_ws*p_ws + p_wsm1*p_wsm1) +  \
    zzcon5 * (p_up1[4]*p_rho_ip1 -                              \
            2.0*p_u[4]*p_rho_i +                                \
              p_um1[4]*p_rho_im1) -                             \
    tz2 * ((c1*p_up1[4] - c2*p_squarep1)*p_wsp1 -               \
           (c1*p_um1[4] - c2*p_squarem1)*p_wsm1);

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m]- dssp * 
      (5.0*p_u[m] - 4.0*p_up1[m] + p_up2[m]);
  }

  k = 2;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp * 
      (-4.0*p_um1[m] + 6.0*p_u[m] -
        4.0*p_up1[m] + p_up2[m]);
  }

  for (k = 3; k <= gp2-4; k++) {
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = p_rhs[m] - dssp * 
        ( p_um2[m] - 4.0*p_um1[m] + 
        6.0*p_u[m] - 4.0*p_up1[m] + 
          p_up2[m] );
    }
  }

  k = gp2-3;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 
      6.0*p_u[m] - 4.0*p_up1[m] );
  }

  k = gp2-2;
  LOOP_PROLOGUE_HALF
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 5.0*p_u[m] );
  }

#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY
}


__kernel void compute_rhs6(__global double *g_rhs,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k, m;

#if COMPUTE_RHS6_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2) || i > (gp0-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
  }

#elif COMPUTE_RHS6_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= (gp0-2); i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
    }
  }

#else //COMPUTE_RHS6_DIM == 1
  k = get_global_id(0) + 1;
  if (k > (gp2-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= (gp1-2); j++) {
    for (i = 1; i <= (gp0-2); i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
      }
    }
  }
#endif
}


#if (X_SOLVE_DIM != 3 || Y_SOLVE_DIM != 3 || Z_SOLVE_DIM != 3)
void lhsinit(__global double lhs[][3][5][5], int ni)
{
  int i, m, n;

  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  i = 0;
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
    lhs[i][1][n][n] = 1.0;
  }
  i = ni;
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
    lhs[i][1][n][n] = 1.0;
  }
}
#endif


//---------------------------------------------------------------------
// subtracts bvec=bvec - ablock*avec
//---------------------------------------------------------------------
void p_matvec_sub(/*__global*/ double ablock[5][5],
                  /*__global*/ double avec[5],
                  /*__global*/ double bvec[5])
{
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - ablock[0][0]*avec[0]
                    - ablock[1][0]*avec[1]
                    - ablock[2][0]*avec[2]
                    - ablock[3][0]*avec[3]
                    - ablock[4][0]*avec[4];
  bvec[1] = bvec[1] - ablock[0][1]*avec[0]
                    - ablock[1][1]*avec[1]
                    - ablock[2][1]*avec[2]
                    - ablock[3][1]*avec[3]
                    - ablock[4][1]*avec[4];
  bvec[2] = bvec[2] - ablock[0][2]*avec[0]
                    - ablock[1][2]*avec[1]
                    - ablock[2][2]*avec[2]
                    - ablock[3][2]*avec[3]
                    - ablock[4][2]*avec[4];
  bvec[3] = bvec[3] - ablock[0][3]*avec[0]
                    - ablock[1][3]*avec[1]
                    - ablock[2][3]*avec[2]
                    - ablock[3][3]*avec[3]
                    - ablock[4][3]*avec[4];
  bvec[4] = bvec[4] - ablock[0][4]*avec[0]
                    - ablock[1][4]*avec[1]
                    - ablock[2][4]*avec[2]
                    - ablock[3][4]*avec[3]
                    - ablock[4][4]*avec[4];
}


//---------------------------------------------------------------------
// subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
//---------------------------------------------------------------------
void p_matmul_sub(/*__global*/ double ablock[5][5],
                  /*__global*/ double bblock[5][5],
                  /*__global*/ double cblock[5][5])
{
  cblock[0][0] = cblock[0][0] - ablock[0][0]*bblock[0][0]
                              - ablock[1][0]*bblock[0][1]
                              - ablock[2][0]*bblock[0][2]
                              - ablock[3][0]*bblock[0][3]
                              - ablock[4][0]*bblock[0][4];
  cblock[0][1] = cblock[0][1] - ablock[0][1]*bblock[0][0]
                              - ablock[1][1]*bblock[0][1]
                              - ablock[2][1]*bblock[0][2]
                              - ablock[3][1]*bblock[0][3]
                              - ablock[4][1]*bblock[0][4];
  cblock[0][2] = cblock[0][2] - ablock[0][2]*bblock[0][0]
                              - ablock[1][2]*bblock[0][1]
                              - ablock[2][2]*bblock[0][2]
                              - ablock[3][2]*bblock[0][3]
                              - ablock[4][2]*bblock[0][4];
  cblock[0][3] = cblock[0][3] - ablock[0][3]*bblock[0][0]
                              - ablock[1][3]*bblock[0][1]
                              - ablock[2][3]*bblock[0][2]
                              - ablock[3][3]*bblock[0][3]
                              - ablock[4][3]*bblock[0][4];
  cblock[0][4] = cblock[0][4] - ablock[0][4]*bblock[0][0]
                              - ablock[1][4]*bblock[0][1]
                              - ablock[2][4]*bblock[0][2]
                              - ablock[3][4]*bblock[0][3]
                              - ablock[4][4]*bblock[0][4];
  cblock[1][0] = cblock[1][0] - ablock[0][0]*bblock[1][0]
                              - ablock[1][0]*bblock[1][1]
                              - ablock[2][0]*bblock[1][2]
                              - ablock[3][0]*bblock[1][3]
                              - ablock[4][0]*bblock[1][4];
  cblock[1][1] = cblock[1][1] - ablock[0][1]*bblock[1][0]
                              - ablock[1][1]*bblock[1][1]
                              - ablock[2][1]*bblock[1][2]
                              - ablock[3][1]*bblock[1][3]
                              - ablock[4][1]*bblock[1][4];
  cblock[1][2] = cblock[1][2] - ablock[0][2]*bblock[1][0]
                              - ablock[1][2]*bblock[1][1]
                              - ablock[2][2]*bblock[1][2]
                              - ablock[3][2]*bblock[1][3]
                              - ablock[4][2]*bblock[1][4];
  cblock[1][3] = cblock[1][3] - ablock[0][3]*bblock[1][0]
                              - ablock[1][3]*bblock[1][1]
                              - ablock[2][3]*bblock[1][2]
                              - ablock[3][3]*bblock[1][3]
                              - ablock[4][3]*bblock[1][4];
  cblock[1][4] = cblock[1][4] - ablock[0][4]*bblock[1][0]
                              - ablock[1][4]*bblock[1][1]
                              - ablock[2][4]*bblock[1][2]
                              - ablock[3][4]*bblock[1][3]
                              - ablock[4][4]*bblock[1][4];
  cblock[2][0] = cblock[2][0] - ablock[0][0]*bblock[2][0]
                              - ablock[1][0]*bblock[2][1]
                              - ablock[2][0]*bblock[2][2]
                              - ablock[3][0]*bblock[2][3]
                              - ablock[4][0]*bblock[2][4];
  cblock[2][1] = cblock[2][1] - ablock[0][1]*bblock[2][0]
                              - ablock[1][1]*bblock[2][1]
                              - ablock[2][1]*bblock[2][2]
                              - ablock[3][1]*bblock[2][3]
                              - ablock[4][1]*bblock[2][4];
  cblock[2][2] = cblock[2][2] - ablock[0][2]*bblock[2][0]
                              - ablock[1][2]*bblock[2][1]
                              - ablock[2][2]*bblock[2][2]
                              - ablock[3][2]*bblock[2][3]
                              - ablock[4][2]*bblock[2][4];
  cblock[2][3] = cblock[2][3] - ablock[0][3]*bblock[2][0]
                              - ablock[1][3]*bblock[2][1]
                              - ablock[2][3]*bblock[2][2]
                              - ablock[3][3]*bblock[2][3]
                              - ablock[4][3]*bblock[2][4];
  cblock[2][4] = cblock[2][4] - ablock[0][4]*bblock[2][0]
                              - ablock[1][4]*bblock[2][1]
                              - ablock[2][4]*bblock[2][2]
                              - ablock[3][4]*bblock[2][3]
                              - ablock[4][4]*bblock[2][4];
  cblock[3][0] = cblock[3][0] - ablock[0][0]*bblock[3][0]
                              - ablock[1][0]*bblock[3][1]
                              - ablock[2][0]*bblock[3][2]
                              - ablock[3][0]*bblock[3][3]
                              - ablock[4][0]*bblock[3][4];
  cblock[3][1] = cblock[3][1] - ablock[0][1]*bblock[3][0]
                              - ablock[1][1]*bblock[3][1]
                              - ablock[2][1]*bblock[3][2]
                              - ablock[3][1]*bblock[3][3]
                              - ablock[4][1]*bblock[3][4];
  cblock[3][2] = cblock[3][2] - ablock[0][2]*bblock[3][0]
                              - ablock[1][2]*bblock[3][1]
                              - ablock[2][2]*bblock[3][2]
                              - ablock[3][2]*bblock[3][3]
                              - ablock[4][2]*bblock[3][4];
  cblock[3][3] = cblock[3][3] - ablock[0][3]*bblock[3][0]
                              - ablock[1][3]*bblock[3][1]
                              - ablock[2][3]*bblock[3][2]
                              - ablock[3][3]*bblock[3][3]
                              - ablock[4][3]*bblock[3][4];
  cblock[3][4] = cblock[3][4] - ablock[0][4]*bblock[3][0]
                              - ablock[1][4]*bblock[3][1]
                              - ablock[2][4]*bblock[3][2]
                              - ablock[3][4]*bblock[3][3]
                              - ablock[4][4]*bblock[3][4];
  cblock[4][0] = cblock[4][0] - ablock[0][0]*bblock[4][0]
                              - ablock[1][0]*bblock[4][1]
                              - ablock[2][0]*bblock[4][2]
                              - ablock[3][0]*bblock[4][3]
                              - ablock[4][0]*bblock[4][4];
  cblock[4][1] = cblock[4][1] - ablock[0][1]*bblock[4][0]
                              - ablock[1][1]*bblock[4][1]
                              - ablock[2][1]*bblock[4][2]
                              - ablock[3][1]*bblock[4][3]
                              - ablock[4][1]*bblock[4][4];
  cblock[4][2] = cblock[4][2] - ablock[0][2]*bblock[4][0]
                              - ablock[1][2]*bblock[4][1]
                              - ablock[2][2]*bblock[4][2]
                              - ablock[3][2]*bblock[4][3]
                              - ablock[4][2]*bblock[4][4];
  cblock[4][3] = cblock[4][3] - ablock[0][3]*bblock[4][0]
                              - ablock[1][3]*bblock[4][1]
                              - ablock[2][3]*bblock[4][2]
                              - ablock[3][3]*bblock[4][3]
                              - ablock[4][3]*bblock[4][4];
  cblock[4][4] = cblock[4][4] - ablock[0][4]*bblock[4][0]
                              - ablock[1][4]*bblock[4][1]
                              - ablock[2][4]*bblock[4][2]
                              - ablock[3][4]*bblock[4][3]
                              - ablock[4][4]*bblock[4][4];
}


void p_binvcrhs(/*__global*/ double lhs[5][5],
                /*__global*/ double c[5][5],
                /*__global*/ double r[5])
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0];
  lhs[1][0] = lhs[1][0]*pivot;
  lhs[2][0] = lhs[2][0]*pivot;
  lhs[3][0] = lhs[3][0]*pivot;
  lhs[4][0] = lhs[4][0]*pivot;
  c[0][0] = c[0][0]*pivot;
  c[1][0] = c[1][0]*pivot;
  c[2][0] = c[2][0]*pivot;
  c[3][0] = c[3][0]*pivot;
  c[4][0] = c[4][0]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1];
  lhs[1][1]= lhs[1][1] - coeff*lhs[1][0];
  lhs[2][1]= lhs[2][1] - coeff*lhs[2][0];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][0];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][0];
  c[0][1] = c[0][1] - coeff*c[0][0];
  c[1][1] = c[1][1] - coeff*c[1][0];
  c[2][1] = c[2][1] - coeff*c[2][0];
  c[3][1] = c[3][1] - coeff*c[3][0];
  c[4][1] = c[4][1] - coeff*c[4][0];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2];
  lhs[1][2]= lhs[1][2] - coeff*lhs[1][0];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][0];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][0];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][0];
  c[0][2] = c[0][2] - coeff*c[0][0];
  c[1][2] = c[1][2] - coeff*c[1][0];
  c[2][2] = c[2][2] - coeff*c[2][0];
  c[3][2] = c[3][2] - coeff*c[3][0];
  c[4][2] = c[4][2] - coeff*c[4][0];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3];
  lhs[1][3]= lhs[1][3] - coeff*lhs[1][0];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][0];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][0];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][0];
  c[0][3] = c[0][3] - coeff*c[0][0];
  c[1][3] = c[1][3] - coeff*c[1][0];
  c[2][3] = c[2][3] - coeff*c[2][0];
  c[3][3] = c[3][3] - coeff*c[3][0];
  c[4][3] = c[4][3] - coeff*c[4][0];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4];
  lhs[1][4]= lhs[1][4] - coeff*lhs[1][0];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][0];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][0];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][0];
  c[0][4] = c[0][4] - coeff*c[0][0];
  c[1][4] = c[1][4] - coeff*c[1][0];
  c[2][4] = c[2][4] - coeff*c[2][0];
  c[3][4] = c[3][4] - coeff*c[3][0];
  c[4][4] = c[4][4] - coeff*c[4][0];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1];
  lhs[2][1] = lhs[2][1]*pivot;
  lhs[3][1] = lhs[3][1]*pivot;
  lhs[4][1] = lhs[4][1]*pivot;
  c[0][1] = c[0][1]*pivot;
  c[1][1] = c[1][1]*pivot;
  c[2][1] = c[2][1]*pivot;
  c[3][1] = c[3][1]*pivot;
  c[4][1] = c[4][1]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0];
  lhs[2][0]= lhs[2][0] - coeff*lhs[2][1];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][1];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][1];
  c[0][0] = c[0][0] - coeff*c[0][1];
  c[1][0] = c[1][0] - coeff*c[1][1];
  c[2][0] = c[2][0] - coeff*c[2][1];
  c[3][0] = c[3][0] - coeff*c[3][1];
  c[4][0] = c[4][0] - coeff*c[4][1];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][1];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][1];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][1];
  c[0][2] = c[0][2] - coeff*c[0][1];
  c[1][2] = c[1][2] - coeff*c[1][1];
  c[2][2] = c[2][2] - coeff*c[2][1];
  c[3][2] = c[3][2] - coeff*c[3][1];
  c[4][2] = c[4][2] - coeff*c[4][1];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][1];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][1];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][1];
  c[0][3] = c[0][3] - coeff*c[0][1];
  c[1][3] = c[1][3] - coeff*c[1][1];
  c[2][3] = c[2][3] - coeff*c[2][1];
  c[3][3] = c[3][3] - coeff*c[3][1];
  c[4][3] = c[4][3] - coeff*c[4][1];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][1];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][1];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][1];
  c[0][4] = c[0][4] - coeff*c[0][1];
  c[1][4] = c[1][4] - coeff*c[1][1];
  c[2][4] = c[2][4] - coeff*c[2][1];
  c[3][4] = c[3][4] - coeff*c[3][1];
  c[4][4] = c[4][4] - coeff*c[4][1];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2];
  lhs[3][2] = lhs[3][2]*pivot;
  lhs[4][2] = lhs[4][2]*pivot;
  c[0][2] = c[0][2]*pivot;
  c[1][2] = c[1][2]*pivot;
  c[2][2] = c[2][2]*pivot;
  c[3][2] = c[3][2]*pivot;
  c[4][2] = c[4][2]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][2];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][2];
  c[0][0] = c[0][0] - coeff*c[0][2];
  c[1][0] = c[1][0] - coeff*c[1][2];
  c[2][0] = c[2][0] - coeff*c[2][2];
  c[3][0] = c[3][0] - coeff*c[3][2];
  c[4][0] = c[4][0] - coeff*c[4][2];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][2];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][2];
  c[0][1] = c[0][1] - coeff*c[0][2];
  c[1][1] = c[1][1] - coeff*c[1][2];
  c[2][1] = c[2][1] - coeff*c[2][2];
  c[3][1] = c[3][1] - coeff*c[3][2];
  c[4][1] = c[4][1] - coeff*c[4][2];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][2];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][2];
  c[0][3] = c[0][3] - coeff*c[0][2];
  c[1][3] = c[1][3] - coeff*c[1][2];
  c[2][3] = c[2][3] - coeff*c[2][2];
  c[3][3] = c[3][3] - coeff*c[3][2];
  c[4][3] = c[4][3] - coeff*c[4][2];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][2];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][2];
  c[0][4] = c[0][4] - coeff*c[0][2];
  c[1][4] = c[1][4] - coeff*c[1][2];
  c[2][4] = c[2][4] - coeff*c[2][2];
  c[3][4] = c[3][4] - coeff*c[3][2];
  c[4][4] = c[4][4] - coeff*c[4][2];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3];
  lhs[4][3] = lhs[4][3]*pivot;
  c[0][3] = c[0][3]*pivot;
  c[1][3] = c[1][3]*pivot;
  c[2][3] = c[2][3]*pivot;
  c[3][3] = c[3][3]*pivot;
  c[4][3] = c[4][3]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][3];
  c[0][0] = c[0][0] - coeff*c[0][3];
  c[1][0] = c[1][0] - coeff*c[1][3];
  c[2][0] = c[2][0] - coeff*c[2][3];
  c[3][0] = c[3][0] - coeff*c[3][3];
  c[4][0] = c[4][0] - coeff*c[4][3];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][3];
  c[0][1] = c[0][1] - coeff*c[0][3];
  c[1][1] = c[1][1] - coeff*c[1][3];
  c[2][1] = c[2][1] - coeff*c[2][3];
  c[3][1] = c[3][1] - coeff*c[3][3];
  c[4][1] = c[4][1] - coeff*c[4][3];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][3];
  c[0][2] = c[0][2] - coeff*c[0][3];
  c[1][2] = c[1][2] - coeff*c[1][3];
  c[2][2] = c[2][2] - coeff*c[2][3];
  c[3][2] = c[3][2] - coeff*c[3][3];
  c[4][2] = c[4][2] - coeff*c[4][3];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][3];
  c[0][4] = c[0][4] - coeff*c[0][3];
  c[1][4] = c[1][4] - coeff*c[1][3];
  c[2][4] = c[2][4] - coeff*c[2][3];
  c[3][4] = c[3][4] - coeff*c[3][3];
  c[4][4] = c[4][4] - coeff*c[4][3];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4];
  c[0][4] = c[0][4]*pivot;
  c[1][4] = c[1][4]*pivot;
  c[2][4] = c[2][4]*pivot;
  c[3][4] = c[3][4]*pivot;
  c[4][4] = c[4][4]*pivot;
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0];
  c[0][0] = c[0][0] - coeff*c[0][4];
  c[1][0] = c[1][0] - coeff*c[1][4];
  c[2][0] = c[2][0] - coeff*c[2][4];
  c[3][0] = c[3][0] - coeff*c[3][4];
  c[4][0] = c[4][0] - coeff*c[4][4];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1];
  c[0][1] = c[0][1] - coeff*c[0][4];
  c[1][1] = c[1][1] - coeff*c[1][4];
  c[2][1] = c[2][1] - coeff*c[2][4];
  c[3][1] = c[3][1] - coeff*c[3][4];
  c[4][1] = c[4][1] - coeff*c[4][4];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2];
  c[0][2] = c[0][2] - coeff*c[0][4];
  c[1][2] = c[1][2] - coeff*c[1][4];
  c[2][2] = c[2][2] - coeff*c[2][4];
  c[3][2] = c[3][2] - coeff*c[3][4];
  c[4][2] = c[4][2] - coeff*c[4][4];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3];
  c[0][3] = c[0][3] - coeff*c[0][4];
  c[1][3] = c[1][3] - coeff*c[1][4];
  c[2][3] = c[2][3] - coeff*c[2][4];
  c[3][3] = c[3][3] - coeff*c[3][4];
  c[4][3] = c[4][3] - coeff*c[4][4];
  r[3]   = r[3]   - coeff*r[4];
}


void p_binvrhs(/*__global*/ double lhs[5][5], /*__global*/ double r[5])
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0];
  lhs[1][0] = lhs[1][0]*pivot;
  lhs[2][0] = lhs[2][0]*pivot;
  lhs[3][0] = lhs[3][0]*pivot;
  lhs[4][0] = lhs[4][0]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1];
  lhs[1][1]= lhs[1][1] - coeff*lhs[1][0];
  lhs[2][1]= lhs[2][1] - coeff*lhs[2][0];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][0];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][0];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2];
  lhs[1][2]= lhs[1][2] - coeff*lhs[1][0];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][0];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][0];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][0];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3];
  lhs[1][3]= lhs[1][3] - coeff*lhs[1][0];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][0];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][0];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][0];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4];
  lhs[1][4]= lhs[1][4] - coeff*lhs[1][0];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][0];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][0];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][0];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1];
  lhs[2][1] = lhs[2][1]*pivot;
  lhs[3][1] = lhs[3][1]*pivot;
  lhs[4][1] = lhs[4][1]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0];
  lhs[2][0]= lhs[2][0] - coeff*lhs[2][1];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][1];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][1];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][1];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][1];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][1];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][1];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][1];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][1];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][1];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][1];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][1];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2];
  lhs[3][2] = lhs[3][2]*pivot;
  lhs[4][2] = lhs[4][2]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][2];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][2];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][2];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][2];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][2];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][2];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][2];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][2];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3];
  lhs[4][3] = lhs[4][3]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][3];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][3];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][3];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][3];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4];
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3];
  r[3]   = r[3]   - coeff*r[4];
}

void load_matrix(double p_matrix[5][5], __global double g_matrix[5][5])
{
/*  int i, j;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      p_matrix[i][j] = g_matrix[i][j];*/
  p_matrix[0][0] = g_matrix[0][0];
  p_matrix[0][1] = g_matrix[0][1];
  p_matrix[0][2] = g_matrix[0][2];
  p_matrix[0][3] = g_matrix[0][3];
  p_matrix[0][4] = g_matrix[0][4];
  p_matrix[1][0] = g_matrix[1][0];
  p_matrix[1][1] = g_matrix[1][1];
  p_matrix[1][2] = g_matrix[1][2];
  p_matrix[1][3] = g_matrix[1][3];
  p_matrix[1][4] = g_matrix[1][4];
  p_matrix[2][0] = g_matrix[2][0];
  p_matrix[2][1] = g_matrix[2][1];
  p_matrix[2][2] = g_matrix[2][2];
  p_matrix[2][3] = g_matrix[2][3];
  p_matrix[2][4] = g_matrix[2][4];
  p_matrix[3][0] = g_matrix[3][0];
  p_matrix[3][1] = g_matrix[3][1];
  p_matrix[3][2] = g_matrix[3][2];
  p_matrix[3][3] = g_matrix[3][3];
  p_matrix[3][4] = g_matrix[3][4];
  p_matrix[4][0] = g_matrix[4][0];
  p_matrix[4][1] = g_matrix[4][1];
  p_matrix[4][2] = g_matrix[4][2];
  p_matrix[4][3] = g_matrix[4][3];
  p_matrix[4][4] = g_matrix[4][4];
}

void save_matrix(__global double g_matrix[5][5], double p_matrix[5][5])
{
/*  int i, j;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      g_matrix[i][j] = p_matrix[i][j];*/
  g_matrix[0][0] = p_matrix[0][0];
  g_matrix[0][1] = p_matrix[0][1];
  g_matrix[0][2] = p_matrix[0][2];
  g_matrix[0][3] = p_matrix[0][3];
  g_matrix[0][4] = p_matrix[0][4];
  g_matrix[1][0] = p_matrix[1][0];
  g_matrix[1][1] = p_matrix[1][1];
  g_matrix[1][2] = p_matrix[1][2];
  g_matrix[1][3] = p_matrix[1][3];
  g_matrix[1][4] = p_matrix[1][4];
  g_matrix[2][0] = p_matrix[2][0];
  g_matrix[2][1] = p_matrix[2][1];
  g_matrix[2][2] = p_matrix[2][2];
  g_matrix[2][3] = p_matrix[2][3];
  g_matrix[2][4] = p_matrix[2][4];
  g_matrix[3][0] = p_matrix[3][0];
  g_matrix[3][1] = p_matrix[3][1];
  g_matrix[3][2] = p_matrix[3][2];
  g_matrix[3][3] = p_matrix[3][3];
  g_matrix[3][4] = p_matrix[3][4];
  g_matrix[4][0] = p_matrix[4][0];
  g_matrix[4][1] = p_matrix[4][1];
  g_matrix[4][2] = p_matrix[4][2];
  g_matrix[4][3] = p_matrix[4][3];
  g_matrix[4][4] = p_matrix[4][4];
}

void copy_matrix(double p_matrix[5][5], double p_source[5][5])
{
/*  int i, j;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      p_matrix[i][j] = p_source[i][j];*/
  p_matrix[0][0] = p_source[0][0];
  p_matrix[0][1] = p_source[0][1];
  p_matrix[0][2] = p_source[0][2];
  p_matrix[0][3] = p_source[0][3];
  p_matrix[0][4] = p_source[0][4];
  p_matrix[1][0] = p_source[1][0];
  p_matrix[1][1] = p_source[1][1];
  p_matrix[1][2] = p_source[1][2];
  p_matrix[1][3] = p_source[1][3];
  p_matrix[1][4] = p_source[1][4];
  p_matrix[2][0] = p_source[2][0];
  p_matrix[2][1] = p_source[2][1];
  p_matrix[2][2] = p_source[2][2];
  p_matrix[2][3] = p_source[2][3];
  p_matrix[2][4] = p_source[2][4];
  p_matrix[3][0] = p_source[3][0];
  p_matrix[3][1] = p_source[3][1];
  p_matrix[3][2] = p_source[3][2];
  p_matrix[3][3] = p_source[3][3];
  p_matrix[3][4] = p_source[3][4];
  p_matrix[4][0] = p_source[4][0];
  p_matrix[4][1] = p_source[4][1];
  p_matrix[4][2] = p_source[4][2];
  p_matrix[4][3] = p_source[4][3];
  p_matrix[4][4] = p_source[4][4];

}

void load_vector(double p_vector[5], __global double g_vector[5])
{
/*  int i;
  for (i = 0; i < 5; i++)
    p_vector[i] = g_vector[i];*/
  p_vector[0] = g_vector[0];
  p_vector[1] = g_vector[1];
  p_vector[2] = g_vector[2];
  p_vector[3] = g_vector[3];
  p_vector[4] = g_vector[4];
}

void save_vector(__global double g_vector[5], double p_vector[5])
{
/*  int i;
  for (i = 0; i < 5; i++)
    g_vector[i] = p_vector[i];*/
  g_vector[0] = p_vector[0];
  g_vector[1] = p_vector[1];
  g_vector[2] = p_vector[2];
  g_vector[3] = p_vector[3];
  g_vector[4] = p_vector[4];
}

void copy_vector(double p_vector[5], double p_source[5])
{
/*  int i;
  for (i = 0; i < 5; i++)
    p_vector[i] = p_source[i];*/
  p_vector[0] = p_source[0];
  p_vector[1] = p_source[1];
  p_vector[2] = p_source[2];
  p_vector[3] = p_source[3];
  p_vector[4] = p_source[4];
}


#if X_SOLVE_DIM == 3
__kernel void x_solve1(__global double *g_qs,
                       __global double *g_rho_i,
                       __global double *g_square,
                       __global double *g_u,
                       __global double *g_fjac,
                       __global double *g_njac,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2, tmp3;
  double p_u[5];

  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0);
  if (k > (gp2-2) || j > (gp1-2) || i >= gp0) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  int my_id = (k-1)*(gp1-2) + (j-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;
  p_u[0] = u[k][j][i][0];
  p_u[1] = u[k][j][i][1];
  p_u[2] = u[k][j][i][2];
  p_u[3] = u[k][j][i][3];
  p_u[4] = u[k][j][i][4];
  //-------------------------------------------------------------------
  // 
  //-------------------------------------------------------------------
  fjac[i][0][0] = 0.0;
  fjac[i][1][0] = 1.0;
  fjac[i][2][0] = 0.0;
  fjac[i][3][0] = 0.0;
  fjac[i][4][0] = 0.0;

  fjac[i][0][1] = -(p_u[1] * tmp2 * p_u[1])
    + c2 * qs[k][j][i];
  fjac[i][1][1] = ( 2.0 - c2 ) * ( p_u[1] / p_u[0] );
  fjac[i][2][1] = - c2 * ( p_u[2] * tmp1 );
  fjac[i][3][1] = - c2 * ( p_u[3] * tmp1 );
  fjac[i][4][1] = c2;

  fjac[i][0][2] = - ( p_u[1]*p_u[2] ) * tmp2;
  fjac[i][1][2] = p_u[2] * tmp1;
  fjac[i][2][2] = p_u[1] * tmp1;
  fjac[i][3][2] = 0.0;
  fjac[i][4][2] = 0.0;

  fjac[i][0][3] = - ( p_u[1]*p_u[3] ) * tmp2;
  fjac[i][1][3] = p_u[3] * tmp1;
  fjac[i][2][3] = 0.0;
  fjac[i][3][3] = p_u[1] * tmp1;
  fjac[i][4][3] = 0.0;

  fjac[i][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
    * ( p_u[1] * tmp2 );
  fjac[i][1][4] = c1 *  p_u[4] * tmp1 
    - c2 * ( p_u[1]*p_u[1] * tmp2 + qs[k][j][i] );
  fjac[i][2][4] = - c2 * ( p_u[2]*p_u[1] ) * tmp2;
  fjac[i][3][4] = - c2 * ( p_u[3]*p_u[1] ) * tmp2;
  fjac[i][4][4] = c1 * ( p_u[1] * tmp1 );

  njac[i][0][0] = 0.0;
  njac[i][1][0] = 0.0;
  njac[i][2][0] = 0.0;
  njac[i][3][0] = 0.0;
  njac[i][4][0] = 0.0;

  njac[i][0][1] = - con43 * c3c4 * tmp2 * p_u[1];
  njac[i][1][1] =   con43 * c3c4 * tmp1;
  njac[i][2][1] =   0.0;
  njac[i][3][1] =   0.0;
  njac[i][4][1] =   0.0;

  njac[i][0][2] = - c3c4 * tmp2 * p_u[2];
  njac[i][1][2] =   0.0;
  njac[i][2][2] =   c3c4 * tmp1;
  njac[i][3][2] =   0.0;
  njac[i][4][2] =   0.0;

  njac[i][0][3] = - c3c4 * tmp2 * p_u[3];
  njac[i][1][3] =   0.0;
  njac[i][2][3] =   0.0;
  njac[i][3][3] =   c3c4 * tmp1;
  njac[i][4][3] =   0.0;

  njac[i][0][4] = - ( con43 * c3c4
      - c1345 ) * tmp3 * (p_u[1]*p_u[1])
    - ( c3c4 - c1345 ) * tmp3 * (p_u[2]*p_u[2])
    - ( c3c4 - c1345 ) * tmp3 * (p_u[3]*p_u[3])
    - c1345 * tmp2 * p_u[4];

  njac[i][1][4] = ( con43 * c3c4
      - c1345 ) * tmp2 * p_u[1];
  njac[i][2][4] = ( c3c4 - c1345 ) * tmp2 * p_u[2];
  njac[i][3][4] = ( c3c4 - c1345 ) * tmp2 * p_u[3];
  njac[i][4][4] = ( c1345 ) * tmp1;
}

__kernel void x_solve2(__global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k, n, m;

  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  i = get_global_id(0);
  if (i == 1) i = gp0-1;
  
  int my_id = (k-1)*(gp1-2) + (j-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
    lhs[i][1][n][n] = 1.0;
  }
}

__kernel void x_solve3(__global double *g_fjac,
                       __global double *g_njac,
                       __global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2;

  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2) || i > (gp0-2)) return;

  int my_id = (k-1)*(gp1-2) + (j-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  tmp1 = dt * tx1;
  tmp2 = dt * tx2;

  lhs[i][AA][0][0] = - tmp2 * fjac[i-1][0][0]
    - tmp1 * njac[i-1][0][0]
    - tmp1 * dx1; 
  lhs[i][AA][1][0] = - tmp2 * fjac[i-1][1][0]
    - tmp1 * njac[i-1][1][0];
  lhs[i][AA][2][0] = - tmp2 * fjac[i-1][2][0]
    - tmp1 * njac[i-1][2][0];
  lhs[i][AA][3][0] = - tmp2 * fjac[i-1][3][0]
    - tmp1 * njac[i-1][3][0];
  lhs[i][AA][4][0] = - tmp2 * fjac[i-1][4][0]
    - tmp1 * njac[i-1][4][0];

  lhs[i][AA][0][1] = - tmp2 * fjac[i-1][0][1]
    - tmp1 * njac[i-1][0][1];
  lhs[i][AA][1][1] = - tmp2 * fjac[i-1][1][1]
    - tmp1 * njac[i-1][1][1]
    - tmp1 * dx2;
  lhs[i][AA][2][1] = - tmp2 * fjac[i-1][2][1]
    - tmp1 * njac[i-1][2][1];
  lhs[i][AA][3][1] = - tmp2 * fjac[i-1][3][1]
    - tmp1 * njac[i-1][3][1];
  lhs[i][AA][4][1] = - tmp2 * fjac[i-1][4][1]
    - tmp1 * njac[i-1][4][1];

  lhs[i][AA][0][2] = - tmp2 * fjac[i-1][0][2]
    - tmp1 * njac[i-1][0][2];
  lhs[i][AA][1][2] = - tmp2 * fjac[i-1][1][2]
    - tmp1 * njac[i-1][1][2];
  lhs[i][AA][2][2] = - tmp2 * fjac[i-1][2][2]
    - tmp1 * njac[i-1][2][2]
    - tmp1 * dx3;
  lhs[i][AA][3][2] = - tmp2 * fjac[i-1][3][2]
    - tmp1 * njac[i-1][3][2];
  lhs[i][AA][4][2] = - tmp2 * fjac[i-1][4][2]
    - tmp1 * njac[i-1][4][2];

  lhs[i][AA][0][3] = - tmp2 * fjac[i-1][0][3]
    - tmp1 * njac[i-1][0][3];
  lhs[i][AA][1][3] = - tmp2 * fjac[i-1][1][3]
    - tmp1 * njac[i-1][1][3];
  lhs[i][AA][2][3] = - tmp2 * fjac[i-1][2][3]
    - tmp1 * njac[i-1][2][3];
  lhs[i][AA][3][3] = - tmp2 * fjac[i-1][3][3]
    - tmp1 * njac[i-1][3][3]
    - tmp1 * dx4;
  lhs[i][AA][4][3] = - tmp2 * fjac[i-1][4][3]
    - tmp1 * njac[i-1][4][3];

  lhs[i][AA][0][4] = - tmp2 * fjac[i-1][0][4]
    - tmp1 * njac[i-1][0][4];
  lhs[i][AA][1][4] = - tmp2 * fjac[i-1][1][4]
    - tmp1 * njac[i-1][1][4];
  lhs[i][AA][2][4] = - tmp2 * fjac[i-1][2][4]
    - tmp1 * njac[i-1][2][4];
  lhs[i][AA][3][4] = - tmp2 * fjac[i-1][3][4]
    - tmp1 * njac[i-1][3][4];
  lhs[i][AA][4][4] = - tmp2 * fjac[i-1][4][4]
    - tmp1 * njac[i-1][4][4]
    - tmp1 * dx5;

  lhs[i][BB][0][0] = 1.0
    + tmp1 * 2.0 * njac[i][0][0]
    + tmp1 * 2.0 * dx1;
  lhs[i][BB][1][0] = tmp1 * 2.0 * njac[i][1][0];
  lhs[i][BB][2][0] = tmp1 * 2.0 * njac[i][2][0];
  lhs[i][BB][3][0] = tmp1 * 2.0 * njac[i][3][0];
  lhs[i][BB][4][0] = tmp1 * 2.0 * njac[i][4][0];

  lhs[i][BB][0][1] = tmp1 * 2.0 * njac[i][0][1];
  lhs[i][BB][1][1] = 1.0
    + tmp1 * 2.0 * njac[i][1][1]
    + tmp1 * 2.0 * dx2;
  lhs[i][BB][2][1] = tmp1 * 2.0 * njac[i][2][1];
  lhs[i][BB][3][1] = tmp1 * 2.0 * njac[i][3][1];
  lhs[i][BB][4][1] = tmp1 * 2.0 * njac[i][4][1];

  lhs[i][BB][0][2] = tmp1 * 2.0 * njac[i][0][2];
  lhs[i][BB][1][2] = tmp1 * 2.0 * njac[i][1][2];
  lhs[i][BB][2][2] = 1.0
    + tmp1 * 2.0 * njac[i][2][2]
    + tmp1 * 2.0 * dx3;
  lhs[i][BB][3][2] = tmp1 * 2.0 * njac[i][3][2];
  lhs[i][BB][4][2] = tmp1 * 2.0 * njac[i][4][2];

  lhs[i][BB][0][3] = tmp1 * 2.0 * njac[i][0][3];
  lhs[i][BB][1][3] = tmp1 * 2.0 * njac[i][1][3];
  lhs[i][BB][2][3] = tmp1 * 2.0 * njac[i][2][3];
  lhs[i][BB][3][3] = 1.0
    + tmp1 * 2.0 * njac[i][3][3]
    + tmp1 * 2.0 * dx4;
  lhs[i][BB][4][3] = tmp1 * 2.0 * njac[i][4][3];

  lhs[i][BB][0][4] = tmp1 * 2.0 * njac[i][0][4];
  lhs[i][BB][1][4] = tmp1 * 2.0 * njac[i][1][4];
  lhs[i][BB][2][4] = tmp1 * 2.0 * njac[i][2][4];
  lhs[i][BB][3][4] = tmp1 * 2.0 * njac[i][3][4];
  lhs[i][BB][4][4] = 1.0
    + tmp1 * 2.0 * njac[i][4][4]
    + tmp1 * 2.0 * dx5;

  lhs[i][CC][0][0] =  tmp2 * fjac[i+1][0][0]
    - tmp1 * njac[i+1][0][0]
    - tmp1 * dx1;
  lhs[i][CC][1][0] =  tmp2 * fjac[i+1][1][0]
    - tmp1 * njac[i+1][1][0];
  lhs[i][CC][2][0] =  tmp2 * fjac[i+1][2][0]
    - tmp1 * njac[i+1][2][0];
  lhs[i][CC][3][0] =  tmp2 * fjac[i+1][3][0]
    - tmp1 * njac[i+1][3][0];
  lhs[i][CC][4][0] =  tmp2 * fjac[i+1][4][0]
    - tmp1 * njac[i+1][4][0];

  lhs[i][CC][0][1] =  tmp2 * fjac[i+1][0][1]
    - tmp1 * njac[i+1][0][1];
  lhs[i][CC][1][1] =  tmp2 * fjac[i+1][1][1]
    - tmp1 * njac[i+1][1][1]
    - tmp1 * dx2;
  lhs[i][CC][2][1] =  tmp2 * fjac[i+1][2][1]
    - tmp1 * njac[i+1][2][1];
  lhs[i][CC][3][1] =  tmp2 * fjac[i+1][3][1]
    - tmp1 * njac[i+1][3][1];
  lhs[i][CC][4][1] =  tmp2 * fjac[i+1][4][1]
    - tmp1 * njac[i+1][4][1];

  lhs[i][CC][0][2] =  tmp2 * fjac[i+1][0][2]
    - tmp1 * njac[i+1][0][2];
  lhs[i][CC][1][2] =  tmp2 * fjac[i+1][1][2]
    - tmp1 * njac[i+1][1][2];
  lhs[i][CC][2][2] =  tmp2 * fjac[i+1][2][2]
    - tmp1 * njac[i+1][2][2]
    - tmp1 * dx3;
  lhs[i][CC][3][2] =  tmp2 * fjac[i+1][3][2]
    - tmp1 * njac[i+1][3][2];
  lhs[i][CC][4][2] =  tmp2 * fjac[i+1][4][2]
    - tmp1 * njac[i+1][4][2];

  lhs[i][CC][0][3] =  tmp2 * fjac[i+1][0][3]
    - tmp1 * njac[i+1][0][3];
  lhs[i][CC][1][3] =  tmp2 * fjac[i+1][1][3]
    - tmp1 * njac[i+1][1][3];
  lhs[i][CC][2][3] =  tmp2 * fjac[i+1][2][3]
    - tmp1 * njac[i+1][2][3];
  lhs[i][CC][3][3] =  tmp2 * fjac[i+1][3][3]
    - tmp1 * njac[i+1][3][3]
    - tmp1 * dx4;
  lhs[i][CC][4][3] =  tmp2 * fjac[i+1][4][3]
    - tmp1 * njac[i+1][4][3];

  lhs[i][CC][0][4] =  tmp2 * fjac[i+1][0][4]
    - tmp1 * njac[i+1][0][4];
  lhs[i][CC][1][4] =  tmp2 * fjac[i+1][1][4]
    - tmp1 * njac[i+1][1][4];
  lhs[i][CC][2][4] =  tmp2 * fjac[i+1][2][4]
    - tmp1 * njac[i+1][2][4];
  lhs[i][CC][3][4] =  tmp2 * fjac[i+1][3][4]
    - tmp1 * njac[i+1][3][4];
  lhs[i][CC][4][4] =  tmp2 * fjac[i+1][4][4]
    - tmp1 * njac[i+1][4][4]
    - tmp1 * dx5;
}

__kernel void x_solve(__global double *g_rhs,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, isize;
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*(gp1-2) + (j-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  isize = gp0-1;

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(IMAX) and rhs'(IMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[k][j][0] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  load_matrix(p_lhsBB, lhs[0][BB]);
  load_matrix(p_lhsCC, lhs[0][CC]);
  load_vector(p_rhs, rhs[k][j][0]);
  p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
  save_matrix(lhs[0][BB], p_lhsBB);
  save_matrix(lhs[0][CC], p_lhsCC);
  save_vector(rhs[k][j][0], p_rhs);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {
    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[i][AA]);
    load_matrix(p_lhsBB, lhs[i][BB]);
    load_matrix(p_lhsCC, lhs[i][CC]);
    load_vector(p_rhs, rhs[k][j][i]);

    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

    save_matrix(lhs[i][BB], p_lhsBB);
    save_matrix(lhs[i][CC], p_lhsCC);
    save_vector(rhs[k][j][i], p_rhs);
  }

  copy_matrix(p_lhsm1CC, p_lhsCC);
  copy_vector(p_rhsm1, p_rhs);
  load_matrix(p_lhsAA, lhs[isize][AA]);
  load_matrix(p_lhsBB, lhs[isize][BB]);
  load_matrix(p_lhsCC, lhs[isize][CC]);
  load_vector(p_rhs, rhs[k][j][isize]);

  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  p_binvrhs( p_lhsBB, p_rhs );

  save_matrix(lhs[isize][BB], p_lhsBB);
  save_matrix(lhs[isize][CC], p_lhsCC);
  save_vector(rhs[k][j][isize], p_rhs);

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(isize)=rhs(isize)
  // else assume U(isize) is loaded in un pack backsub_info
  // so just use it
  // after u(istart) will be sent to next cell
  //---------------------------------------------------------------------
  for (i = isize-1; i >=0; i--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      p_rhsm1[m] = rhs[k][j][i][m];
      for (n = 0; n < BLOCK_SIZE; n++) {
        p_rhsm1[m] = p_rhsm1[m] 
          - lhs[i][CC][n][m]*p_rhs[n];
      }
      rhs[k][j][i][m] = p_rhsm1[m];
    }
    copy_vector(p_rhs, p_rhsm1);
  }
}

#elif X_SOLVE_DIM == 2
__kernel void x_solve(__global double *g_qs,
                      __global double *g_rho_i,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, isize;
  double tmp1, tmp2, tmp3;

  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*(gp1-2) + (j-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  isize = gp0-1;

  for (i = 0; i <= isize; i++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    //-------------------------------------------------------------------
    // 
    //-------------------------------------------------------------------
    fjac[i][0][0] = 0.0;
    fjac[i][1][0] = 1.0;
    fjac[i][2][0] = 0.0;
    fjac[i][3][0] = 0.0;
    fjac[i][4][0] = 0.0;

    fjac[i][0][1] = -(u[k][j][i][1] * tmp2 * u[k][j][i][1])
      + c2 * qs[k][j][i];
    fjac[i][1][1] = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
    fjac[i][2][1] = - c2 * ( u[k][j][i][2] * tmp1 );
    fjac[i][3][1] = - c2 * ( u[k][j][i][3] * tmp1 );
    fjac[i][4][1] = c2;

    fjac[i][0][2] = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac[i][1][2] = u[k][j][i][2] * tmp1;
    fjac[i][2][2] = u[k][j][i][1] * tmp1;
    fjac[i][3][2] = 0.0;
    fjac[i][4][2] = 0.0;

    fjac[i][0][3] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[i][1][3] = u[k][j][i][3] * tmp1;
    fjac[i][2][3] = 0.0;
    fjac[i][3][3] = u[k][j][i][1] * tmp1;
    fjac[i][4][3] = 0.0;

    fjac[i][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * ( u[k][j][i][1] * tmp2 );
    fjac[i][1][4] = c1 *  u[k][j][i][4] * tmp1 
      - c2 * ( u[k][j][i][1]*u[k][j][i][1] * tmp2 + qs[k][j][i] );
    fjac[i][2][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * tmp2;
    fjac[i][3][4] = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * tmp2;
    fjac[i][4][4] = c1 * ( u[k][j][i][1] * tmp1 );

    njac[i][0][0] = 0.0;
    njac[i][1][0] = 0.0;
    njac[i][2][0] = 0.0;
    njac[i][3][0] = 0.0;
    njac[i][4][0] = 0.0;

    njac[i][0][1] = - con43 * c3c4 * tmp2 * u[k][j][i][1];
    njac[i][1][1] =   con43 * c3c4 * tmp1;
    njac[i][2][1] =   0.0;
    njac[i][3][1] =   0.0;
    njac[i][4][1] =   0.0;

    njac[i][0][2] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[i][1][2] =   0.0;
    njac[i][2][2] =   c3c4 * tmp1;
    njac[i][3][2] =   0.0;
    njac[i][4][2] =   0.0;

    njac[i][0][3] = - c3c4 * tmp2 * u[k][j][i][3];
    njac[i][1][3] =   0.0;
    njac[i][2][3] =   0.0;
    njac[i][3][3] =   c3c4 * tmp1;
    njac[i][4][3] =   0.0;

    njac[i][0][4] = - ( con43 * c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[i][1][4] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][1];
    njac[i][2][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[i][3][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac[i][4][4] = ( c1345 ) * tmp1;
  }
  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in x direction
  //---------------------------------------------------------------------
  lhsinit(lhs, isize);

  for (i = 1; i <= isize-1; i++) {
    tmp1 = dt * tx1;
    tmp2 = dt * tx2;

    lhs[i][AA][0][0] = - tmp2 * fjac[i-1][0][0]
      - tmp1 * njac[i-1][0][0]
      - tmp1 * dx1; 
    lhs[i][AA][1][0] = - tmp2 * fjac[i-1][1][0]
      - tmp1 * njac[i-1][1][0];
    lhs[i][AA][2][0] = - tmp2 * fjac[i-1][2][0]
      - tmp1 * njac[i-1][2][0];
    lhs[i][AA][3][0] = - tmp2 * fjac[i-1][3][0]
      - tmp1 * njac[i-1][3][0];
    lhs[i][AA][4][0] = - tmp2 * fjac[i-1][4][0]
      - tmp1 * njac[i-1][4][0];

    lhs[i][AA][0][1] = - tmp2 * fjac[i-1][0][1]
      - tmp1 * njac[i-1][0][1];
    lhs[i][AA][1][1] = - tmp2 * fjac[i-1][1][1]
      - tmp1 * njac[i-1][1][1]
      - tmp1 * dx2;
    lhs[i][AA][2][1] = - tmp2 * fjac[i-1][2][1]
      - tmp1 * njac[i-1][2][1];
    lhs[i][AA][3][1] = - tmp2 * fjac[i-1][3][1]
      - tmp1 * njac[i-1][3][1];
    lhs[i][AA][4][1] = - tmp2 * fjac[i-1][4][1]
      - tmp1 * njac[i-1][4][1];

    lhs[i][AA][0][2] = - tmp2 * fjac[i-1][0][2]
      - tmp1 * njac[i-1][0][2];
    lhs[i][AA][1][2] = - tmp2 * fjac[i-1][1][2]
      - tmp1 * njac[i-1][1][2];
    lhs[i][AA][2][2] = - tmp2 * fjac[i-1][2][2]
      - tmp1 * njac[i-1][2][2]
      - tmp1 * dx3;
    lhs[i][AA][3][2] = - tmp2 * fjac[i-1][3][2]
      - tmp1 * njac[i-1][3][2];
    lhs[i][AA][4][2] = - tmp2 * fjac[i-1][4][2]
      - tmp1 * njac[i-1][4][2];

    lhs[i][AA][0][3] = - tmp2 * fjac[i-1][0][3]
      - tmp1 * njac[i-1][0][3];
    lhs[i][AA][1][3] = - tmp2 * fjac[i-1][1][3]
      - tmp1 * njac[i-1][1][3];
    lhs[i][AA][2][3] = - tmp2 * fjac[i-1][2][3]
      - tmp1 * njac[i-1][2][3];
    lhs[i][AA][3][3] = - tmp2 * fjac[i-1][3][3]
      - tmp1 * njac[i-1][3][3]
      - tmp1 * dx4;
    lhs[i][AA][4][3] = - tmp2 * fjac[i-1][4][3]
      - tmp1 * njac[i-1][4][3];

    lhs[i][AA][0][4] = - tmp2 * fjac[i-1][0][4]
      - tmp1 * njac[i-1][0][4];
    lhs[i][AA][1][4] = - tmp2 * fjac[i-1][1][4]
      - tmp1 * njac[i-1][1][4];
    lhs[i][AA][2][4] = - tmp2 * fjac[i-1][2][4]
      - tmp1 * njac[i-1][2][4];
    lhs[i][AA][3][4] = - tmp2 * fjac[i-1][3][4]
      - tmp1 * njac[i-1][3][4];
    lhs[i][AA][4][4] = - tmp2 * fjac[i-1][4][4]
      - tmp1 * njac[i-1][4][4]
      - tmp1 * dx5;

    lhs[i][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[i][0][0]
      + tmp1 * 2.0 * dx1;
    lhs[i][BB][1][0] = tmp1 * 2.0 * njac[i][1][0];
    lhs[i][BB][2][0] = tmp1 * 2.0 * njac[i][2][0];
    lhs[i][BB][3][0] = tmp1 * 2.0 * njac[i][3][0];
    lhs[i][BB][4][0] = tmp1 * 2.0 * njac[i][4][0];

    lhs[i][BB][0][1] = tmp1 * 2.0 * njac[i][0][1];
    lhs[i][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[i][1][1]
      + tmp1 * 2.0 * dx2;
    lhs[i][BB][2][1] = tmp1 * 2.0 * njac[i][2][1];
    lhs[i][BB][3][1] = tmp1 * 2.0 * njac[i][3][1];
    lhs[i][BB][4][1] = tmp1 * 2.0 * njac[i][4][1];

    lhs[i][BB][0][2] = tmp1 * 2.0 * njac[i][0][2];
    lhs[i][BB][1][2] = tmp1 * 2.0 * njac[i][1][2];
    lhs[i][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[i][2][2]
      + tmp1 * 2.0 * dx3;
    lhs[i][BB][3][2] = tmp1 * 2.0 * njac[i][3][2];
    lhs[i][BB][4][2] = tmp1 * 2.0 * njac[i][4][2];

    lhs[i][BB][0][3] = tmp1 * 2.0 * njac[i][0][3];
    lhs[i][BB][1][3] = tmp1 * 2.0 * njac[i][1][3];
    lhs[i][BB][2][3] = tmp1 * 2.0 * njac[i][2][3];
    lhs[i][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[i][3][3]
      + tmp1 * 2.0 * dx4;
    lhs[i][BB][4][3] = tmp1 * 2.0 * njac[i][4][3];

    lhs[i][BB][0][4] = tmp1 * 2.0 * njac[i][0][4];
    lhs[i][BB][1][4] = tmp1 * 2.0 * njac[i][1][4];
    lhs[i][BB][2][4] = tmp1 * 2.0 * njac[i][2][4];
    lhs[i][BB][3][4] = tmp1 * 2.0 * njac[i][3][4];
    lhs[i][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[i][4][4]
      + tmp1 * 2.0 * dx5;

    lhs[i][CC][0][0] =  tmp2 * fjac[i+1][0][0]
      - tmp1 * njac[i+1][0][0]
      - tmp1 * dx1;
    lhs[i][CC][1][0] =  tmp2 * fjac[i+1][1][0]
      - tmp1 * njac[i+1][1][0];
    lhs[i][CC][2][0] =  tmp2 * fjac[i+1][2][0]
      - tmp1 * njac[i+1][2][0];
    lhs[i][CC][3][0] =  tmp2 * fjac[i+1][3][0]
      - tmp1 * njac[i+1][3][0];
    lhs[i][CC][4][0] =  tmp2 * fjac[i+1][4][0]
      - tmp1 * njac[i+1][4][0];

    lhs[i][CC][0][1] =  tmp2 * fjac[i+1][0][1]
      - tmp1 * njac[i+1][0][1];
    lhs[i][CC][1][1] =  tmp2 * fjac[i+1][1][1]
      - tmp1 * njac[i+1][1][1]
      - tmp1 * dx2;
    lhs[i][CC][2][1] =  tmp2 * fjac[i+1][2][1]
      - tmp1 * njac[i+1][2][1];
    lhs[i][CC][3][1] =  tmp2 * fjac[i+1][3][1]
      - tmp1 * njac[i+1][3][1];
    lhs[i][CC][4][1] =  tmp2 * fjac[i+1][4][1]
      - tmp1 * njac[i+1][4][1];

    lhs[i][CC][0][2] =  tmp2 * fjac[i+1][0][2]
      - tmp1 * njac[i+1][0][2];
    lhs[i][CC][1][2] =  tmp2 * fjac[i+1][1][2]
      - tmp1 * njac[i+1][1][2];
    lhs[i][CC][2][2] =  tmp2 * fjac[i+1][2][2]
      - tmp1 * njac[i+1][2][2]
      - tmp1 * dx3;
    lhs[i][CC][3][2] =  tmp2 * fjac[i+1][3][2]
      - tmp1 * njac[i+1][3][2];
    lhs[i][CC][4][2] =  tmp2 * fjac[i+1][4][2]
      - tmp1 * njac[i+1][4][2];

    lhs[i][CC][0][3] =  tmp2 * fjac[i+1][0][3]
      - tmp1 * njac[i+1][0][3];
    lhs[i][CC][1][3] =  tmp2 * fjac[i+1][1][3]
      - tmp1 * njac[i+1][1][3];
    lhs[i][CC][2][3] =  tmp2 * fjac[i+1][2][3]
      - tmp1 * njac[i+1][2][3];
    lhs[i][CC][3][3] =  tmp2 * fjac[i+1][3][3]
      - tmp1 * njac[i+1][3][3]
      - tmp1 * dx4;
    lhs[i][CC][4][3] =  tmp2 * fjac[i+1][4][3]
      - tmp1 * njac[i+1][4][3];

    lhs[i][CC][0][4] =  tmp2 * fjac[i+1][0][4]
      - tmp1 * njac[i+1][0][4];
    lhs[i][CC][1][4] =  tmp2 * fjac[i+1][1][4]
      - tmp1 * njac[i+1][1][4];
    lhs[i][CC][2][4] =  tmp2 * fjac[i+1][2][4]
      - tmp1 * njac[i+1][2][4];
    lhs[i][CC][3][4] =  tmp2 * fjac[i+1][3][4]
      - tmp1 * njac[i+1][3][4];
    lhs[i][CC][4][4] =  tmp2 * fjac[i+1][4][4]
      - tmp1 * njac[i+1][4][4]
      - tmp1 * dx5;
  }

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(IMAX) and rhs'(IMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[k][j][0] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[k][j][0] );

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {
    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    matvec_sub(lhs[i][AA], rhs[k][j][i-1], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    matmul_sub(lhs[i][AA], lhs[i-1][CC], lhs[i][BB]);


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[i][BB], lhs[i][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[isize][AA], rhs[k][j][isize-1], rhs[k][j][isize]);

  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  matmul_sub(lhs[isize][AA], lhs[isize-1][CC], lhs[isize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[isize][BB], rhs[k][j][isize] );

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(isize)=rhs(isize)
  // else assume U(isize) is loaded in un pack backsub_info
  // so just use it
  // after u(istart) will be sent to next cell
  //---------------------------------------------------------------------
  for (i = isize-1; i >=0; i--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[i][CC][n][m]*rhs[k][j][i+1][n];
      }
    }
  }
}

#else //X_SOLVE_DIM == 1
__kernel void x_solve(__global double *g_qs,
                      __global double *g_rho_i,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, isize;
  double tmp1, tmp2, tmp3;
  double p_u[5];
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  k = get_global_id(0) + 1;
  if (k > (gp2-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = k - 1;
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  isize = gp0-1;

  for (j = 1; j <= gp1-2; j++) {
    for (i = 0; i <= isize; i++) {
      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;
      p_u[0] = u[k][j][i][0];
      p_u[1] = u[k][j][i][1];
      p_u[2] = u[k][j][i][2];
      p_u[3] = u[k][j][i][3];
      p_u[4] = u[k][j][i][4];
      //-------------------------------------------------------------------
      // 
      //-------------------------------------------------------------------
      fjac[i][0][0] = 0.0;
      fjac[i][1][0] = 1.0;
      fjac[i][2][0] = 0.0;
      fjac[i][3][0] = 0.0;
      fjac[i][4][0] = 0.0;

      fjac[i][0][1] = -(p_u[1] * tmp2 * p_u[1])
        + c2 * qs[k][j][i];
      fjac[i][1][1] = ( 2.0 - c2 ) * ( p_u[1] / p_u[0] );
      fjac[i][2][1] = - c2 * ( p_u[2] * tmp1 );
      fjac[i][3][1] = - c2 * ( p_u[3] * tmp1 );
      fjac[i][4][1] = c2;

      fjac[i][0][2] = - ( p_u[1]*p_u[2] ) * tmp2;
      fjac[i][1][2] = p_u[2] * tmp1;
      fjac[i][2][2] = p_u[1] * tmp1;
      fjac[i][3][2] = 0.0;
      fjac[i][4][2] = 0.0;

      fjac[i][0][3] = - ( p_u[1]*p_u[3] ) * tmp2;
      fjac[i][1][3] = p_u[3] * tmp1;
      fjac[i][2][3] = 0.0;
      fjac[i][3][3] = p_u[1] * tmp1;
      fjac[i][4][3] = 0.0;

      fjac[i][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
        * ( p_u[1] * tmp2 );
      fjac[i][1][4] = c1 *  p_u[4] * tmp1 
        - c2 * ( p_u[1]*p_u[1] * tmp2 + qs[k][j][i] );
      fjac[i][2][4] = - c2 * ( p_u[2]*p_u[1] ) * tmp2;
      fjac[i][3][4] = - c2 * ( p_u[3]*p_u[1] ) * tmp2;
      fjac[i][4][4] = c1 * ( p_u[1] * tmp1 );

      njac[i][0][0] = 0.0;
      njac[i][1][0] = 0.0;
      njac[i][2][0] = 0.0;
      njac[i][3][0] = 0.0;
      njac[i][4][0] = 0.0;

      njac[i][0][1] = - con43 * c3c4 * tmp2 * p_u[1];
      njac[i][1][1] =   con43 * c3c4 * tmp1;
      njac[i][2][1] =   0.0;
      njac[i][3][1] =   0.0;
      njac[i][4][1] =   0.0;

      njac[i][0][2] = - c3c4 * tmp2 * p_u[2];
      njac[i][1][2] =   0.0;
      njac[i][2][2] =   c3c4 * tmp1;
      njac[i][3][2] =   0.0;
      njac[i][4][2] =   0.0;

      njac[i][0][3] = - c3c4 * tmp2 * p_u[3];
      njac[i][1][3] =   0.0;
      njac[i][2][3] =   0.0;
      njac[i][3][3] =   c3c4 * tmp1;
      njac[i][4][3] =   0.0;

      njac[i][0][4] = - ( con43 * c3c4
          - c1345 ) * tmp3 * (p_u[1]*p_u[1])
        - ( c3c4 - c1345 ) * tmp3 * (p_u[2]*p_u[2])
        - ( c3c4 - c1345 ) * tmp3 * (p_u[3]*p_u[3])
        - c1345 * tmp2 * p_u[4];

      njac[i][1][4] = ( con43 * c3c4
          - c1345 ) * tmp2 * p_u[1];
      njac[i][2][4] = ( c3c4 - c1345 ) * tmp2 * p_u[2];
      njac[i][3][4] = ( c3c4 - c1345 ) * tmp2 * p_u[3];
      njac[i][4][4] = ( c1345 ) * tmp1;
    }
    //---------------------------------------------------------------------
    // now jacobians set, so form left hand side in x direction
    //---------------------------------------------------------------------
    lhsinit(lhs, isize);
    for (i = 1; i <= isize-1; i++) {
      tmp1 = dt * tx1;
      tmp2 = dt * tx2;

      lhs[i][AA][0][0] = - tmp2 * fjac[i-1][0][0]
        - tmp1 * njac[i-1][0][0]
        - tmp1 * dx1; 
      lhs[i][AA][1][0] = - tmp2 * fjac[i-1][1][0]
        - tmp1 * njac[i-1][1][0];
      lhs[i][AA][2][0] = - tmp2 * fjac[i-1][2][0]
        - tmp1 * njac[i-1][2][0];
      lhs[i][AA][3][0] = - tmp2 * fjac[i-1][3][0]
        - tmp1 * njac[i-1][3][0];
      lhs[i][AA][4][0] = - tmp2 * fjac[i-1][4][0]
        - tmp1 * njac[i-1][4][0];

      lhs[i][AA][0][1] = - tmp2 * fjac[i-1][0][1]
        - tmp1 * njac[i-1][0][1];
      lhs[i][AA][1][1] = - tmp2 * fjac[i-1][1][1]
        - tmp1 * njac[i-1][1][1]
        - tmp1 * dx2;
      lhs[i][AA][2][1] = - tmp2 * fjac[i-1][2][1]
        - tmp1 * njac[i-1][2][1];
      lhs[i][AA][3][1] = - tmp2 * fjac[i-1][3][1]
        - tmp1 * njac[i-1][3][1];
      lhs[i][AA][4][1] = - tmp2 * fjac[i-1][4][1]
        - tmp1 * njac[i-1][4][1];

      lhs[i][AA][0][2] = - tmp2 * fjac[i-1][0][2]
        - tmp1 * njac[i-1][0][2];
      lhs[i][AA][1][2] = - tmp2 * fjac[i-1][1][2]
        - tmp1 * njac[i-1][1][2];
      lhs[i][AA][2][2] = - tmp2 * fjac[i-1][2][2]
        - tmp1 * njac[i-1][2][2]
        - tmp1 * dx3;
      lhs[i][AA][3][2] = - tmp2 * fjac[i-1][3][2]
        - tmp1 * njac[i-1][3][2];
      lhs[i][AA][4][2] = - tmp2 * fjac[i-1][4][2]
        - tmp1 * njac[i-1][4][2];

      lhs[i][AA][0][3] = - tmp2 * fjac[i-1][0][3]
        - tmp1 * njac[i-1][0][3];
      lhs[i][AA][1][3] = - tmp2 * fjac[i-1][1][3]
        - tmp1 * njac[i-1][1][3];
      lhs[i][AA][2][3] = - tmp2 * fjac[i-1][2][3]
        - tmp1 * njac[i-1][2][3];
      lhs[i][AA][3][3] = - tmp2 * fjac[i-1][3][3]
        - tmp1 * njac[i-1][3][3]
        - tmp1 * dx4;
      lhs[i][AA][4][3] = - tmp2 * fjac[i-1][4][3]
        - tmp1 * njac[i-1][4][3];

      lhs[i][AA][0][4] = - tmp2 * fjac[i-1][0][4]
        - tmp1 * njac[i-1][0][4];
      lhs[i][AA][1][4] = - tmp2 * fjac[i-1][1][4]
        - tmp1 * njac[i-1][1][4];
      lhs[i][AA][2][4] = - tmp2 * fjac[i-1][2][4]
        - tmp1 * njac[i-1][2][4];
      lhs[i][AA][3][4] = - tmp2 * fjac[i-1][3][4]
        - tmp1 * njac[i-1][3][4];
      lhs[i][AA][4][4] = - tmp2 * fjac[i-1][4][4]
        - tmp1 * njac[i-1][4][4]
        - tmp1 * dx5;

      lhs[i][BB][0][0] = 1.0
        + tmp1 * 2.0 * njac[i][0][0]
        + tmp1 * 2.0 * dx1;
      lhs[i][BB][1][0] = tmp1 * 2.0 * njac[i][1][0];
      lhs[i][BB][2][0] = tmp1 * 2.0 * njac[i][2][0];
      lhs[i][BB][3][0] = tmp1 * 2.0 * njac[i][3][0];
      lhs[i][BB][4][0] = tmp1 * 2.0 * njac[i][4][0];

      lhs[i][BB][0][1] = tmp1 * 2.0 * njac[i][0][1];
      lhs[i][BB][1][1] = 1.0
        + tmp1 * 2.0 * njac[i][1][1]
        + tmp1 * 2.0 * dx2;
      lhs[i][BB][2][1] = tmp1 * 2.0 * njac[i][2][1];
      lhs[i][BB][3][1] = tmp1 * 2.0 * njac[i][3][1];
      lhs[i][BB][4][1] = tmp1 * 2.0 * njac[i][4][1];

      lhs[i][BB][0][2] = tmp1 * 2.0 * njac[i][0][2];
      lhs[i][BB][1][2] = tmp1 * 2.0 * njac[i][1][2];
      lhs[i][BB][2][2] = 1.0
        + tmp1 * 2.0 * njac[i][2][2]
        + tmp1 * 2.0 * dx3;
      lhs[i][BB][3][2] = tmp1 * 2.0 * njac[i][3][2];
      lhs[i][BB][4][2] = tmp1 * 2.0 * njac[i][4][2];

      lhs[i][BB][0][3] = tmp1 * 2.0 * njac[i][0][3];
      lhs[i][BB][1][3] = tmp1 * 2.0 * njac[i][1][3];
      lhs[i][BB][2][3] = tmp1 * 2.0 * njac[i][2][3];
      lhs[i][BB][3][3] = 1.0
        + tmp1 * 2.0 * njac[i][3][3]
        + tmp1 * 2.0 * dx4;
      lhs[i][BB][4][3] = tmp1 * 2.0 * njac[i][4][3];

      lhs[i][BB][0][4] = tmp1 * 2.0 * njac[i][0][4];
      lhs[i][BB][1][4] = tmp1 * 2.0 * njac[i][1][4];
      lhs[i][BB][2][4] = tmp1 * 2.0 * njac[i][2][4];
      lhs[i][BB][3][4] = tmp1 * 2.0 * njac[i][3][4];
      lhs[i][BB][4][4] = 1.0
        + tmp1 * 2.0 * njac[i][4][4]
        + tmp1 * 2.0 * dx5;

      lhs[i][CC][0][0] =  tmp2 * fjac[i+1][0][0]
        - tmp1 * njac[i+1][0][0]
        - tmp1 * dx1;
      lhs[i][CC][1][0] =  tmp2 * fjac[i+1][1][0]
        - tmp1 * njac[i+1][1][0];
      lhs[i][CC][2][0] =  tmp2 * fjac[i+1][2][0]
        - tmp1 * njac[i+1][2][0];
      lhs[i][CC][3][0] =  tmp2 * fjac[i+1][3][0]
        - tmp1 * njac[i+1][3][0];
      lhs[i][CC][4][0] =  tmp2 * fjac[i+1][4][0]
        - tmp1 * njac[i+1][4][0];

      lhs[i][CC][0][1] =  tmp2 * fjac[i+1][0][1]
        - tmp1 * njac[i+1][0][1];
      lhs[i][CC][1][1] =  tmp2 * fjac[i+1][1][1]
        - tmp1 * njac[i+1][1][1]
        - tmp1 * dx2;
      lhs[i][CC][2][1] =  tmp2 * fjac[i+1][2][1]
        - tmp1 * njac[i+1][2][1];
      lhs[i][CC][3][1] =  tmp2 * fjac[i+1][3][1]
        - tmp1 * njac[i+1][3][1];
      lhs[i][CC][4][1] =  tmp2 * fjac[i+1][4][1]
        - tmp1 * njac[i+1][4][1];

      lhs[i][CC][0][2] =  tmp2 * fjac[i+1][0][2]
        - tmp1 * njac[i+1][0][2];
      lhs[i][CC][1][2] =  tmp2 * fjac[i+1][1][2]
        - tmp1 * njac[i+1][1][2];
      lhs[i][CC][2][2] =  tmp2 * fjac[i+1][2][2]
        - tmp1 * njac[i+1][2][2]
        - tmp1 * dx3;
      lhs[i][CC][3][2] =  tmp2 * fjac[i+1][3][2]
        - tmp1 * njac[i+1][3][2];
      lhs[i][CC][4][2] =  tmp2 * fjac[i+1][4][2]
        - tmp1 * njac[i+1][4][2];

      lhs[i][CC][0][3] =  tmp2 * fjac[i+1][0][3]
        - tmp1 * njac[i+1][0][3];
      lhs[i][CC][1][3] =  tmp2 * fjac[i+1][1][3]
        - tmp1 * njac[i+1][1][3];
      lhs[i][CC][2][3] =  tmp2 * fjac[i+1][2][3]
        - tmp1 * njac[i+1][2][3];
      lhs[i][CC][3][3] =  tmp2 * fjac[i+1][3][3]
        - tmp1 * njac[i+1][3][3]
        - tmp1 * dx4;
      lhs[i][CC][4][3] =  tmp2 * fjac[i+1][4][3]
        - tmp1 * njac[i+1][4][3];

      lhs[i][CC][0][4] =  tmp2 * fjac[i+1][0][4]
        - tmp1 * njac[i+1][0][4];
      lhs[i][CC][1][4] =  tmp2 * fjac[i+1][1][4]
        - tmp1 * njac[i+1][1][4];
      lhs[i][CC][2][4] =  tmp2 * fjac[i+1][2][4]
        - tmp1 * njac[i+1][2][4];
      lhs[i][CC][3][4] =  tmp2 * fjac[i+1][3][4]
        - tmp1 * njac[i+1][3][4];
      lhs[i][CC][4][4] =  tmp2 * fjac[i+1][4][4]
        - tmp1 * njac[i+1][4][4]
        - tmp1 * dx5;
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // performs guaussian elimination on this cell.
    // 
    // assumes that unpacking routines for non-first cells 
    // preload C' and rhs' from previous cell.
    // 
    // assumed send happens outside this routine, but that
    // c'(IMAX) and rhs'(IMAX) will be sent to next cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // outer most do loops - sweeping in i direction
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[k][j][0] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    load_matrix(p_lhsBB, lhs[0][BB]);
    load_matrix(p_lhsCC, lhs[0][CC]);
    load_vector(p_rhs, rhs[k][j][0]);
    p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
    save_matrix(lhs[0][BB], p_lhsBB);
    save_matrix(lhs[0][CC], p_lhsCC);
    save_vector(rhs[k][j][0], p_rhs);

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last 
    //---------------------------------------------------------------------
    for (i = 1; i <= isize-1; i++) {
      copy_matrix(p_lhsm1CC, p_lhsCC);
      copy_vector(p_rhsm1, p_rhs);
      load_matrix(p_lhsAA, lhs[i][AA]);
      load_matrix(p_lhsBB, lhs[i][BB]);
      load_matrix(p_lhsCC, lhs[i][CC]);
      load_vector(p_rhs, rhs[k][j][i]);

      //-------------------------------------------------------------------
      // rhs(i) = rhs(i) - A*rhs(i-1)
      //-------------------------------------------------------------------
      p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

      //-------------------------------------------------------------------
      // B(i) = B(i) - C(i-1)*A(i)
      //-------------------------------------------------------------------
      p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);


      //-------------------------------------------------------------------
      // multiply c[k][j][i] by b_inverse and copy back to c
      // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
      //-------------------------------------------------------------------
      p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

      save_matrix(lhs[i][BB], p_lhsBB);
      save_matrix(lhs[i][CC], p_lhsCC);
      save_vector(rhs[k][j][i], p_rhs);
    }

    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[isize][AA]);
    load_matrix(p_lhsBB, lhs[isize][BB]);
    load_matrix(p_lhsCC, lhs[isize][CC]);
    load_vector(p_rhs, rhs[k][j][isize]);

    //---------------------------------------------------------------------
    // rhs(isize) = rhs(isize) - A*rhs(isize-1)
    //---------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //---------------------------------------------------------------------
    // B(isize) = B(isize) - C(isize-1)*A(isize)
    //---------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //---------------------------------------------------------------------
    // multiply rhs() by b_inverse() and copy to rhs
    //---------------------------------------------------------------------
    p_binvrhs( p_lhsBB, p_rhs );

    save_matrix(lhs[isize][BB], p_lhsBB);
    save_matrix(lhs[isize][CC], p_lhsCC);
    save_vector(rhs[k][j][isize], p_rhs);

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(isize)=rhs(isize)
    // else assume U(isize) is loaded in un pack backsub_info
    // so just use it
    // after u(istart) will be sent to next cell
    //---------------------------------------------------------------------
    for (i = isize-1; i >=0; i--) {
      for (m = 0; m < BLOCK_SIZE; m++) {
        p_rhsm1[m] = rhs[k][j][i][m];
        for (n = 0; n < BLOCK_SIZE; n++) {
          p_rhsm1[m] = p_rhsm1[m] 
            - lhs[i][CC][n][m]*p_rhs[n];
        }
        rhs[k][j][i][m] = p_rhsm1[m];
      }
      copy_vector(p_rhs, p_rhsm1);
    }
  }
}
#endif


#if Y_SOLVE_DIM == 3
__kernel void y_solve1(__global double *g_qs,
                       __global double *g_rho_i,
                       __global double *g_square,
                       __global double *g_u,
                       __global double *g_fjac,
                       __global double *g_njac,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2, tmp3;
  double p_u[5];

  k = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  j = get_global_id(0);
  if (k > (gp2-2) || i > (gp0-2) || j >= gp1) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  int my_id = (k-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;
  p_u[0] = u[k][j][i][0];
  p_u[1] = u[k][j][i][1];
  p_u[2] = u[k][j][i][2];
  p_u[3] = u[k][j][i][3];
  p_u[4] = u[k][j][i][4];

  fjac[j][0][0] = 0.0;
  fjac[j][1][0] = 0.0;
  fjac[j][2][0] = 1.0;
  fjac[j][3][0] = 0.0;
  fjac[j][4][0] = 0.0;

  fjac[j][0][1] = - ( p_u[1]*p_u[2] ) * tmp2;
  fjac[j][1][1] = p_u[2] * tmp1;
  fjac[j][2][1] = p_u[1] * tmp1;
  fjac[j][3][1] = 0.0;
  fjac[j][4][1] = 0.0;

  fjac[j][0][2] = - ( p_u[2]*p_u[2]*tmp2)
    + c2 * qs[k][j][i];
  fjac[j][1][2] = - c2 *  p_u[1] * tmp1;
  fjac[j][2][2] = ( 2.0 - c2 ) *  p_u[2] * tmp1;
  fjac[j][3][2] = - c2 * p_u[3] * tmp1;
  fjac[j][4][2] = c2;

  fjac[j][0][3] = - ( p_u[2]*p_u[3] ) * tmp2;
  fjac[j][1][3] = 0.0;
  fjac[j][2][3] = p_u[3] * tmp1;
  fjac[j][3][3] = p_u[2] * tmp1;
  fjac[j][4][3] = 0.0;

  fjac[j][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
    * p_u[2] * tmp2;
  fjac[j][1][4] = - c2 * p_u[1]*p_u[2] * tmp2;
  fjac[j][2][4] = c1 * p_u[4] * tmp1 
    - c2 * ( qs[k][j][i] + p_u[2]*p_u[2] * tmp2 );
  fjac[j][3][4] = - c2 * ( p_u[2]*p_u[3] ) * tmp2;
  fjac[j][4][4] = c1 * p_u[2] * tmp1;

  njac[j][0][0] = 0.0;
  njac[j][1][0] = 0.0;
  njac[j][2][0] = 0.0;
  njac[j][3][0] = 0.0;
  njac[j][4][0] = 0.0;

  njac[j][0][1] = - c3c4 * tmp2 * p_u[1];
  njac[j][1][1] =   c3c4 * tmp1;
  njac[j][2][1] =   0.0;
  njac[j][3][1] =   0.0;
  njac[j][4][1] =   0.0;

  njac[j][0][2] = - con43 * c3c4 * tmp2 * p_u[2];
  njac[j][1][2] =   0.0;
  njac[j][2][2] =   con43 * c3c4 * tmp1;
  njac[j][3][2] =   0.0;
  njac[j][4][2] =   0.0;

  njac[j][0][3] = - c3c4 * tmp2 * p_u[3];
  njac[j][1][3] =   0.0;
  njac[j][2][3] =   0.0;
  njac[j][3][3] =   c3c4 * tmp1;
  njac[j][4][3] =   0.0;

  njac[j][0][4] = - (  c3c4
      - c1345 ) * tmp3 * (p_u[1]*p_u[1])
    - ( con43 * c3c4
        - c1345 ) * tmp3 * (p_u[2]*p_u[2])
    - ( c3c4 - c1345 ) * tmp3 * (p_u[3]*p_u[3])
    - c1345 * tmp2 * p_u[4];

  njac[j][1][4] = (  c3c4 - c1345 ) * tmp2 * p_u[1];
  njac[j][2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * p_u[2];
  njac[j][3][4] = ( c3c4 - c1345 ) * tmp2 * p_u[3];
  njac[j][4][4] = ( c1345 ) * tmp1;
}

__kernel void y_solve2(__global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k, n, m;

  k = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  if (k > (gp2-2) || i > (gp0-2)) return;

  j = get_global_id(0);
  if (j == 1) j = gp1-1;
  
  int my_id = (k-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[j][0][n][m] = 0.0;
      lhs[j][1][n][m] = 0.0;
      lhs[j][2][n][m] = 0.0;
    }
    lhs[j][1][n][n] = 1.0;
  }
}

__kernel void y_solve3(__global double *g_fjac,
                       __global double *g_njac,
                       __global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2;

  k = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || i > (gp0-2) || j > (gp1-2)) return;

  int my_id = (k-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  tmp1 = dt * ty1;
  tmp2 = dt * ty2;

  lhs[j][AA][0][0] = - tmp2 * fjac[j-1][0][0]
    - tmp1 * njac[j-1][0][0]
    - tmp1 * dy1; 
  lhs[j][AA][1][0] = - tmp2 * fjac[j-1][1][0]
    - tmp1 * njac[j-1][1][0];
  lhs[j][AA][2][0] = - tmp2 * fjac[j-1][2][0]
    - tmp1 * njac[j-1][2][0];
  lhs[j][AA][3][0] = - tmp2 * fjac[j-1][3][0]
    - tmp1 * njac[j-1][3][0];
  lhs[j][AA][4][0] = - tmp2 * fjac[j-1][4][0]
    - tmp1 * njac[j-1][4][0];

  lhs[j][AA][0][1] = - tmp2 * fjac[j-1][0][1]
    - tmp1 * njac[j-1][0][1];
  lhs[j][AA][1][1] = - tmp2 * fjac[j-1][1][1]
    - tmp1 * njac[j-1][1][1]
    - tmp1 * dy2;
  lhs[j][AA][2][1] = - tmp2 * fjac[j-1][2][1]
    - tmp1 * njac[j-1][2][1];
  lhs[j][AA][3][1] = - tmp2 * fjac[j-1][3][1]
    - tmp1 * njac[j-1][3][1];
  lhs[j][AA][4][1] = - tmp2 * fjac[j-1][4][1]
    - tmp1 * njac[j-1][4][1];

  lhs[j][AA][0][2] = - tmp2 * fjac[j-1][0][2]
    - tmp1 * njac[j-1][0][2];
  lhs[j][AA][1][2] = - tmp2 * fjac[j-1][1][2]
    - tmp1 * njac[j-1][1][2];
  lhs[j][AA][2][2] = - tmp2 * fjac[j-1][2][2]
    - tmp1 * njac[j-1][2][2]
    - tmp1 * dy3;
  lhs[j][AA][3][2] = - tmp2 * fjac[j-1][3][2]
    - tmp1 * njac[j-1][3][2];
  lhs[j][AA][4][2] = - tmp2 * fjac[j-1][4][2]
    - tmp1 * njac[j-1][4][2];

  lhs[j][AA][0][3] = - tmp2 * fjac[j-1][0][3]
    - tmp1 * njac[j-1][0][3];
  lhs[j][AA][1][3] = - tmp2 * fjac[j-1][1][3]
    - tmp1 * njac[j-1][1][3];
  lhs[j][AA][2][3] = - tmp2 * fjac[j-1][2][3]
    - tmp1 * njac[j-1][2][3];
  lhs[j][AA][3][3] = - tmp2 * fjac[j-1][3][3]
    - tmp1 * njac[j-1][3][3]
    - tmp1 * dy4;
  lhs[j][AA][4][3] = - tmp2 * fjac[j-1][4][3]
    - tmp1 * njac[j-1][4][3];

  lhs[j][AA][0][4] = - tmp2 * fjac[j-1][0][4]
    - tmp1 * njac[j-1][0][4];
  lhs[j][AA][1][4] = - tmp2 * fjac[j-1][1][4]
    - tmp1 * njac[j-1][1][4];
  lhs[j][AA][2][4] = - tmp2 * fjac[j-1][2][4]
    - tmp1 * njac[j-1][2][4];
  lhs[j][AA][3][4] = - tmp2 * fjac[j-1][3][4]
    - tmp1 * njac[j-1][3][4];
  lhs[j][AA][4][4] = - tmp2 * fjac[j-1][4][4]
    - tmp1 * njac[j-1][4][4]
    - tmp1 * dy5;

  lhs[j][BB][0][0] = 1.0
    + tmp1 * 2.0 * njac[j][0][0]
    + tmp1 * 2.0 * dy1;
  lhs[j][BB][1][0] = tmp1 * 2.0 * njac[j][1][0];
  lhs[j][BB][2][0] = tmp1 * 2.0 * njac[j][2][0];
  lhs[j][BB][3][0] = tmp1 * 2.0 * njac[j][3][0];
  lhs[j][BB][4][0] = tmp1 * 2.0 * njac[j][4][0];

  lhs[j][BB][0][1] = tmp1 * 2.0 * njac[j][0][1];
  lhs[j][BB][1][1] = 1.0
    + tmp1 * 2.0 * njac[j][1][1]
    + tmp1 * 2.0 * dy2;
  lhs[j][BB][2][1] = tmp1 * 2.0 * njac[j][2][1];
  lhs[j][BB][3][1] = tmp1 * 2.0 * njac[j][3][1];
  lhs[j][BB][4][1] = tmp1 * 2.0 * njac[j][4][1];

  lhs[j][BB][0][2] = tmp1 * 2.0 * njac[j][0][2];
  lhs[j][BB][1][2] = tmp1 * 2.0 * njac[j][1][2];
  lhs[j][BB][2][2] = 1.0
    + tmp1 * 2.0 * njac[j][2][2]
    + tmp1 * 2.0 * dy3;
  lhs[j][BB][3][2] = tmp1 * 2.0 * njac[j][3][2];
  lhs[j][BB][4][2] = tmp1 * 2.0 * njac[j][4][2];

  lhs[j][BB][0][3] = tmp1 * 2.0 * njac[j][0][3];
  lhs[j][BB][1][3] = tmp1 * 2.0 * njac[j][1][3];
  lhs[j][BB][2][3] = tmp1 * 2.0 * njac[j][2][3];
  lhs[j][BB][3][3] = 1.0
    + tmp1 * 2.0 * njac[j][3][3]
    + tmp1 * 2.0 * dy4;
  lhs[j][BB][4][3] = tmp1 * 2.0 * njac[j][4][3];

  lhs[j][BB][0][4] = tmp1 * 2.0 * njac[j][0][4];
  lhs[j][BB][1][4] = tmp1 * 2.0 * njac[j][1][4];
  lhs[j][BB][2][4] = tmp1 * 2.0 * njac[j][2][4];
  lhs[j][BB][3][4] = tmp1 * 2.0 * njac[j][3][4];
  lhs[j][BB][4][4] = 1.0
    + tmp1 * 2.0 * njac[j][4][4] 
    + tmp1 * 2.0 * dy5;

  lhs[j][CC][0][0] =  tmp2 * fjac[j+1][0][0]
    - tmp1 * njac[j+1][0][0]
    - tmp1 * dy1;
  lhs[j][CC][1][0] =  tmp2 * fjac[j+1][1][0]
    - tmp1 * njac[j+1][1][0];
  lhs[j][CC][2][0] =  tmp2 * fjac[j+1][2][0]
    - tmp1 * njac[j+1][2][0];
  lhs[j][CC][3][0] =  tmp2 * fjac[j+1][3][0]
    - tmp1 * njac[j+1][3][0];
  lhs[j][CC][4][0] =  tmp2 * fjac[j+1][4][0]
    - tmp1 * njac[j+1][4][0];

  lhs[j][CC][0][1] =  tmp2 * fjac[j+1][0][1]
    - tmp1 * njac[j+1][0][1];
  lhs[j][CC][1][1] =  tmp2 * fjac[j+1][1][1]
    - tmp1 * njac[j+1][1][1]
    - tmp1 * dy2;
  lhs[j][CC][2][1] =  tmp2 * fjac[j+1][2][1]
    - tmp1 * njac[j+1][2][1];
  lhs[j][CC][3][1] =  tmp2 * fjac[j+1][3][1]
    - tmp1 * njac[j+1][3][1];
  lhs[j][CC][4][1] =  tmp2 * fjac[j+1][4][1]
    - tmp1 * njac[j+1][4][1];

  lhs[j][CC][0][2] =  tmp2 * fjac[j+1][0][2]
    - tmp1 * njac[j+1][0][2];
  lhs[j][CC][1][2] =  tmp2 * fjac[j+1][1][2]
    - tmp1 * njac[j+1][1][2];
  lhs[j][CC][2][2] =  tmp2 * fjac[j+1][2][2]
    - tmp1 * njac[j+1][2][2]
    - tmp1 * dy3;
  lhs[j][CC][3][2] =  tmp2 * fjac[j+1][3][2]
    - tmp1 * njac[j+1][3][2];
  lhs[j][CC][4][2] =  tmp2 * fjac[j+1][4][2]
    - tmp1 * njac[j+1][4][2];

  lhs[j][CC][0][3] =  tmp2 * fjac[j+1][0][3]
    - tmp1 * njac[j+1][0][3];
  lhs[j][CC][1][3] =  tmp2 * fjac[j+1][1][3]
    - tmp1 * njac[j+1][1][3];
  lhs[j][CC][2][3] =  tmp2 * fjac[j+1][2][3]
    - tmp1 * njac[j+1][2][3];
  lhs[j][CC][3][3] =  tmp2 * fjac[j+1][3][3]
    - tmp1 * njac[j+1][3][3]
    - tmp1 * dy4;
  lhs[j][CC][4][3] =  tmp2 * fjac[j+1][4][3]
    - tmp1 * njac[j+1][4][3];

  lhs[j][CC][0][4] =  tmp2 * fjac[j+1][0][4]
    - tmp1 * njac[j+1][0][4];
  lhs[j][CC][1][4] =  tmp2 * fjac[j+1][1][4]
    - tmp1 * njac[j+1][1][4];
  lhs[j][CC][2][4] =  tmp2 * fjac[j+1][2][4]
    - tmp1 * njac[j+1][2][4];
  lhs[j][CC][3][4] =  tmp2 * fjac[j+1][3][4]
    - tmp1 * njac[j+1][3][4];
  lhs[j][CC][4][4] =  tmp2 * fjac[j+1][4][4]
    - tmp1 * njac[j+1][4][4]
    - tmp1 * dy5;
}

__kernel void y_solve(__global double *g_rhs,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, jsize;
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  k = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || i > (gp0-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  jsize = gp1-1;

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(JMAX) and rhs'(JMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[k][0][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  load_matrix(p_lhsBB, lhs[0][BB]);
  load_matrix(p_lhsCC, lhs[0][CC]);
  load_vector(p_rhs, rhs[k][0][i]);
  p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
  save_matrix(lhs[0][BB], p_lhsBB);
  save_matrix(lhs[0][CC], p_lhsCC);
  save_vector(rhs[k][0][i], p_rhs);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (j = 1; j <= jsize-1; j++) {
    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[j][AA]);
    load_matrix(p_lhsBB, lhs[j][BB]);
    load_matrix(p_lhsCC, lhs[j][CC]);
    load_vector(p_rhs, rhs[k][j][i]);

    //-------------------------------------------------------------------
    // subtract A*lhs_vector(j-1) from lhs_vector(j)
    // 
    // rhs(j) = rhs(j) - A*rhs(j-1)
    //-------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //-------------------------------------------------------------------
    // B(j) = B(j) - C(j-1)*A(j)
    //-------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
    //-------------------------------------------------------------------
    p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

    save_matrix(lhs[j][BB], p_lhsBB);
    save_matrix(lhs[j][CC], p_lhsCC);
    save_vector(rhs[k][j][i], p_rhs);
  }

  copy_matrix(p_lhsm1CC, p_lhsCC);
  copy_vector(p_rhsm1, p_rhs);
  load_matrix(p_lhsAA, lhs[jsize][AA]);
  load_matrix(p_lhsBB, lhs[jsize][BB]);
  load_matrix(p_lhsCC, lhs[jsize][CC]);
  load_vector(p_rhs, rhs[k][jsize][i]);

  //---------------------------------------------------------------------
  // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
  //---------------------------------------------------------------------
  p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

  //---------------------------------------------------------------------
  // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
  // matmul_sub(AA,i,jsize,k,c,
  // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
  //---------------------------------------------------------------------
  p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

  //---------------------------------------------------------------------
  // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
  //---------------------------------------------------------------------
  p_binvrhs( p_lhsBB, p_rhs );

  save_matrix(lhs[jsize][BB], p_lhsBB);
  save_matrix(lhs[jsize][CC], p_lhsCC);
  save_vector(rhs[k][jsize][i], p_rhs);

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(jsize)=rhs(jsize)
  // else assume U(jsize) is loaded in un pack backsub_info
  // so just use it
  // after u(jstart) will be sent to next cell
  //---------------------------------------------------------------------
  for (j = jsize-1; j >= 0; j--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      p_rhsm1[m] = rhs[k][j][i][m];
      for (n = 0; n < BLOCK_SIZE; n++) {
        p_rhsm1[m] = p_rhsm1[m]
          - lhs[j][CC][n][m]*p_rhs[n];
      }
      rhs[k][j][i][m] = p_rhsm1[m];
    }
    copy_vector(p_rhs, p_rhsm1);
  }
}


#elif Y_SOLVE_DIM == 2
__kernel void y_solve(__global double *g_qs,
                      __global double *g_rho_i,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, jsize;
  double tmp1, tmp2, tmp3;

  k = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || i > (gp0-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  jsize = gp1-1;

  for (j = 0; j <= jsize; j++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[j][0][0] = 0.0;
    fjac[j][1][0] = 0.0;
    fjac[j][2][0] = 1.0;
    fjac[j][3][0] = 0.0;
    fjac[j][4][0] = 0.0;

    fjac[j][0][1] = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac[j][1][1] = u[k][j][i][2] * tmp1;
    fjac[j][2][1] = u[k][j][i][1] * tmp1;
    fjac[j][3][1] = 0.0;
    fjac[j][4][1] = 0.0;

    fjac[j][0][2] = - ( u[k][j][i][2]*u[k][j][i][2]*tmp2)
      + c2 * qs[k][j][i];
    fjac[j][1][2] = - c2 *  u[k][j][i][1] * tmp1;
    fjac[j][2][2] = ( 2.0 - c2 ) *  u[k][j][i][2] * tmp1;
    fjac[j][3][2] = - c2 * u[k][j][i][3] * tmp1;
    fjac[j][4][2] = c2;

    fjac[j][0][3] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[j][1][3] = 0.0;
    fjac[j][2][3] = u[k][j][i][3] * tmp1;
    fjac[j][3][3] = u[k][j][i][2] * tmp1;
    fjac[j][4][3] = 0.0;

    fjac[j][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][2] * tmp2;
    fjac[j][1][4] = - c2 * u[k][j][i][1]*u[k][j][i][2] * tmp2;
    fjac[j][2][4] = c1 * u[k][j][i][4] * tmp1 
      - c2 * ( qs[k][j][i] + u[k][j][i][2]*u[k][j][i][2] * tmp2 );
    fjac[j][3][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[j][4][4] = c1 * u[k][j][i][2] * tmp1;

    njac[j][0][0] = 0.0;
    njac[j][1][0] = 0.0;
    njac[j][2][0] = 0.0;
    njac[j][3][0] = 0.0;
    njac[j][4][0] = 0.0;

    njac[j][0][1] = - c3c4 * tmp2 * u[k][j][i][1];
    njac[j][1][1] =   c3c4 * tmp1;
    njac[j][2][1] =   0.0;
    njac[j][3][1] =   0.0;
    njac[j][4][1] =   0.0;

    njac[j][0][2] = - con43 * c3c4 * tmp2 * u[k][j][i][2];
    njac[j][1][2] =   0.0;
    njac[j][2][2] =   con43 * c3c4 * tmp1;
    njac[j][3][2] =   0.0;
    njac[j][4][2] =   0.0;

    njac[j][0][3] = - c3c4 * tmp2 * u[k][j][i][3];
    njac[j][1][3] =   0.0;
    njac[j][2][3] =   0.0;
    njac[j][3][3] =   c3c4 * tmp1;
    njac[j][4][3] =   0.0;

    njac[j][0][4] = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[j][1][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac[j][2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[j][3][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac[j][4][4] = ( c1345 ) * tmp1;
  }

  //---------------------------------------------------------------------
  // now joacobians set, so form left hand side in y direction
  //---------------------------------------------------------------------
  lhsinit(lhs, jsize);
  for (j = 1; j <= jsize-1; j++) {
    tmp1 = dt * ty1;
    tmp2 = dt * ty2;

    lhs[j][AA][0][0] = - tmp2 * fjac[j-1][0][0]
      - tmp1 * njac[j-1][0][0]
      - tmp1 * dy1; 
    lhs[j][AA][1][0] = - tmp2 * fjac[j-1][1][0]
      - tmp1 * njac[j-1][1][0];
    lhs[j][AA][2][0] = - tmp2 * fjac[j-1][2][0]
      - tmp1 * njac[j-1][2][0];
    lhs[j][AA][3][0] = - tmp2 * fjac[j-1][3][0]
      - tmp1 * njac[j-1][3][0];
    lhs[j][AA][4][0] = - tmp2 * fjac[j-1][4][0]
      - tmp1 * njac[j-1][4][0];

    lhs[j][AA][0][1] = - tmp2 * fjac[j-1][0][1]
      - tmp1 * njac[j-1][0][1];
    lhs[j][AA][1][1] = - tmp2 * fjac[j-1][1][1]
      - tmp1 * njac[j-1][1][1]
      - tmp1 * dy2;
    lhs[j][AA][2][1] = - tmp2 * fjac[j-1][2][1]
      - tmp1 * njac[j-1][2][1];
    lhs[j][AA][3][1] = - tmp2 * fjac[j-1][3][1]
      - tmp1 * njac[j-1][3][1];
    lhs[j][AA][4][1] = - tmp2 * fjac[j-1][4][1]
      - tmp1 * njac[j-1][4][1];

    lhs[j][AA][0][2] = - tmp2 * fjac[j-1][0][2]
      - tmp1 * njac[j-1][0][2];
    lhs[j][AA][1][2] = - tmp2 * fjac[j-1][1][2]
      - tmp1 * njac[j-1][1][2];
    lhs[j][AA][2][2] = - tmp2 * fjac[j-1][2][2]
      - tmp1 * njac[j-1][2][2]
      - tmp1 * dy3;
    lhs[j][AA][3][2] = - tmp2 * fjac[j-1][3][2]
      - tmp1 * njac[j-1][3][2];
    lhs[j][AA][4][2] = - tmp2 * fjac[j-1][4][2]
      - tmp1 * njac[j-1][4][2];

    lhs[j][AA][0][3] = - tmp2 * fjac[j-1][0][3]
      - tmp1 * njac[j-1][0][3];
    lhs[j][AA][1][3] = - tmp2 * fjac[j-1][1][3]
      - tmp1 * njac[j-1][1][3];
    lhs[j][AA][2][3] = - tmp2 * fjac[j-1][2][3]
      - tmp1 * njac[j-1][2][3];
    lhs[j][AA][3][3] = - tmp2 * fjac[j-1][3][3]
      - tmp1 * njac[j-1][3][3]
      - tmp1 * dy4;
    lhs[j][AA][4][3] = - tmp2 * fjac[j-1][4][3]
      - tmp1 * njac[j-1][4][3];

    lhs[j][AA][0][4] = - tmp2 * fjac[j-1][0][4]
      - tmp1 * njac[j-1][0][4];
    lhs[j][AA][1][4] = - tmp2 * fjac[j-1][1][4]
      - tmp1 * njac[j-1][1][4];
    lhs[j][AA][2][4] = - tmp2 * fjac[j-1][2][4]
      - tmp1 * njac[j-1][2][4];
    lhs[j][AA][3][4] = - tmp2 * fjac[j-1][3][4]
      - tmp1 * njac[j-1][3][4];
    lhs[j][AA][4][4] = - tmp2 * fjac[j-1][4][4]
      - tmp1 * njac[j-1][4][4]
      - tmp1 * dy5;

    lhs[j][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[j][0][0]
      + tmp1 * 2.0 * dy1;
    lhs[j][BB][1][0] = tmp1 * 2.0 * njac[j][1][0];
    lhs[j][BB][2][0] = tmp1 * 2.0 * njac[j][2][0];
    lhs[j][BB][3][0] = tmp1 * 2.0 * njac[j][3][0];
    lhs[j][BB][4][0] = tmp1 * 2.0 * njac[j][4][0];

    lhs[j][BB][0][1] = tmp1 * 2.0 * njac[j][0][1];
    lhs[j][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[j][1][1]
      + tmp1 * 2.0 * dy2;
    lhs[j][BB][2][1] = tmp1 * 2.0 * njac[j][2][1];
    lhs[j][BB][3][1] = tmp1 * 2.0 * njac[j][3][1];
    lhs[j][BB][4][1] = tmp1 * 2.0 * njac[j][4][1];

    lhs[j][BB][0][2] = tmp1 * 2.0 * njac[j][0][2];
    lhs[j][BB][1][2] = tmp1 * 2.0 * njac[j][1][2];
    lhs[j][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[j][2][2]
      + tmp1 * 2.0 * dy3;
    lhs[j][BB][3][2] = tmp1 * 2.0 * njac[j][3][2];
    lhs[j][BB][4][2] = tmp1 * 2.0 * njac[j][4][2];

    lhs[j][BB][0][3] = tmp1 * 2.0 * njac[j][0][3];
    lhs[j][BB][1][3] = tmp1 * 2.0 * njac[j][1][3];
    lhs[j][BB][2][3] = tmp1 * 2.0 * njac[j][2][3];
    lhs[j][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[j][3][3]
      + tmp1 * 2.0 * dy4;
    lhs[j][BB][4][3] = tmp1 * 2.0 * njac[j][4][3];

    lhs[j][BB][0][4] = tmp1 * 2.0 * njac[j][0][4];
    lhs[j][BB][1][4] = tmp1 * 2.0 * njac[j][1][4];
    lhs[j][BB][2][4] = tmp1 * 2.0 * njac[j][2][4];
    lhs[j][BB][3][4] = tmp1 * 2.0 * njac[j][3][4];
    lhs[j][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[j][4][4] 
      + tmp1 * 2.0 * dy5;

    lhs[j][CC][0][0] =  tmp2 * fjac[j+1][0][0]
      - tmp1 * njac[j+1][0][0]
      - tmp1 * dy1;
    lhs[j][CC][1][0] =  tmp2 * fjac[j+1][1][0]
      - tmp1 * njac[j+1][1][0];
    lhs[j][CC][2][0] =  tmp2 * fjac[j+1][2][0]
      - tmp1 * njac[j+1][2][0];
    lhs[j][CC][3][0] =  tmp2 * fjac[j+1][3][0]
      - tmp1 * njac[j+1][3][0];
    lhs[j][CC][4][0] =  tmp2 * fjac[j+1][4][0]
      - tmp1 * njac[j+1][4][0];

    lhs[j][CC][0][1] =  tmp2 * fjac[j+1][0][1]
      - tmp1 * njac[j+1][0][1];
    lhs[j][CC][1][1] =  tmp2 * fjac[j+1][1][1]
      - tmp1 * njac[j+1][1][1]
      - tmp1 * dy2;
    lhs[j][CC][2][1] =  tmp2 * fjac[j+1][2][1]
      - tmp1 * njac[j+1][2][1];
    lhs[j][CC][3][1] =  tmp2 * fjac[j+1][3][1]
      - tmp1 * njac[j+1][3][1];
    lhs[j][CC][4][1] =  tmp2 * fjac[j+1][4][1]
      - tmp1 * njac[j+1][4][1];

    lhs[j][CC][0][2] =  tmp2 * fjac[j+1][0][2]
      - tmp1 * njac[j+1][0][2];
    lhs[j][CC][1][2] =  tmp2 * fjac[j+1][1][2]
      - tmp1 * njac[j+1][1][2];
    lhs[j][CC][2][2] =  tmp2 * fjac[j+1][2][2]
      - tmp1 * njac[j+1][2][2]
      - tmp1 * dy3;
    lhs[j][CC][3][2] =  tmp2 * fjac[j+1][3][2]
      - tmp1 * njac[j+1][3][2];
    lhs[j][CC][4][2] =  tmp2 * fjac[j+1][4][2]
      - tmp1 * njac[j+1][4][2];

    lhs[j][CC][0][3] =  tmp2 * fjac[j+1][0][3]
      - tmp1 * njac[j+1][0][3];
    lhs[j][CC][1][3] =  tmp2 * fjac[j+1][1][3]
      - tmp1 * njac[j+1][1][3];
    lhs[j][CC][2][3] =  tmp2 * fjac[j+1][2][3]
      - tmp1 * njac[j+1][2][3];
    lhs[j][CC][3][3] =  tmp2 * fjac[j+1][3][3]
      - tmp1 * njac[j+1][3][3]
      - tmp1 * dy4;
    lhs[j][CC][4][3] =  tmp2 * fjac[j+1][4][3]
      - tmp1 * njac[j+1][4][3];

    lhs[j][CC][0][4] =  tmp2 * fjac[j+1][0][4]
      - tmp1 * njac[j+1][0][4];
    lhs[j][CC][1][4] =  tmp2 * fjac[j+1][1][4]
      - tmp1 * njac[j+1][1][4];
    lhs[j][CC][2][4] =  tmp2 * fjac[j+1][2][4]
      - tmp1 * njac[j+1][2][4];
    lhs[j][CC][3][4] =  tmp2 * fjac[j+1][3][4]
      - tmp1 * njac[j+1][3][4];
    lhs[j][CC][4][4] =  tmp2 * fjac[j+1][4][4]
      - tmp1 * njac[j+1][4][4]
      - tmp1 * dy5;
  }

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(JMAX) and rhs'(JMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[k][0][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[k][0][i] );

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (j = 1; j <= jsize-1; j++) {
    //-------------------------------------------------------------------
    // subtract A*lhs_vector(j-1) from lhs_vector(j)
    // 
    // rhs(j) = rhs(j) - A*rhs(j-1)
    //-------------------------------------------------------------------
    matvec_sub(lhs[j][AA], rhs[k][j-1][i], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(j) = B(j) - C(j-1)*A(j)
    //-------------------------------------------------------------------
    matmul_sub(lhs[j][AA], lhs[j-1][CC], lhs[j][BB]);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[j][BB], lhs[j][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[jsize][AA], rhs[k][jsize-1][i], rhs[k][jsize][i]);

  //---------------------------------------------------------------------
  // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
  // matmul_sub(AA,i,jsize,k,c,
  // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
  //---------------------------------------------------------------------
  matmul_sub(lhs[jsize][AA], lhs[jsize-1][CC], lhs[jsize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[jsize][BB], rhs[k][jsize][i] );

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(jsize)=rhs(jsize)
  // else assume U(jsize) is loaded in un pack backsub_info
  // so just use it
  // after u(jstart) will be sent to next cell
  //---------------------------------------------------------------------
  for (j = jsize-1; j >= 0; j--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[j][CC][n][m]*rhs[k][j+1][i][n];
      }
    }
  }
}

#else //Y_SOLVE_DIM == 1
__kernel void y_solve(__global double *g_qs,
                      __global double *g_rho_i,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, jsize;
  double tmp1, tmp2, tmp3;
  double p_u[5];
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  k = get_global_id(0) + 1;
  if (k > (gp2-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = k - 1;
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  jsize = gp1-1;

  for (i = 1; i <= gp0-2; i++) {
    for (j = 0; j <= jsize; j++) {
      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;
      p_u[0] = u[k][j][i][0];
      p_u[1] = u[k][j][i][1];
      p_u[2] = u[k][j][i][2];
      p_u[3] = u[k][j][i][3];
      p_u[4] = u[k][j][i][4];

      fjac[j][0][0] = 0.0;
      fjac[j][1][0] = 0.0;
      fjac[j][2][0] = 1.0;
      fjac[j][3][0] = 0.0;
      fjac[j][4][0] = 0.0;

      fjac[j][0][1] = - ( p_u[1]*p_u[2] ) * tmp2;
      fjac[j][1][1] = p_u[2] * tmp1;
      fjac[j][2][1] = p_u[1] * tmp1;
      fjac[j][3][1] = 0.0;
      fjac[j][4][1] = 0.0;

      fjac[j][0][2] = - ( p_u[2]*p_u[2]*tmp2)
        + c2 * qs[k][j][i];
      fjac[j][1][2] = - c2 *  p_u[1] * tmp1;
      fjac[j][2][2] = ( 2.0 - c2 ) *  p_u[2] * tmp1;
      fjac[j][3][2] = - c2 * p_u[3] * tmp1;
      fjac[j][4][2] = c2;

      fjac[j][0][3] = - ( p_u[2]*p_u[3] ) * tmp2;
      fjac[j][1][3] = 0.0;
      fjac[j][2][3] = p_u[3] * tmp1;
      fjac[j][3][3] = p_u[2] * tmp1;
      fjac[j][4][3] = 0.0;

      fjac[j][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
        * p_u[2] * tmp2;
      fjac[j][1][4] = - c2 * p_u[1]*p_u[2] * tmp2;
      fjac[j][2][4] = c1 * p_u[4] * tmp1 
        - c2 * ( qs[k][j][i] + p_u[2]*p_u[2] * tmp2 );
      fjac[j][3][4] = - c2 * ( p_u[2]*p_u[3] ) * tmp2;
      fjac[j][4][4] = c1 * p_u[2] * tmp1;

      njac[j][0][0] = 0.0;
      njac[j][1][0] = 0.0;
      njac[j][2][0] = 0.0;
      njac[j][3][0] = 0.0;
      njac[j][4][0] = 0.0;

      njac[j][0][1] = - c3c4 * tmp2 * p_u[1];
      njac[j][1][1] =   c3c4 * tmp1;
      njac[j][2][1] =   0.0;
      njac[j][3][1] =   0.0;
      njac[j][4][1] =   0.0;

      njac[j][0][2] = - con43 * c3c4 * tmp2 * p_u[2];
      njac[j][1][2] =   0.0;
      njac[j][2][2] =   con43 * c3c4 * tmp1;
      njac[j][3][2] =   0.0;
      njac[j][4][2] =   0.0;

      njac[j][0][3] = - c3c4 * tmp2 * p_u[3];
      njac[j][1][3] =   0.0;
      njac[j][2][3] =   0.0;
      njac[j][3][3] =   c3c4 * tmp1;
      njac[j][4][3] =   0.0;

      njac[j][0][4] = - (  c3c4
          - c1345 ) * tmp3 * (p_u[1]*p_u[1])
        - ( con43 * c3c4
            - c1345 ) * tmp3 * (p_u[2]*p_u[2])
        - ( c3c4 - c1345 ) * tmp3 * (p_u[3]*p_u[3])
        - c1345 * tmp2 * p_u[4];

      njac[j][1][4] = (  c3c4 - c1345 ) * tmp2 * p_u[1];
      njac[j][2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * p_u[2];
      njac[j][3][4] = ( c3c4 - c1345 ) * tmp2 * p_u[3];
      njac[j][4][4] = ( c1345 ) * tmp1;
    }

    //---------------------------------------------------------------------
    // now joacobians set, so form left hand side in y direction
    //---------------------------------------------------------------------
    lhsinit(lhs, jsize);
    for (j = 1; j <= jsize-1; j++) {
      tmp1 = dt * ty1;
      tmp2 = dt * ty2;

      lhs[j][AA][0][0] = - tmp2 * fjac[j-1][0][0]
        - tmp1 * njac[j-1][0][0]
        - tmp1 * dy1; 
      lhs[j][AA][1][0] = - tmp2 * fjac[j-1][1][0]
        - tmp1 * njac[j-1][1][0];
      lhs[j][AA][2][0] = - tmp2 * fjac[j-1][2][0]
        - tmp1 * njac[j-1][2][0];
      lhs[j][AA][3][0] = - tmp2 * fjac[j-1][3][0]
        - tmp1 * njac[j-1][3][0];
      lhs[j][AA][4][0] = - tmp2 * fjac[j-1][4][0]
        - tmp1 * njac[j-1][4][0];

      lhs[j][AA][0][1] = - tmp2 * fjac[j-1][0][1]
        - tmp1 * njac[j-1][0][1];
      lhs[j][AA][1][1] = - tmp2 * fjac[j-1][1][1]
        - tmp1 * njac[j-1][1][1]
        - tmp1 * dy2;
      lhs[j][AA][2][1] = - tmp2 * fjac[j-1][2][1]
        - tmp1 * njac[j-1][2][1];
      lhs[j][AA][3][1] = - tmp2 * fjac[j-1][3][1]
        - tmp1 * njac[j-1][3][1];
      lhs[j][AA][4][1] = - tmp2 * fjac[j-1][4][1]
        - tmp1 * njac[j-1][4][1];

      lhs[j][AA][0][2] = - tmp2 * fjac[j-1][0][2]
        - tmp1 * njac[j-1][0][2];
      lhs[j][AA][1][2] = - tmp2 * fjac[j-1][1][2]
        - tmp1 * njac[j-1][1][2];
      lhs[j][AA][2][2] = - tmp2 * fjac[j-1][2][2]
        - tmp1 * njac[j-1][2][2]
        - tmp1 * dy3;
      lhs[j][AA][3][2] = - tmp2 * fjac[j-1][3][2]
        - tmp1 * njac[j-1][3][2];
      lhs[j][AA][4][2] = - tmp2 * fjac[j-1][4][2]
        - tmp1 * njac[j-1][4][2];

      lhs[j][AA][0][3] = - tmp2 * fjac[j-1][0][3]
        - tmp1 * njac[j-1][0][3];
      lhs[j][AA][1][3] = - tmp2 * fjac[j-1][1][3]
        - tmp1 * njac[j-1][1][3];
      lhs[j][AA][2][3] = - tmp2 * fjac[j-1][2][3]
        - tmp1 * njac[j-1][2][3];
      lhs[j][AA][3][3] = - tmp2 * fjac[j-1][3][3]
        - tmp1 * njac[j-1][3][3]
        - tmp1 * dy4;
      lhs[j][AA][4][3] = - tmp2 * fjac[j-1][4][3]
        - tmp1 * njac[j-1][4][3];

      lhs[j][AA][0][4] = - tmp2 * fjac[j-1][0][4]
        - tmp1 * njac[j-1][0][4];
      lhs[j][AA][1][4] = - tmp2 * fjac[j-1][1][4]
        - tmp1 * njac[j-1][1][4];
      lhs[j][AA][2][4] = - tmp2 * fjac[j-1][2][4]
        - tmp1 * njac[j-1][2][4];
      lhs[j][AA][3][4] = - tmp2 * fjac[j-1][3][4]
        - tmp1 * njac[j-1][3][4];
      lhs[j][AA][4][4] = - tmp2 * fjac[j-1][4][4]
        - tmp1 * njac[j-1][4][4]
        - tmp1 * dy5;

      lhs[j][BB][0][0] = 1.0
        + tmp1 * 2.0 * njac[j][0][0]
        + tmp1 * 2.0 * dy1;
      lhs[j][BB][1][0] = tmp1 * 2.0 * njac[j][1][0];
      lhs[j][BB][2][0] = tmp1 * 2.0 * njac[j][2][0];
      lhs[j][BB][3][0] = tmp1 * 2.0 * njac[j][3][0];
      lhs[j][BB][4][0] = tmp1 * 2.0 * njac[j][4][0];

      lhs[j][BB][0][1] = tmp1 * 2.0 * njac[j][0][1];
      lhs[j][BB][1][1] = 1.0
        + tmp1 * 2.0 * njac[j][1][1]
        + tmp1 * 2.0 * dy2;
      lhs[j][BB][2][1] = tmp1 * 2.0 * njac[j][2][1];
      lhs[j][BB][3][1] = tmp1 * 2.0 * njac[j][3][1];
      lhs[j][BB][4][1] = tmp1 * 2.0 * njac[j][4][1];

      lhs[j][BB][0][2] = tmp1 * 2.0 * njac[j][0][2];
      lhs[j][BB][1][2] = tmp1 * 2.0 * njac[j][1][2];
      lhs[j][BB][2][2] = 1.0
        + tmp1 * 2.0 * njac[j][2][2]
        + tmp1 * 2.0 * dy3;
      lhs[j][BB][3][2] = tmp1 * 2.0 * njac[j][3][2];
      lhs[j][BB][4][2] = tmp1 * 2.0 * njac[j][4][2];

      lhs[j][BB][0][3] = tmp1 * 2.0 * njac[j][0][3];
      lhs[j][BB][1][3] = tmp1 * 2.0 * njac[j][1][3];
      lhs[j][BB][2][3] = tmp1 * 2.0 * njac[j][2][3];
      lhs[j][BB][3][3] = 1.0
        + tmp1 * 2.0 * njac[j][3][3]
        + tmp1 * 2.0 * dy4;
      lhs[j][BB][4][3] = tmp1 * 2.0 * njac[j][4][3];

      lhs[j][BB][0][4] = tmp1 * 2.0 * njac[j][0][4];
      lhs[j][BB][1][4] = tmp1 * 2.0 * njac[j][1][4];
      lhs[j][BB][2][4] = tmp1 * 2.0 * njac[j][2][4];
      lhs[j][BB][3][4] = tmp1 * 2.0 * njac[j][3][4];
      lhs[j][BB][4][4] = 1.0
        + tmp1 * 2.0 * njac[j][4][4] 
        + tmp1 * 2.0 * dy5;

      lhs[j][CC][0][0] =  tmp2 * fjac[j+1][0][0]
        - tmp1 * njac[j+1][0][0]
        - tmp1 * dy1;
      lhs[j][CC][1][0] =  tmp2 * fjac[j+1][1][0]
        - tmp1 * njac[j+1][1][0];
      lhs[j][CC][2][0] =  tmp2 * fjac[j+1][2][0]
        - tmp1 * njac[j+1][2][0];
      lhs[j][CC][3][0] =  tmp2 * fjac[j+1][3][0]
        - tmp1 * njac[j+1][3][0];
      lhs[j][CC][4][0] =  tmp2 * fjac[j+1][4][0]
        - tmp1 * njac[j+1][4][0];

      lhs[j][CC][0][1] =  tmp2 * fjac[j+1][0][1]
        - tmp1 * njac[j+1][0][1];
      lhs[j][CC][1][1] =  tmp2 * fjac[j+1][1][1]
        - tmp1 * njac[j+1][1][1]
        - tmp1 * dy2;
      lhs[j][CC][2][1] =  tmp2 * fjac[j+1][2][1]
        - tmp1 * njac[j+1][2][1];
      lhs[j][CC][3][1] =  tmp2 * fjac[j+1][3][1]
        - tmp1 * njac[j+1][3][1];
      lhs[j][CC][4][1] =  tmp2 * fjac[j+1][4][1]
        - tmp1 * njac[j+1][4][1];

      lhs[j][CC][0][2] =  tmp2 * fjac[j+1][0][2]
        - tmp1 * njac[j+1][0][2];
      lhs[j][CC][1][2] =  tmp2 * fjac[j+1][1][2]
        - tmp1 * njac[j+1][1][2];
      lhs[j][CC][2][2] =  tmp2 * fjac[j+1][2][2]
        - tmp1 * njac[j+1][2][2]
        - tmp1 * dy3;
      lhs[j][CC][3][2] =  tmp2 * fjac[j+1][3][2]
        - tmp1 * njac[j+1][3][2];
      lhs[j][CC][4][2] =  tmp2 * fjac[j+1][4][2]
        - tmp1 * njac[j+1][4][2];

      lhs[j][CC][0][3] =  tmp2 * fjac[j+1][0][3]
        - tmp1 * njac[j+1][0][3];
      lhs[j][CC][1][3] =  tmp2 * fjac[j+1][1][3]
        - tmp1 * njac[j+1][1][3];
      lhs[j][CC][2][3] =  tmp2 * fjac[j+1][2][3]
        - tmp1 * njac[j+1][2][3];
      lhs[j][CC][3][3] =  tmp2 * fjac[j+1][3][3]
        - tmp1 * njac[j+1][3][3]
        - tmp1 * dy4;
      lhs[j][CC][4][3] =  tmp2 * fjac[j+1][4][3]
        - tmp1 * njac[j+1][4][3];

      lhs[j][CC][0][4] =  tmp2 * fjac[j+1][0][4]
        - tmp1 * njac[j+1][0][4];
      lhs[j][CC][1][4] =  tmp2 * fjac[j+1][1][4]
        - tmp1 * njac[j+1][1][4];
      lhs[j][CC][2][4] =  tmp2 * fjac[j+1][2][4]
        - tmp1 * njac[j+1][2][4];
      lhs[j][CC][3][4] =  tmp2 * fjac[j+1][3][4]
        - tmp1 * njac[j+1][3][4];
      lhs[j][CC][4][4] =  tmp2 * fjac[j+1][4][4]
        - tmp1 * njac[j+1][4][4]
        - tmp1 * dy5;
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // performs guaussian elimination on this cell.
    // 
    // assumes that unpacking routines for non-first cells 
    // preload C' and rhs' from previous cell.
    // 
    // assumed send happens outside this routine, but that
    // c'(JMAX) and rhs'(JMAX) will be sent to next cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[k][0][i] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    load_matrix(p_lhsBB, lhs[0][BB]);
    load_matrix(p_lhsCC, lhs[0][CC]);
    load_vector(p_rhs, rhs[k][0][i]);
    p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
    save_matrix(lhs[0][BB], p_lhsBB);
    save_matrix(lhs[0][CC], p_lhsCC);
    save_vector(rhs[k][0][i], p_rhs);

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last 
    //---------------------------------------------------------------------
    for (j = 1; j <= jsize-1; j++) {
      copy_matrix(p_lhsm1CC, p_lhsCC);
      copy_vector(p_rhsm1, p_rhs);
      load_matrix(p_lhsAA, lhs[j][AA]);
      load_matrix(p_lhsBB, lhs[j][BB]);
      load_matrix(p_lhsCC, lhs[j][CC]);
      load_vector(p_rhs, rhs[k][j][i]);

      //-------------------------------------------------------------------
      // subtract A*lhs_vector(j-1) from lhs_vector(j)
      // 
      // rhs(j) = rhs(j) - A*rhs(j-1)
      //-------------------------------------------------------------------
      p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

      //-------------------------------------------------------------------
      // B(j) = B(j) - C(j-1)*A(j)
      //-------------------------------------------------------------------
      p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

      //-------------------------------------------------------------------
      // multiply c[k][j][i] by b_inverse and copy back to c
      // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
      //-------------------------------------------------------------------
      p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

      save_matrix(lhs[j][BB], p_lhsBB);
      save_matrix(lhs[j][CC], p_lhsCC);
      save_vector(rhs[k][j][i], p_rhs);
    }

    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[jsize][AA]);
    load_matrix(p_lhsBB, lhs[jsize][BB]);
    load_matrix(p_lhsCC, lhs[jsize][CC]);
    load_vector(p_rhs, rhs[k][jsize][i]);

    //---------------------------------------------------------------------
    // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
    //---------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //---------------------------------------------------------------------
    // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
    // matmul_sub(AA,i,jsize,k,c,
    // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
    //---------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //---------------------------------------------------------------------
    // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
    //---------------------------------------------------------------------
    p_binvrhs( p_lhsBB, p_rhs );

    save_matrix(lhs[jsize][BB], p_lhsBB);
    save_matrix(lhs[jsize][CC], p_lhsCC);
    save_vector(rhs[k][jsize][i], p_rhs);

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(jsize)=rhs(jsize)
    // else assume U(jsize) is loaded in un pack backsub_info
    // so just use it
    // after u(jstart) will be sent to next cell
    //---------------------------------------------------------------------
    for (j = jsize-1; j >= 0; j--) {
      for (m = 0; m < BLOCK_SIZE; m++) {
        p_rhsm1[m] = rhs[k][j][i][m];
        for (n = 0; n < BLOCK_SIZE; n++) {
          p_rhsm1[m] = p_rhsm1[m]
            - lhs[j][CC][n][m]*p_rhs[n];
        }
        rhs[k][j][i][m] = p_rhsm1[m];
      }
      copy_vector(p_rhs, p_rhsm1);
    }
  }
}
#endif


#if Z_SOLVE_DIM == 3
__kernel void z_solve1(__global double *g_qs,
                       __global double *g_square,
                       __global double *g_u,
                       __global double *g_fjac,
                       __global double *g_njac,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2, tmp3;
  double p_u[5];

  j = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  k = get_global_id(0);
  if (j > (gp1-2) || i > (gp0-2) || k >= gp2) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

  int my_id = (j-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  tmp1 = 1.0 / u[k][j][i][0];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;
  p_u[0] = u[k][j][i][0];
  p_u[1] = u[k][j][i][1];
  p_u[2] = u[k][j][i][2];
  p_u[3] = u[k][j][i][3];
  p_u[4] = u[k][j][i][4];

  fjac[k][0][0] = 0.0;
  fjac[k][1][0] = 0.0;
  fjac[k][2][0] = 0.0;
  fjac[k][3][0] = 1.0;
  fjac[k][4][0] = 0.0;

  fjac[k][0][1] = - ( p_u[1]*p_u[3] ) * tmp2;
  fjac[k][1][1] = p_u[3] * tmp1;
  fjac[k][2][1] = 0.0;
  fjac[k][3][1] = p_u[1] * tmp1;
  fjac[k][4][1] = 0.0;

  fjac[k][0][2] = - ( p_u[2]*p_u[3] ) * tmp2;
  fjac[k][1][2] = 0.0;
  fjac[k][2][2] = p_u[3] * tmp1;
  fjac[k][3][2] = p_u[2] * tmp1;
  fjac[k][4][2] = 0.0;

  fjac[k][0][3] = - (p_u[3]*p_u[3] * tmp2 ) 
    + c2 * qs[k][j][i];
  fjac[k][1][3] = - c2 *  p_u[1] * tmp1;
  fjac[k][2][3] = - c2 *  p_u[2] * tmp1;
  fjac[k][3][3] = ( 2.0 - c2 ) *  p_u[3] * tmp1;
  fjac[k][4][3] = c2;

  fjac[k][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
    * p_u[3] * tmp2;
  fjac[k][1][4] = - c2 * ( p_u[1]*p_u[3] ) * tmp2;
  fjac[k][2][4] = - c2 * ( p_u[2]*p_u[3] ) * tmp2;
  fjac[k][3][4] = c1 * ( p_u[4] * tmp1 )
    - c2 * ( qs[k][j][i] + p_u[3]*p_u[3] * tmp2 );
  fjac[k][4][4] = c1 * p_u[3] * tmp1;

  njac[k][0][0] = 0.0;
  njac[k][1][0] = 0.0;
  njac[k][2][0] = 0.0;
  njac[k][3][0] = 0.0;
  njac[k][4][0] = 0.0;

  njac[k][0][1] = - c3c4 * tmp2 * p_u[1];
  njac[k][1][1] =   c3c4 * tmp1;
  njac[k][2][1] =   0.0;
  njac[k][3][1] =   0.0;
  njac[k][4][1] =   0.0;

  njac[k][0][2] = - c3c4 * tmp2 * p_u[2];
  njac[k][1][2] =   0.0;
  njac[k][2][2] =   c3c4 * tmp1;
  njac[k][3][2] =   0.0;
  njac[k][4][2] =   0.0;

  njac[k][0][3] = - con43 * c3c4 * tmp2 * p_u[3];
  njac[k][1][3] =   0.0;
  njac[k][2][3] =   0.0;
  njac[k][3][3] =   con43 * c3 * c4 * tmp1;
  njac[k][4][3] =   0.0;

  njac[k][0][4] = - (  c3c4
      - c1345 ) * tmp3 * (p_u[1]*p_u[1])
    - ( c3c4 - c1345 ) * tmp3 * (p_u[2]*p_u[2])
    - ( con43 * c3c4
        - c1345 ) * tmp3 * (p_u[3]*p_u[3])
    - c1345 * tmp2 * p_u[4];

  njac[k][1][4] = (  c3c4 - c1345 ) * tmp2 * p_u[1];
  njac[k][2][4] = (  c3c4 - c1345 ) * tmp2 * p_u[2];
  njac[k][3][4] = ( con43 * c3c4
      - c1345 ) * tmp2 * p_u[3];
  njac[k][4][4] = ( c1345 )* tmp1;
}

__kernel void z_solve2(__global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k, n, m;

  j = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  if (j > (gp1-2) || i > (gp0-2)) return;

  k = get_global_id(0);
  if (k == 1) k = gp2-1;
  
  int my_id = (j-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[k][0][n][m] = 0.0;
      lhs[k][1][n][m] = 0.0;
      lhs[k][2][n][m] = 0.0;
    }
    lhs[k][1][n][n] = 1.0;
  }
}

__kernel void z_solve3(__global double *g_fjac,
                       __global double *g_njac,
                       __global double *g_lhs,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k;
  double tmp1, tmp2;

  j = get_global_id(2) + 1;
  i = get_global_id(1) + 1;
  k = get_global_id(0) + 1;
  if (j > (gp1-2) || i > (gp0-2) || k > (gp2-2)) return;

  int my_id = (j-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  tmp1 = dt * tz1;
  tmp2 = dt * tz2;

  lhs[k][AA][0][0] = - tmp2 * fjac[k-1][0][0]
    - tmp1 * njac[k-1][0][0]
    - tmp1 * dz1; 
  lhs[k][AA][1][0] = - tmp2 * fjac[k-1][1][0]
    - tmp1 * njac[k-1][1][0];
  lhs[k][AA][2][0] = - tmp2 * fjac[k-1][2][0]
    - tmp1 * njac[k-1][2][0];
  lhs[k][AA][3][0] = - tmp2 * fjac[k-1][3][0]
    - tmp1 * njac[k-1][3][0];
  lhs[k][AA][4][0] = - tmp2 * fjac[k-1][4][0]
    - tmp1 * njac[k-1][4][0];

  lhs[k][AA][0][1] = - tmp2 * fjac[k-1][0][1]
    - tmp1 * njac[k-1][0][1];
  lhs[k][AA][1][1] = - tmp2 * fjac[k-1][1][1]
    - tmp1 * njac[k-1][1][1]
    - tmp1 * dz2;
  lhs[k][AA][2][1] = - tmp2 * fjac[k-1][2][1]
    - tmp1 * njac[k-1][2][1];
  lhs[k][AA][3][1] = - tmp2 * fjac[k-1][3][1]
    - tmp1 * njac[k-1][3][1];
  lhs[k][AA][4][1] = - tmp2 * fjac[k-1][4][1]
    - tmp1 * njac[k-1][4][1];

  lhs[k][AA][0][2] = - tmp2 * fjac[k-1][0][2]
    - tmp1 * njac[k-1][0][2];
  lhs[k][AA][1][2] = - tmp2 * fjac[k-1][1][2]
    - tmp1 * njac[k-1][1][2];
  lhs[k][AA][2][2] = - tmp2 * fjac[k-1][2][2]
    - tmp1 * njac[k-1][2][2]
    - tmp1 * dz3;
  lhs[k][AA][3][2] = - tmp2 * fjac[k-1][3][2]
    - tmp1 * njac[k-1][3][2];
  lhs[k][AA][4][2] = - tmp2 * fjac[k-1][4][2]
    - tmp1 * njac[k-1][4][2];

  lhs[k][AA][0][3] = - tmp2 * fjac[k-1][0][3]
    - tmp1 * njac[k-1][0][3];
  lhs[k][AA][1][3] = - tmp2 * fjac[k-1][1][3]
    - tmp1 * njac[k-1][1][3];
  lhs[k][AA][2][3] = - tmp2 * fjac[k-1][2][3]
    - tmp1 * njac[k-1][2][3];
  lhs[k][AA][3][3] = - tmp2 * fjac[k-1][3][3]
    - tmp1 * njac[k-1][3][3]
    - tmp1 * dz4;
  lhs[k][AA][4][3] = - tmp2 * fjac[k-1][4][3]
    - tmp1 * njac[k-1][4][3];

  lhs[k][AA][0][4] = - tmp2 * fjac[k-1][0][4]
    - tmp1 * njac[k-1][0][4];
  lhs[k][AA][1][4] = - tmp2 * fjac[k-1][1][4]
    - tmp1 * njac[k-1][1][4];
  lhs[k][AA][2][4] = - tmp2 * fjac[k-1][2][4]
    - tmp1 * njac[k-1][2][4];
  lhs[k][AA][3][4] = - tmp2 * fjac[k-1][3][4]
    - tmp1 * njac[k-1][3][4];
  lhs[k][AA][4][4] = - tmp2 * fjac[k-1][4][4]
    - tmp1 * njac[k-1][4][4]
    - tmp1 * dz5;

  lhs[k][BB][0][0] = 1.0
    + tmp1 * 2.0 * njac[k][0][0]
    + tmp1 * 2.0 * dz1;
  lhs[k][BB][1][0] = tmp1 * 2.0 * njac[k][1][0];
  lhs[k][BB][2][0] = tmp1 * 2.0 * njac[k][2][0];
  lhs[k][BB][3][0] = tmp1 * 2.0 * njac[k][3][0];
  lhs[k][BB][4][0] = tmp1 * 2.0 * njac[k][4][0];

  lhs[k][BB][0][1] = tmp1 * 2.0 * njac[k][0][1];
  lhs[k][BB][1][1] = 1.0
    + tmp1 * 2.0 * njac[k][1][1]
    + tmp1 * 2.0 * dz2;
  lhs[k][BB][2][1] = tmp1 * 2.0 * njac[k][2][1];
  lhs[k][BB][3][1] = tmp1 * 2.0 * njac[k][3][1];
  lhs[k][BB][4][1] = tmp1 * 2.0 * njac[k][4][1];

  lhs[k][BB][0][2] = tmp1 * 2.0 * njac[k][0][2];
  lhs[k][BB][1][2] = tmp1 * 2.0 * njac[k][1][2];
  lhs[k][BB][2][2] = 1.0
    + tmp1 * 2.0 * njac[k][2][2]
    + tmp1 * 2.0 * dz3;
  lhs[k][BB][3][2] = tmp1 * 2.0 * njac[k][3][2];
  lhs[k][BB][4][2] = tmp1 * 2.0 * njac[k][4][2];

  lhs[k][BB][0][3] = tmp1 * 2.0 * njac[k][0][3];
  lhs[k][BB][1][3] = tmp1 * 2.0 * njac[k][1][3];
  lhs[k][BB][2][3] = tmp1 * 2.0 * njac[k][2][3];
  lhs[k][BB][3][3] = 1.0
    + tmp1 * 2.0 * njac[k][3][3]
    + tmp1 * 2.0 * dz4;
  lhs[k][BB][4][3] = tmp1 * 2.0 * njac[k][4][3];

  lhs[k][BB][0][4] = tmp1 * 2.0 * njac[k][0][4];
  lhs[k][BB][1][4] = tmp1 * 2.0 * njac[k][1][4];
  lhs[k][BB][2][4] = tmp1 * 2.0 * njac[k][2][4];
  lhs[k][BB][3][4] = tmp1 * 2.0 * njac[k][3][4];
  lhs[k][BB][4][4] = 1.0
    + tmp1 * 2.0 * njac[k][4][4] 
    + tmp1 * 2.0 * dz5;

  lhs[k][CC][0][0] =  tmp2 * fjac[k+1][0][0]
    - tmp1 * njac[k+1][0][0]
    - tmp1 * dz1;
  lhs[k][CC][1][0] =  tmp2 * fjac[k+1][1][0]
    - tmp1 * njac[k+1][1][0];
  lhs[k][CC][2][0] =  tmp2 * fjac[k+1][2][0]
    - tmp1 * njac[k+1][2][0];
  lhs[k][CC][3][0] =  tmp2 * fjac[k+1][3][0]
    - tmp1 * njac[k+1][3][0];
  lhs[k][CC][4][0] =  tmp2 * fjac[k+1][4][0]
    - tmp1 * njac[k+1][4][0];

  lhs[k][CC][0][1] =  tmp2 * fjac[k+1][0][1]
    - tmp1 * njac[k+1][0][1];
  lhs[k][CC][1][1] =  tmp2 * fjac[k+1][1][1]
    - tmp1 * njac[k+1][1][1]
    - tmp1 * dz2;
  lhs[k][CC][2][1] =  tmp2 * fjac[k+1][2][1]
    - tmp1 * njac[k+1][2][1];
  lhs[k][CC][3][1] =  tmp2 * fjac[k+1][3][1]
    - tmp1 * njac[k+1][3][1];
  lhs[k][CC][4][1] =  tmp2 * fjac[k+1][4][1]
    - tmp1 * njac[k+1][4][1];

  lhs[k][CC][0][2] =  tmp2 * fjac[k+1][0][2]
    - tmp1 * njac[k+1][0][2];
  lhs[k][CC][1][2] =  tmp2 * fjac[k+1][1][2]
    - tmp1 * njac[k+1][1][2];
  lhs[k][CC][2][2] =  tmp2 * fjac[k+1][2][2]
    - tmp1 * njac[k+1][2][2]
    - tmp1 * dz3;
  lhs[k][CC][3][2] =  tmp2 * fjac[k+1][3][2]
    - tmp1 * njac[k+1][3][2];
  lhs[k][CC][4][2] =  tmp2 * fjac[k+1][4][2]
    - tmp1 * njac[k+1][4][2];

  lhs[k][CC][0][3] =  tmp2 * fjac[k+1][0][3]
    - tmp1 * njac[k+1][0][3];
  lhs[k][CC][1][3] =  tmp2 * fjac[k+1][1][3]
    - tmp1 * njac[k+1][1][3];
  lhs[k][CC][2][3] =  tmp2 * fjac[k+1][2][3]
    - tmp1 * njac[k+1][2][3];
  lhs[k][CC][3][3] =  tmp2 * fjac[k+1][3][3]
    - tmp1 * njac[k+1][3][3]
    - tmp1 * dz4;
  lhs[k][CC][4][3] =  tmp2 * fjac[k+1][4][3]
    - tmp1 * njac[k+1][4][3];

  lhs[k][CC][0][4] =  tmp2 * fjac[k+1][0][4]
    - tmp1 * njac[k+1][0][4];
  lhs[k][CC][1][4] =  tmp2 * fjac[k+1][1][4]
    - tmp1 * njac[k+1][1][4];
  lhs[k][CC][2][4] =  tmp2 * fjac[k+1][2][4]
    - tmp1 * njac[k+1][2][4];
  lhs[k][CC][3][4] =  tmp2 * fjac[k+1][3][4]
    - tmp1 * njac[k+1][3][4];
  lhs[k][CC][4][4] =  tmp2 * fjac[k+1][4][4]
    - tmp1 * njac[k+1][4][4]
    - tmp1 * dz5;
}

__kernel void z_solve(__global double *g_rhs,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, ksize;
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (j > (gp1-2) || i > (gp0-2)) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (j-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  ksize = gp2-1;

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[0][j][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  load_matrix(p_lhsBB, lhs[0][BB]);
  load_matrix(p_lhsCC, lhs[0][CC]);
  load_vector(p_rhs, rhs[0][j][i]);
  p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
  save_matrix(lhs[0][BB], p_lhsBB);
  save_matrix(lhs[0][CC], p_lhsCC);
  save_vector(rhs[0][j][i], p_rhs);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (k = 1; k <= ksize-1; k++) {
    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[k][AA]);
    load_matrix(p_lhsBB, lhs[k][BB]);
    load_matrix(p_lhsCC, lhs[k][CC]);
    load_vector(p_rhs, rhs[k][j][i]);

    //-------------------------------------------------------------------
    // subtract A*lhs_vector(k-1) from lhs_vector(k)
    // 
    // rhs(k) = rhs(k) - A*rhs(k-1)
    //-------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //-------------------------------------------------------------------
    // B(k) = B(k) - C(k-1)*A(k)
    // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
    //-------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
    //-------------------------------------------------------------------
    p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

    save_matrix(lhs[k][BB], p_lhsBB);
    save_matrix(lhs[k][CC], p_lhsCC);
    save_vector(rhs[k][j][i], p_rhs);
  }

  copy_matrix(p_lhsm1CC, p_lhsCC);
  copy_vector(p_rhsm1, p_rhs);
  load_matrix(p_lhsAA, lhs[ksize][AA]);
  load_matrix(p_lhsBB, lhs[ksize][BB]);
  load_matrix(p_lhsCC, lhs[ksize][CC]);
  load_vector(p_rhs, rhs[ksize][j][i]);

  //---------------------------------------------------------------------
  // Now finish up special cases for last cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
  //---------------------------------------------------------------------
  p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

  //---------------------------------------------------------------------
  // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
  // matmul_sub(AA,i,j,ksize,c,
  // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
  //---------------------------------------------------------------------
  p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

  //---------------------------------------------------------------------
  // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
  //---------------------------------------------------------------------
  p_binvrhs( p_lhsBB, p_rhs );

  save_matrix(lhs[ksize][BB], p_lhsBB);
  save_matrix(lhs[ksize][CC], p_lhsCC);
  save_vector(rhs[ksize][j][i], p_rhs);

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(ksize)=rhs(ksize)
  // else assume U(ksize) is loaded in un pack backsub_info
  // so just use it
  // after u(kstart) will be sent to next cell
  //---------------------------------------------------------------------

  for (k = ksize-1; k >= 0; k--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      p_rhsm1[m] = rhs[k][j][i][m];
      for (n = 0; n < BLOCK_SIZE; n++) {
        p_rhsm1[m] = p_rhsm1[m]
          - lhs[k][CC][n][m]*p_rhs[n];
      }
      rhs[k][j][i][m] = p_rhsm1[m];
    }
    copy_vector(p_rhs, p_rhsm1);
  }
}


#elif Z_SOLVE_DIM == 2
__kernel void z_solve(__global double *g_qs,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, ksize;
  double tmp1, tmp2, tmp3;

  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (j > (gp1-2) || i > (gp0-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (j-1)*(gp0-2) + (i-1);
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  ksize = gp2-1;

  for (k = 0; k <= ksize; k++) {
    tmp1 = 1.0 / u[k][j][i][0];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[k][0][0] = 0.0;
    fjac[k][1][0] = 0.0;
    fjac[k][2][0] = 0.0;
    fjac[k][3][0] = 1.0;
    fjac[k][4][0] = 0.0;

    fjac[k][0][1] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][1] = u[k][j][i][3] * tmp1;
    fjac[k][2][1] = 0.0;
    fjac[k][3][1] = u[k][j][i][1] * tmp1;
    fjac[k][4][1] = 0.0;

    fjac[k][0][2] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][2] = 0.0;
    fjac[k][2][2] = u[k][j][i][3] * tmp1;
    fjac[k][3][2] = u[k][j][i][2] * tmp1;
    fjac[k][4][2] = 0.0;

    fjac[k][0][3] = - (u[k][j][i][3]*u[k][j][i][3] * tmp2 ) 
      + c2 * qs[k][j][i];
    fjac[k][1][3] = - c2 *  u[k][j][i][1] * tmp1;
    fjac[k][2][3] = - c2 *  u[k][j][i][2] * tmp1;
    fjac[k][3][3] = ( 2.0 - c2 ) *  u[k][j][i][3] * tmp1;
    fjac[k][4][3] = c2;

    fjac[k][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][3] * tmp2;
    fjac[k][1][4] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][2][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][3][4] = c1 * ( u[k][j][i][4] * tmp1 )
      - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * tmp2 );
    fjac[k][4][4] = c1 * u[k][j][i][3] * tmp1;

    njac[k][0][0] = 0.0;
    njac[k][1][0] = 0.0;
    njac[k][2][0] = 0.0;
    njac[k][3][0] = 0.0;
    njac[k][4][0] = 0.0;

    njac[k][0][1] = - c3c4 * tmp2 * u[k][j][i][1];
    njac[k][1][1] =   c3c4 * tmp1;
    njac[k][2][1] =   0.0;
    njac[k][3][1] =   0.0;
    njac[k][4][1] =   0.0;

    njac[k][0][2] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[k][1][2] =   0.0;
    njac[k][2][2] =   c3c4 * tmp1;
    njac[k][3][2] =   0.0;
    njac[k][4][2] =   0.0;

    njac[k][0][3] = - con43 * c3c4 * tmp2 * u[k][j][i][3];
    njac[k][1][3] =   0.0;
    njac[k][2][3] =   0.0;
    njac[k][3][3] =   con43 * c3 * c4 * tmp1;
    njac[k][4][3] =   0.0;

    njac[k][0][4] = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[k][1][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac[k][2][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[k][3][4] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][3];
    njac[k][4][4] = ( c1345 )* tmp1;
  }

  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in z direction
  //---------------------------------------------------------------------
  lhsinit(lhs, ksize);
  for (k = 1; k <= ksize-1; k++) {
    tmp1 = dt * tz1;
    tmp2 = dt * tz2;

    lhs[k][AA][0][0] = - tmp2 * fjac[k-1][0][0]
      - tmp1 * njac[k-1][0][0]
      - tmp1 * dz1; 
    lhs[k][AA][1][0] = - tmp2 * fjac[k-1][1][0]
      - tmp1 * njac[k-1][1][0];
    lhs[k][AA][2][0] = - tmp2 * fjac[k-1][2][0]
      - tmp1 * njac[k-1][2][0];
    lhs[k][AA][3][0] = - tmp2 * fjac[k-1][3][0]
      - tmp1 * njac[k-1][3][0];
    lhs[k][AA][4][0] = - tmp2 * fjac[k-1][4][0]
      - tmp1 * njac[k-1][4][0];

    lhs[k][AA][0][1] = - tmp2 * fjac[k-1][0][1]
      - tmp1 * njac[k-1][0][1];
    lhs[k][AA][1][1] = - tmp2 * fjac[k-1][1][1]
      - tmp1 * njac[k-1][1][1]
      - tmp1 * dz2;
    lhs[k][AA][2][1] = - tmp2 * fjac[k-1][2][1]
      - tmp1 * njac[k-1][2][1];
    lhs[k][AA][3][1] = - tmp2 * fjac[k-1][3][1]
      - tmp1 * njac[k-1][3][1];
    lhs[k][AA][4][1] = - tmp2 * fjac[k-1][4][1]
      - tmp1 * njac[k-1][4][1];

    lhs[k][AA][0][2] = - tmp2 * fjac[k-1][0][2]
      - tmp1 * njac[k-1][0][2];
    lhs[k][AA][1][2] = - tmp2 * fjac[k-1][1][2]
      - tmp1 * njac[k-1][1][2];
    lhs[k][AA][2][2] = - tmp2 * fjac[k-1][2][2]
      - tmp1 * njac[k-1][2][2]
      - tmp1 * dz3;
    lhs[k][AA][3][2] = - tmp2 * fjac[k-1][3][2]
      - tmp1 * njac[k-1][3][2];
    lhs[k][AA][4][2] = - tmp2 * fjac[k-1][4][2]
      - tmp1 * njac[k-1][4][2];

    lhs[k][AA][0][3] = - tmp2 * fjac[k-1][0][3]
      - tmp1 * njac[k-1][0][3];
    lhs[k][AA][1][3] = - tmp2 * fjac[k-1][1][3]
      - tmp1 * njac[k-1][1][3];
    lhs[k][AA][2][3] = - tmp2 * fjac[k-1][2][3]
      - tmp1 * njac[k-1][2][3];
    lhs[k][AA][3][3] = - tmp2 * fjac[k-1][3][3]
      - tmp1 * njac[k-1][3][3]
      - tmp1 * dz4;
    lhs[k][AA][4][3] = - tmp2 * fjac[k-1][4][3]
      - tmp1 * njac[k-1][4][3];

    lhs[k][AA][0][4] = - tmp2 * fjac[k-1][0][4]
      - tmp1 * njac[k-1][0][4];
    lhs[k][AA][1][4] = - tmp2 * fjac[k-1][1][4]
      - tmp1 * njac[k-1][1][4];
    lhs[k][AA][2][4] = - tmp2 * fjac[k-1][2][4]
      - tmp1 * njac[k-1][2][4];
    lhs[k][AA][3][4] = - tmp2 * fjac[k-1][3][4]
      - tmp1 * njac[k-1][3][4];
    lhs[k][AA][4][4] = - tmp2 * fjac[k-1][4][4]
      - tmp1 * njac[k-1][4][4]
      - tmp1 * dz5;

    lhs[k][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[k][0][0]
      + tmp1 * 2.0 * dz1;
    lhs[k][BB][1][0] = tmp1 * 2.0 * njac[k][1][0];
    lhs[k][BB][2][0] = tmp1 * 2.0 * njac[k][2][0];
    lhs[k][BB][3][0] = tmp1 * 2.0 * njac[k][3][0];
    lhs[k][BB][4][0] = tmp1 * 2.0 * njac[k][4][0];

    lhs[k][BB][0][1] = tmp1 * 2.0 * njac[k][0][1];
    lhs[k][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[k][1][1]
      + tmp1 * 2.0 * dz2;
    lhs[k][BB][2][1] = tmp1 * 2.0 * njac[k][2][1];
    lhs[k][BB][3][1] = tmp1 * 2.0 * njac[k][3][1];
    lhs[k][BB][4][1] = tmp1 * 2.0 * njac[k][4][1];

    lhs[k][BB][0][2] = tmp1 * 2.0 * njac[k][0][2];
    lhs[k][BB][1][2] = tmp1 * 2.0 * njac[k][1][2];
    lhs[k][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[k][2][2]
      + tmp1 * 2.0 * dz3;
    lhs[k][BB][3][2] = tmp1 * 2.0 * njac[k][3][2];
    lhs[k][BB][4][2] = tmp1 * 2.0 * njac[k][4][2];

    lhs[k][BB][0][3] = tmp1 * 2.0 * njac[k][0][3];
    lhs[k][BB][1][3] = tmp1 * 2.0 * njac[k][1][3];
    lhs[k][BB][2][3] = tmp1 * 2.0 * njac[k][2][3];
    lhs[k][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[k][3][3]
      + tmp1 * 2.0 * dz4;
    lhs[k][BB][4][3] = tmp1 * 2.0 * njac[k][4][3];

    lhs[k][BB][0][4] = tmp1 * 2.0 * njac[k][0][4];
    lhs[k][BB][1][4] = tmp1 * 2.0 * njac[k][1][4];
    lhs[k][BB][2][4] = tmp1 * 2.0 * njac[k][2][4];
    lhs[k][BB][3][4] = tmp1 * 2.0 * njac[k][3][4];
    lhs[k][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[k][4][4] 
      + tmp1 * 2.0 * dz5;

    lhs[k][CC][0][0] =  tmp2 * fjac[k+1][0][0]
      - tmp1 * njac[k+1][0][0]
      - tmp1 * dz1;
    lhs[k][CC][1][0] =  tmp2 * fjac[k+1][1][0]
      - tmp1 * njac[k+1][1][0];
    lhs[k][CC][2][0] =  tmp2 * fjac[k+1][2][0]
      - tmp1 * njac[k+1][2][0];
    lhs[k][CC][3][0] =  tmp2 * fjac[k+1][3][0]
      - tmp1 * njac[k+1][3][0];
    lhs[k][CC][4][0] =  tmp2 * fjac[k+1][4][0]
      - tmp1 * njac[k+1][4][0];

    lhs[k][CC][0][1] =  tmp2 * fjac[k+1][0][1]
      - tmp1 * njac[k+1][0][1];
    lhs[k][CC][1][1] =  tmp2 * fjac[k+1][1][1]
      - tmp1 * njac[k+1][1][1]
      - tmp1 * dz2;
    lhs[k][CC][2][1] =  tmp2 * fjac[k+1][2][1]
      - tmp1 * njac[k+1][2][1];
    lhs[k][CC][3][1] =  tmp2 * fjac[k+1][3][1]
      - tmp1 * njac[k+1][3][1];
    lhs[k][CC][4][1] =  tmp2 * fjac[k+1][4][1]
      - tmp1 * njac[k+1][4][1];

    lhs[k][CC][0][2] =  tmp2 * fjac[k+1][0][2]
      - tmp1 * njac[k+1][0][2];
    lhs[k][CC][1][2] =  tmp2 * fjac[k+1][1][2]
      - tmp1 * njac[k+1][1][2];
    lhs[k][CC][2][2] =  tmp2 * fjac[k+1][2][2]
      - tmp1 * njac[k+1][2][2]
      - tmp1 * dz3;
    lhs[k][CC][3][2] =  tmp2 * fjac[k+1][3][2]
      - tmp1 * njac[k+1][3][2];
    lhs[k][CC][4][2] =  tmp2 * fjac[k+1][4][2]
      - tmp1 * njac[k+1][4][2];

    lhs[k][CC][0][3] =  tmp2 * fjac[k+1][0][3]
      - tmp1 * njac[k+1][0][3];
    lhs[k][CC][1][3] =  tmp2 * fjac[k+1][1][3]
      - tmp1 * njac[k+1][1][3];
    lhs[k][CC][2][3] =  tmp2 * fjac[k+1][2][3]
      - tmp1 * njac[k+1][2][3];
    lhs[k][CC][3][3] =  tmp2 * fjac[k+1][3][3]
      - tmp1 * njac[k+1][3][3]
      - tmp1 * dz4;
    lhs[k][CC][4][3] =  tmp2 * fjac[k+1][4][3]
      - tmp1 * njac[k+1][4][3];

    lhs[k][CC][0][4] =  tmp2 * fjac[k+1][0][4]
      - tmp1 * njac[k+1][0][4];
    lhs[k][CC][1][4] =  tmp2 * fjac[k+1][1][4]
      - tmp1 * njac[k+1][1][4];
    lhs[k][CC][2][4] =  tmp2 * fjac[k+1][2][4]
      - tmp1 * njac[k+1][2][4];
    lhs[k][CC][3][4] =  tmp2 * fjac[k+1][3][4]
      - tmp1 * njac[k+1][3][4];
    lhs[k][CC][4][4] =  tmp2 * fjac[k+1][4][4]
      - tmp1 * njac[k+1][4][4]
      - tmp1 * dz5;
  }

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[0][j][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[0][j][i] );

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (k = 1; k <= ksize-1; k++) {
    //-------------------------------------------------------------------
    // subtract A*lhs_vector(k-1) from lhs_vector(k)
    // 
    // rhs(k) = rhs(k) - A*rhs(k-1)
    //-------------------------------------------------------------------
    matvec_sub(lhs[k][AA], rhs[k-1][j][i], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(k) = B(k) - C(k-1)*A(k)
    // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
    //-------------------------------------------------------------------
    matmul_sub(lhs[k][AA], lhs[k-1][CC], lhs[k][BB]);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[k][BB], lhs[k][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // Now finish up special cases for last cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[ksize][AA], rhs[ksize-1][j][i], rhs[ksize][j][i]);

  //---------------------------------------------------------------------
  // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
  // matmul_sub(AA,i,j,ksize,c,
  // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
  //---------------------------------------------------------------------
  matmul_sub(lhs[ksize][AA], lhs[ksize-1][CC], lhs[ksize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[ksize][BB], rhs[ksize][j][i] );

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(ksize)=rhs(ksize)
  // else assume U(ksize) is loaded in un pack backsub_info
  // so just use it
  // after u(kstart) will be sent to next cell
  //---------------------------------------------------------------------

  for (k = ksize-1; k >= 0; k--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[k][CC][n][m]*rhs[k+1][j][i][n];
      }
    }
  }
}

#else //Z_SOLVE_DIM == 1
__kernel void z_solve(__global double *g_qs,
                      __global double *g_square,
                      __global double *g_u,
                      __global double *g_rhs,
                      __global double *g_fjac,
                      __global double *g_njac,
                      __global double *g_lhs,
                      int gp0,
                      int gp1,
                      int gp2)
{
  int i, j, k, m, n, ksize;
  double tmp1, tmp2, tmp3;
  double p_u[5];
  double p_lhsAA[5][5], p_lhsBB[5][5], p_lhsCC[5][5], p_rhs[5];
  double p_lhsm1CC[5][5], p_rhsm1[5];

  j = get_global_id(0) + 1;
  if (j > (gp1-2)) return;

  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = j - 1;
  int my_offset = my_id * (PROBLEM_SIZE+1) * 5 * 5;
  __global double (*fjac)[5][5] = 
    (__global double (*)[5][5])&g_fjac[my_offset];
  __global double (*njac)[5][5] = 
    (__global double (*)[5][5])&g_njac[my_offset];

  my_offset = my_id * (PROBLEM_SIZE+1) * 3 * 5 * 5;
  __global double (*lhs)[3][5][5] =
    (__global double (*)[3][5][5])&g_lhs[my_offset];

  ksize = gp2-1;

  for (i = 1; i <= gp0-2; i++) {
    for (k = 0; k <= ksize; k++) {
      tmp1 = 1.0 / u[k][j][i][0];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;
      p_u[0] = u[k][j][i][0];
      p_u[1] = u[k][j][i][1];
      p_u[2] = u[k][j][i][2];
      p_u[3] = u[k][j][i][3];
      p_u[4] = u[k][j][i][4];

      fjac[k][0][0] = 0.0;
      fjac[k][1][0] = 0.0;
      fjac[k][2][0] = 0.0;
      fjac[k][3][0] = 1.0;
      fjac[k][4][0] = 0.0;

      fjac[k][0][1] = - ( p_u[1]*p_u[3] ) * tmp2;
      fjac[k][1][1] = p_u[3] * tmp1;
      fjac[k][2][1] = 0.0;
      fjac[k][3][1] = p_u[1] * tmp1;
      fjac[k][4][1] = 0.0;

      fjac[k][0][2] = - ( p_u[2]*p_u[3] ) * tmp2;
      fjac[k][1][2] = 0.0;
      fjac[k][2][2] = p_u[3] * tmp1;
      fjac[k][3][2] = p_u[2] * tmp1;
      fjac[k][4][2] = 0.0;

      fjac[k][0][3] = - (p_u[3]*p_u[3] * tmp2 ) 
        + c2 * qs[k][j][i];
      fjac[k][1][3] = - c2 *  p_u[1] * tmp1;
      fjac[k][2][3] = - c2 *  p_u[2] * tmp1;
      fjac[k][3][3] = ( 2.0 - c2 ) *  p_u[3] * tmp1;
      fjac[k][4][3] = c2;

      fjac[k][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * p_u[4] )
        * p_u[3] * tmp2;
      fjac[k][1][4] = - c2 * ( p_u[1]*p_u[3] ) * tmp2;
      fjac[k][2][4] = - c2 * ( p_u[2]*p_u[3] ) * tmp2;
      fjac[k][3][4] = c1 * ( p_u[4] * tmp1 )
        - c2 * ( qs[k][j][i] + p_u[3]*p_u[3] * tmp2 );
      fjac[k][4][4] = c1 * p_u[3] * tmp1;

      njac[k][0][0] = 0.0;
      njac[k][1][0] = 0.0;
      njac[k][2][0] = 0.0;
      njac[k][3][0] = 0.0;
      njac[k][4][0] = 0.0;

      njac[k][0][1] = - c3c4 * tmp2 * p_u[1];
      njac[k][1][1] =   c3c4 * tmp1;
      njac[k][2][1] =   0.0;
      njac[k][3][1] =   0.0;
      njac[k][4][1] =   0.0;

      njac[k][0][2] = - c3c4 * tmp2 * p_u[2];
      njac[k][1][2] =   0.0;
      njac[k][2][2] =   c3c4 * tmp1;
      njac[k][3][2] =   0.0;
      njac[k][4][2] =   0.0;

      njac[k][0][3] = - con43 * c3c4 * tmp2 * p_u[3];
      njac[k][1][3] =   0.0;
      njac[k][2][3] =   0.0;
      njac[k][3][3] =   con43 * c3 * c4 * tmp1;
      njac[k][4][3] =   0.0;

      njac[k][0][4] = - (  c3c4
          - c1345 ) * tmp3 * (p_u[1]*p_u[1])
        - ( c3c4 - c1345 ) * tmp3 * (p_u[2]*p_u[2])
        - ( con43 * c3c4
            - c1345 ) * tmp3 * (p_u[3]*p_u[3])
        - c1345 * tmp2 * p_u[4];

      njac[k][1][4] = (  c3c4 - c1345 ) * tmp2 * p_u[1];
      njac[k][2][4] = (  c3c4 - c1345 ) * tmp2 * p_u[2];
      njac[k][3][4] = ( con43 * c3c4
          - c1345 ) * tmp2 * p_u[3];
      njac[k][4][4] = ( c1345 )* tmp1;
    }

    //---------------------------------------------------------------------
    // now jacobians set, so form left hand side in z direction
    //---------------------------------------------------------------------
    lhsinit(lhs, ksize);
    for (k = 1; k <= ksize-1; k++) {
      tmp1 = dt * tz1;
      tmp2 = dt * tz2;

      lhs[k][AA][0][0] = - tmp2 * fjac[k-1][0][0]
        - tmp1 * njac[k-1][0][0]
        - tmp1 * dz1; 
      lhs[k][AA][1][0] = - tmp2 * fjac[k-1][1][0]
        - tmp1 * njac[k-1][1][0];
      lhs[k][AA][2][0] = - tmp2 * fjac[k-1][2][0]
        - tmp1 * njac[k-1][2][0];
      lhs[k][AA][3][0] = - tmp2 * fjac[k-1][3][0]
        - tmp1 * njac[k-1][3][0];
      lhs[k][AA][4][0] = - tmp2 * fjac[k-1][4][0]
        - tmp1 * njac[k-1][4][0];

      lhs[k][AA][0][1] = - tmp2 * fjac[k-1][0][1]
        - tmp1 * njac[k-1][0][1];
      lhs[k][AA][1][1] = - tmp2 * fjac[k-1][1][1]
        - tmp1 * njac[k-1][1][1]
        - tmp1 * dz2;
      lhs[k][AA][2][1] = - tmp2 * fjac[k-1][2][1]
        - tmp1 * njac[k-1][2][1];
      lhs[k][AA][3][1] = - tmp2 * fjac[k-1][3][1]
        - tmp1 * njac[k-1][3][1];
      lhs[k][AA][4][1] = - tmp2 * fjac[k-1][4][1]
        - tmp1 * njac[k-1][4][1];

      lhs[k][AA][0][2] = - tmp2 * fjac[k-1][0][2]
        - tmp1 * njac[k-1][0][2];
      lhs[k][AA][1][2] = - tmp2 * fjac[k-1][1][2]
        - tmp1 * njac[k-1][1][2];
      lhs[k][AA][2][2] = - tmp2 * fjac[k-1][2][2]
        - tmp1 * njac[k-1][2][2]
        - tmp1 * dz3;
      lhs[k][AA][3][2] = - tmp2 * fjac[k-1][3][2]
        - tmp1 * njac[k-1][3][2];
      lhs[k][AA][4][2] = - tmp2 * fjac[k-1][4][2]
        - tmp1 * njac[k-1][4][2];

      lhs[k][AA][0][3] = - tmp2 * fjac[k-1][0][3]
        - tmp1 * njac[k-1][0][3];
      lhs[k][AA][1][3] = - tmp2 * fjac[k-1][1][3]
        - tmp1 * njac[k-1][1][3];
      lhs[k][AA][2][3] = - tmp2 * fjac[k-1][2][3]
        - tmp1 * njac[k-1][2][3];
      lhs[k][AA][3][3] = - tmp2 * fjac[k-1][3][3]
        - tmp1 * njac[k-1][3][3]
        - tmp1 * dz4;
      lhs[k][AA][4][3] = - tmp2 * fjac[k-1][4][3]
        - tmp1 * njac[k-1][4][3];

      lhs[k][AA][0][4] = - tmp2 * fjac[k-1][0][4]
        - tmp1 * njac[k-1][0][4];
      lhs[k][AA][1][4] = - tmp2 * fjac[k-1][1][4]
        - tmp1 * njac[k-1][1][4];
      lhs[k][AA][2][4] = - tmp2 * fjac[k-1][2][4]
        - tmp1 * njac[k-1][2][4];
      lhs[k][AA][3][4] = - tmp2 * fjac[k-1][3][4]
        - tmp1 * njac[k-1][3][4];
      lhs[k][AA][4][4] = - tmp2 * fjac[k-1][4][4]
        - tmp1 * njac[k-1][4][4]
        - tmp1 * dz5;

      lhs[k][BB][0][0] = 1.0
        + tmp1 * 2.0 * njac[k][0][0]
        + tmp1 * 2.0 * dz1;
      lhs[k][BB][1][0] = tmp1 * 2.0 * njac[k][1][0];
      lhs[k][BB][2][0] = tmp1 * 2.0 * njac[k][2][0];
      lhs[k][BB][3][0] = tmp1 * 2.0 * njac[k][3][0];
      lhs[k][BB][4][0] = tmp1 * 2.0 * njac[k][4][0];

      lhs[k][BB][0][1] = tmp1 * 2.0 * njac[k][0][1];
      lhs[k][BB][1][1] = 1.0
        + tmp1 * 2.0 * njac[k][1][1]
        + tmp1 * 2.0 * dz2;
      lhs[k][BB][2][1] = tmp1 * 2.0 * njac[k][2][1];
      lhs[k][BB][3][1] = tmp1 * 2.0 * njac[k][3][1];
      lhs[k][BB][4][1] = tmp1 * 2.0 * njac[k][4][1];

      lhs[k][BB][0][2] = tmp1 * 2.0 * njac[k][0][2];
      lhs[k][BB][1][2] = tmp1 * 2.0 * njac[k][1][2];
      lhs[k][BB][2][2] = 1.0
        + tmp1 * 2.0 * njac[k][2][2]
        + tmp1 * 2.0 * dz3;
      lhs[k][BB][3][2] = tmp1 * 2.0 * njac[k][3][2];
      lhs[k][BB][4][2] = tmp1 * 2.0 * njac[k][4][2];

      lhs[k][BB][0][3] = tmp1 * 2.0 * njac[k][0][3];
      lhs[k][BB][1][3] = tmp1 * 2.0 * njac[k][1][3];
      lhs[k][BB][2][3] = tmp1 * 2.0 * njac[k][2][3];
      lhs[k][BB][3][3] = 1.0
        + tmp1 * 2.0 * njac[k][3][3]
        + tmp1 * 2.0 * dz4;
      lhs[k][BB][4][3] = tmp1 * 2.0 * njac[k][4][3];

      lhs[k][BB][0][4] = tmp1 * 2.0 * njac[k][0][4];
      lhs[k][BB][1][4] = tmp1 * 2.0 * njac[k][1][4];
      lhs[k][BB][2][4] = tmp1 * 2.0 * njac[k][2][4];
      lhs[k][BB][3][4] = tmp1 * 2.0 * njac[k][3][4];
      lhs[k][BB][4][4] = 1.0
        + tmp1 * 2.0 * njac[k][4][4] 
        + tmp1 * 2.0 * dz5;

      lhs[k][CC][0][0] =  tmp2 * fjac[k+1][0][0]
        - tmp1 * njac[k+1][0][0]
        - tmp1 * dz1;
      lhs[k][CC][1][0] =  tmp2 * fjac[k+1][1][0]
        - tmp1 * njac[k+1][1][0];
      lhs[k][CC][2][0] =  tmp2 * fjac[k+1][2][0]
        - tmp1 * njac[k+1][2][0];
      lhs[k][CC][3][0] =  tmp2 * fjac[k+1][3][0]
        - tmp1 * njac[k+1][3][0];
      lhs[k][CC][4][0] =  tmp2 * fjac[k+1][4][0]
        - tmp1 * njac[k+1][4][0];

      lhs[k][CC][0][1] =  tmp2 * fjac[k+1][0][1]
        - tmp1 * njac[k+1][0][1];
      lhs[k][CC][1][1] =  tmp2 * fjac[k+1][1][1]
        - tmp1 * njac[k+1][1][1]
        - tmp1 * dz2;
      lhs[k][CC][2][1] =  tmp2 * fjac[k+1][2][1]
        - tmp1 * njac[k+1][2][1];
      lhs[k][CC][3][1] =  tmp2 * fjac[k+1][3][1]
        - tmp1 * njac[k+1][3][1];
      lhs[k][CC][4][1] =  tmp2 * fjac[k+1][4][1]
        - tmp1 * njac[k+1][4][1];

      lhs[k][CC][0][2] =  tmp2 * fjac[k+1][0][2]
        - tmp1 * njac[k+1][0][2];
      lhs[k][CC][1][2] =  tmp2 * fjac[k+1][1][2]
        - tmp1 * njac[k+1][1][2];
      lhs[k][CC][2][2] =  tmp2 * fjac[k+1][2][2]
        - tmp1 * njac[k+1][2][2]
        - tmp1 * dz3;
      lhs[k][CC][3][2] =  tmp2 * fjac[k+1][3][2]
        - tmp1 * njac[k+1][3][2];
      lhs[k][CC][4][2] =  tmp2 * fjac[k+1][4][2]
        - tmp1 * njac[k+1][4][2];

      lhs[k][CC][0][3] =  tmp2 * fjac[k+1][0][3]
        - tmp1 * njac[k+1][0][3];
      lhs[k][CC][1][3] =  tmp2 * fjac[k+1][1][3]
        - tmp1 * njac[k+1][1][3];
      lhs[k][CC][2][3] =  tmp2 * fjac[k+1][2][3]
        - tmp1 * njac[k+1][2][3];
      lhs[k][CC][3][3] =  tmp2 * fjac[k+1][3][3]
        - tmp1 * njac[k+1][3][3]
        - tmp1 * dz4;
      lhs[k][CC][4][3] =  tmp2 * fjac[k+1][4][3]
        - tmp1 * njac[k+1][4][3];

      lhs[k][CC][0][4] =  tmp2 * fjac[k+1][0][4]
        - tmp1 * njac[k+1][0][4];
      lhs[k][CC][1][4] =  tmp2 * fjac[k+1][1][4]
        - tmp1 * njac[k+1][1][4];
      lhs[k][CC][2][4] =  tmp2 * fjac[k+1][2][4]
        - tmp1 * njac[k+1][2][4];
      lhs[k][CC][3][4] =  tmp2 * fjac[k+1][3][4]
        - tmp1 * njac[k+1][3][4];
      lhs[k][CC][4][4] =  tmp2 * fjac[k+1][4][4]
        - tmp1 * njac[k+1][4][4]
        - tmp1 * dz5;
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // performs guaussian elimination on this cell.
    // 
    // assumes that unpacking routines for non-first cells 
    // preload C' and rhs' from previous cell.
    // 
    // assumed send happens outside this routine, but that
    // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // outer most do loops - sweeping in i direction
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[0][j][i] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    load_matrix(p_lhsBB, lhs[0][BB]);
    load_matrix(p_lhsCC, lhs[0][CC]);
    load_vector(p_rhs, rhs[0][j][i]);
    p_binvcrhs( p_lhsBB, p_lhsCC, p_rhs );
    save_matrix(lhs[0][BB], p_lhsBB);
    save_matrix(lhs[0][CC], p_lhsCC);
    save_vector(rhs[0][j][i], p_rhs);

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last 
    //---------------------------------------------------------------------
    for (k = 1; k <= ksize-1; k++) {
      copy_matrix(p_lhsm1CC, p_lhsCC);
      copy_vector(p_rhsm1, p_rhs);
      load_matrix(p_lhsAA, lhs[k][AA]);
      load_matrix(p_lhsBB, lhs[k][BB]);
      load_matrix(p_lhsCC, lhs[k][CC]);
      load_vector(p_rhs, rhs[k][j][i]);

      //-------------------------------------------------------------------
      // subtract A*lhs_vector(k-1) from lhs_vector(k)
      // 
      // rhs(k) = rhs(k) - A*rhs(k-1)
      //-------------------------------------------------------------------
      p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

      //-------------------------------------------------------------------
      // B(k) = B(k) - C(k-1)*A(k)
      // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
      //-------------------------------------------------------------------
      p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

      //-------------------------------------------------------------------
      // multiply c[k][j][i] by b_inverse and copy back to c
      // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
      //-------------------------------------------------------------------
      p_binvcrhs(p_lhsBB, p_lhsCC, p_rhs);

      save_matrix(lhs[k][BB], p_lhsBB);
      save_matrix(lhs[k][CC], p_lhsCC);
      save_vector(rhs[k][j][i], p_rhs);
    }

    copy_matrix(p_lhsm1CC, p_lhsCC);
    copy_vector(p_rhsm1, p_rhs);
    load_matrix(p_lhsAA, lhs[ksize][AA]);
    load_matrix(p_lhsBB, lhs[ksize][BB]);
    load_matrix(p_lhsCC, lhs[ksize][CC]);
    load_vector(p_rhs, rhs[ksize][j][i]);

    //---------------------------------------------------------------------
    // Now finish up special cases for last cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
    //---------------------------------------------------------------------
    p_matvec_sub(p_lhsAA, p_rhsm1, p_rhs);

    //---------------------------------------------------------------------
    // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
    // matmul_sub(AA,i,j,ksize,c,
    // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
    //---------------------------------------------------------------------
    p_matmul_sub(p_lhsAA, p_lhsm1CC, p_lhsBB);

    //---------------------------------------------------------------------
    // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
    //---------------------------------------------------------------------
    p_binvrhs( p_lhsBB, p_rhs );

    save_matrix(lhs[ksize][BB], p_lhsBB);
    save_matrix(lhs[ksize][CC], p_lhsCC);
    save_vector(rhs[ksize][j][i], p_rhs);

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(ksize)=rhs(ksize)
    // else assume U(ksize) is loaded in un pack backsub_info
    // so just use it
    // after u(kstart) will be sent to next cell
    //---------------------------------------------------------------------

    for (k = ksize-1; k >= 0; k--) {
      for (m = 0; m < BLOCK_SIZE; m++) {
        p_rhsm1[m] = rhs[k][j][i][m];
        for (n = 0; n < BLOCK_SIZE; n++) {
          p_rhsm1[m] = p_rhsm1[m]
            - lhs[k][CC][n][m]*p_rhs[n];
        }
        rhs[k][j][i][m] = p_rhsm1[m];
      }
      copy_vector(p_rhs, p_rhsm1);
    }
  }
}
#endif


__kernel void add(__global double *g_u,
                  __global double *g_rhs,
                  int gp0,
                  int gp1,
                  int gp2)
{
  int i, j, k, m;

#if ADD_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2) || i > (gp0-2)) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (m = 0; m < 5; m++) {
    u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
  }

#elif ADD_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > (gp2-2) || j > (gp1-2)) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= (gp0-2); i++) {
    for (m = 0; m < 5; m++) {
      u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
    }
  }

#else //ADD_DIM == 1
  k = get_global_id(0) + 1;
  if (k > (gp2-2)) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= (gp1-2); j++) {
    for (i = 1; i <= (gp0-2); i++) {
      for (m = 0; m < 5; m++) {
        u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
      }
    }
  }

#endif
}
