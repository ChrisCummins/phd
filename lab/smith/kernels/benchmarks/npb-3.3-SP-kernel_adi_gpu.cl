//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB SP code. This OpenCL    //
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

#include "sp.h"

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
                           __global double *g_speed,
                           __global double *g_square,
                           int gp0,
                           int gp1,
                           int gp2)
{
  int i, j, k;
  double aux, rho_inv;
  double p_u[5];
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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  p_u[0] = u[k][j][i][0];
  p_u[1] = u[k][j][i][1];
  p_u[2] = u[k][j][i][2];
  p_u[3] = u[k][j][i][3];
  p_u[4] = u[k][j][i][4];

  rho_inv = 1.0/p_u[0];
  rho_i[k][j][i] = rho_inv;
  us[k][j][i] = p_u[1] * rho_inv;
  vs[k][j][i] = p_u[2] * rho_inv;
  ws[k][j][i] = p_u[3] * rho_inv;
  p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
  square[k][j][i] = p_square;
  qs[k][j][i] = p_square * rho_inv;
  //-------------------------------------------------------------------
  // (don't need speed and ainx until the lhs computation)
  //-------------------------------------------------------------------
  aux = c1c2*rho_inv* (p_u[4] - p_square);
  speed[k][j][i] = sqrt(aux);

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  for (i = 0; i < gp0; i++) {
    p_u[0] = u[k][j][i][0];
    p_u[1] = u[k][j][i][1];
    p_u[2] = u[k][j][i][2];
    p_u[3] = u[k][j][i][3];
    p_u[4] = u[k][j][i][4];

    rho_inv = 1.0/p_u[0];
    rho_i[k][j][i] = rho_inv;
    us[k][j][i] = p_u[1] * rho_inv;
    vs[k][j][i] = p_u[2] * rho_inv;
    ws[k][j][i] = p_u[3] * rho_inv;
    p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
    square[k][j][i] = p_square;
    qs[k][j][i] = p_square * rho_inv;
    //-------------------------------------------------------------------
    // (don't need speed and ainx until the lhs computation)
    //-------------------------------------------------------------------
    aux = c1c2*rho_inv* (p_u[4] - p_square);
    speed[k][j][i] = sqrt(aux);
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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;

  for (j = 0; j < gp1; j++) {
    for (i = 0; i < gp0; i++) {
      p_u[0] = u[k][j][i][0];
      p_u[1] = u[k][j][i][1];
      p_u[2] = u[k][j][i][2];
      p_u[3] = u[k][j][i][3];
      p_u[4] = u[k][j][i][4];

      rho_inv = 1.0/p_u[0];
      rho_i[k][j][i] = rho_inv;
      us[k][j][i] = p_u[1] * rho_inv;
      vs[k][j][i] = p_u[2] * rho_inv;
      ws[k][j][i] = p_u[3] * rho_inv;
      p_square = 0.5* (p_u[1]*p_u[1] + p_u[2]*p_u[2] + p_u[3]*p_u[3]) * rho_inv;
      square[k][j][i] = p_square;
      qs[k][j][i] = p_square * rho_inv;
      //-------------------------------------------------------------------
      // (don't need speed and ainx until the lhs computation)
      //-------------------------------------------------------------------
      aux = c1c2*rho_inv* (p_u[4] - p_square);
      speed[k][j][i] = sqrt(aux);
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
                           int nx2,
                           int ny2,
                           int nz2)
{
  int i, j, k, m;

#if COMPUTE_RHS2_DIM == 3
  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);
  if (k > (nz2+1) || j > (ny2+1) || i > (nx2+1)) return;

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
  if (k > (nz2+1) || j > (ny2+1)) return;

  __global double (*forcing)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_forcing;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 0; i <= nx2+1; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = forcing[k][j][i][m];
    }
  }

#else //COMPUTE_RHS2_DIM == 1
  k = get_global_id(0);
  if (k > (nz2+1)) return;

  __global double (*forcing)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_forcing;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 0; j <= ny2+1; j++) {
    for (i = 0; i <= nx2+1; i++) {
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
                           int nx2,
                           int ny2,
                           int nz2)
{
  int i, j, k, m;
  double p_rhs[5];
  double p_u[5], p_up1[5], p_up2[5], p_um1[5], p_um2[5];
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

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

  //---------------------------------------------------------------------
  // add fourth order xi-direction dissipation               
  //---------------------------------------------------------------------
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

  for (i = 3; i <= nx2-2; i++) {
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = p_rhs[m] - dssp * 
        ( p_um2[m] - 4.0*p_um1[m] + 
        6.0*p_u[m] - 4.0*p_up1[m] + 
          p_up2[m] );
    }
  }

  i = nx2-1;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 
      6.0*p_u[m] - 4.0*p_up1[m] );
  }

  i = nx2;
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
                           int nx2,
                           int ny2,
                           int nz2)
{
  int i, j, k, m;
  double p_rhs[5];
  double p_u[5], p_up1[5], p_up2[5], p_um1[5], p_um2[5];
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

#if COMPUTE_RHS4_DIM == 2
  k = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || i > nx2) return;

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

  for (j = 3; j <= ny2-2; j++) {
    LOOP_PROLOGUE_FULL
    LOOP_BODY
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = p_rhs[m] - dssp * 
        ( p_um2[m] - 4.0*p_um1[m] + 
        6.0*p_u[m] - 4.0*p_up1[m] + 
          p_up2[m] );
    }
  }

  j = ny2-1;
  LOOP_PROLOGUE_FULL
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 
      6.0*p_u[m] - 4.0*p_up1[m] );
  }

  j = ny2;
  LOOP_PROLOGUE_HALF
  LOOP_BODY
  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = p_rhs[m] - dssp *
      ( p_um2[m] - 4.0*p_um1[m] + 5.0*p_u[m] );
  }

#undef LOOP_PROLOGUE_FULL
#undef LOOP_PROLOGUE_HALF
#undef LOOP_BODY

#else //COMPUTE_RHS4_DIM == 1

  k = get_global_id(0) + 1;
  if (k > nz2) return;

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
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      p_vs = vs[k][j][i];
      p_vsp1  = vs[k][j+1][i];
      p_vsm1  = vs[k][j-1][i];

      rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 * 
        (u[k][j+1][i][0] - 2.0*u[k][j][i][0] + u[k][j-1][i][0]) -
        ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);

      rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 * 
        (u[k][j+1][i][1] - 2.0*u[k][j][i][1] + u[k][j-1][i][1]) +
        yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
        ty2 * (u[k][j+1][i][1]*p_vsp1 - u[k][j-1][i][1]*p_vsm1);

      rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 * 
        (u[k][j+1][i][2] - 2.0*u[k][j][i][2] + u[k][j-1][i][2]) +
        yycon2*con43 * (p_vsp1 - 2.0*p_vs + p_vsm1) -
        ty2 * (u[k][j+1][i][2]*p_vsp1 - u[k][j-1][i][2]*p_vsm1 +
              (u[k][j+1][i][4] - square[k][j+1][i] - 
               u[k][j-1][i][4] + square[k][j-1][i]) * c2);

      rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 * 
        (u[k][j+1][i][3] - 2.0*u[k][j][i][3] + u[k][j-1][i][3]) +
        yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
        ty2 * (u[k][j+1][i][3]*p_vsp1 - u[k][j-1][i][3]*p_vsm1);

      rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 * 
        (u[k][j+1][i][4] - 2.0*u[k][j][i][4] + u[k][j-1][i][4]) +
        yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
        yycon4 * (p_vsp1*p_vsp1       - 2.0*p_vs*p_vs + p_vsm1*p_vsm1) +
        yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
                2.0*u[k][j][i][4]*rho_i[k][j][i] +
                  u[k][j-1][i][4]*rho_i[k][j-1][i]) -
        ty2 * ((c1*u[k][j+1][i][4] - c2*square[k][j+1][i]) * p_vsp1 -
               (c1*u[k][j-1][i][4] - c2*square[k][j-1][i]) * p_vsm1);
    }
  }

  //---------------------------------------------------------------------
  // add fourth order eta-direction dissipation         
  //---------------------------------------------------------------------
  j = 1;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
        ( 5.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
    }
  }

  j = 2;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
        (-4.0*u[k][j-1][i][m] + 6.0*u[k][j][i][m] -
          4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
    }
  }

  for (j = 3; j <= ny2-2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
          6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + 
            u[k][j+2][i][m] );
      }
    }
  }

  j = ny2-1;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
        ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
        6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] );
    }
  }

  j = ny2;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
        ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 5.0*u[k][j][i][m] );
    }
  }
#endif
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
  double p_u[5], p_up1[5], p_up2[5], p_um1[5], p_um2[5];
  double p_us, p_usp1, p_usm1;
  double p_vs, p_vsp1, p_vsm1;
  double p_ws, p_wsp1, p_wsm1;
  double p_qs, p_qsp1, p_qsm1;
  double p_rho_i, p_rho_ip1, p_rho_im1;
  double p_square, p_squarep1, p_squarem1;

  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (j > (gp1-2) || i > (gp0-2)) return;

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
                           int nx2,
                           int ny2,
                           int nz2)
{
  int i, j, k, m;

#if COMPUTE_RHS6_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (m = 0; m < 5; m++) {
    rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
  }

#elif COMPUTE_RHS6_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
    }
  }

#else //COMPUTE_RHS6_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
      }
    }
  }
#endif
}


__kernel void txinvr(__global const double *g_us,
                     __global const double *g_vs,
                     __global const double *g_ws,
                     __global const double *g_qs,
                     __global const double *g_rho_i,
                     __global const double *g_speed,
                     __global double *g_rhs,
                     int nx2,
                     int ny2,
                     int nz2)
{
  int i, j, k;
  double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

#if TXINVR_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  ru1 = rho_i[k][j][i];
  uu = us[k][j][i];
  vv = vs[k][j][i];
  ww = ws[k][j][i];
  ac = speed[k][j][i];
  ac2inv = ac*ac;

  r1 = rhs[k][j][i][0];
  r2 = rhs[k][j][i][1];
  r3 = rhs[k][j][i][2];
  r4 = rhs[k][j][i][3];
  r5 = rhs[k][j][i][4];

  t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
  t2 = bt * ru1 * ( uu * r1 - r2 );
  t3 = ( bt * ru1 * ac ) * t1;

  rhs[k][j][i][0] = r1 - t1;
  rhs[k][j][i][1] = - ru1 * ( ww*r1 - r4 );
  rhs[k][j][i][2] =   ru1 * ( vv*r1 - r3 );
  rhs[k][j][i][3] = - t2 + t3;
  rhs[k][j][i][4] =   t2 + t3;

#elif TXINVR_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    ru1 = rho_i[k][j][i];
    uu = us[k][j][i];
    vv = vs[k][j][i];
    ww = ws[k][j][i];
    ac = speed[k][j][i];
    ac2inv = ac*ac;

    r1 = rhs[k][j][i][0];
    r2 = rhs[k][j][i][1];
    r3 = rhs[k][j][i][2];
    r4 = rhs[k][j][i][3];
    r5 = rhs[k][j][i][4];

    t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
    t2 = bt * ru1 * ( uu * r1 - r2 );
    t3 = ( bt * ru1 * ac ) * t1;

    rhs[k][j][i][0] = r1 - t1;
    rhs[k][j][i][1] = - ru1 * ( ww*r1 - r4 );
    rhs[k][j][i][2] =   ru1 * ( vv*r1 - r3 );
    rhs[k][j][i][3] = - t2 + t3;
    rhs[k][j][i][4] =   t2 + t3;
  }

#else //TXINVR_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      ru1 = rho_i[k][j][i];
      uu = us[k][j][i];
      vv = vs[k][j][i];
      ww = ws[k][j][i];
      ac = speed[k][j][i];
      ac2inv = ac*ac;

      r1 = rhs[k][j][i][0];
      r2 = rhs[k][j][i][1];
      r3 = rhs[k][j][i][2];
      r4 = rhs[k][j][i][3];
      r5 = rhs[k][j][i][4];

      t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
      t2 = bt * ru1 * ( uu * r1 - r2 );
      t3 = ( bt * ru1 * ac ) * t1;

      rhs[k][j][i][0] = r1 - t1;
      rhs[k][j][i][1] = - ru1 * ( ww*r1 - r4 );
      rhs[k][j][i][2] =   ru1 * ( vv*r1 - r3 );
      rhs[k][j][i][3] = - t2 + t3;
      rhs[k][j][i][4] =   t2 + t3;
    }
  }
#endif
}


//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__kernel void x_solve(__global double *g_us,
                      __global double *g_rho_i,
                      __global double *g_speed,
                      __global double *g_rhs,
                      __global double *g_cv,
                      __global double *g_rhon,
                      __global double *g_lhs,
                      __global double *g_lhsp,
                      __global double *g_lhsm,
                      int nx2,
                      int ny2,
                      int nz2,
                      int gp0)
{
  int i, j, k, i1, i2, m;
  double ru1, fac1, fac2;
  double p_lhs[5], p_lhsp1[5], p_lhsp2[5];
  double p_lhsp[5], p_lhspp1[5], p_lhspp2[5];
  double p_lhsm[5], p_lhsmp1[5], p_lhsmp2[5];
  double p_rhs[5], p_rhsp1[5], p_rhsp2[5];
  double p_cv, p_cvp1, p_cvm1;
  double p_rhon, p_rhonp1, p_rhonm1;
  double p_speed, p_speedp1, p_speedm1;

#if X_SOLVE_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*ny2 + (j-1);
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhon = (__global double *)&g_rhon[my_offset];

  my_offset = my_id * (IMAXP+1) * 5;
  __global double (*lhs)[5]  = (__global double (*)[5])&g_lhs[my_offset];
  __global double (*lhsp)[5] = (__global double (*)[5])&g_lhsp[my_offset];
  __global double (*lhsm)[5] = (__global double (*)[5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs [0][m] = 0.0;
    lhsp[0][m] = 0.0;
    lhsm[0][m] = 0.0;
    lhs [nx2+1][m] = 0.0;
    lhsp[nx2+1][m] = 0.0;
    lhsm[nx2+1][m] = 0.0;
  }
  lhs [0][2] = 1.0;
  lhsp[0][2] = 1.0;
  lhsm[0][2] = 1.0;
  lhs [nx2+1][2] = 1.0;
  lhsp[nx2+1][2] = 1.0;
  lhsm[nx2+1][2] = 1.0;

  //---------------------------------------------------------------------
  // Computes the left hand side for the three x-factors  
  // first fill the lhs for the u-eigenvalue                   
  // add fourth order dissipation                             
  // subsequently, fill the other factors (u+c), (u-c) by adding to 
  // the first  
  //---------------------------------------------------------------------
  ru1 = c3c4*rho_i[k][j][0];
  cv[0] = p_cv = us[k][j][0];
  rhon[0] = p_rhon = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
  p_speed = speed[k][j][0];

  ru1 = c3c4*rho_i[k][j][1];
  cv[1] = p_cvp1 = us[k][j][1];
  rhon[1] = p_rhonp1 = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
  p_speedp1 = speed[k][j][1];

#define LOOP_PROLOGUE                                                      \
  p_cvm1   = p_cv;    p_cv   = p_cvp1;                                     \
  p_rhonm1 = p_rhon;  p_rhon = p_rhonp1;                                   \
  ru1 = c3c4*rho_i[k][j][i+1];                                             \
  cv[i+1] = p_cvp1 = us[k][j][i+1];                                        \
  rhon[i+1] = p_rhonp1 = max(max(dx2+con43*ru1,dx5+c1c5*ru1),              \
                             max(dxmax+ru1,dx1));                          \
  p_speedm1 = p_speed; p_speed = p_speedp1; p_speedp1 = speed[k][j][i+1];

#define LOOP_EPILOGUE                         \
  lhsp[i][0] = p_lhs[0];                      \
  lhsp[i][1] = p_lhs[1] - dttx2 * p_speedm1;  \
  lhsp[i][2] = p_lhs[2];                      \
  lhsp[i][3] = p_lhs[3] + dttx2 * p_speedp1;  \
  lhsp[i][4] = p_lhs[4];                      \
  lhsm[i][0] = p_lhs[0];                      \
  lhsm[i][1] = p_lhs[1] + dttx2 * p_speedm1;  \
  lhsm[i][2] = p_lhs[2];                      \
  lhsm[i][3] = p_lhs[3] - dttx2 * p_speedp1;  \
  lhsm[i][4] = p_lhs[4];

  i = 1;
    LOOP_PROLOGUE
    lhs[i][0] = p_lhs[0] =  0.0;
    lhs[i][1] = p_lhs[1] = -dttx2 * p_cvm1 - dttx1 * p_rhonm1;
    lhs[i][2] = p_lhs[2] =  1.0 + c2dttx1 * p_rhon            + comz5;
    lhs[i][3] = p_lhs[3] =  dttx2 * p_cvp1 - dttx1 * p_rhonp1 - comz4;
    lhs[i][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  i = 2;
    LOOP_PROLOGUE
    lhs[i][0] = p_lhs[0] =  0.0;
    lhs[i][1] = p_lhs[1] = -dttx2 * p_cvm1 - dttx1 * p_rhonm1 - comz4;
    lhs[i][2] = p_lhs[2] =  1.0 + c2dttx1 * p_rhon            + comz6;
    lhs[i][3] = p_lhs[3] =  dttx2 * p_cvp1 - dttx1 * p_rhonp1 - comz4;
    lhs[i][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  for (i = 3; i <= gp0-4; i++) {
    LOOP_PROLOGUE
    lhs[i][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[i][1] = p_lhs[1] = -dttx2 * p_cvm1 - dttx1 * p_rhonm1 - comz4;
    lhs[i][2] = p_lhs[2] =  1.0 + c2dttx1 * p_rhon            + comz6;
    lhs[i][3] = p_lhs[3] =  dttx2 * p_cvp1 - dttx1 * p_rhonp1 - comz4;
    lhs[i][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE
  }

  i = gp0-3;
    LOOP_PROLOGUE
    lhs[i][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[i][1] = p_lhs[1] = -dttx2 * p_cvm1 - dttx1 * p_rhonm1 - comz4;
    lhs[i][2] = p_lhs[2] =  1.0 + c2dttx1 * p_rhon            + comz6;
    lhs[i][3] = p_lhs[3] =  dttx2 * p_cvp1 - dttx1 * p_rhonp1 - comz4;
    lhs[i][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

  i = gp0-2;
    LOOP_PROLOGUE
    lhs[i][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[i][1] = p_lhs[1] = -dttx2 * p_cvm1 - dttx1 * p_rhonm1 - comz4;
    lhs[i][2] = p_lhs[2] =  1.0 + c2dttx1 * p_rhon            + comz5;
    lhs[i][3] = p_lhs[3] =  dttx2 * p_cvp1 - dttx1 * p_rhonp1;
    lhs[i][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

#undef LOOP_PROLOGUE
#undef LOOP_EPILOGUE

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // perform the Thomas algorithm; first, FORWARD ELIMINATION     
  //---------------------------------------------------------------------
  p_lhsp1[2] = lhs[0][2]; p_lhsp1[3] = lhs[0][3];
  p_lhsp2[1] = lhs[1][1]; p_lhsp2[2] = lhs[1][2];

  p_rhsp1[0] = rhs[k][j][0][0]; p_rhsp1[1] = rhs[k][j][0][1];
  p_rhsp1[2] = rhs[k][j][0][2];
  p_rhsp2[0] = rhs[k][j][1][0]; p_rhsp2[1] = rhs[k][j][1][1];
  p_rhsp2[2] = rhs[k][j][1][2];

  for (i = 0; i <= gp0-3; i++) {
    i1 = i + 1;
    i2 = i + 2;

    p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
    p_lhs[4] = lhs[i][4];
    p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

    fac1 = 1.0/p_lhs[2];
    lhs[i][3] = p_lhs[3] = fac1*p_lhs[3];
    lhs[i][4] = p_lhs[4] = fac1*p_lhs[4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
    }

    p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
    p_lhsp1[3] = lhs[i1][3];
    p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

    lhs[i1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
    p_lhsp1[3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
    }

    p_lhsp2[0] = lhs[i2][0]; p_lhsp2[1] = lhs[i2][1]; p_lhsp2[2] = lhs[i2][2];
    p_rhsp2[0] = rhs[k][j][i2][0]; p_rhsp2[1] = rhs[k][j][i2][1];
    p_rhsp2[2] = rhs[k][j][i2][2];

    lhs[i2][1] = p_lhsp2[1] = p_lhsp2[1] - p_lhsp2[0]*p_lhs[3];
    p_lhsp2[2] = p_lhsp2[2] - p_lhsp2[0]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp2[m] - p_lhsp2[0]*p_rhs[m];
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  i  = gp0-2;
  i1 = gp0-1;

  p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
  p_lhs[4] = lhs[i][4];
  p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

  fac1 = 1.0/p_lhs[2];
  lhs[i][3] = p_lhs[3] = fac1*p_lhs[3];
  lhs[i][4] = p_lhs[4] = fac1*p_lhs[4];
  for (m = 0; m < 3; m++) {
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
  }

  p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
  p_lhsp1[3] = lhs[i1][3];
  p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

  lhs[i1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
  lhs[i1][3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
  }

  //---------------------------------------------------------------------
  // scale the last row immediately 
  //---------------------------------------------------------------------
  fac2 = 1.0/p_lhsp1[2];
  for (m = 0; m < 3; m++) {
    rhs[k][j][i1][m] = fac2*p_rhsp1[m];
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors                 
  //---------------------------------------------------------------------
  p_lhspp1[2] = lhsp[0][2]; p_lhspp1[3] = lhsp[0][3];
  p_lhspp2[1] = lhsp[1][1]; p_lhspp2[2] = lhsp[1][2];
  p_lhsmp1[2] = lhsm[0][2]; p_lhsmp1[3] = lhsm[0][3];
  p_lhsmp2[1] = lhsm[1][1]; p_lhsmp2[2] = lhsm[1][2];

  p_rhsp1[3] = rhs[k][j][0][3]; p_rhsp1[4] = rhs[k][j][0][4];
  p_rhsp2[3] = rhs[k][j][1][3]; p_rhsp2[4] = rhs[k][j][1][4];

  for (i = 0; i <= gp0-3; i++) {
    i1 = i + 1;
    i2 = i + 2;

    m = 3;

    p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[i][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsp[2];
    lhsp[i][3] = p_lhsp[3] = fac1*p_lhsp[3];
    lhsp[i][4] = p_lhsp[4] = fac1*p_lhsp[4];
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];

    p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[i1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsp[i1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
    p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

    p_lhspp2[0] = lhsp[i2][0]; p_lhspp2[1] = lhsp[i2][1]; p_lhspp2[2] = lhsp[i2][2];
    p_rhsp2[m] = rhs[k][j][i2][m];
    lhsp[i2][1] = p_lhspp2[1] = p_lhspp2[1] - p_lhspp2[0]*p_lhsp[3];
    p_lhspp2[2] = p_lhspp2[2] - p_lhspp2[0]*p_lhsp[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhspp2[0]*p_rhs[m];

    m = 4;

    p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[i][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsm[2];
    lhsm[i][3] = p_lhsm[3] = fac1*p_lhsm[3];
    lhsm[i][4] = p_lhsm[4] = fac1*p_lhsm[4];
    rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

    p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[i1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsm[i1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
    p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

    p_lhsmp2[0] = lhsm[i2][0]; p_lhsmp2[1] = lhsm[i2][1]; p_lhsmp2[2] = lhsm[i2][2];
    p_rhsp2[m] = rhs[k][j][i2][m];
    lhsm[i2][1] = p_lhsmp2[1] = p_lhsmp2[1] - p_lhsmp2[0]*p_lhsm[3];
    p_lhsmp2[2] = p_lhsmp2[2] - p_lhsmp2[0]*p_lhsm[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhsmp2[0]*p_rhs[m];
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  i  = gp0-2;
  i1 = gp0-1;

  m = 3;

  p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[i][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsp[2];
  lhsp[i][3] = p_lhsp[3] = fac1*p_lhsp[3];
  lhsp[i][4] = p_lhsp[4] = fac1*p_lhsp[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[i1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsp[i1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
  lhsp[i1][3] = p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

  m = 4;

  p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[i][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsm[2];
  lhsm[i][3] = p_lhsm[3] = fac1*p_lhsm[3];
  lhsm[i][4] = p_lhsm[4] = fac1*p_lhsm[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[i1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsm[i1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
  lhsm[i1][3] = p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs[k][j][i1][3] = p_rhsp1[3]/p_lhspp1[2];
  rhs[k][j][i1][4] = p_rhsp1[4]/p_lhsmp1[2];

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  i  = gp0-2;
  i1 = gp0-1;
  p_lhs[3] = lhs[i][3];

  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = rhs[k][j][i1][m];
    rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - p_lhs[3]*p_rhsp1[m];
  }

  p_rhsp1[3] = rhs[k][j][i1][3];
  rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - lhsp[i][3]*p_rhsp1[3];
  p_rhsp1[4] = rhs[k][j][i1][4];
  rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - lhsm[i][3]*p_rhsp1[4];

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (i = gp0-3; i >= 0; i--) {
    i1 = i + 1;
    i2 = i + 2;
    p_lhs[3] = lhs[i][3];
    p_lhs[4] = lhs[i][4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp1[m]; p_rhsp1[m] = p_rhs[m];
      rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - 
                        p_lhs[3]*p_rhsp1[m] -
                        p_lhs[4]*p_rhsp2[m];
    }

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    p_rhsp2[3] = p_rhsp1[3]; p_rhsp1[3] = p_rhs[3];
    rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - 
                      lhsp[i][3]*p_rhsp1[3] -
                      lhsp[i][4]*p_rhsp2[3];
    p_rhsp2[4] = p_rhsp1[4]; p_rhsp1[4] = p_rhs[4];
    rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - 
                      lhsm[i][3]*p_rhsp1[4] -
                      lhsm[i][4]*p_rhsp2[4];
  }

#else //X_SOLVE_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = k - 1;
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhon = (__global double *)&g_rhon[my_offset];

  my_offset = my_id * (IMAXP+1) * (IMAXP+1) * 5;
  __global double (*lhs)[IMAXP+1][5]  = 
    (__global double (*)[IMAXP+1][5])&g_lhs[my_offset];
  __global double (*lhsp)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsp[my_offset];
  __global double (*lhsm)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (m = 0; m < 5; m++) {
      lhs [j][0][m] = 0.0;
      lhsp[j][0][m] = 0.0;
      lhsm[j][0][m] = 0.0;
      lhs [j][nx2+1][m] = 0.0;
      lhsp[j][nx2+1][m] = 0.0;
      lhsm[j][nx2+1][m] = 0.0;
    }
    lhs [j][0][2] = 1.0;
    lhsp[j][0][2] = 1.0;
    lhsm[j][0][2] = 1.0;
    lhs [j][nx2+1][2] = 1.0;
    lhsp[j][nx2+1][2] = 1.0;
    lhsm[j][nx2+1][2] = 1.0;
  }

  //---------------------------------------------------------------------
  // Computes the left hand side for the three x-factors  
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // first fill the lhs for the u-eigenvalue                   
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (i = 0; i < gp0; i++) {
      ru1 = c3c4*rho_i[k][j][i];
      cv[i] = us[k][j][i];
      rhon[i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
    }

    for (i = 1; i <= nx2; i++) {
      lhs[j][i][0] =  0.0;
      lhs[j][i][1] = -dttx2 * cv[i-1] - dttx1 * rhon[i-1];
      lhs[j][i][2] =  1.0 + c2dttx1 * rhon[i];
      lhs[j][i][3] =  dttx2 * cv[i+1] - dttx1 * rhon[i+1];
      lhs[j][i][4] =  0.0;
    }
  }

  //---------------------------------------------------------------------
  // add fourth order dissipation                             
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    i = 1;
    lhs[j][i][2] = lhs[j][i][2] + comz5;
    lhs[j][i][3] = lhs[j][i][3] - comz4;
    lhs[j][i][4] = lhs[j][i][4] + comz1;

    lhs[j][i+1][1] = lhs[j][i+1][1] - comz4;
    lhs[j][i+1][2] = lhs[j][i+1][2] + comz6;
    lhs[j][i+1][3] = lhs[j][i+1][3] - comz4;
    lhs[j][i+1][4] = lhs[j][i+1][4] + comz1;
  }

  for (j = 1; j <= ny2; j++) {
    for (i = 3; i <= gp0-4; i++) {
      lhs[j][i][0] = lhs[j][i][0] + comz1;
      lhs[j][i][1] = lhs[j][i][1] - comz4;
      lhs[j][i][2] = lhs[j][i][2] + comz6;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i][4] = lhs[j][i][4] + comz1;
    }
  }

  for (j = 1; j <= ny2; j++) {
    i = gp0-3;
    lhs[j][i][0] = lhs[j][i][0] + comz1;
    lhs[j][i][1] = lhs[j][i][1] - comz4;
    lhs[j][i][2] = lhs[j][i][2] + comz6;
    lhs[j][i][3] = lhs[j][i][3] - comz4;

    lhs[j][i+1][0] = lhs[j][i+1][0] + comz1;
    lhs[j][i+1][1] = lhs[j][i+1][1] - comz4;
    lhs[j][i+1][2] = lhs[j][i+1][2] + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, fill the other factors (u+c), (u-c) by adding to 
  // the first  
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      lhsp[j][i][0] = lhs[j][i][0];
      lhsp[j][i][1] = lhs[j][i][1] - dttx2 * speed[k][j][i-1];
      lhsp[j][i][2] = lhs[j][i][2];
      lhsp[j][i][3] = lhs[j][i][3] + dttx2 * speed[k][j][i+1];
      lhsp[j][i][4] = lhs[j][i][4];
      lhsm[j][i][0] = lhs[j][i][0];
      lhsm[j][i][1] = lhs[j][i][1] + dttx2 * speed[k][j][i-1];
      lhsm[j][i][2] = lhs[j][i][2];
      lhsm[j][i][3] = lhs[j][i][3] - dttx2 * speed[k][j][i+1];
      lhsm[j][i][4] = lhs[j][i][4];
    }
  }

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // perform the Thomas algorithm; first, FORWARD ELIMINATION     
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (i = 0; i <= gp0-3; i++) {
      i1 = i + 1;
      i2 = i + 2;
      fac1 = 1.0/lhs[j][i][2];
      lhs[j][i][3] = fac1*lhs[j][i][3];
      lhs[j][i][4] = fac1*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1]*lhs[j][i][3];
      lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1]*rhs[k][j][i][m];
      }
      lhs[j][i2][1] = lhs[j][i2][1] - lhs[j][i2][0]*lhs[j][i][3];
      lhs[j][i2][2] = lhs[j][i2][2] - lhs[j][i2][0]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs[j][i2][0]*rhs[k][j][i][m];
      }
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    i  = gp0-2;
    i1 = gp0-1;
    fac1 = 1.0/lhs[j][i][2];
    lhs[j][i][3] = fac1*lhs[j][i][3];
    lhs[j][i][4] = fac1*lhs[j][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
    }
    lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1]*lhs[j][i][3];
    lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1]*lhs[j][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1]*rhs[k][j][i][m];
    }

    //---------------------------------------------------------------------
    // scale the last row immediately 
    //---------------------------------------------------------------------
    fac2 = 1.0/lhs[j][i1][2];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i1][m] = fac2*rhs[k][j][i1][m];
    }
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors                 
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (i = 0; i <= gp0-3; i++) {
      i1 = i + 1;
      i2 = i + 2;

      m = 3;
      fac1 = 1.0/lhsp[j][i][2];
      lhsp[j][i][3]    = fac1*lhsp[j][i][3];
      lhsp[j][i][4]    = fac1*lhsp[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[j][i1][2]   = lhsp[j][i1][2] - lhsp[j][i1][1]*lhsp[j][i][3];
      lhsp[j][i1][3]   = lhsp[j][i1][3] - lhsp[j][i1][1]*lhsp[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1]*rhs[k][j][i][m];
      lhsp[j][i2][1]   = lhsp[j][i2][1] - lhsp[j][i2][0]*lhsp[j][i][3];
      lhsp[j][i2][2]   = lhsp[j][i2][2] - lhsp[j][i2][0]*lhsp[j][i][4];
      rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp[j][i2][0]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[j][i][2];
      lhsm[j][i][3]    = fac1*lhsm[j][i][3];
      lhsm[j][i][4]    = fac1*lhsm[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[j][i1][2]   = lhsm[j][i1][2] - lhsm[j][i1][1]*lhsm[j][i][3];
      lhsm[j][i1][3]   = lhsm[j][i1][3] - lhsm[j][i1][1]*lhsm[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1]*rhs[k][j][i][m];
      lhsm[j][i2][1]   = lhsm[j][i2][1] - lhsm[j][i2][0]*lhsm[j][i][3];
      lhsm[j][i2][2]   = lhsm[j][i2][2] - lhsm[j][i2][0]*lhsm[j][i][4];
      rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm[j][i2][0]*rhs[k][j][i][m];
    }
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    i  = gp0-2;
    i1 = gp0-1;

    m = 3;
    fac1 = 1.0/lhsp[j][i][2];
    lhsp[j][i][3]    = fac1*lhsp[j][i][3];
    lhsp[j][i][4]    = fac1*lhsp[j][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsp[j][i1][2]   = lhsp[j][i1][2] - lhsp[j][i1][1]*lhsp[j][i][3];
    lhsp[j][i1][3]   = lhsp[j][i1][3] - lhsp[j][i1][1]*lhsp[j][i][4];
    rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1]*rhs[k][j][i][m];

    m = 4;
    fac1 = 1.0/lhsm[j][i][2];
    lhsm[j][i][3]    = fac1*lhsm[j][i][3];
    lhsm[j][i][4]    = fac1*lhsm[j][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsm[j][i1][2]   = lhsm[j][i1][2] - lhsm[j][i1][1]*lhsm[j][i][3];
    lhsm[j][i1][3]   = lhsm[j][i1][3] - lhsm[j][i1][1]*lhsm[j][i][4];
    rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1]*rhs[k][j][i][m];

    //---------------------------------------------------------------------
    // Scale the last row immediately
    //---------------------------------------------------------------------
    rhs[k][j][i1][3] = rhs[k][j][i1][3]/lhsp[j][i1][2];
    rhs[k][j][i1][4] = rhs[k][j][i1][4]/lhsm[j][i1][2];
  }

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    i  = gp0-2;
    i1 = gp0-1;
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3]*rhs[k][j][i1][m];
    }

    rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3]*rhs[k][j][i1][3];
    rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3]*rhs[k][j][i1][4];
  }

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (j = 1; j <= ny2; j++) {
    for (i = gp0-3; i >= 0; i--) {
      i1 = i + 1;
      i2 = i + 2;
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - 
                          lhs[j][i][3]*rhs[k][j][i1][m] -
                          lhs[j][i][4]*rhs[k][j][i2][m];
      }

      //-------------------------------------------------------------------
      // And the remaining two
      //-------------------------------------------------------------------
      rhs[k][j][i][3] = rhs[k][j][i][3] - 
                        lhsp[j][i][3]*rhs[k][j][i1][3] -
                        lhsp[j][i][4]*rhs[k][j][i2][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - 
                        lhsm[j][i][3]*rhs[k][j][i1][4] -
                        lhsm[j][i][4]*rhs[k][j][i2][4];
    }
  }
#endif
}


//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication              
//---------------------------------------------------------------------
__kernel void ninvr(__global double *g_rhs,
                    int nx2,
                    int ny2,
                    int nz2)
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

#if NINVR_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  r1 = rhs[k][j][i][0];
  r2 = rhs[k][j][i][1];
  r3 = rhs[k][j][i][2];
  r4 = rhs[k][j][i][3];
  r5 = rhs[k][j][i][4];

  t1 = bt * r3;
  t2 = 0.5 * ( r4 + r5 );

  rhs[k][j][i][0] = -r2;
  rhs[k][j][i][1] =  r1;
  rhs[k][j][i][2] = bt * ( r4 - r5 );
  rhs[k][j][i][3] = -t1 + t2;
  rhs[k][j][i][4] =  t1 + t2;

#elif NINVR_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    r1 = rhs[k][j][i][0];
    r2 = rhs[k][j][i][1];
    r3 = rhs[k][j][i][2];
    r4 = rhs[k][j][i][3];
    r5 = rhs[k][j][i][4];

    t1 = bt * r3;
    t2 = 0.5 * ( r4 + r5 );

    rhs[k][j][i][0] = -r2;
    rhs[k][j][i][1] =  r1;
    rhs[k][j][i][2] = bt * ( r4 - r5 );
    rhs[k][j][i][3] = -t1 + t2;
    rhs[k][j][i][4] =  t1 + t2;
  }

#else //NINVR_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      r1 = rhs[k][j][i][0];
      r2 = rhs[k][j][i][1];
      r3 = rhs[k][j][i][2];
      r4 = rhs[k][j][i][3];
      r5 = rhs[k][j][i][4];

      t1 = bt * r3;
      t2 = 0.5 * ( r4 + r5 );

      rhs[k][j][i][0] = -r2;
      rhs[k][j][i][1] =  r1;
      rhs[k][j][i][2] = bt * ( r4 - r5 );
      rhs[k][j][i][3] = -t1 + t2;
      rhs[k][j][i][4] =  t1 + t2;
    }
  }
#endif
}


//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__kernel void y_solve(__global double *g_vs,
                      __global double *g_rho_i,
                      __global double *g_speed,
                      __global double *g_rhs,
                      __global double *g_cv,
                      __global double *g_rhoq,
                      __global double *g_lhs,
                      __global double *g_lhsp,
                      __global double *g_lhsm,
                      int nx2,
                      int ny2,
                      int nz2,
                      int gp1)
{
  int i, j, k, j1, j2, m;
  double ru1, fac1, fac2;
  double p_lhs[5], p_lhsp1[5], p_lhsp2[5];
  double p_lhsp[5], p_lhspp1[5], p_lhspp2[5];
  double p_lhsm[5], p_lhsmp1[5], p_lhsmp2[5];
  double p_rhs[5], p_rhsp1[5], p_rhsp2[5];
  double p_cv, p_cvp1, p_cvm1;
  double p_rhoq, p_rhoqp1, p_rhoqm1;
  double p_speed, p_speedp1, p_speedm1;

#if Y_SOLVE_DIM == 2
  k = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || i > nx2) return;

  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (k-1)*nx2 + (i-1);
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhoq = (__global double *)&g_rhoq[my_offset];

  my_offset = my_id * (IMAXP+1) * 5;
  __global double (*lhs)[5]  = (__global double (*)[5])&g_lhs[my_offset];
  __global double (*lhsp)[5] = (__global double (*)[5])&g_lhsp[my_offset];
  __global double (*lhsm)[5] = (__global double (*)[5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs [0][m] = 0.0;
    lhsp[0][m] = 0.0;
    lhsm[0][m] = 0.0;
    lhs [ny2+1][m] = 0.0;
    lhsp[ny2+1][m] = 0.0;
    lhsm[ny2+1][m] = 0.0;
  }
  lhs [0][2] = 1.0;
  lhsp[0][2] = 1.0;
  lhsm[0][2] = 1.0;
  lhs [ny2+1][2] = 1.0;
  lhsp[ny2+1][2] = 1.0;
  lhsm[ny2+1][2] = 1.0;

  //---------------------------------------------------------------------
  // Computes the left hand side for the three y-factors   
  // first fill the lhs for the u-eigenvalue         
  // add fourth order dissipation                             
  // subsequently, for (the other two factors                    
  //---------------------------------------------------------------------
  ru1 = c3c4*rho_i[k][0][i];
  cv[0] = p_cv = vs[k][0][i];
  rhoq[0] = p_rhoq = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
  p_speed = speed[k][0][i];

  ru1 = c3c4*rho_i[k][1][i];
  cv[1] = p_cvp1 = vs[k][1][i];
  rhoq[1] = p_rhoqp1 = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
  p_speedp1 = speed[k][1][i];

#define LOOP_PROLOGUE                                                      \
  p_cvm1   = p_cv;    p_cv   = p_cvp1;                                     \
  p_rhoqm1 = p_rhoq;  p_rhoq = p_rhoqp1;                                   \
  ru1 = c3c4*rho_i[k][j+1][i];                                             \
  cv[j+1] = p_cvp1 = vs[k][j+1][i];                                        \
  rhoq[j+1] = p_rhoqp1 = max(max(dy3+con43*ru1, dy5+c1c5*ru1),             \
                             max(dymax+ru1, dy1));                         \
  p_speedm1 = p_speed; p_speed = p_speedp1; p_speedp1 = speed[k][j+1][i];

#define LOOP_EPILOGUE                         \
  lhsp[j][0] = p_lhs[0];                      \
  lhsp[j][1] = p_lhs[1] - dtty2 * p_speedm1;  \
  lhsp[j][2] = p_lhs[2];                      \
  lhsp[j][3] = p_lhs[3] + dtty2 * p_speedp1;  \
  lhsp[j][4] = p_lhs[4];                      \
  lhsm[j][0] = p_lhs[0];                      \
  lhsm[j][1] = p_lhs[1] + dtty2 * p_speedm1;  \
  lhsm[j][2] = p_lhs[2];                      \
  lhsm[j][3] = p_lhs[3] - dtty2 * p_speedp1;  \
  lhsm[j][4] = p_lhs[4];

  j = 1;
    LOOP_PROLOGUE
    lhs[j][0] = p_lhs[0] =  0.0;
    lhs[j][1] = p_lhs[1] = -dtty2 * p_cvm1 - dtty1 * p_rhoqm1;
    lhs[j][2] = p_lhs[2] =  1.0 + c2dtty1 * p_rhoq            + comz5;
    lhs[j][3] = p_lhs[3] =  dtty2 * p_cvp1 - dtty1 * p_rhoqp1 - comz4;
    lhs[j][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  j = 2;
    LOOP_PROLOGUE
    lhs[j][0] = p_lhs[0] =  0.0;
    lhs[j][1] = p_lhs[1] = -dtty2 * p_cvm1 - dtty1 * p_rhoqm1 - comz4;
    lhs[j][2] = p_lhs[2] =  1.0 + c2dtty1 * p_rhoq            + comz6;
    lhs[j][3] = p_lhs[3] =  dtty2 * p_cvp1 - dtty1 * p_rhoqp1 - comz4;
    lhs[j][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  for (j = 3; j <= gp1-4; j++) {
    LOOP_PROLOGUE
    lhs[j][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[j][1] = p_lhs[1] = -dtty2 * p_cvm1 - dtty1 * p_rhoqm1 - comz4;
    lhs[j][2] = p_lhs[2] =  1.0 + c2dtty1 * p_rhoq            + comz6;
    lhs[j][3] = p_lhs[3] =  dtty2 * p_cvp1 - dtty1 * p_rhoqp1 - comz4;
    lhs[j][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE
  }

  j = gp1-3;
    LOOP_PROLOGUE
    lhs[j][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[j][1] = p_lhs[1] = -dtty2 * p_cvm1 - dtty1 * p_rhoqm1 - comz4;
    lhs[j][2] = p_lhs[2] =  1.0 + c2dtty1 * p_rhoq            + comz6;
    lhs[j][3] = p_lhs[3] =  dtty2 * p_cvp1 - dtty1 * p_rhoqp1 - comz4;
    lhs[j][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

  j = gp1-2;
    LOOP_PROLOGUE
    lhs[j][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[j][1] = p_lhs[1] = -dtty2 * p_cvm1 - dtty1 * p_rhoqm1 - comz4;
    lhs[j][2] = p_lhs[2] =  1.0 + c2dtty1 * p_rhoq            + comz5;
    lhs[j][3] = p_lhs[3] =  dtty2 * p_cvp1 - dtty1 * p_rhoqp1;
    lhs[j][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

#undef LOOP_PROLOGUE
#undef LOOP_EPILOGUE

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------
  p_lhsp1[2] = lhs[0][2]; p_lhsp1[3] = lhs[0][3];
  p_lhsp2[1] = lhs[1][1]; p_lhsp2[2] = lhs[1][2];

  p_rhsp1[0] = rhs[k][0][i][0]; p_rhsp1[1] = rhs[k][0][i][1];
  p_rhsp1[2] = rhs[k][0][i][2];
  p_rhsp2[0] = rhs[k][1][i][0]; p_rhsp2[1] = rhs[k][1][i][1];
  p_rhsp2[2] = rhs[k][1][i][2];

  for (j = 0; j <= gp1-3; j++) {
    j1 = j + 1;
    j2 = j + 2;

    p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
    p_lhs[4] = lhs[j][4];
    p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

    fac1 = 1.0/p_lhs[2];
    lhs[j][3] = p_lhs[3] = fac1*p_lhs[3];
    lhs[j][4] = p_lhs[4] = fac1*p_lhs[4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
    }

    p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
    p_lhsp1[3] = lhs[j1][3];
    p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

    lhs[j1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
    p_lhsp1[3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
    }

    p_lhsp2[0] = lhs[j2][0]; p_lhsp2[1] = lhs[j2][1]; p_lhsp2[2] = lhs[j2][2];
    p_rhsp2[0] = rhs[k][j2][i][0]; p_rhsp2[1] = rhs[k][j2][i][1];
    p_rhsp2[2] = rhs[k][j2][i][2];

    lhs[j2][1] = p_lhsp2[1] = p_lhsp2[1] - p_lhsp2[0]*p_lhs[3];
    p_lhsp2[2] = p_lhsp2[2] - p_lhsp2[0]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp2[m] - p_lhsp2[0]*p_rhs[m];
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;

  p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
  p_lhs[4] = lhs[j][4];
  p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

  fac1 = 1.0/p_lhs[2];
  lhs[j][3] = p_lhs[3] = fac1*p_lhs[3];
  lhs[j][4] = p_lhs[4] = fac1*p_lhs[4];
  for (m = 0; m < 3; m++) {
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
  }

  p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
  p_lhsp1[3] = lhs[j1][3];
  p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

  lhs[j1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
  lhs[j1][3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
  }

  //---------------------------------------------------------------------
  // scale the last row immediately 
  //---------------------------------------------------------------------
  fac2 = 1.0/p_lhsp1[2];
  for (m = 0; m < 3; m++) {
    rhs[k][j1][i][m] = fac2*p_rhsp1[m];
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors                 
  //---------------------------------------------------------------------
  p_lhspp1[2] = lhsp[0][2]; p_lhspp1[3] = lhsp[0][3];
  p_lhspp2[1] = lhsp[1][1]; p_lhspp2[2] = lhsp[1][2];
  p_lhsmp1[2] = lhsm[0][2]; p_lhsmp1[3] = lhsm[0][3];
  p_lhsmp2[1] = lhsm[1][1]; p_lhsmp2[2] = lhsm[1][2];

  p_rhsp1[3] = rhs[k][0][i][3]; p_rhsp1[4] = rhs[k][0][i][4];
  p_rhsp2[3] = rhs[k][1][i][3]; p_rhsp2[4] = rhs[k][1][i][4];

  for (j = 0; j <= gp1-3; j++) {
    j1 = j + 1;
    j2 = j + 2;

    m = 3;

    p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[j][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsp[2];
    lhsp[j][3] = p_lhsp[3] = fac1*p_lhsp[3];
    lhsp[j][4] = p_lhsp[4] = fac1*p_lhsp[4];
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];

    p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[j1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsp[j1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
    p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

    p_lhspp2[0] = lhsp[j2][0]; p_lhspp2[1] = lhsp[j2][1]; p_lhspp2[2] = lhsp[j2][2];
    p_rhsp2[m] = rhs[k][j2][i][m];
    lhsp[j2][1] = p_lhspp2[1] = p_lhspp2[1] - p_lhspp2[0]*p_lhsp[3];
    p_lhspp2[2] = p_lhspp2[2] - p_lhspp2[0]*p_lhsp[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhspp2[0]*p_rhs[m];

    m = 4;

    p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[j][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsm[2];
    lhsm[j][3] = p_lhsm[3] = fac1*p_lhsm[3];
    lhsm[j][4] = p_lhsm[4] = fac1*p_lhsm[4];
    rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

    p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[j1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsm[j1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
    p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

    p_lhsmp2[0] = lhsm[j2][0]; p_lhsmp2[1] = lhsm[j2][1]; p_lhsmp2[2] = lhsm[j2][2];
    p_rhsp2[m] = rhs[k][j2][i][m];
    lhsm[j2][1] = p_lhsmp2[1] = p_lhsmp2[1] - p_lhsmp2[0]*p_lhsm[3];
    p_lhsmp2[2] = p_lhsmp2[2] - p_lhsmp2[0]*p_lhsm[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhsmp2[0]*p_rhs[m];

  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;

  m = 3;

  p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[j][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsp[2];
  lhsp[j][3] = p_lhsp[3] = fac1*p_lhsp[3];
  lhsp[j][4] = p_lhsp[4] = fac1*p_lhsp[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[j1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsp[j1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
  lhsp[j1][3] = p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

  m = 4;

  p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[j][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsm[2];
  lhsm[j][3] = p_lhsm[3] = fac1*p_lhsm[3];
  lhsm[j][4] = p_lhsm[4] = fac1*p_lhsm[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[j1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsm[j1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
  lhsm[j1][3] = p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs[k][j1][i][3] = p_rhsp1[3]/p_lhspp1[2];
  rhs[k][j1][i][4] = p_rhsp1[4]/p_lhsmp1[2];

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
  p_lhs[3] = lhs[j][3];
  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = rhs[k][j1][i][m];
    rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - p_lhs[3]*rhs[k][j1][i][m];
  }

  p_rhsp1[3] = rhs[k][j1][i][3];
  rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - lhsp[j][3]*p_rhsp1[3];
  p_rhsp1[4] = rhs[k][j1][i][4];
  rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - lhsm[j][3]*p_rhsp1[4];

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (j = gp1-3; j >= 0; j--) {
    j1 = j + 1;
    j2 = j + 2;
    p_lhs[3] = lhs[j][3];
    p_lhs[4] = lhs[j][4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp1[m]; p_rhsp1[m] = p_rhs[m];
      rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - 
                        p_lhs[3]*p_rhsp1[m] -
                        p_lhs[4]*p_rhsp2[m];
    }

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    p_rhsp2[3] = p_rhsp1[3]; p_rhsp1[3] = p_rhs[3];
    rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - 
                      lhsp[j][3]*p_rhsp1[3] -
                      lhsp[j][4]*p_rhsp2[3];
    p_rhsp2[4] = p_rhsp1[4]; p_rhsp1[4] = p_rhs[4];
    rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - 
                      lhsm[j][3]*p_rhsp1[4] -
                      lhsm[j][4]*p_rhsp2[4];
  }

#else //Y_SOLVE_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = k - 1;
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhoq = (__global double *)&g_rhoq[my_offset];

  my_offset = my_id * (IMAXP+1) * (IMAXP+1) * 5;
  __global double (*lhs)[IMAXP+1][5]  = 
    (__global double (*)[IMAXP+1][5])&g_lhs[my_offset];
  __global double (*lhsp)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsp[my_offset];
  __global double (*lhsm)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      lhs [0][i][m] = 0.0;
      lhsp[0][i][m] = 0.0;
      lhsm[0][i][m] = 0.0;
      lhs [ny2+1][i][m] = 0.0;
      lhsp[ny2+1][i][m] = 0.0;
      lhsm[ny2+1][i][m] = 0.0;
    }
    lhs [0][i][2] = 1.0;
    lhsp[0][i][2] = 1.0;
    lhsm[0][i][2] = 1.0;
    lhs [ny2+1][i][2] = 1.0;
    lhsp[ny2+1][i][2] = 1.0;
    lhsm[ny2+1][i][2] = 1.0;
  }

  //---------------------------------------------------------------------
  // Computes the left hand side for the three y-factors   
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // first fill the lhs for the u-eigenvalue         
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    for (j = 0; j < gp1; j++) {
      ru1 = c3c4*rho_i[k][j][i];
      cv[j] = vs[k][j][i];
      rhoq[j] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
    }

    for (j = 1; j <= gp1-2; j++) {
      lhs[j][i][0] =  0.0;
      lhs[j][i][1] = -dtty2 * cv[j-1] - dtty1 * rhoq[j-1];
      lhs[j][i][2] =  1.0 + c2dtty1 * rhoq[j];
      lhs[j][i][3] =  dtty2 * cv[j+1] - dtty1 * rhoq[j+1];
      lhs[j][i][4] =  0.0;
    }
  }

  //---------------------------------------------------------------------
  // add fourth order dissipation                             
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    j = 1;
    lhs[j][i][2] = lhs[j][i][2] + comz5;
    lhs[j][i][3] = lhs[j][i][3] - comz4;
    lhs[j][i][4] = lhs[j][i][4] + comz1;

    lhs[j+1][i][1] = lhs[j+1][i][1] - comz4;
    lhs[j+1][i][2] = lhs[j+1][i][2] + comz6;
    lhs[j+1][i][3] = lhs[j+1][i][3] - comz4;
    lhs[j+1][i][4] = lhs[j+1][i][4] + comz1;
  }

  for (j = 3; j <= gp1-4; j++) {
    for (i = 1; i <= nx2; i++) {
      lhs[j][i][0] = lhs[j][i][0] + comz1;
      lhs[j][i][1] = lhs[j][i][1] - comz4;
      lhs[j][i][2] = lhs[j][i][2] + comz6;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i][4] = lhs[j][i][4] + comz1;
    }
  }

  for (i = 1; i <= nx2; i++) {
    j = gp1-3;
    lhs[j][i][0] = lhs[j][i][0] + comz1;
    lhs[j][i][1] = lhs[j][i][1] - comz4;
    lhs[j][i][2] = lhs[j][i][2] + comz6;
    lhs[j][i][3] = lhs[j][i][3] - comz4;

    lhs[j+1][i][0] = lhs[j+1][i][0] + comz1;
    lhs[j+1][i][1] = lhs[j+1][i][1] - comz4;
    lhs[j+1][i][2] = lhs[j+1][i][2] + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, for (the other two factors                    
  //---------------------------------------------------------------------
  for (j = 1; j <= gp1-2; j++) {
    for (i = 1; i <= nx2; i++) {
      lhsp[j][i][0] = lhs[j][i][0];
      lhsp[j][i][1] = lhs[j][i][1] - dtty2 * speed[k][j-1][i];
      lhsp[j][i][2] = lhs[j][i][2];
      lhsp[j][i][3] = lhs[j][i][3] + dtty2 * speed[k][j+1][i];
      lhsp[j][i][4] = lhs[j][i][4];
      lhsm[j][i][0] = lhs[j][i][0];
      lhsm[j][i][1] = lhs[j][i][1] + dtty2 * speed[k][j-1][i];
      lhsm[j][i][2] = lhs[j][i][2];
      lhsm[j][i][3] = lhs[j][i][3] - dtty2 * speed[k][j+1][i];
      lhsm[j][i][4] = lhs[j][i][4];
    }
  }


  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------
  for (j = 0; j <= gp1-3; j++) {
    j1 = j + 1;
    j2 = j + 2;
    for (i = 1; i <= nx2; i++) {
      fac1 = 1.0/lhs[j][i][2];
      lhs[j][i][3] = fac1*lhs[j][i][3];
      lhs[j][i][4] = fac1*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[j1][i][2] = lhs[j1][i][2] - lhs[j1][i][1]*lhs[j][i][3];
      lhs[j1][i][3] = lhs[j1][i][3] - lhs[j1][i][1]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhs[j1][i][1]*rhs[k][j][i][m];
      }
      lhs[j2][i][1] = lhs[j2][i][1] - lhs[j2][i][0]*lhs[j][i][3];
      lhs[j2][i][2] = lhs[j2][i][2] - lhs[j2][i][0]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhs[j2][i][0]*rhs[k][j][i][m];
      }
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
  for (i = 1; i <= nx2; i++) {
    fac1 = 1.0/lhs[j][i][2];
    lhs[j][i][3] = fac1*lhs[j][i][3];
    lhs[j][i][4] = fac1*lhs[j][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
    }
    lhs[j1][i][2] = lhs[j1][i][2] - lhs[j1][i][1]*lhs[j][i][3];
    lhs[j1][i][3] = lhs[j1][i][3] - lhs[j1][i][1]*lhs[j][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhs[j1][i][1]*rhs[k][j][i][m];
    }
    //---------------------------------------------------------------------
    // scale the last row immediately 
    //---------------------------------------------------------------------
    fac2 = 1.0/lhs[j1][i][2];
    for (m = 0; m < 3; m++) {
      rhs[k][j1][i][m] = fac2*rhs[k][j1][i][m];
    }
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors                 
  //---------------------------------------------------------------------
  for (j = 0; j <= gp1-3; j++) {
    j1 = j + 1;
    j2 = j + 2;
    for (i = 1; i <= nx2; i++) {
      m = 3;
      fac1 = 1.0/lhsp[j][i][2];
      lhsp[j][i][3]    = fac1*lhsp[j][i][3];
      lhsp[j][i][4]    = fac1*lhsp[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[j1][i][2]   = lhsp[j1][i][2] - lhsp[j1][i][1]*lhsp[j][i][3];
      lhsp[j1][i][3]   = lhsp[j1][i][3] - lhsp[j1][i][1]*lhsp[j][i][4];
      rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsp[j1][i][1]*rhs[k][j][i][m];
      lhsp[j2][i][1]   = lhsp[j2][i][1] - lhsp[j2][i][0]*lhsp[j][i][3];
      lhsp[j2][i][2]   = lhsp[j2][i][2] - lhsp[j2][i][0]*lhsp[j][i][4];
      rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhsp[j2][i][0]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[j][i][2];
      lhsm[j][i][3]    = fac1*lhsm[j][i][3];
      lhsm[j][i][4]    = fac1*lhsm[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[j1][i][2]   = lhsm[j1][i][2] - lhsm[j1][i][1]*lhsm[j][i][3];
      lhsm[j1][i][3]   = lhsm[j1][i][3] - lhsm[j1][i][1]*lhsm[j][i][4];
      rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsm[j1][i][1]*rhs[k][j][i][m];
      lhsm[j2][i][1]   = lhsm[j2][i][1] - lhsm[j2][i][0]*lhsm[j][i][3];
      lhsm[j2][i][2]   = lhsm[j2][i][2] - lhsm[j2][i][0]*lhsm[j][i][4];
      rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhsm[j2][i][0]*rhs[k][j][i][m];
    }
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
  for (i = 1; i <= nx2; i++) {
    m = 3;
    fac1 = 1.0/lhsp[j][i][2];
    lhsp[j][i][3]    = fac1*lhsp[j][i][3];
    lhsp[j][i][4]    = fac1*lhsp[j][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsp[j1][i][2]   = lhsp[j1][i][2] - lhsp[j1][i][1]*lhsp[j][i][3];
    lhsp[j1][i][3]   = lhsp[j1][i][3] - lhsp[j1][i][1]*lhsp[j][i][4];
    rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsp[j1][i][1]*rhs[k][j][i][m];

    m = 4;
    fac1 = 1.0/lhsm[j][i][2];
    lhsm[j][i][3]    = fac1*lhsm[j][i][3];
    lhsm[j][i][4]    = fac1*lhsm[j][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsm[j1][i][2]   = lhsm[j1][i][2] - lhsm[j1][i][1]*lhsm[j][i][3];
    lhsm[j1][i][3]   = lhsm[j1][i][3] - lhsm[j1][i][1]*lhsm[j][i][4];
    rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsm[j1][i][1]*rhs[k][j][i][m];

    //---------------------------------------------------------------------
    // Scale the last row immediately 
    //---------------------------------------------------------------------
    rhs[k][j1][i][3]   = rhs[k][j1][i][3]/lhsp[j1][i][2];
    rhs[k][j1][i][4]   = rhs[k][j1][i][4]/lhsm[j1][i][2];
  }


  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3]*rhs[k][j1][i][m];
    }

    rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3]*rhs[k][j1][i][3];
    rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3]*rhs[k][j1][i][4];
  }

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (j = gp1-3; j >= 0; j--) {
    j1 = j + 1;
    j2 = j + 2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - 
                          lhs[j][i][3]*rhs[k][j1][i][m] -
                          lhs[j][i][4]*rhs[k][j2][i][m];
      }

      //-------------------------------------------------------------------
      // And the remaining two
      //-------------------------------------------------------------------
      rhs[k][j][i][3] = rhs[k][j][i][3] - 
                        lhsp[j][i][3]*rhs[k][j1][i][3] -
                        lhsp[j][i][4]*rhs[k][j2][i][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - 
                        lhsm[j][i][3]*rhs[k][j1][i][4] -
                        lhsm[j][i][4]*rhs[k][j2][i][4];
    }
  }
#endif
}


//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__kernel void pinvr(__global double *g_rhs,
                    int nx2,
                    int ny2,
                    int nz2)
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

#if PINVR_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  r1 = rhs[k][j][i][0];
  r2 = rhs[k][j][i][1];
  r3 = rhs[k][j][i][2];
  r4 = rhs[k][j][i][3];
  r5 = rhs[k][j][i][4];

  t1 = bt * r1;
  t2 = 0.5 * ( r4 + r5 );

  rhs[k][j][i][0] =  bt * ( r4 - r5 );
  rhs[k][j][i][1] = -r3;
  rhs[k][j][i][2] =  r2;
  rhs[k][j][i][3] = -t1 + t2;
  rhs[k][j][i][4] =  t1 + t2;

#elif PINVR_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    r1 = rhs[k][j][i][0];
    r2 = rhs[k][j][i][1];
    r3 = rhs[k][j][i][2];
    r4 = rhs[k][j][i][3];
    r5 = rhs[k][j][i][4];

    t1 = bt * r1;
    t2 = 0.5 * ( r4 + r5 );

    rhs[k][j][i][0] =  bt * ( r4 - r5 );
    rhs[k][j][i][1] = -r3;
    rhs[k][j][i][2] =  r2;
    rhs[k][j][i][3] = -t1 + t2;
    rhs[k][j][i][4] =  t1 + t2;
  }

#else //PINVR_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      r1 = rhs[k][j][i][0];
      r2 = rhs[k][j][i][1];
      r3 = rhs[k][j][i][2];
      r4 = rhs[k][j][i][3];
      r5 = rhs[k][j][i][4];

      t1 = bt * r1;
      t2 = 0.5 * ( r4 + r5 );

      rhs[k][j][i][0] =  bt * ( r4 - r5 );
      rhs[k][j][i][1] = -r3;
      rhs[k][j][i][2] =  r2;
      rhs[k][j][i][3] = -t1 + t2;
      rhs[k][j][i][4] =  t1 + t2;
    }
  }
#endif
}


//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__kernel void z_solve(__global double *g_ws,
                      __global double *g_rho_i,
                      __global double *g_speed,
                      __global double *g_rhs,
                      __global double *g_cv,
                      __global double *g_rhos,
                      __global double *g_lhs,
                      __global double *g_lhsp,
                      __global double *g_lhsm,
                      int nx2,
                      int ny2,
                      int nz2,
                      int gp2)
{
  int i, j, k, k1, k2, m;
  double ru1, fac1, fac2;
  double p_lhs[5], p_lhsp1[5], p_lhsp2[5];
  double p_lhsp[5], p_lhspp1[5], p_lhspp2[5];
  double p_lhsm[5], p_lhsmp1[5], p_lhsmp2[5];
  double p_rhs[5], p_rhsp1[5], p_rhsp2[5];
  double p_cv, p_cvp1, p_cvm1;
  double p_rhos, p_rhosp1, p_rhosm1;
  double p_speed, p_speedp1, p_speedm1;

#if Z_SOLVE_DIM == 2
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (j > ny2 || i > nx2) return;

  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = (j-1)*nx2 + (i-1);
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhos = (__global double *)&g_rhos[my_offset];

  my_offset = my_id * (IMAXP+1) * 5;
  __global double (*lhs)[5]  = (__global double (*)[5])&g_lhs[my_offset];
  __global double (*lhsp)[5] = (__global double (*)[5])&g_lhsp[my_offset];
  __global double (*lhsm)[5] = (__global double (*)[5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs [0][m] = 0.0;
    lhsp[0][m] = 0.0;
    lhsm[0][m] = 0.0;
    lhs [nz2+1][m] = 0.0;
    lhsp[nz2+1][m] = 0.0;
    lhsm[nz2+1][m] = 0.0;
  }
  lhs [0][2] = 1.0;
  lhsp[0][2] = 1.0;
  lhsm[0][2] = 1.0;
  lhs [nz2+1][2] = 1.0;
  lhsp[nz2+1][2] = 1.0;
  lhsm[nz2+1][2] = 1.0;

  //---------------------------------------------------------------------
  // Computes the left hand side for the three z-factors   
  // first fill the lhs for the u-eigenvalue                          
  // add fourth order dissipation                                  
  // subsequently, fill the other factors (u+c), (u-c) 
  //---------------------------------------------------------------------
  ru1 = c3c4*rho_i[0][j][i];
  cv[0] = p_cv = ws[0][j][i];
  rhos[0] = p_rhos = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
  p_speed = speed[0][j][i];

  ru1 = c3c4*rho_i[1][j][i];
  cv[1] = p_cvp1 = ws[1][j][i];
  rhos[1] = p_rhosp1 = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
  p_speedp1 = speed[1][j][i];

#define LOOP_PROLOGUE                                                      \
  p_cvm1   = p_cv;    p_cv   = p_cvp1;                                     \
  p_rhosm1 = p_rhos;  p_rhos = p_rhosp1;                                   \
  ru1 = c3c4*rho_i[k+1][j][i];                                             \
  cv[k+1] = p_cvp1 = ws[k+1][j][i];                                        \
  rhos[k+1] = p_rhosp1 = max(max(dz4+con43*ru1, dz5+c1c5*ru1),             \
                             max(dzmax+ru1, dz1));                         \
  p_speedm1 = p_speed; p_speed = p_speedp1; p_speedp1 = speed[k+1][j][i];

#define LOOP_EPILOGUE                         \
  lhsp[k][0] = p_lhs[0];                      \
  lhsp[k][1] = p_lhs[1] - dttz2 * p_speedm1;  \
  lhsp[k][2] = p_lhs[2];                      \
  lhsp[k][3] = p_lhs[3] + dttz2 * p_speedp1;  \
  lhsp[k][4] = p_lhs[4];                      \
  lhsm[k][0] = p_lhs[0];                      \
  lhsm[k][1] = p_lhs[1] + dttz2 * p_speedm1;  \
  lhsm[k][2] = p_lhs[2];                      \
  lhsm[k][3] = p_lhs[3] - dttz2 * p_speedp1;  \
  lhsm[k][4] = p_lhs[4];

  k = 1;
    LOOP_PROLOGUE
    lhs[k][0] = p_lhs[0] =  0.0;
    lhs[k][1] = p_lhs[1] = -dttz2 * p_cvm1 - dttz1 * p_rhosm1;
    lhs[k][2] = p_lhs[2] =  1.0 + c2dttz1 * p_rhos            + comz5;
    lhs[k][3] = p_lhs[3] =  dttz2 * p_cvp1 - dttz1 * p_rhosp1 - comz4;
    lhs[k][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  k = 2;
    LOOP_PROLOGUE
    lhs[k][0] = p_lhs[0] =  0.0;
    lhs[k][1] = p_lhs[1] = -dttz2 * p_cvm1 - dttz1 * p_rhosm1 - comz4;
    lhs[k][2] = p_lhs[2] =  1.0 + c2dttz1 * p_rhos            + comz6;
    lhs[k][3] = p_lhs[3] =  dttz2 * p_cvp1 - dttz1 * p_rhosp1 - comz4;
    lhs[k][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE

  for (k = 3; k <= nz2-2; k++) {
    LOOP_PROLOGUE
    lhs[k][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[k][1] = p_lhs[1] = -dttz2 * p_cvm1 - dttz1 * p_rhosm1 - comz4;
    lhs[k][2] = p_lhs[2] =  1.0 + c2dttz1 * p_rhos            + comz6;
    lhs[k][3] = p_lhs[3] =  dttz2 * p_cvp1 - dttz1 * p_rhosp1 - comz4;
    lhs[k][4] = p_lhs[4] =  0.0                               + comz1;
    LOOP_EPILOGUE
  }

  k = nz2-1;
    LOOP_PROLOGUE
    lhs[k][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[k][1] = p_lhs[1] = -dttz2 * p_cvm1 - dttz1 * p_rhosm1 - comz4;
    lhs[k][2] = p_lhs[2] =  1.0 + c2dttz1 * p_rhos            + comz6;
    lhs[k][3] = p_lhs[3] =  dttz2 * p_cvp1 - dttz1 * p_rhosp1 - comz4;
    lhs[k][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

  k = nz2;
    LOOP_PROLOGUE
    lhs[k][0] = p_lhs[0] =  0.0                               + comz1;
    lhs[k][1] = p_lhs[1] = -dttz2 * p_cvm1 - dttz1 * p_rhosm1 - comz4;
    lhs[k][2] = p_lhs[2] =  1.0 + c2dttz1 * p_rhos            + comz5;
    lhs[k][3] = p_lhs[3] =  dttz2 * p_cvp1 - dttz1 * p_rhosp1;
    lhs[k][4] = p_lhs[4] =  0.0;
    LOOP_EPILOGUE

#undef LOOP_PROLOGUE
#undef LOOP_EPILOGUE

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------
  p_lhsp1[2] = lhs[0][2]; p_lhsp1[3] = lhs[0][3];
  p_lhsp2[1] = lhs[1][1]; p_lhsp2[2] = lhs[1][2];

  p_rhsp1[0] = rhs[0][j][i][0]; p_rhsp1[1] = rhs[0][j][i][1];
  p_rhsp1[2] = rhs[0][j][i][2];
  p_rhsp2[0] = rhs[1][j][i][0]; p_rhsp2[1] = rhs[1][j][i][1];
  p_rhsp2[2] = rhs[1][j][i][2];

  for (k = 0; k <= gp2-3; k++) {
    k1 = k + 1;
    k2 = k + 2;

    p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
    p_lhs[4] = lhs[k][4];
    p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

    fac1 = 1.0/p_lhs[2];
    lhs[k][3] = p_lhs[3] = fac1*p_lhs[3];
    lhs[k][4] = p_lhs[4] = fac1*p_lhs[4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
    }

    p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
    p_lhsp1[3] = lhs[k1][3];
    p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

    lhs[k1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
    p_lhsp1[3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
    }

    p_lhsp2[0] = lhs[k2][0]; p_lhsp2[1] = lhs[k2][1]; p_lhsp2[2] = lhs[k2][2];
    p_rhsp2[0] = rhs[k2][j][i][0]; p_rhsp2[1] = rhs[k2][j][i][1];
    p_rhsp2[2] = rhs[k2][j][i][2];

    lhs[k2][1] = p_lhsp2[1] = p_lhsp2[1] - p_lhsp2[0]*p_lhs[3];
    p_lhsp2[2] = p_lhsp2[2] - p_lhsp2[0]*p_lhs[4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp2[m] - p_lhsp2[0]*p_rhs[m];
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;

  p_lhs[2] = p_lhsp1[2]; p_lhs[3] = p_lhsp1[3];
  p_lhs[4] = lhs[k][4];
  p_rhs[0] = p_rhsp1[0]; p_rhs[1] = p_rhsp1[1]; p_rhs[2] = p_rhsp1[2];

  fac1 = 1.0/p_lhs[2];
  lhs[k][3] = p_lhs[3] = fac1*p_lhs[3];
  lhs[k][4] = p_lhs[4] = fac1*p_lhs[4];
  for (m = 0; m < 3; m++) {
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];
  }

  p_lhsp1[1] = p_lhsp2[1]; p_lhsp1[2] = p_lhsp2[2];
  p_lhsp1[3] = lhs[k1][3];
  p_rhsp1[0] = p_rhsp2[0]; p_rhsp1[1] = p_rhsp2[1]; p_rhsp1[2] = p_rhsp2[2];

  lhs[k1][2] = p_lhsp1[2] = p_lhsp1[2] - p_lhsp1[1]*p_lhs[3];
  lhs[k1][3] = p_lhsp1[3] - p_lhsp1[1]*p_lhs[4];
  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = p_rhsp1[m] - p_lhsp1[1]*p_rhs[m];
  }

  //---------------------------------------------------------------------
  // scale the last row immediately 
  //---------------------------------------------------------------------
  fac2 = 1.0/p_lhsp1[2];
  for (m = 0; m < 3; m++) {
    rhs[k1][j][i][m] = fac2*p_rhsp1[m];
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors               
  //---------------------------------------------------------------------
  p_lhspp1[2] = lhsp[0][2]; p_lhspp1[3] = lhsp[0][3];
  p_lhspp2[1] = lhsp[1][1]; p_lhspp2[2] = lhsp[1][2];
  p_lhsmp1[2] = lhsm[0][2]; p_lhsmp1[3] = lhsm[0][3];
  p_lhsmp2[1] = lhsm[1][1]; p_lhsmp2[2] = lhsm[1][2];

  p_rhsp1[3] = rhs[0][j][i][3]; p_rhsp1[4] = rhs[0][j][i][4];
  p_rhsp2[3] = rhs[1][j][i][3]; p_rhsp2[4] = rhs[1][j][i][4];

  for (k = 0; k <= gp2-3; k++) {
    k1 = k + 1;
    k2 = k + 2;

    m = 3;

    p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[k][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsp[2];
    lhsp[k][3] = p_lhsp[3] = fac1*p_lhsp[3];
    lhsp[k][4] = p_lhsp[4] = fac1*p_lhsp[4];
    rhs[k][j][i][m] = p_rhs[m] = fac1*p_rhs[m];

    p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[k1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsp[k1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
    p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

    p_lhspp2[0] = lhsp[k2][0]; p_lhspp2[1] = lhsp[k2][1]; p_lhspp2[2] = lhsp[k2][2];
    p_rhsp2[m] = rhs[k2][j][i][m];
    lhsp[k2][1] = p_lhspp2[1] = p_lhspp2[1] - p_lhspp2[0]*p_lhsp[3];
    p_lhspp2[2] = p_lhspp2[2] - p_lhspp2[0]*p_lhsp[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhspp2[0]*p_rhs[m];

    m = 4;

    p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[k][4];
    p_rhs[m] = p_rhsp1[m];
    fac1 = 1.0/p_lhsm[2];
    lhsm[k][3] = p_lhsm[3] = fac1*p_lhsm[3];
    lhsm[k][4] = p_lhsm[4] = fac1*p_lhsm[4];
    rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

    p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[k1][3];
    p_rhsp1[m] = p_rhsp2[m];
    lhsm[k1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
    p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
    p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

    p_lhsmp2[0] = lhsm[k2][0]; p_lhsmp2[1] = lhsm[k2][1]; p_lhsmp2[2] = lhsm[k2][2];
    p_rhsp2[m] = rhs[k2][j][i][m];
    lhsm[k2][1] = p_lhsmp2[1] = p_lhsmp2[1] - p_lhsmp2[0]*p_lhsm[3];
    p_lhsmp2[2] = p_lhsmp2[2] - p_lhsmp2[0]*p_lhsm[4];
    p_rhsp2[m] = p_rhsp2[m] - p_lhsmp2[0]*p_rhs[m];
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;

  m = 3;

  p_lhsp[2] = p_lhspp1[2]; p_lhsp[3] = p_lhspp1[3]; p_lhsp[4] = lhsp[k][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsp[2];
  lhsp[k][3] = p_lhsp[3] = fac1*p_lhsp[3];
  lhsp[k][4] = p_lhsp[4] = fac1*p_lhsp[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhspp1[1] = p_lhspp2[1]; p_lhspp1[2] = p_lhspp2[2]; p_lhspp1[3] = lhsp[k1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsp[k1][2] = p_lhspp1[2] = p_lhspp1[2] - p_lhspp1[1]*p_lhsp[3];
  lhsp[k1][3] = p_lhspp1[3] = p_lhspp1[3] - p_lhspp1[1]*p_lhsp[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhspp1[1]*p_rhs[m];

  m = 4;

  p_lhsm[2] = p_lhsmp1[2]; p_lhsm[3] = p_lhsmp1[3]; p_lhsm[4] = lhsm[k][4];
  p_rhs[m] = p_rhsp1[m];
  fac1 = 1.0/p_lhsm[2];
  lhsm[k][3] = p_lhsm[3] = fac1*p_lhsm[3];
  lhsm[k][4] = p_lhsm[4] = fac1*p_lhsm[4];
  rhs[k][j][i][m]  = p_rhs[m] = fac1*p_rhs[m];

  p_lhsmp1[1] = p_lhsmp2[1]; p_lhsmp1[2] = p_lhsmp2[2]; p_lhsmp1[3] = lhsm[k1][3];
  p_rhsp1[m] = p_rhsp2[m];
  lhsm[k1][2] = p_lhsmp1[2] = p_lhsmp1[2] - p_lhsmp1[1]*p_lhsm[3];
  lhsm[k1][3] = p_lhsmp1[3] = p_lhsmp1[3] - p_lhsmp1[1]*p_lhsm[4];
  p_rhsp1[m] = p_rhsp1[m] - p_lhsmp1[1]*p_rhs[m];

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs[k1][j][i][3] = p_rhsp1[3]/p_lhspp1[2];
  rhs[k1][j][i][4] = p_rhsp1[4]/p_lhsmp1[2];

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;
  p_lhs[3] = lhs[k][3];
  for (m = 0; m < 3; m++) {
    p_rhsp1[m] = rhs[k1][j][i][m];
    rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - p_lhs[3]*p_rhsp1[m];
  }

  p_rhsp1[3] = rhs[k1][j][i][3];
  rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - lhsp[k][3]*p_rhsp1[3];
  p_rhsp1[4] = rhs[k1][j][i][4];
  rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - lhsm[k][3]*p_rhsp1[4];

  //---------------------------------------------------------------------
  // Whether or not this is the last processor, we always have
  // to complete the back-substitution 
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (k = gp2-3; k >= 0; k--) {
    k1 = k + 1;
    k2 = k + 2;
    p_lhs[3] = lhs[k][3];
    p_lhs[4] = lhs[k][4];
    for (m = 0; m < 3; m++) {
      p_rhsp2[m] = p_rhsp1[m]; p_rhsp1[m] = p_rhs[m];
      rhs[k][j][i][m] = p_rhs[m] = rhs[k][j][i][m] - 
                        p_lhs[3]*p_rhsp1[m] -
                        p_lhs[4]*p_rhsp2[m];
    }

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    p_rhsp2[3] = p_rhsp1[3]; p_rhsp1[3] = p_rhs[3];
    rhs[k][j][i][3] = p_rhs[3] = rhs[k][j][i][3] - 
                      lhsp[k][3]*p_rhsp1[3] -
                      lhsp[k][4]*p_rhsp2[3];
    p_rhsp2[4] = p_rhsp1[4]; p_rhsp1[4] = p_rhs[4];
    rhs[k][j][i][4] = p_rhs[4] = rhs[k][j][i][4] - 
                      lhsm[k][3]*p_rhsp1[4] -
                      lhsm[k][4]*p_rhsp2[4];
  }

#else //Z_SOLVE_DIM == 1
  j = get_global_id(0) + 1;
  if (j > ny2) return;

  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  int my_id = j - 1;
  int my_offset = my_id * PROBLEM_SIZE;
  __global double *cv   = (__global double *)&g_cv[my_offset];
  __global double *rhos = (__global double *)&g_rhos[my_offset];

  my_offset = my_id * (IMAXP+1) * (IMAXP+1) * 5;
  __global double (*lhs)[IMAXP+1][5]  = 
    (__global double (*)[IMAXP+1][5])&g_lhs[my_offset];
  __global double (*lhsp)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsp[my_offset];
  __global double (*lhsm)[IMAXP+1][5] = 
    (__global double (*)[IMAXP+1][5])&g_lhsm[my_offset];

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      lhs [0][i][m] = 0.0;
      lhsp[0][i][m] = 0.0;
      lhsm[0][i][m] = 0.0;
      lhs [nz2+1][i][m] = 0.0;
      lhsp[nz2+1][i][m] = 0.0;
      lhsm[nz2+1][i][m] = 0.0;
    }
    lhs [0][i][2] = 1.0;
    lhsp[0][i][2] = 1.0;
    lhsm[0][i][2] = 1.0;
    lhs [nz2+1][i][2] = 1.0;
    lhsp[nz2+1][i][2] = 1.0;
    lhsm[nz2+1][i][2] = 1.0;
  }

  //---------------------------------------------------------------------
  // Computes the left hand side for the three z-factors   
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // first fill the lhs for the u-eigenvalue                          
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    for (k = 0; k <= nz2+1; k++) {
      ru1 = c3c4*rho_i[k][j][i];
      cv[k] = ws[k][j][i];
      rhos[k] = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
    }

    for (k = 1; k <= nz2; k++) {
      lhs[k][i][0] =  0.0;
      lhs[k][i][1] = -dttz2 * cv[k-1] - dttz1 * rhos[k-1];
      lhs[k][i][2] =  1.0 + c2dttz1 * rhos[k];
      lhs[k][i][3] =  dttz2 * cv[k+1] - dttz1 * rhos[k+1];
      lhs[k][i][4] =  0.0;
    }
  }

  //---------------------------------------------------------------------
  // add fourth order dissipation                                  
  //---------------------------------------------------------------------
  for (i = 1; i <= nx2; i++) {
    k = 1;
    lhs[k][i][2] = lhs[k][i][2] + comz5;
    lhs[k][i][3] = lhs[k][i][3] - comz4;
    lhs[k][i][4] = lhs[k][i][4] + comz1;

    k = 2;
    lhs[k][i][1] = lhs[k][i][1] - comz4;
    lhs[k][i][2] = lhs[k][i][2] + comz6;
    lhs[k][i][3] = lhs[k][i][3] - comz4;
    lhs[k][i][4] = lhs[k][i][4] + comz1;
  }

  for (k = 3; k <= nz2-2; k++) {
    for (i = 1; i <= nx2; i++) {
      lhs[k][i][0] = lhs[k][i][0] + comz1;
      lhs[k][i][1] = lhs[k][i][1] - comz4;
      lhs[k][i][2] = lhs[k][i][2] + comz6;
      lhs[k][i][3] = lhs[k][i][3] - comz4;
      lhs[k][i][4] = lhs[k][i][4] + comz1;
    }
  }

  for (i = 1; i <= nx2; i++) {
    k = nz2-1;
    lhs[k][i][0] = lhs[k][i][0] + comz1;
    lhs[k][i][1] = lhs[k][i][1] - comz4;
    lhs[k][i][2] = lhs[k][i][2] + comz6;
    lhs[k][i][3] = lhs[k][i][3] - comz4;

    k = nz2;
    lhs[k][i][0] = lhs[k][i][0] + comz1;
    lhs[k][i][1] = lhs[k][i][1] - comz4;
    lhs[k][i][2] = lhs[k][i][2] + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, fill the other factors (u+c), (u-c) 
  //---------------------------------------------------------------------
  for (k = 1; k <= nz2; k++) {
    for (i = 1; i <= nx2; i++) {
      lhsp[k][i][0] = lhs[k][i][0];
      lhsp[k][i][1] = lhs[k][i][1] - dttz2 * speed[k-1][j][i];
      lhsp[k][i][2] = lhs[k][i][2];
      lhsp[k][i][3] = lhs[k][i][3] + dttz2 * speed[k+1][j][i];
      lhsp[k][i][4] = lhs[k][i][4];
      lhsm[k][i][0] = lhs[k][i][0];
      lhsm[k][i][1] = lhs[k][i][1] + dttz2 * speed[k-1][j][i];
      lhsm[k][i][2] = lhs[k][i][2];
      lhsm[k][i][3] = lhs[k][i][3] - dttz2 * speed[k+1][j][i];
      lhsm[k][i][4] = lhs[k][i][4];
    }
  }


  //---------------------------------------------------------------------
  // FORWARD ELIMINATION  
  //---------------------------------------------------------------------
  for (k = 0; k <= gp2-3; k++) {
    k1 = k + 1;
    k2 = k + 2;
    for (i = 1; i <= nx2; i++) {
      fac1 = 1.0/lhs[k][i][2];
      lhs[k][i][3] = fac1*lhs[k][i][3];
      lhs[k][i][4] = fac1*lhs[k][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[k1][i][2] = lhs[k1][i][2] - lhs[k1][i][1]*lhs[k][i][3];
      lhs[k1][i][3] = lhs[k1][i][3] - lhs[k1][i][1]*lhs[k][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhs[k1][i][1]*rhs[k][j][i][m];
      }
      lhs[k2][i][1] = lhs[k2][i][1] - lhs[k2][i][0]*lhs[k][i][3];
      lhs[k2][i][2] = lhs[k2][i][2] - lhs[k2][i][0]*lhs[k][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhs[k2][i][0]*rhs[k][j][i][m];
      }
    }
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different, 
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;
  for (i = 1; i <= nx2; i++) {
    fac1 = 1.0/lhs[k][i][2];
    lhs[k][i][3] = fac1*lhs[k][i][3];
    lhs[k][i][4] = fac1*lhs[k][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
    }
    lhs[k1][i][2] = lhs[k1][i][2] - lhs[k1][i][1]*lhs[k][i][3];
    lhs[k1][i][3] = lhs[k1][i][3] - lhs[k1][i][1]*lhs[k][i][4];
    for (m = 0; m < 3; m++) {
      rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhs[k1][i][1]*rhs[k][j][i][m];
    }

    //---------------------------------------------------------------------
    // scale the last row immediately
    //---------------------------------------------------------------------
    fac2 = 1.0/lhs[k1][i][2];
    for (m = 0; m < 3; m++) {
      rhs[k1][j][i][m] = fac2*rhs[k1][j][i][m];
    }
  }

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors               
  //---------------------------------------------------------------------
  for (k = 0; k <= gp2-3; k++) {
    k1 = k + 1;
    k2 = k + 2;
    for (i = 1; i <= nx2; i++) {
      m = 3;
      fac1 = 1.0/lhsp[k][i][2];
      lhsp[k][i][3]    = fac1*lhsp[k][i][3];
      lhsp[k][i][4]    = fac1*lhsp[k][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[k1][i][2]   = lhsp[k1][i][2] - lhsp[k1][i][1]*lhsp[k][i][3];
      lhsp[k1][i][3]   = lhsp[k1][i][3] - lhsp[k1][i][1]*lhsp[k][i][4];
      rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsp[k1][i][1]*rhs[k][j][i][m];
      lhsp[k2][i][1]   = lhsp[k2][i][1] - lhsp[k2][i][0]*lhsp[k][i][3];
      lhsp[k2][i][2]   = lhsp[k2][i][2] - lhsp[k2][i][0]*lhsp[k][i][4];
      rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhsp[k2][i][0]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[k][i][2];
      lhsm[k][i][3]    = fac1*lhsm[k][i][3];
      lhsm[k][i][4]    = fac1*lhsm[k][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[k1][i][2]   = lhsm[k1][i][2] - lhsm[k1][i][1]*lhsm[k][i][3];
      lhsm[k1][i][3]   = lhsm[k1][i][3] - lhsm[k1][i][1]*lhsm[k][i][4];
      rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsm[k1][i][1]*rhs[k][j][i][m];
      lhsm[k2][i][1]   = lhsm[k2][i][1] - lhsm[k2][i][0]*lhsm[k][i][3];
      lhsm[k2][i][2]   = lhsm[k2][i][2] - lhsm[k2][i][0]*lhsm[k][i][4];
      rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhsm[k2][i][0]*rhs[k][j][i][m];
    }
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;
  for (i = 1; i <= nx2; i++) {
    m = 3;
    fac1 = 1.0/lhsp[k][i][2];
    lhsp[k][i][3]    = fac1*lhsp[k][i][3];
    lhsp[k][i][4]    = fac1*lhsp[k][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsp[k1][i][2]   = lhsp[k1][i][2] - lhsp[k1][i][1]*lhsp[k][i][3];
    lhsp[k1][i][3]   = lhsp[k1][i][3] - lhsp[k1][i][1]*lhsp[k][i][4];
    rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsp[k1][i][1]*rhs[k][j][i][m];

    m = 4;
    fac1 = 1.0/lhsm[k][i][2];
    lhsm[k][i][3]    = fac1*lhsm[k][i][3];
    lhsm[k][i][4]    = fac1*lhsm[k][i][4];
    rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
    lhsm[k1][i][2]   = lhsm[k1][i][2] - lhsm[k1][i][1]*lhsm[k][i][3];
    lhsm[k1][i][3]   = lhsm[k1][i][3] - lhsm[k1][i][1]*lhsm[k][i][4];
    rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsm[k1][i][1]*rhs[k][j][i][m];

    //---------------------------------------------------------------------
    // Scale the last row immediately (some of this is overkill
    // if this is the last cell)
    //---------------------------------------------------------------------
    rhs[k1][j][i][3] = rhs[k1][j][i][3]/lhsp[k1][i][2];
    rhs[k1][j][i][4] = rhs[k1][j][i][4]/lhsm[k1][i][2];
  }


  //---------------------------------------------------------------------
  // BACKSUBSTITUTION 
  //---------------------------------------------------------------------
  k  = gp2-2;
  k1 = gp2-1;
  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 3; m++) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[k][i][3]*rhs[k1][j][i][m];
    }

    rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[k][i][3]*rhs[k1][j][i][3];
    rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[k][i][3]*rhs[k1][j][i][4];
  }

  //---------------------------------------------------------------------
  // Whether or not this is the last processor, we always have
  // to complete the back-substitution 
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (k = gp2-3; k >= 0; k--) {
    k1 = k + 1;
    k2 = k + 2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - 
                          lhs[k][i][3]*rhs[k1][j][i][m] -
                          lhs[k][i][4]*rhs[k2][j][i][m];
      }

      //-------------------------------------------------------------------
      // And the remaining two
      //-------------------------------------------------------------------
      rhs[k][j][i][3] = rhs[k][j][i][3] - 
                        lhsp[k][i][3]*rhs[k1][j][i][3] -
                        lhsp[k][i][4]*rhs[k2][j][i][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - 
                        lhsm[k][i][3]*rhs[k1][j][i][4] -
                        lhsm[k][i][4]*rhs[k2][j][i][4];
    }
  }
#endif
}


//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__kernel void tzetar(__global double *g_u,
                     __global double *g_us,
                     __global double *g_vs,
                     __global double *g_ws,
                     __global double *g_qs,
                     __global double *g_speed,
                     __global double *g_rhs,
                     int nx2,
                     int ny2,
                     int nz2)
{
  int i, j, k;
  double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
  double btuz, ac2u, uzik1;

#if TZETAR_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  xvel = us[k][j][i];
  yvel = vs[k][j][i];
  zvel = ws[k][j][i];
  ac   = speed[k][j][i];

  ac2u = ac*ac;

  r1 = rhs[k][j][i][0];
  r2 = rhs[k][j][i][1];
  r3 = rhs[k][j][i][2];
  r4 = rhs[k][j][i][3];
  r5 = rhs[k][j][i][4];     

  uzik1 = u[k][j][i][0];
  btuz  = bt * uzik1;

  t1 = btuz/ac * (r4 + r5);
  t2 = r3 + t1;
  t3 = btuz * (r4 - r5);

  rhs[k][j][i][0] = t2;
  rhs[k][j][i][1] = -uzik1*r2 + xvel*t2;
  rhs[k][j][i][2] =  uzik1*r1 + yvel*t2;
  rhs[k][j][i][3] =  zvel*t2  + t3;
  rhs[k][j][i][4] =  uzik1*(-xvel*r2 + yvel*r1) + 
                     qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;

#elif TZETAR_DIM == 2
  k = get_global_id(1) + 1;
  j = get_global_id(0) + 1;
  if (k > nz2 || j > ny2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    xvel = us[k][j][i];
    yvel = vs[k][j][i];
    zvel = ws[k][j][i];
    ac   = speed[k][j][i];

    ac2u = ac*ac;

    r1 = rhs[k][j][i][0];
    r2 = rhs[k][j][i][1];
    r3 = rhs[k][j][i][2];
    r4 = rhs[k][j][i][3];
    r5 = rhs[k][j][i][4];     

    uzik1 = u[k][j][i][0];
    btuz  = bt * uzik1;

    t1 = btuz/ac * (r4 + r5);
    t2 = r3 + t1;
    t3 = btuz * (r4 - r5);

    rhs[k][j][i][0] = t2;
    rhs[k][j][i][1] = -uzik1*r2 + xvel*t2;
    rhs[k][j][i][2] =  uzik1*r1 + yvel*t2;
    rhs[k][j][i][3] =  zvel*t2  + t3;
    rhs[k][j][i][4] =  uzik1*(-xvel*r2 + yvel*r1) + 
                       qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
  }

#else //TZETAR_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

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
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      xvel = us[k][j][i];
      yvel = vs[k][j][i];
      zvel = ws[k][j][i];
      ac   = speed[k][j][i];

      ac2u = ac*ac;

      r1 = rhs[k][j][i][0];
      r2 = rhs[k][j][i][1];
      r3 = rhs[k][j][i][2];
      r4 = rhs[k][j][i][3];
      r5 = rhs[k][j][i][4];     

      uzik1 = u[k][j][i][0];
      btuz  = bt * uzik1;

      t1 = btuz/ac * (r4 + r5);
      t2 = r3 + t1;
      t3 = btuz * (r4 - r5);

      rhs[k][j][i][0] = t2;
      rhs[k][j][i][1] = -uzik1*r2 + xvel*t2;
      rhs[k][j][i][2] =  uzik1*r1 + yvel*t2;
      rhs[k][j][i][3] =  zvel*t2  + t3;
      rhs[k][j][i][4] =  uzik1*(-xvel*r2 + yvel*r1) + 
                         qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
    }
  }
#endif
}


__kernel void add(__global double *g_u,
                  __global double *g_rhs,
                  int nx2,
                  int ny2,
                  int nz2)
{
  int i, j, k, m;

#if ADD_DIM == 3
  k = get_global_id(2) + 1;
  j = get_global_id(1) + 1;
  i = get_global_id(0) + 1;
  if (k > nz2 || j > ny2 || i > nx2) return;

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
  if (k > nz2 || j > ny2) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (i = 1; i <= nx2; i++) {
    for (m = 0; m < 5; m++) {
      u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
    }
  }

#else //ADD_DIM == 1
  k = get_global_id(0) + 1;
  if (k > nz2) return;

  __global double (*u)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
  __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
      }
    }
  }

#endif
}

