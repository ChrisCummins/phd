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
                    __constant double *g_ce)
{
  int m;
  __constant double (*ce)[13] = (__constant double (*)[13])g_ce;

  for (m = 0; m < 5; m++) {
    dtemp[m] =  ce[m][0] +
      xi*(ce[m][1] + xi*(ce[m][4] + xi*(ce[m][7] + xi*ce[m][10]))) +
      eta*(ce[m][2] + eta*(ce[m][5] + eta*(ce[m][8] + eta*ce[m][11])))+
      zeta*(ce[m][3] + zeta*(ce[m][6] + zeta*(ce[m][9] + 
      zeta*ce[m][12])));
  }
}


__kernel void error_norm(__global double *g_u,
                         __constant double *g_ce,
                         __global double *g_rms,
                         __local double *l_rms,
                         int gp0,
                         int gp1,
                         int gp2)
{
  int i, j, k, m, lid;
  double xi, eta, zeta, u_exact[5], add;
  __local double *rms_local;

  k = get_global_id(0) + 1;
  lid = get_local_id(0);
  rms_local = &l_rms[lid * 5];

  for (m = 0; m < 5; m++) {
    rms_local[m] = 0.0;
  }

  if (k < gp2) {
    __global double (*u)[JMAXP+1][IMAXP+1][5] = 
      (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;

    zeta = (double)k * dnzm1;
    for (j = 0; j < gp1; j++) {
      eta = (double)j * dnym1;
      for (i = 0; i < gp0; i++) {
        xi = (double)i * dnxm1;
        exact_solution(xi, eta, zeta, u_exact, g_ce);

        for (m = 0; m < 5; m++) {
          add = u[k][j][i][m]-u_exact[m];
          rms_local[m] = rms_local[m] + add*add;
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < get_local_size(0); i++) {
      __local double *rms_other = &l_rms[i * 5];
      for (m = 0; m < 5; m++) {
        rms_local[m] += rms_other[m];
      }
    }

    __global double *rms = &g_rms[get_group_id(0) * 5];
    for (m = 0; m < 5; m++) {
      rms[m] = rms_local[m];
    }
  }
}


__kernel void rhs_norm(__global double *g_rhs,
                       __global double *g_rms,
                       __local double *l_rms,
                       int gp0,
                       int gp1,
                       int gp2)
{
  int i, j, k, m, lid;
  double add;
  __local double *rms_local;

  k = get_global_id(0) + 1;
  lid = get_local_id(0);
  rms_local = &l_rms[lid * 5];

  for (m = 0; m < 5; m++) {
    rms_local[m] = 0.0;
  }

  if (k <= (gp2-2)) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = 
      (__global double (*)[JMAXP+1][IMAXP+1][5])g_rhs;

    for (j = 1; j <= (gp1-2); j++) {
      for (i = 1; i <= (gp0-2); i++) {
        for (m = 0; m < 5; m++) {
          add = rhs[k][j][i][m];
          rms_local[m] = rms_local[m] + add*add;
        } 
      } 
    } 
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < get_local_size(0); i++) {
      __local double *rms_other = &l_rms[i * 5];
      for (m = 0; m < 5; m++) {
        rms_local[m] += rms_other[m];
      }
    }

    __global double *rms = &g_rms[get_group_id(0) * 5];
    for (m = 0; m < 5; m++) {
      rms[m] = rms_local[m];
    }
  }
}
