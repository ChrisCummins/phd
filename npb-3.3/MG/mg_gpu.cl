//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB MG code. This OpenCL    //
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

#include "mg.h"
#include "mg_dim.h"

#define PSINV_DIM       PSINV_DIM_GPU
#define RESID_DIM       RESID_DIM_GPU
#define RPRJ3_DIM       RPRJ3_DIM_GPU
#define INTERP_1_DIM    INTERP_1_DIM_GPU
#define NORM2U3_DIM     NORM2U3_DIM_GPU
#define COMM3_1_DIM     COMM3_1_DIM_GPU
#define COMM3_2_DIM     COMM3_2_DIM_GPU
#define COMM3_3_DIM     COMM3_3_DIM_GPU
#define ZERO3_DIM       ZERO3_DIM_GPU


__kernel void kernel_zero3(__global double *z,
                           int n1, int n2, int n3, int offset)
{
#if ZERO3_DIM == 3
  int i3 = get_global_id(2);
  int i2 = get_global_id(1);
  int i1 = get_global_id(0);
  if (i1 >= n1) return;

  z[i3*n2*n1 + i2*n1 + i1 + offset] = 0.0;

#elif ZERO3_DIM == 2
  int i3 = get_global_id(1);
  int i2 = get_global_id(0);
  if (i2 >= n2) return;

  for (int i1 = 0; i1 < n1; i1++) {
    z[i3*n2*n1 + i2*n1 + i1 + offset] = 0.0;
  }

#elif ZERO3_DIM == 1
  int i3 = get_global_id(0);
  if (i3 >= n3) return;

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      z[i3*n2*n1 + i2*n1 + i1 + offset] = 0.0;
    }
  }

#else
#error "ERROR: ZERO3_DIM"
#endif
}


__kernel void kernel_comm3_1(__global double *u,
                             int n1, int n2, int n3, int offset)
{
#if COMM3_1_DIM == 2
  int i3 = get_global_id(1) + 1;
  int i2 = get_global_id(0) + 1;
  if (i2 >= n2-1) return;

  u[i3*n2*n1+i2*n1+0+offset]    = u[i3*n2*n1+i2*n1+n1-2+offset];
  u[i3*n2*n1+i2*n1+n1-1+offset] = u[i3*n2*n1+i2*n1+1+offset];

#elif COMM3_1_DIM == 1
  int i3 = get_group_id(0) + 1;
  int i2 = get_local_id(0) + 1;

  while (i2 < n2-1) {
    u[i3*n2*n1+i2*n1+0+offset]    = u[i3*n2*n1+i2*n1+n1-2+offset];
    u[i3*n2*n1+i2*n1+n1-1+offset] = u[i3*n2*n1+i2*n1+1+offset];

    i2 += get_local_size(0);
  }

#else
#error "ERROR: COMM3_1_dIM"
#endif
}

__kernel void kernel_comm3_2(__global double *u,
                             int n1, int n2, int n3, int offset)
{
#if COMM3_2_DIM == 2
  int i3 = get_global_id(1) + 1;
  int i1 = get_global_id(0);
  if (i1 >= n1) return;

  u[i3*n2*n1+0*n1+i1+offset]      = u[i3*n2*n1+(n2-2)*n1+i1+offset];
  u[i3*n2*n1+(n2-1)*n1+i1+offset] = u[i3*n2*n1+1*n1+i1+offset];

#elif COMM3_2_DIM == 1
  int i3 = get_group_id(0) + 1;
  int i1 = get_local_id(0);

  while (i1 < n1) {
    u[i3*n2*n1+0*n1+i1+offset]      = u[i3*n2*n1+(n2-2)*n1+i1+offset];
    u[i3*n2*n1+(n2-1)*n1+i1+offset] = u[i3*n2*n1+1*n1+i1+offset];

    i1 += get_local_size(0);
  }

#else
#error "ERROR: COMM3_2_dIM"
#endif
}

__kernel void kernel_comm3_3(__global double *u,
                             int n1, int n2, int n3, int offset)
{
#if COMM3_3_DIM == 2
  int i2 = get_global_id(1);
  int i1 = get_global_id(0);
  if (i1 >= n1) return;

  u[   0*n2*n1+i2*n1+i1+offset]   = u[(n3-2)*n2*n1+i2*n1+i1+offset];
  u[(n3-1)*n2*n1+i2*n1+i1+offset] = u[   1*n2*n1+i2*n1+i1+offset];

#elif COMM3_3_DIM == 1
  int i2 = get_group_id(0);
  int i1 = get_local_id(0);

  while (i1 < n1) {
    u[   0*n2*n1+i2*n1+i1+offset]   = u[(n3-2)*n2*n1+i2*n1+i1+offset];
    u[(n3-1)*n2*n1+i2*n1+i1+offset] = u[   1*n2*n1+i2*n1+i1+offset];

    i1 += get_local_size(0);
  }

#else
#error "ERROR: COMM3_3_dIM"
#endif
}


__kernel void kernel_zran3_1(__global double* oz,
			                 __global double* starts,
			                 int n1,
			                 int n2,
			                 int n3,
			                 int offset,
			                 int e2,
			                 int e3,
			                 int d1,
			                 double a1)
{

	int i3 = get_global_id(0);
	int i2;
	double xx, x1;
	const double a = pow(5.0, 13.0);

	if (i3 < e3 && i3 >= 1)
	{
		x1 = starts[i3];
		for (i2 = 1; i2 < e2; i2++) {
		  xx = x1;
		  vranlc(d1, &xx, a, &(oz[i3*n2*n1+i2*n1+1]));
		  randlc(&x1, a1);
		}
	}

}


__kernel void kernel_psinv(__global double *r,
                           __global double *u,
                           __global double *c,
                           int n1,
                           int n2,
                           int n3,
                           int offset)
{
#if PSINV_DIM == 2
  int i3 = get_global_id(1) + 1;
  int i2 = get_group_id(0) + 1;
  int lid = get_local_id(0);
#elif PSINV_DIM == 1
  int i3 = get_group_id(0) / (n2-2) + 1;
  int i2 = get_group_id(0) % (n2-2) + 1;
  int lid = get_local_id(0);
#else
#error "ERROR: PSINV_DIM"
#endif

  __local double r1[M], r2[M];

  int i1;
  for (i1 = lid; i1 < n1; i1 += get_local_size(0)) {
    r1[i1] = r[i3*n2*n1+(i2-1)*n2+i1+offset]
           + r[i3*n2*n1+(i2+1)*n1+i1+offset]
           + r[(i3-1)*n2*n1+i2*n1+i1+offset]
           + r[(i3+1)*n2*n1+i2*n1+i1+offset];
    r2[i1] = r[(i3-1)*n2*n1+(i2-1)*n1+i1+offset]
           + r[(i3-1)*n2*n1+(i2+1)*n1+i1+offset]
           + r[(i3+1)*n2*n1+(i2-1)*n1+i1+offset]
           + r[(i3+1)*n2*n1+(i2+1)*n1+i1+offset];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i1 = lid+1; i1 < n1-1; i1 += get_local_size(0)) {
    u[i3*n2*n1+i2*n1+i1+offset] = u[i3*n2*n1+i2*n1+i1+offset]
      + c[0] * r[i3*n2*n1+i2*n1+i1+offset]
      + c[1] * ( r[i3*n2*n1+i2*n1+i1-1+offset]
               + r[i3*n2*n1+i2*n1+i1+1+offset]
               + r1[i1] )
      + c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] );
  }
}


__kernel void kernel_resid(__global double *r,
                           __global double *u,
                           __global double *v,
                           __global double *a,
                           int n1,
                           int n2,
                           int n3,
                           int offset)
{
#if RESID_DIM == 2
  int i3 = get_global_id(1) + 1;
  int i2 = get_group_id(0) + 1;
  int lid = get_local_id(0);
#elif RESID_DIM == 1
  int i3 = get_group_id(0) / (n2-2) + 1;
  int i2 = get_group_id(0) % (n2-2) + 1;
  int lid = get_local_id(0);
#else
#error "ERROR: RESID_DIM"
#endif

  __local double u1[M], u2[M];

  int i1;
  for (i1 = lid; i1 < n1; i1 += get_local_size(0)) {
    u1[i1] = u[i3*n2*n1+(i2-1)*n1+i1+offset]
           + u[i3*n2*n1+(i2+1)*n1+i1+offset]
           + u[(i3-1)*n2*n1+i2*n1+i1+offset]
           + u[(i3+1)*n2*n1+i2*n1+i1+offset];
    u2[i1] = u[(i3-1)*n2*n1+(i2-1)*n1+i1+offset]
           + u[(i3-1)*n2*n1+(i2+1)*n1+i1+offset]
           + u[(i3+1)*n2*n1+(i2-1)*n1+i1+offset]
           + u[(i3+1)*n2*n1+(i2+1)*n1+i1+offset];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i1 = lid+1; i1 < n1-1; i1 += get_local_size(0)) {
    r[i3*n2*n1+i2*n1+i1+offset] = v[i3*n2*n1+i2*n1+i1+offset]
      - a[0] * u[i3*n2*n1+i2*n1+i1+offset]
      - a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
      - a[3] * ( u2[i1-1] + u2[i1+1] );
  }
}


//A little tricky here, we know this kernel only involves buffer r, just different offsets
__kernel void kernel_rprj3(__global double *base_r,
                           int m1k, int m2k, int m3k,
                           int m1j, int m2j, int m3j,
                           int offset_r1, int offset_r2,
                           int d1, int d2, int d3)
{
  int j3, j2, j1, i3, i2, i1, j;
  double x2, y2;
  __local double x1[M], y1[M];

  __global double *r = base_r + offset_r1;
  __global double *s = base_r + offset_r2;

#if RPRJ3_DIM == 2
  j3 = get_global_id(1) + 1;
  j2 = get_group_id(0) + 1;
  j1 = get_local_id(0) + 1;
#elif RPRJ3_DIM == 1
  j3 = get_group_id(0) / (m2j-2) + 1;
  j2 = get_group_id(0) % (m2j-2) + 1;
  j1 = get_local_id(0) + 1;
#else
#error "ERROR: RPRJ3_DIM"
#endif

  i3 = 2*j3-d3;
  i2 = 2*j2-d2;
  i1 = 2*j1-d1;
  x1[i1] = r[(i3+1)*m2k*m1k+i2*m1k+i1]
         + r[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
         + r[i3*m2k*m1k+(i2+1)*m1k+i1]
         + r[(i3+2)*m2k*m1k+(i2+1)*m1k+i1];
  y1[i1] = r[i3*m2k*m1k+i2*m1k+i1]
         + r[(i3+2)*m2k*m1k+i2*m1k+i1]
         + r[i3*m2k*m1k+(i2+2)*m1k+i1]
         + r[(i3+2)*m2k*m1k+(i2+2)*m1k+i1];

  barrier(CLK_LOCAL_MEM_FENCE);

  if (j1 < m1j-1)
  {
    i1 = 2*j1-d1;
    y2 = r[i3*m2k*m1k+i2*m1k+i1+1]
       + r[(i3+2)*m2k*m1k+i2*m1k+i1+1]
       + r[i3*m2k*m1k+(i2+2)*m1k+i1+1]
       + r[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1];
    x2 = r[(i3+1)*m2k*m1k+i2*m1k+i1+1]
       + r[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
       + r[i3*m2k*m1k+(i2+1)*m1k+i1+1]
       + r[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1];
    s[j3*m2j*m1j+j2*m1j+j1] =
        0.5 * r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
      + 0.25 * (r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
              + r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2] + x2)
      + 0.125 * (x1[i1] + x1[i1+2] + y2)
      + 0.0625 * (y1[i1] + y1[i1+2]);
  }
}


__kernel void kernel_interp_1(__global double *base_u,
                              int mm1, int mm2, int mm3,
                              int n1, int n2, int n3,
                              int offset_u1, int offset_u2)
{
  int i3, i2, i1;
  __local double z1[M], z2[M], z3[M];

  __global double *z = base_u + offset_u1;
  __global double *u = base_u + offset_u2;

#if INTERP_1_DIM == 2
  i3 = get_global_id(1);
  i2 = get_group_id(0);
  i1 = get_local_id(0);
#elif INTERP_1_DIM == 1
  i3 = get_group_id(0) / (mm2-1);
  i2 = get_group_id(0) % (mm2-1);
  i1 = get_local_id(0);
#else
#error "ERROR: INTERP_1_DIM"
#endif

  z1[i1] = z[i3*mm2*mm1+(i2+1)*mm1+i1] + z[i3*mm2*mm1+i2*mm1+i1];
  z2[i1] = z[(i3+1)*mm2*mm1+i2*mm1+i1] + z[i3*mm2*mm1+i2*mm1+i1];
  z3[i1] = z[(i3+1)*mm2*mm1+(i2+1)*mm1+i1] 
         + z[(i3+1)*mm2*mm1+i2*mm1+i1] + z1[i1];

  barrier(CLK_LOCAL_MEM_FENCE);

  if (i1 < mm1-1)
  {
    double z321 = z[i3*mm2*mm1+i2*mm1+i1];
    u[2*i3*n2*n1+2*i2*n1+2*i1] += z321;
    u[2*i3*n2*n1+2*i2*n1+2*i1+1] += 0.5 * (z[i3*mm2*mm1+i2*mm1+i1+1] + z321);

    u[2*i3*n2*n1+(2*i2+1)*n1+2*i1] += 0.5 * z1[i1];
    u[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1] += 0.25 * (z1[i1] + z1[i1+1]);

    u[(2*i3+1)*n2*n1+2*i2*n1+2*i1] += 0.5 * z2[i1];
    u[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1] += 0.25 * (z2[i1] + z2[i1+1]);

    u[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1] += 0.25 * z3[i1];
    u[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1] += 0.125 * (z3[i1] + z3[i1+1]);
  }
}


__kernel void kernel_interp_2(__global double* base_o_u,
							int mm1, int mm2, int mm3,
							int n1, int n2, int n3,
							int offset_u1, int offset_u2,
							int d1, int d2, int d3,
							int t1, int t2, int t3
			                 )
{
	int i3, i2, i1;
	__global double* oz = base_o_u + offset_u1;
	__global double* ou = base_o_u + offset_u2;

	i3 = get_global_id(0);
	i2 = get_global_id(1);

	if (i3 >= d3 && i3 <= mm3-1 && i2 >= d2 && i2 <= mm2-1)
	{
		for (i1 = d1; i1 <= mm1-1; i1++) {
			ou[(2*i3-d3-1)*n2*n1+(2*i2-d2-1)*n1+2*i1-d1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-d2-1)*n1+2*i1-d1-1]
			   + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1];
		}
		for (i1 = 1; i1 <= mm1-1; i1++) {
			ou[(2*i3-d3-1)*n2*n1+(2*i2-d2-1)*n1+2*i1-t1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-d2-1)*n1+2*i1-t1-1]
			+ 0.5 * (oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
	}
}

__kernel void kernel_interp_3(__global double* base_o_u,
							int mm1, int mm2, int mm3,
							int n1, int n2, int n3,
							int offset_u1, int offset_u2,
							int d1, int d2, int d3,
							int t1, int t2, int t3
			                 )
{
	int i3, i2, i1;
	__global double* oz = base_o_u + offset_u1;
	__global double* ou = base_o_u + offset_u2;

	i3 = get_global_id(0);
	i2 = get_global_id(1);

	if (i3 >= d3 && i3 <= mm3-1 && i2 >= 1 && i2 <= mm2-1)
	{
		for (i1 = d1; i1 <= mm1-1; i1++) {
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1]
			+ 0.5 * (oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
		for (i1 = 1; i1 <= mm1-1; i1++) {
		  ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1]
			+ 0.25 * (oz[(i3-1)*mm2*mm1+i2*mm1+i1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1]
					+ oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
	}
}

__kernel void kernel_interp_4(__global double* base_o_u,
							int mm1, int mm2, int mm3,
							int n1, int n2, int n3,
							int offset_u1, int offset_u2,
							int d1, int d2, int d3,
							int t1, int t2, int t3
			                 )
{
	int i3, i2, i1;
	__global double* oz = base_o_u + offset_u1;
	__global double* ou = base_o_u + offset_u2;

	i3 = get_global_id(0);
	i2 = get_global_id(1);

	if (i3 >= 1 && i3 <= mm3-1 && i2 >= d2 && i2 <= mm2-1)
	{
		for (i1 = d1; i1 <= mm1-1; i1++) {
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1]
			+ 0.5 * (oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
		for (i1 = 1; i1 <= mm1-1; i1++) {
		  ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1] =
			ou[(2*i3-d3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1]
			+ 0.25 * (oz[(i3-1)*mm2*mm1+i2*mm1+i1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1]
					+ oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}

	}
}

__kernel void kernel_interp_5(__global double* base_o_u,
							int mm1, int mm2, int mm3,
							int n1, int n2, int n3,
							int offset_u1, int offset_u2,
							int d1, int d2, int d3,
							int t1, int t2, int t3
			                 )
{
	int i3, i2, i1;
	__global double* oz = base_o_u + offset_u1;
	__global double* ou = base_o_u + offset_u2;

	i3 = get_global_id(0);
	i2 = get_global_id(1);

	if (i3 >= 1 && i3 <= mm3-1 && i2 >= 1 && i2 <= mm2-1)
	{
		for (i1 = d1; i1 <= mm1-1; i1++) {
			ou[(2*i3-t3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1] =
				ou[(2*i3-t3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-d1-1]
			+ 0.25 * (oz[i3*mm2*mm1+i2*mm1+i1-1] + oz[i3*mm2*mm1+(i2-1)*mm1+i1-1]
					+ oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
		for (i1 = 1; i1 <= mm1-1; i1++) {
			ou[(2*i3-t3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1] =
				ou[(2*i3-t3-1)*n2*n1+(2*i2-t2-1)*n1+2*i1-t1-1]
			+ 0.125 * (oz[i3*mm2*mm1+i2*mm1+i1  ] + oz[i3*mm2*mm1+(i2-1)*mm1+i1  ]
					 + oz[i3*mm2*mm1+i2*mm1+i1-1] + oz[i3*mm2*mm1+(i2-1)*mm1+i1-1]
					 + oz[(i3-1)*mm2*mm1+i2*mm1+i1  ] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1  ]
					 + oz[(i3-1)*mm2*mm1+i2*mm1+i1-1] + oz[(i3-1)*mm2*mm1+(i2-1)*mm1+i1-1]);
		}
	}
}

__kernel
void kernel_norm2u3(__global double *r,
                    const int n1, const int n2, const int n3,
                    __global double *res_sum,
                    __global double *res_max,
                    __local double *scratch_sum,
                    __local double *scratch_max)
{
#if NORM2U3_DIM == 2
  int i3 = get_global_id(1) + 1;
  int i2 = get_group_id(0) + 1;
  int i1 = get_local_id(0) + 1;
#elif NORM2U3_DIM == 1
  int i3 = get_group_id(0) / (n2-2) + 1;
  int i2 = get_group_id(0) % (n2-2) + 1;
  int i1 = get_local_id(0) + 1;
#else
#error "ERROR: NORM2U3_DIM"
#endif

  double s = 0.0;
  double my_rnmu = 0.0;
  double a;
  while (i1 < n1-1) {
    double r321 = r[i3*n2*n1+i2*n1+i1];
    //s = s + pow(r321, 2.0);
    s = s + r321 * r321;
    a = fabs(r321);
    my_rnmu = (a > my_rnmu) ? a : my_rnmu;

    i1 += get_local_size(0);
  }
  int lid = get_local_id(0);
  scratch_sum[lid] = s;
  scratch_max[lid] = my_rnmu;

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction in a work-group
  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      scratch_sum[lid] += scratch_sum[lid + i];
      scratch_max[lid] = (scratch_max[lid] > scratch_max[lid + i])
                       ? scratch_max[lid] : scratch_max[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    int idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    res_sum[idx] = scratch_sum[0];
    res_max[idx] = scratch_max[0];
  }
}


#define mm 10

__kernel void kernel_zran3_2(__global double* base_oz,
							__global double *g_ten,
							__global int *g_j1,
							__global int *g_j2,
							__global int *g_j3,
							int n1, int n2, int n3,
							int offset
							)
{
	int i1, i2, i3;
	int i;
	__global double *oz = base_oz + offset;
	i3 = get_global_id(0);
	if (i3 >= n3) return;

	__global double (*ten)[mm][2] = (__global double(*)[mm][2])g_ten;
	__global int (*j1)[mm][2] = (__global int(*)[mm][2])g_j1;
	__global int (*j2)[mm][2] = (__global int(*)[mm][2])g_j2;
	__global int (*j3)[mm][2] = (__global int(*)[mm][2])g_j3;

	  for (i = 0; i < mm; i++) {
	    ten[i3][i][1] = 0.0;
	    j1[i3][i][1] = 0;
	    j2[i3][i][1] = 0;
	    j3[i3][i][1] = 0;
	    ten[i3][i][0] = 1.0;
	    j1[i3][i][0] = 0;
	    j2[i3][i][0] = 0;
	    j3[i3][i][0] = 0;
	  }

	  i3 = i3+1;
  for (i2 = 1; i2 < n2-1; i2++)
	for (i1 = 1; i1 < n1-1; i1++)
	if (i3 >= 1 && i3 < n3-1)
	{
		__global double *zi3 = oz + i3*n2*n1;

		if (zi3[i2*n1+i1] > ten[i3-1][0][1]) {
		  ten[i3-1][0][1] = zi3[i2*n1+i1];
		  j1[i3-1][0][1] = i1;
		  j2[i3-1][0][1] = i2;
		  j3[i3-1][0][1] = i3;
		  bubble(ten[i3-1], j1[i3-1], j2[i3-1], j3[i3-1], mm, 1);
		}
		if (zi3[i2*n1+i1] < ten[i3-1][0][0]) {
		  ten[i3-1][0][0] = zi3[i2*n1+i1];
		  j1[i3-1][0][0] = i1;
		  j2[i3-1][0][0] = i2;
		  j3[i3-1][0][0] = i3;
		  bubble(ten[i3-1], j1[i3-1], j2[i3-1], j3[i3-1], mm, 0);
		}
	}

}


__kernel void kernel_zran3_3(__global double* oz,
							 int n1, int n2, int n3, int offset)
{
	int i1, i2, i3;
	i3 = get_global_id(0);
	i2 = get_global_id(1);
	i1 = get_global_id(2);
	if (i3 < n3 && i2 < n2 && i1 < n1) oz[i3*n2*n1+i2*n1+i1 + offset] = 0.0;
}

