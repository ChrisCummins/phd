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

#ifndef __LU_H__
#define __LU_H__

#if CLASS == 'S'
#define ISIZ1  12
#define ISIZ2  12
#define ISIZ3  12
#define ITMAX_DEFAULT  50
#define INORM_DEFAULT  50
#define DT_DEFAULT     0.5

#elif CLASS == 'W'
#define ISIZ1  33
#define ISIZ2  33
#define ISIZ3  33
#define ITMAX_DEFAULT  300
#define INORM_DEFAULT  300
#define DT_DEFAULT     1.5e-3

#elif CLASS == 'A'
#define ISIZ1  64
#define ISIZ2  64
#define ISIZ3  64
#define ITMAX_DEFAULT  250
#define INORM_DEFAULT  250
#define DT_DEFAULT     2.0

#elif CLASS == 'B'
#define ISIZ1  102
#define ISIZ2  102
#define ISIZ3  102
#define ITMAX_DEFAULT  250
#define INORM_DEFAULT  250
#define DT_DEFAULT     2.0

#elif CLASS == 'C'
#define ISIZ1  162
#define ISIZ2  162
#define ISIZ3  162
#define ITMAX_DEFAULT  250
#define INORM_DEFAULT  250
#define DT_DEFAULT     2.0

#elif CLASS == 'D'
#define ISIZ1  408
#define ISIZ2  408
#define ISIZ3  408
#define ITMAX_DEFAULT  300
#define INORM_DEFAULT  300
#define DT_DEFAULT     1.0

#elif CLASS == 'E'
#define ISIZ1  1020
#define ISIZ2  1020
#define ISIZ3  1020
#define ITMAX_DEFAULT  300
#define INORM_DEFAULT  300
#define DT_DEFAULT     0.5

#else
#error "Unknown class!"
#endif

//---------------------------------------------------------------------
// parameters which can be overridden in runtime config file
// isiz1,isiz2,isiz3 give the maximum size
// ipr = 1 to print out verbose information
// omega = 2.0 is correct for all classes
// tolrsd is tolerance levels for steady state residuals
//---------------------------------------------------------------------
#define IPR_DEFAULT     1
#define OMEGA_DEFAULT   1.2
#define TOLRSD1_DEF     1.0e-08
#define TOLRSD2_DEF     1.0e-08
#define TOLRSD3_DEF     1.0e-08
#define TOLRSD4_DEF     1.0e-08
#define TOLRSD5_DEF     1.0e-08

#define C1              1.40e+00
#define C2              0.40e+00
#define C3              1.00e-01
#define C4              1.00e+00
#define C5              1.40e+00


//---------------------------------------------------------------------
// from setcoeff()
//---------------------------------------------------------------------
#define nx0     ISIZ1
#define ny0     ISIZ2
#define nz0     ISIZ3

#define ist     1
#define iend    (nx - 1)
#define jst     1
#define jend    (ny - 1)

#define dt      DT_DEFAULT
#define omega   OMEGA_DEFAULT

//---------------------------------------------------------------------
// set up coefficients
//---------------------------------------------------------------------
#define dxi     (1.0 / ( nx0 - 1 ))
#define deta    (1.0 / ( ny0 - 1 ))
#define dzeta   (1.0 / ( nz0 - 1 ))

#define tx1     (1.0 / ( dxi * dxi ))
#define tx2     (1.0 / ( 2.0 * dxi ))
#define tx3     (1.0 / dxi)

#define ty1     (1.0 / ( deta * deta ))
#define ty2     (1.0 / ( 2.0 * deta ))
#define ty3     (1.0 / deta)

#define tz1     (1.0 / ( dzeta * dzeta ))
#define tz2     (1.0 / ( 2.0 * dzeta ))
#define tz3     (1.0 / dzeta)

//---------------------------------------------------------------------
// diffusion coefficients
//---------------------------------------------------------------------
#define dx1     0.75
#define dx2     dx1
#define dx3     dx1
#define dx4     dx1
#define dx5     dx1

#define dy1     0.75
#define dy2     dy1
#define dy3     dy1
#define dy4     dy1
#define dy5     dy1

#define dz1     1.00
#define dz2     dz1
#define dz3     dz1
#define dz4     dz1
#define dz5     dz1

//---------------------------------------------------------------------
// fourth difference dissipation
//---------------------------------------------------------------------
#define dssp    (( max(max(dx1, dy1), dz1) ) / 4.0)
//---------------------------------------------------------------------


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#else
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
#endif

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif

#include "lu_dim.h"

/* Below constants should be the same as those in lu.c!! */
#ifdef USE_CPU
#define SETBV1_DIM      SETBV1_DIM_CPU
#define SETBV2_DIM      SETBV2_DIM_CPU
#define SETBV3_DIM      SETBV3_DIM_CPU
#define SETIV_DIM       SETIV_DIM_CPU
#define ERHS1_DIM       ERHS1_DIM_CPU
#define ERHS2_DIM       ERHS2_DIM_CPU
#define ERHS3_DIM       ERHS3_DIM_CPU
#define ERHS4_DIM       ERHS4_DIM_CPU
#define PINTGR1_DIM     PINTGR1_DIM_CPU
#define PINTGR2_DIM     PINTGR2_DIM_CPU
#define PINTGR3_DIM     PINTGR3_DIM_CPU
#define RHS_DIM         RHS_DIM_CPU
#define RHSX_DIM        RHSX_DIM_CPU
#define RHSY_DIM        RHSY_DIM_CPU
#define RHSZ_DIM        RHSZ_DIM_CPU
#define SSOR2_DIM       SSOR2_DIM_CPU
#define SSOR3_DIM       SSOR3_DIM_CPU

#else //GPU
#define SETBV1_DIM      SETBV1_DIM_GPU
#define SETBV2_DIM      SETBV2_DIM_GPU
#define SETBV3_DIM      SETBV3_DIM_GPU
#define SETIV_DIM       SETIV_DIM_GPU
#define ERHS1_DIM       ERHS1_DIM_GPU
#define ERHS2_DIM       ERHS2_DIM_GPU
#define ERHS3_DIM       ERHS3_DIM_GPU
#define ERHS4_DIM       ERHS4_DIM_GPU
#define PINTGR1_DIM     PINTGR1_DIM_GPU
#define PINTGR2_DIM     PINTGR2_DIM_GPU
#define PINTGR3_DIM     PINTGR3_DIM_GPU
#define RHS_DIM         RHS_DIM_GPU
#define RHSX_DIM        RHSX_DIM_GPU
#define RHSY_DIM        RHSY_DIM_GPU
#define RHSZ_DIM        RHSZ_DIM_GPU
#define SSOR2_DIM       SSOR2_DIM_GPU
#define SSOR3_DIM       SSOR3_DIM_GPU

#endif


#endif //__LU_H__

