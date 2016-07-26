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

#ifndef __CG_H__
#define __CG_H__

//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------

#ifndef CLASS
#error "CLASS is not defined"
#endif

//----------
//  Class S:
//----------
#if CLASS == 'S'
#define NA        1400
#define NONZER    7
#define SHIFT     10
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class W:
//----------
#if CLASS == 'W'
#define NA        7000
#define NONZER    8
#define SHIFT     12
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class A:
//----------
#if CLASS == 'A'
#define NA        14000
#define NONZER    11
#define SHIFT     20
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class B:
//----------
#if CLASS == 'B'
#define NA        75000
#define NONZER    13
#define SHIFT     60
#define NITER     75
#define RCOND     1.0e-1
#endif

//----------
//  Class C:
//----------
#if CLASS == 'C'
#define NA        150000
#define NONZER    15
#define SHIFT     110
#define NITER     75
#define RCOND     1.0e-1
#endif

//----------
//  Class D:
//----------
#if CLASS == 'D'
#define NA        1500000
#define NONZER    21
#define SHIFT     500
#define NITER     100
#define RCOND     1.0e-1
#endif

//----------
//  Class E:
//----------
#if CLASS == 'E'
#define NA        9000000
#define NONZER    26
#define SHIFT     1500
#define NITER     100
#define RCOND     1.0e-1
#endif

#define NZ    (NA*(NONZER+1)*(NONZER+1))
#define NAZ   (NA*(NONZER+1))

#define TRUE    1
#define FALSE   0

typedef bool logical;

#endif //__CG_H__

