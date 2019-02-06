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

#ifndef __LU_DIM_H__
#define __LU_DIM_H__

#define SETBV1_DIM_CPU    2
#define SETBV2_DIM_CPU    2
#define SETBV3_DIM_CPU    2
#define SETIV_DIM_CPU     2
#define ERHS1_DIM_CPU     3
#define ERHS2_DIM_CPU     2
#define ERHS3_DIM_CPU     2
#define ERHS4_DIM_CPU     2
#define PINTGR1_DIM_CPU   1
#define PINTGR2_DIM_CPU   1
#define PINTGR3_DIM_CPU   1
#define RHS_DIM_CPU       1
#define RHSX_DIM_CPU      1
#define RHSY_DIM_CPU      1
#define RHSZ_DIM_CPU      1
#define SSOR2_DIM_CPU     1
#define SSOR3_DIM_CPU     1

#define SETBV1_DIM_GPU    2
#define SETBV2_DIM_GPU    3
#define SETBV3_DIM_GPU    3
#define SETIV_DIM_GPU     3
#define ERHS1_DIM_GPU     3
#define ERHS2_DIM_GPU     2
#define ERHS3_DIM_GPU     2
#define ERHS4_DIM_GPU     2
#define PINTGR1_DIM_GPU   2
#define PINTGR2_DIM_GPU   2
#define PINTGR3_DIM_GPU   2
#define RHS_DIM_GPU       3
#define RHSX_DIM_GPU      2
#define RHSY_DIM_GPU      2
#define RHSZ_DIM_GPU      2
#define SSOR2_DIM_GPU     3
#define SSOR3_DIM_GPU     3

#endif //__LU_DIM_H__

