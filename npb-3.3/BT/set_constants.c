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

#include "header.h"

void set_constants()
{
  ce[0][0]  = 2.0;
  ce[0][1]  = 0.0;
  ce[0][2]  = 0.0;
  ce[0][3]  = 4.0;
  ce[0][4]  = 5.0;
  ce[0][5]  = 3.0;
  ce[0][6]  = 0.5;
  ce[0][7]  = 0.02;
  ce[0][8]  = 0.01;
  ce[0][9]  = 0.03;
  ce[0][10] = 0.5;
  ce[0][11] = 0.4;
  ce[0][12] = 0.3;

  ce[1][0]  = 1.0;
  ce[1][1]  = 0.0;
  ce[1][2]  = 0.0;
  ce[1][3]  = 0.0;
  ce[1][4]  = 1.0;
  ce[1][5]  = 2.0;
  ce[1][6]  = 3.0;
  ce[1][7]  = 0.01;
  ce[1][8]  = 0.03;
  ce[1][9]  = 0.02;
  ce[1][10] = 0.4;
  ce[1][11] = 0.3;
  ce[1][12] = 0.5;

  ce[2][0]  = 2.0;
  ce[2][1]  = 2.0;
  ce[2][2]  = 0.0;
  ce[2][3]  = 0.0;
  ce[2][4]  = 0.0;
  ce[2][5]  = 2.0;
  ce[2][6]  = 3.0;
  ce[2][7]  = 0.04;
  ce[2][8]  = 0.03;
  ce[2][9]  = 0.05;
  ce[2][10] = 0.3;
  ce[2][11] = 0.5;
  ce[2][12] = 0.4;

  ce[3][0]  = 2.0;
  ce[3][1]  = 2.0;
  ce[3][2]  = 0.0;
  ce[3][3]  = 0.0;
  ce[3][4]  = 0.0;
  ce[3][5]  = 2.0;
  ce[3][6]  = 3.0;
  ce[3][7]  = 0.03;
  ce[3][8]  = 0.05;
  ce[3][9] = 0.04;
  ce[3][10] = 0.2;
  ce[3][11] = 0.1;
  ce[3][12] = 0.3;

  ce[4][0]  = 5.0;
  ce[4][1]  = 4.0;
  ce[4][2]  = 3.0;
  ce[4][3]  = 2.0;
  ce[4][4]  = 0.1;
  ce[4][5]  = 0.4;
  ce[4][6]  = 0.3;
  ce[4][7]  = 0.05;
  ce[4][8]  = 0.04;
  ce[4][9] = 0.03;
  ce[4][10] = 0.1;
  ce[4][11] = 0.3;
  ce[4][12] = 0.2;

  //------------------------------------------------------------------------
  cl_int ecode;
  ecode = clEnqueueWriteBuffer(cmd_queue,
                               m_ce,
                               CL_FALSE,
                               0, sizeof(double)*5*13,
                               ce,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer() for m_ce");
  //------------------------------------------------------------------------
}
