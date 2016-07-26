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
#include "timers.h"

/* CEC profiling. */
#include <stdio.h>
#include <stdlib.h>
static cl_event cec_event;
static void cec_profile(cl_event event, const char* name) {
  clWaitForEvents(1, &event);
  cl_int err;
  cl_ulong start_time, end_time;

  err = clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_QUEUED,
                                sizeof(start_time), &start_time,
                                NULL);
  if (err != CL_SUCCESS) {
    printf("[CEC] fatal: Kernel timer 1!");
    exit(104);
  }

  err = clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(end_time), &end_time,
                                NULL);
  if (err != CL_SUCCESS) {
    printf("[CEC] fatal: Kernel timer 2!");
    exit(105);
  }

  float elapsed_ms = (float)(end_time - start_time) / 1000;
  printf("\n[CEC] %s %.3f\n", name, elapsed_ms);
}
/* END CEC profiling. */

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  cl_int ecode;

  if (timeron) timer_start(t_add);

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_add,
                                 ADD_DIM, NULL,
                                 add_gws,
                                 add_lws,
                                 0, NULL, &cec_event);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  cec_profile(cec_event, "k_add");
  CHECK_FINISH();

  if (timeron) timer_stop(t_add);
}
