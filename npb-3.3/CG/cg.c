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

//---------------------------------------------------------------------
//      program cg
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
#include "cl_util.h"

//#define USE_CHECK_FINISH
//#define TIMER_DETAIL

#ifdef TIMER_DETAIL
enum OPENCL_TIMER {
  T_OPENCL_API = 10,
  T_BUILD,
  T_RELEASE,
  T_BUFFER_CREATE,
  T_BUFFER_READ,
  T_BUFFER_WRITE,
  T_KERNEL_INIT_MEM,
  T_KERNEL_MAKEA_0,
  T_KERNEL_MAKEA_1,
  T_KERNEL_MAKEA_2,
  T_KERNEL_MAKEA_3,
  T_KERNEL_MAKEA_4,
  T_KERNEL_MAKEA_5,
  T_KERNEL_MAKEA_6,
  T_KERNEL_MAKEA_7,
  T_KERNEL_MAIN_0,
  T_KERNEL_MAIN_1,
  T_KERNEL_MAIN_2,
  T_KERNEL_MAIN_3,
  T_KERNEL_MAIN_4,
  T_KERNEL_CONJ_GRAD_0,
  T_KERNEL_CONJ_GRAD_1,
  T_KERNEL_CONJ_GRAD_2,
  T_KERNEL_CONJ_GRAD_3,
  T_KERNEL_CONJ_GRAD_4,
  T_KERNEL_CONJ_GRAD_5,
  T_KERNEL_CONJ_GRAD_6,
  T_KERNEL_CONJ_GRAD_7,
  T_END
};

char *kernel_names[] = {
  "init_mem",
  "makea_0",
  "makea_1",
  "makea_2",
  "makea_3",
  "makea_4",
  "makea_5",
  "makea_6",
  "makea_7",
  "main_0",
  "main_1",
  "main_2",
  "main_3",
  "main_4",
  "conj_grad_0",
  "conj_grad_1",
  "conj_grad_2",
  "conj_grad_3",
  "conj_grad_4",
  "conj_grad_5",
  "conj_grad_6",
  "conj_grad_7"
};

#define DTIMER_START(id)    if (timeron) timer_start(id)
#define DTIMER_STOP(id)     if (timeron) timer_stop(id)
#else
#define DTIMER_START(id)
#define DTIMER_STOP(id)
#endif

#ifdef USE_CHECK_FINISH
#define CHECK_FINISH()      { cl_int ecode = clFinish(cmd_queue); \
                              clu_CheckError(ecode, "clFinish()"); }
#else
#define CHECK_FINISH()
#endif


//---------------------------------------------------------------------
#define max_threads 1024

/* common / partit_size / */
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;

/* common /urando/ */
static double amult;
static double tran;

/* common /timers/ */
static logical timeron;
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// OPENCL Variables
static cl_device_type   device_type;
static cl_device_id     device;
static char            *device_name;
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_program       program;
static cl_program       p_makea;
static size_t work_item_sizes[3];
static size_t max_work_group_size;
static size_t max_compute_units;

#define NUM_K_MAIN        5
#define NUM_K_CONJ_GRAD   8
static cl_kernel k_main[NUM_K_MAIN];
static cl_kernel k_conj_grad[NUM_K_CONJ_GRAD];

static cl_mem m_colidx;
static cl_mem m_rowstr;
static cl_mem m_a;
static cl_mem m_x;
static cl_mem m_z;
static cl_mem m_p;
static cl_mem m_q;
static cl_mem m_r;

static cl_mem m_norm_temp1, m_norm_temp2;
static cl_mem m_rho;
static cl_mem m_d;

static double *g_norm_temp1;
static double *g_norm_temp2;
static double *g_rho;
static double *g_d;
static size_t norm_temp_size;
static size_t rho_size, d_size;

static size_t MAIN_3_LWS, MAIN_3_GWS;
static size_t CG_LWS, CG_GWS;
static size_t CG_LSIZE;

static void setup_opencl(int argc, char *argv[], char Class);
static void release_opencl();
static void init_mem();
#ifdef TIMER_DETAIL
static void print_opencl_timers();
#endif
//---------------------------------------------------------------------


//---------------------------------------------------------------------
static void conj_grad(double *rnorm);
static void makea(int n, int nz, int firstrow, int lastrow);
//---------------------------------------------------------------------


int main(int argc, char *argv[])
{
  int i, j, it;

  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;

  double t, mflops, tmax;
  char Class;
  logical verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_last];

  size_t main_lws[NUM_K_MAIN], main_gws[NUM_K_MAIN];
  int gws;
  cl_int ecode;

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  for (i = 0; i < T_last; i++) {
    timer_clear(i);
  }

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";
    t_names[T_conj_grad] = "conjgd";
    fclose(fp);
  } else {
    timeron = false;
  }

  timer_start(T_init);

  firstrow = 0;
  lastrow  = NA-1;
  firstcol = 0;
  lastcol  = NA-1;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
    Class = 'S';
    zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
    Class = 'W';
    zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
    Class = 'A';
    zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
    Class = 'B';
    zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
    Class = 'C';
    zeta_verify_value = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
    Class = 'D';
    zeta_verify_value = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
    Class = 'E';
    zeta_verify_value = 77.522164599383;
  } else {
    Class = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations:                  %5d\n", NITER);
  printf("\n");

  setup_opencl(argc, argv, Class);

  naa = NA;
  nzz = NZ;

  //---------------------------------------------------------------------
  // Inialize random number generator
  //---------------------------------------------------------------------
  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, amult);

  //---------------------------------------------------------------------
  //
  //---------------------------------------------------------------------
  makea(naa, nzz, firstrow, lastrow);

  //---------------------------------------------------------------------
  // Note: as a result of the above call to makea:
  //    values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
  //    values of colidx which are col indexes go from firstcol --> lastcol
  //    So:
  //    Shift the col index vals from actual (firstcol --> lastcol )
  //    to local, i.e., (1 --> lastcol-firstcol+1)
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAIN_0);
  gws = lastrow - firstrow + 1;
  ecode  = clSetKernelArg(k_main[0], 2, sizeof(int), &firstcol);
  ecode |= clSetKernelArg(k_main[0], 3, sizeof(int), &gws);
  clu_CheckError(ecode, "clSetKernelArg() for main_0");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    main_lws[0] = CG_LWS;
    main_gws[0] = CG_GWS;
  } else {
    main_lws[0] = CG_LSIZE;
    main_gws[0] = CG_LSIZE * gws;
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_main[0],
                                 1, NULL,
                                 &main_gws[0],
                                 &main_lws[0],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_0");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAIN_0);

  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAIN_1);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    main_lws[1] = CG_LWS;
    main_gws[1] = CG_GWS;
  } else {
    main_lws[1] = work_item_sizes[0];
    main_gws[1] = clu_RoundWorkSize((size_t)(NA+1), main_lws[1]);
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_main[1],
                                 1, NULL,
                                 &main_gws[1],
                                 &main_lws[1],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_1");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAIN_1);

  DTIMER_START(T_KERNEL_MAIN_2);
  gws = lastcol - firstcol + 1;
  ecode = clSetKernelArg(k_main[2], 4, sizeof(int), &gws);
  clu_CheckError(ecode, "clSetKernelArg() for main_2");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    main_lws[2] = CG_LWS;
    main_gws[2] = CG_GWS;
  } else {
    main_lws[2] = work_item_sizes[0];
    main_gws[2] = clu_RoundWorkSize((size_t)gws, main_lws[2]);
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_main[2],
                                 1, NULL,
                                 &main_gws[2],
                                 &main_lws[2],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_2");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAIN_2);

  zeta = 0.0;

  //---------------------------------------------------------------------
  //---->
  // Do one iteration untimed to init all code and data page tables
  //---->                    (then reinit, start timing, to niter its)
  //---------------------------------------------------------------------
  for (it = 1; it <= 1; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    conj_grad(&rnorm);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_MAIN_3);
    gws = lastcol - firstcol + 1;
    ecode = clSetKernelArg(k_main[3], 6, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for main_3");

    main_lws[3] = MAIN_3_LWS;
    main_gws[3] = MAIN_3_GWS;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_main[3],
                                   1, NULL,
                                   &main_gws[3],
                                   &main_lws[3],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_3");

    norm_temp1 = 0.0;
    norm_temp2 = 0.0;

    if (device_type == CL_DEVICE_TYPE_CPU) {
      ecode = clFinish(cmd_queue);
      clu_CheckError(ecode, "clFinish()");
      DTIMER_STOP(T_KERNEL_MAIN_3);
    } else {
      CHECK_FINISH();
      DTIMER_STOP(T_KERNEL_MAIN_3);

      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_norm_temp1,
                                  CL_FALSE,
                                  0,
                                  norm_temp_size,
                                  g_norm_temp1,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");

      ecode = clEnqueueReadBuffer(cmd_queue, 
                                  m_norm_temp2,
                                  CL_TRUE,
                                  0,
                                  norm_temp_size,
                                  g_norm_temp2,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    // reduction
    for (j = 0; j < main_gws[3]/main_lws[3]; j++) {
      norm_temp1 += g_norm_temp1[j];
      norm_temp2 += g_norm_temp2[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_MAIN_4);
    gws = lastcol - firstcol + 1;
    ecode  = clSetKernelArg(k_main[4], 2, sizeof(double), &norm_temp2);
    ecode |= clSetKernelArg(k_main[4], 3, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for main_4");

    if (device_type == CL_DEVICE_TYPE_CPU) {
      main_lws[4] = CG_LWS;
      main_gws[4] = CG_GWS;
    } else {
      main_lws[4] = work_item_sizes[0];
      main_gws[4] = clu_RoundWorkSize((size_t)gws, main_lws[4]);
    }
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_main[4],
                                   1, NULL,
                                   &main_gws[4],
                                   &main_lws[4],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_4");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_MAIN_4);
  } // end of do one iteration untimed


  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAIN_1);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_main[1],
                                 1, NULL,
                                 &main_gws[1],
                                 &main_lws[1],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_1");
  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");
  DTIMER_STOP(T_KERNEL_MAIN_1);

  zeta = 0.0;

  timer_stop(T_init);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));

  timer_start(T_bench);


  //---------------------------------------------------------------------
  //---->
  // Main Iteration for inverse power method
  //---->
  //---------------------------------------------------------------------
  for (it = 1; it <= NITER; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    if (timeron) timer_start(T_conj_grad);
    conj_grad(&rnorm);
    if (timeron) timer_stop(T_conj_grad);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_MAIN_3);
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_main[3],
                                   1, NULL,
                                   &main_gws[3],
                                   &main_lws[3],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_3");

    norm_temp1 = 0.0;
    norm_temp2 = 0.0;

    if (device_type == CL_DEVICE_TYPE_CPU) {
      ecode = clFinish(cmd_queue);
      clu_CheckError(ecode, "clFinish()");
      DTIMER_STOP(T_KERNEL_MAIN_3);
    } else {
      CHECK_FINISH();
      DTIMER_STOP(T_KERNEL_MAIN_3);

      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_norm_temp1,
                                  CL_FALSE,
                                  0,
                                  norm_temp_size,
                                  g_norm_temp1,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");

      ecode = clEnqueueReadBuffer(cmd_queue, 
                                  m_norm_temp2,
                                  CL_TRUE,
                                  0,
                                  norm_temp_size,
                                  g_norm_temp2,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    // reduction
    for (j = 0; j < main_gws[3]/main_lws[3]; j++) {
      norm_temp1 += g_norm_temp1[j];
      norm_temp2 += g_norm_temp2[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    zeta = SHIFT + 1.0 / norm_temp1;
    if (it == 1)
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);


    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_MAIN_4);
    ecode = clSetKernelArg(k_main[4], 2, sizeof(double), &norm_temp2);
    clu_CheckError(ecode, "clSetKernelArg() for main_4");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_main[4],
                                   1, NULL,
                                   &main_gws[4],
                                   &main_lws[4],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for main_4");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_MAIN_4);
  } // end of main iter inv pow meth
  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");

  timer_stop(T_bench);

  //---------------------------------------------------------------------
  // End of timed section
  //---------------------------------------------------------------------

  t = timer_read(T_bench);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (Class != 'U') {
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13E\n", zeta);
      printf(" Error is   %20.13E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", zeta);
      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (double)(2*NITER*NA)
                   * (3.0+(double)(NONZER*(NONZER+1))
                     + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
                     + 3.0) / t / 1000000.0;
  } else {
    mflops = 0.0;
  }

  c_print_results("CG", Class, NA, 0, 0,
                  NITER, t,
                  mflops, "          floating point",
                  verified, NPBVERSION, COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  clu_GetDeviceTypeName(device_type), device_name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 0; i < T_last; i++) {
      t = timer_read(i);
      if (i == T_init) {
        printf("  %8s:%9.3f\n", t_names[i], t);
      } else {
        printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
        if (i == T_conj_grad) {
          t = tmax - t;
          printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
        }
      }
    }
  }

  release_opencl();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// Floaging point arrays here are named as in NPB1 spec discussion of
// CG algorithm
//---------------------------------------------------------------------
static void conj_grad(double *rnorm)
{
  int j;
  int cgit, cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  int gws;
  size_t cg_lws[NUM_K_CONJ_GRAD], cg_gws[NUM_K_CONJ_GRAD];
  cl_int ecode;

  rho = 0.0;
  sum = 0.0;

  //---------------------------------------------------------------------
  // Initialize the CG algorithm:
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_CONJ_GRAD_0);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    cg_lws[0] = CG_LWS;
    cg_gws[0] = CG_GWS;
  } else {
    cg_lws[0] = work_item_sizes[0];
    cg_gws[0] = clu_RoundWorkSize((size_t)(naa+1), cg_lws[0]);
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_conj_grad[0],
                                 1, NULL,
                                 &cg_gws[0],
                                 &cg_lws[0],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_0");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_CONJ_GRAD_0);

  //---------------------------------------------------------------------
  // rho = r.r
  // Now, obtain the norm of r: First, sum squares of r elements locally...
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_CONJ_GRAD_1);
  gws = lastcol - firstcol + 1;
  ecode = clSetKernelArg(k_conj_grad[1], 2, sizeof(int), &gws);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_1");

  cg_lws[1] = CG_LWS;
  cg_gws[1] = CG_GWS;
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_conj_grad[1],
                                 1, NULL,
                                 &cg_gws[1],
                                 &cg_lws[1],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_1");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode = clFinish(cmd_queue);
    clu_CheckError(ecode, "clFinish()");
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_1);
  } else {
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_1);

    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_rho,
                                CL_TRUE, 0,
                                rho_size,
                                g_rho,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  }
  
  // reduction
  for (j = 0; j < cg_gws[1]/cg_lws[1]; j++) {
    rho = rho + g_rho[j];
  }

  
  //---------------------------------------------------------------------
  //---->
  // The conj grad iteration loop
  //---->
  //---------------------------------------------------------------------
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    //---------------------------------------------------------------------
    // Save a temporary of rho and initialize reduction variables
    //---------------------------------------------------------------------
    rho0 = rho;
    d = 0.0;
    rho = 0.0;

    //---------------------------------------------------------------------
    // q = A.p
    // The partition submatrix-vector multiply: use workspace w
    //---------------------------------------------------------------------
    //
    // NOTE: this version of the multiply is actually (slightly: maybe %5)
    //       faster on the sp2 on 16 nodes than is the unrolled-by-2 version
    //       below.   On the Cray t3d, the reverse is true, i.e., the
    //       unrolled-by-two version is some 10% faster.
    //       The unrolled-by-8 version below is significantly faster
    //       on the Cray t3d - overall speed of code is 1.5 times faster.

    DTIMER_START(T_KERNEL_CONJ_GRAD_2);
    gws = lastrow - firstrow + 1;
    ecode = clSetKernelArg(k_conj_grad[2], 5, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for conj_grad_2");

    if (device_type == CL_DEVICE_TYPE_CPU) {
      cg_lws[2] = CG_LWS;
      cg_gws[2] = CG_GWS;
    } else {
      cg_lws[2] = CG_LSIZE;
      cg_gws[2] = CG_LSIZE * (lastrow - firstrow + 1);
    }
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_conj_grad[2],
                                   1, NULL,
                                   &cg_gws[2],
                                   &cg_lws[2],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_2");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_2);

    //---------------------------------------------------------------------
    // Obtain p.q
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_CONJ_GRAD_3);
    gws = lastcol - firstcol + 1;
    ecode = clSetKernelArg(k_conj_grad[3], 3, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for conj_grad_3");

    cg_lws[3] = CG_LWS;
    cg_gws[3] = CG_GWS;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_conj_grad[3],
                                   1, NULL,
                                   &cg_gws[3],
                                   &cg_lws[3],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_3");

    if (device_type == CL_DEVICE_TYPE_CPU) {
      ecode = clFinish(cmd_queue);
      clu_CheckError(ecode, "clFinish()");
      DTIMER_STOP(T_KERNEL_CONJ_GRAD_3);
    } else {
      CHECK_FINISH();
      DTIMER_STOP(T_KERNEL_CONJ_GRAD_3);

      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_d,
                                  CL_TRUE, 0,
                                  d_size,
                                  g_d,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    // reduction
    for (j = 0; j < cg_gws[3]/cg_lws[3]; j++) {
      d = d + g_d[j];
    }

    //---------------------------------------------------------------------
    // Obtain alpha = rho / (p.q)
    //---------------------------------------------------------------------
    alpha = rho0 / d;

    //---------------------------------------------------------------------
    // Obtain z = z + alpha*p
    // and    r = r - alpha*q
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_CONJ_GRAD_4);
    gws = lastcol - firstcol + 1;
    ecode  = clSetKernelArg(k_conj_grad[4], 5, sizeof(double), &alpha);
    ecode |= clSetKernelArg(k_conj_grad[4], 6, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for conj_grad_4");

    cg_lws[4] = CG_LWS;
    cg_gws[4] = CG_GWS;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_conj_grad[4],
                                   1, NULL,
                                   &cg_gws[4],
                                   &cg_lws[4],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_4");

    if (device_type == CL_DEVICE_TYPE_CPU) {
      ecode = clFinish(cmd_queue);
      clu_CheckError(ecode, "clFinish()");
      DTIMER_STOP(T_KERNEL_CONJ_GRAD_4);
    } else {
      CHECK_FINISH();
      DTIMER_STOP(T_KERNEL_CONJ_GRAD_4);

      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_rho,
                                  CL_TRUE, 0,
                                  rho_size,
                                  g_rho,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }
    
    // reduction
    for (j = 0; j < cg_gws[4]/cg_lws[4]; j++) {
      rho = rho + g_rho[j];
    }

    //---------------------------------------------------------------------
    // Obtain beta:
    //---------------------------------------------------------------------
    beta = rho / rho0;

    //---------------------------------------------------------------------
    // p = r + beta*p
    //---------------------------------------------------------------------
    DTIMER_START(T_KERNEL_CONJ_GRAD_5);
    gws = lastcol - firstcol + 1;
    ecode  = clSetKernelArg(k_conj_grad[5], 2, sizeof(double), &beta);
    ecode |= clSetKernelArg(k_conj_grad[5], 3, sizeof(int), &gws);
    clu_CheckError(ecode, "clSetKernelArg() for conj_grad_5");
    
    if (device_type == CL_DEVICE_TYPE_CPU) {
      cg_lws[5] = CG_LWS;
      cg_gws[5] = CG_GWS;
    } else {
      cg_lws[5] = work_item_sizes[0];
      cg_gws[5] = clu_RoundWorkSize((size_t)gws, cg_lws[5]);
    }
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_conj_grad[5],
                                   1, NULL,
                                   &cg_gws[5],
                                   &cg_lws[5],
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_5");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_5);
  } // end of do cgit=1,cgitmax

  //---------------------------------------------------------------------
  // Compute residual norm explicitly:  ||r|| = ||x - A.z||
  // First, form A.z
  // The partition submatrix-vector multiply
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_CONJ_GRAD_6);
  gws = lastrow - firstrow + 1;
  ecode = clSetKernelArg(k_conj_grad[6], 5, sizeof(int), &gws);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_6");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    cg_lws[6] = CG_LWS;
    cg_gws[6] = CG_GWS;
  } else {
    cg_lws[6] = CG_LSIZE;
    cg_gws[6] = CG_LSIZE * (lastrow - firstrow + 1);
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_conj_grad[6],
                                 1, NULL,
                                 &cg_gws[6],
                                 &cg_lws[6],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_6");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_CONJ_GRAD_6);

  //---------------------------------------------------------------------
  // At this point, r contains A.z
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_CONJ_GRAD_7);
  gws = lastcol-firstcol+1;
  ecode = clSetKernelArg(k_conj_grad[7], 3, sizeof(int), &gws);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_7");

  cg_lws[7] = CG_LWS;
  cg_gws[7] = CG_GWS;
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_conj_grad[7],
                                 1, NULL,
                                 &cg_gws[7],
                                 &cg_lws[7],
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for conj_grad_7");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode = clFinish(cmd_queue);
    clu_CheckError(ecode, "clFinish()");
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_7);
  } else {
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_CONJ_GRAD_7);

    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_d,
                                CL_TRUE, 0,
                                d_size,
                                g_d,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  }

  // reduction
  for (j = 0; j < cg_gws[7]/cg_lws[7]; j++) {
    sum = sum + g_d[j];
  }

  *rnorm = sqrt(sum);
}


//---------------------------------------------------------------------
// generate the test problem for benchmark 6
// makea generates a sparse matrix with a
// prescribed sparsity distribution
//
// parameter    type        usage
//
// input
//
// n            i           number of cols/rows of matrix
// nz           i           nonzeros as declared array size
// rcond        r*8         condition number
// shift        r*8         main diagonal shift
//
// output
//
// a            r*8         array for nonzeros
// colidx       i           col indices
// rowstr       i           row pointers
//
// workspace
//
// iv, arow, acol i
// aelt           r*8
//---------------------------------------------------------------------
static void makea(int n, int nz, int firstrow, int lastrow)
{
  int nn1;
  int i;

  cl_mem m_acol;
  cl_mem m_arow;
  cl_mem m_aelt;
  cl_mem m_v;
  cl_mem m_iv;
  cl_mem m_last_n;
  cl_mem m_ilow, m_ihigh;

  const int NUM_K_MAKEA = 8;
  cl_kernel k_makea[NUM_K_MAKEA];
  size_t makea_lws[3], makea_gws[3];
  char kname[10];
  cl_int ecode;

  const int MAKEA_GLOBAL_SIZE = max_threads;

  // Create buffers
  DTIMER_START(T_BUFFER_CREATE);
  m_arow = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          (NA+1) * sizeof(int),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_arow");

  m_acol = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          NAZ * sizeof(int),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_acol");

  m_aelt = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          NAZ * sizeof(double),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_aelt");

  m_v = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       NZ * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_v");

  m_iv = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        (NZ+1+NA) * sizeof(int),
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_iv");

  m_last_n = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            max_threads * sizeof(int),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_last_n");

  m_ilow = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          MAKEA_GLOBAL_SIZE * sizeof(int),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_ilow");

  m_ihigh = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          MAKEA_GLOBAL_SIZE * sizeof(int),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_ihigh");
  DTIMER_STOP(T_BUFFER_CREATE);

  // Create kernels
  DTIMER_START(T_OPENCL_API);
  for (i = 0; i < NUM_K_MAKEA; i++) {
    sprintf(kname, "makea_%d", i);
    k_makea[i] = clCreateKernel(p_makea, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }
  DTIMER_STOP(T_OPENCL_API);

  //---------------------------------------------------------------------
  // nn1 is the smallest power of two not less than n
  //---------------------------------------------------------------------
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  //---------------------------------------------------------------------
  // Generate nonzero positions and save for the use in sparse.
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAKEA_0);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    makea_lws[0] = CG_LWS;
    makea_gws[0] = CG_GWS;
  } else {
    makea_lws[0] = 32;
    makea_gws[0] = MAKEA_GLOBAL_SIZE;
  }

  ecode  = clSetKernelArg(k_makea[0], 0, sizeof(cl_mem), &m_arow);
  ecode |= clSetKernelArg(k_makea[0], 1, sizeof(cl_mem), &m_acol);
  ecode |= clSetKernelArg(k_makea[0], 2, sizeof(cl_mem), &m_aelt);
  ecode |= clSetKernelArg(k_makea[0], 3, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[0], 4, sizeof(cl_mem), &m_ihigh);
  ecode |= clSetKernelArg(k_makea[0], 5, sizeof(int), &n);
  ecode |= clSetKernelArg(k_makea[0], 6, sizeof(int), &nn1);
  ecode |= clSetKernelArg(k_makea[0], 7, sizeof(double), &tran);
  ecode |= clSetKernelArg(k_makea[0], 8, sizeof(double), &amult);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[0],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_0);


  //---------------------------------------------------------------------
  // ... make the sparse matrix from list of elements with duplicates
  //     (v and iv are used as  workspace)
  //---------------------------------------------------------------------
  //sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
  //       aelt, firstrow, lastrow, last_n,
  //       v, &iv[0], &iv[nz], RCOND, SHIFT);
  DTIMER_START(T_KERNEL_MAKEA_1);
  ecode  = clSetKernelArg(k_makea[1], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_makea[1], 1, sizeof(cl_mem), &m_arow);
  ecode |= clSetKernelArg(k_makea[1], 2, sizeof(cl_mem), &m_acol);
  ecode |= clSetKernelArg(k_makea[1], 3, sizeof(cl_mem), &m_last_n);
  ecode |= clSetKernelArg(k_makea[1], 4, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[1], 5, sizeof(cl_mem), &m_ihigh);
  ecode |= clSetKernelArg(k_makea[1], 6, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[1],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_1);

  DTIMER_START(T_KERNEL_MAKEA_2);
  ecode  = clSetKernelArg(k_makea[2], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_makea[2], 1, sizeof(cl_mem), &m_last_n);
  ecode |= clSetKernelArg(k_makea[2], 2, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[2], 3, sizeof(cl_mem), &m_ihigh);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[2],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_2);

  int nrows = lastrow - firstrow + 1;
  int nza;
  DTIMER_START(T_BUFFER_READ);
  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_rowstr,
                              CL_TRUE,
                              nrows * sizeof(int),
                              sizeof(int),
                              &nza,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  DTIMER_STOP(T_BUFFER_READ);

  nza = nza - 1;

  //---------------------------------------------------------------------
  // ... rowstr(j) now is the location of the first nonzero
  //     of row j of a
  //---------------------------------------------------------------------
  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  //---------------------------------------------------------------------
  // ... preload data pages
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  // ... generate actual values by summing duplicates
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAKEA_3);
  ecode  = clSetKernelArg(k_makea[3], 0, sizeof(cl_mem), &m_v);
  ecode |= clSetKernelArg(k_makea[3], 1, sizeof(cl_mem), &m_iv);
  ecode |= clSetKernelArg(k_makea[3], 2, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_makea[3], 3, sizeof(cl_mem), &m_arow);
  ecode |= clSetKernelArg(k_makea[3], 4, sizeof(cl_mem), &m_acol);
  ecode |= clSetKernelArg(k_makea[3], 5, sizeof(cl_mem), &m_aelt);
  ecode |= clSetKernelArg(k_makea[3], 6, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[3], 7, sizeof(cl_mem), &m_ihigh);
  ecode |= clSetKernelArg(k_makea[3], 8, sizeof(int), &n);
  ecode |= clSetKernelArg(k_makea[3], 9, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    makea_lws[0] = 128;
    makea_gws[0] = clu_RoundWorkSize((size_t)n, makea_lws[0]);
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[3],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_3);

  //---------------------------------------------------------------------
  // ... remove empty entries and generate final results
  //---------------------------------------------------------------------
  DTIMER_START(T_KERNEL_MAKEA_4);
  ecode  = clSetKernelArg(k_makea[4], 0, sizeof(cl_mem), &m_iv);
  ecode |= clSetKernelArg(k_makea[4], 1, sizeof(cl_mem), &m_last_n);
  ecode |= clSetKernelArg(k_makea[4], 2, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[4], 3, sizeof(cl_mem), &m_ihigh);
  ecode |= clSetKernelArg(k_makea[4], 4, sizeof(int), &n);
  ecode |= clSetKernelArg(k_makea[4], 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    makea_lws[0] = 32;
    makea_gws[0] = MAKEA_GLOBAL_SIZE;
  }
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[4],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_4);

  DTIMER_START(T_KERNEL_MAKEA_5);
  ecode  = clSetKernelArg(k_makea[5], 0, sizeof(cl_mem), &m_iv);
  ecode |= clSetKernelArg(k_makea[5], 1, sizeof(cl_mem), &m_last_n);
  ecode |= clSetKernelArg(k_makea[5], 2, sizeof(cl_mem), &m_ilow);
  ecode |= clSetKernelArg(k_makea[5], 3, sizeof(cl_mem), &m_ihigh);
  ecode |= clSetKernelArg(k_makea[5], 4, sizeof(int), &n);
  ecode |= clSetKernelArg(k_makea[5], 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[5],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_5);

  DTIMER_START(T_KERNEL_MAKEA_6);
  ecode  = clSetKernelArg(k_makea[6], 0, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_makea[6], 1, sizeof(cl_mem), &m_v);
  ecode |= clSetKernelArg(k_makea[6], 2, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_makea[6], 3, sizeof(cl_mem), &m_colidx);
  ecode |= clSetKernelArg(k_makea[6], 4, sizeof(cl_mem), &m_iv);
  ecode |= clSetKernelArg(k_makea[6], 5, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_makea[6], 6, sizeof(int), &nrows);
  clu_CheckError(ecode, "clSetKernelArg()");

  makea_lws[0] = (device_type == CL_DEVICE_TYPE_CPU) ? 32 : 128;
  makea_gws[0] = clu_RoundWorkSize((size_t)nrows, makea_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[6],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_6);

  DTIMER_START(T_KERNEL_MAKEA_7);
  ecode  = clSetKernelArg(k_makea[7], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_makea[7], 1, sizeof(cl_mem), &m_iv);
  ecode |= clSetKernelArg(k_makea[7], 2, sizeof(int), &nrows);
  ecode |= clSetKernelArg(k_makea[7], 3, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");

  makea_lws[0] = (device_type == CL_DEVICE_TYPE_CPU) ? 32 : 128;
  makea_gws[0] = clu_RoundWorkSize((size_t)nrows, makea_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_makea[7],
                                 1, NULL,
                                 makea_gws,
                                 makea_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_MAKEA_7);

  // Release kernel objects
  DTIMER_START(T_RELEASE);
  clReleaseMemObject(m_acol);
  clReleaseMemObject(m_arow);
  clReleaseMemObject(m_aelt);
  clReleaseMemObject(m_v);
  clReleaseMemObject(m_iv);
  clReleaseMemObject(m_last_n);
  clReleaseMemObject(m_ilow);
  clReleaseMemObject(m_ihigh);

  for (i = 0; i < NUM_K_MAKEA; i++) {
    clReleaseKernel(k_makea[i]);
  }
  DTIMER_STOP(T_RELEASE);
}


static void setup_opencl(int argc, char *argv[], char Class)
{
  int i;
  cl_int ecode;
  char *source_dir = "CG";
  if (argc > 1) source_dir = argv[1];

#ifdef TIMER_DETAIL
  if (timeron) {
    for (i = T_OPENCL_API; i < T_END; i++) timer_clear(i);
  }
#endif

  DTIMER_START(T_OPENCL_API);

  // 1. Find the default device type and get a device for the device type
  device_type = clu_GetDefaultDeviceType();
  device      = clu_GetAvailableDevice(device_type);
  device_name = clu_GetDeviceName(device);

  // Device information
  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(work_item_sizes),
                          &work_item_sizes,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(size_t),
                          &max_work_group_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cl_uint),
                          &max_compute_units,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  // FIXME: The below values are experimental.
  size_t default_size = 128;
  if (max_work_group_size > default_size) {
    max_work_group_size = default_size;
    int i;
    for (i = 0; i < 3; i++) {
      if (work_item_sizes[i] > default_size) {
        work_item_sizes[i] = default_size;
      }
    }
  }

  // 2. Create a context for the specified device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  // 3. Create a command queue
  cmd_queue = clCreateCommandQueue(context, device, 0, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  DTIMER_STOP(T_OPENCL_API);

  // 4. Build the program
  DTIMER_START(T_BUILD);
  char *source_file_makea;
  char *source_file;
  char build_option[100];
  if (device_type == CL_DEVICE_TYPE_CPU) {
    MAIN_3_LWS = 1;
    MAIN_3_GWS = max_compute_units;

    CG_LWS = 1;
    CG_GWS = max_compute_units;

    source_file_makea = "cg_cpu_makea.cl";
    source_file = "cg_cpu.cl";
    sprintf(build_option, "-I. -DCLASS=%d", Class);

  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    MAIN_3_LWS = work_item_sizes[0];
    MAIN_3_GWS = clu_RoundWorkSize((size_t)NA, MAIN_3_LWS);
    CG_LWS = work_item_sizes[0];
    CG_GWS = clu_RoundWorkSize((size_t)NA, CG_LWS);

    CG_LSIZE = 64;

    source_file_makea = "cg_gpu_makea.cl";
    source_file = "cg_gpu.cl";
    sprintf(build_option, "-I. -DCLASS=\'%c\' -DLSIZE=%lu -cl-mad-enable",
        Class, CG_LSIZE);

  } else {
    fprintf(stderr, "%s: not supported.", clu_GetDeviceTypeName(device_type));
    exit(EXIT_FAILURE);
  }
  p_makea = clu_MakeProgram(context, device, source_dir, source_file_makea,
                            build_option);
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);
  DTIMER_STOP(T_BUILD);

  // 5. Create buffers
  DTIMER_START(T_BUFFER_CREATE);
  /* common / main_int_mem / */
  m_colidx = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            NZ * sizeof(int),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_colidx");

  m_rowstr = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            (NA+1) * sizeof(int),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rowstr");

  /* common / main_flt_mem / */
  m_a = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       NZ * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_a");

  m_x = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA+2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_x");

  m_z = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA+2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_z");

  m_p = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA+2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_p");

  m_q = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA+2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_q");

  m_r = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA+2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_r");

  // reduction buffers
  norm_temp_size = (MAIN_3_GWS/MAIN_3_LWS) * sizeof(double);
  rho_size = (CG_GWS/CG_LWS) * sizeof(double);
  d_size = (CG_GWS/CG_LWS) * sizeof(double);

  g_norm_temp1 = (double *)malloc(norm_temp_size);
  g_norm_temp2 = (double *)malloc(norm_temp_size);
  g_rho = (double *)malloc(rho_size);
  g_d = (double *)malloc(d_size);

  if (device_type == CL_DEVICE_TYPE_CPU) {
    m_norm_temp1 = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         norm_temp_size,
                         g_norm_temp1, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp1");

    m_norm_temp2 = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         norm_temp_size,
                         g_norm_temp2, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp2");

    m_rho = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         rho_size,
                         g_rho, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_rho");

    m_d = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         d_size,
                         g_d, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_d");
  } else {
    m_norm_temp1 = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         norm_temp_size,
                         NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp1");

    m_norm_temp2 = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         norm_temp_size,
                         NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp2");

    m_rho = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         rho_size,
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_rho");

    m_d = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         d_size,
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_d");
  }
  DTIMER_STOP(T_BUFFER_CREATE);

  // 6. Create kernels and set arguments
  DTIMER_START(T_OPENCL_API);
  char kname[15];
  for (i = 0; i < NUM_K_MAIN; i++) {
    sprintf(kname, "main_%d", i);
    k_main[i] = clCreateKernel(program, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }

  for (i = 0; i < NUM_K_CONJ_GRAD; i++) {
    sprintf(kname, "conj_grad_%d", i);
    k_conj_grad[i] = clCreateKernel(program, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }

  // arguments for main_0
  ecode  = clSetKernelArg(k_main[0], 0, sizeof(cl_mem), &m_colidx);
  ecode |= clSetKernelArg(k_main[0], 1, sizeof(cl_mem), &m_rowstr);
  clu_CheckError(ecode, "clSetKernelArg() for main_0");

  // arguments for main_1
  ecode = clSetKernelArg(k_main[1], 0, sizeof(cl_mem), &m_x);
  clu_CheckError(ecode, "clSetKernelArg() for main_1");

  // arguments for main_2
  ecode  = clSetKernelArg(k_main[2], 0, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_main[2], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_main[2], 2, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_main[2], 3, sizeof(cl_mem), &m_p);
  clu_CheckError(ecode, "clSetKernelArg() for main_2");

  // arguments for main_3
  ecode  = clSetKernelArg(k_main[3], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_main[3], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_main[3], 2, sizeof(cl_mem), &m_norm_temp1);
  ecode |= clSetKernelArg(k_main[3], 3, sizeof(cl_mem), &m_norm_temp2);
  ecode |= clSetKernelArg(k_main[3], 4, sizeof(double) * MAIN_3_LWS, NULL);
  ecode |= clSetKernelArg(k_main[3], 5, sizeof(double) * MAIN_3_LWS, NULL);
  clu_CheckError(ecode, "clSetKernelArg() for main_3");

  // arguments for main_4
  ecode  = clSetKernelArg(k_main[4], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_main[4], 1, sizeof(cl_mem), &m_z);
  clu_CheckError(ecode, "clSetKernelArg() for main_4");

  // arguments for conj_grad_0
  ecode  = clSetKernelArg(k_conj_grad[0], 0, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[0], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_conj_grad[0], 2, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[0], 3, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_conj_grad[0], 4, sizeof(cl_mem), &m_p);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_0");

  // arguments for conj_grad_1
  ecode  = clSetKernelArg(k_conj_grad[1], 0, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[1], 1, sizeof(cl_mem), &m_rho);
  if (device_type != CL_DEVICE_TYPE_CPU)
    ecode |= clSetKernelArg(k_conj_grad[1], 3, sizeof(double)*CG_LWS, NULL);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_1");

  // arguments for conj_grad_2
  ecode  = clSetKernelArg(k_conj_grad[2], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_conj_grad[2], 1, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_conj_grad[2], 2, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[2], 3, sizeof(cl_mem), &m_colidx);
  ecode |= clSetKernelArg(k_conj_grad[2], 4, sizeof(cl_mem), &m_q);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_2");

  // arguments for conj_grad_3
  ecode  = clSetKernelArg(k_conj_grad[3], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[3], 1, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[3], 2, sizeof(cl_mem), &m_d);
  if (device_type != CL_DEVICE_TYPE_CPU)
    ecode |= clSetKernelArg(k_conj_grad[3], 4, sizeof(double)*CG_LWS, NULL);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_3");

  // arguments for conj_grad_4
  ecode  = clSetKernelArg(k_conj_grad[4], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[4], 1, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[4], 2, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[4], 3, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_conj_grad[4], 4, sizeof(cl_mem), &m_rho);
  if (device_type != CL_DEVICE_TYPE_CPU)
    ecode |= clSetKernelArg(k_conj_grad[4], 7, sizeof(double)*CG_LWS, NULL);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_4");

  // arguments for conj_grad_5
  ecode  = clSetKernelArg(k_conj_grad[5], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[5], 1, sizeof(cl_mem), &m_r);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_5");

  // arguments for conj_grad_6
  ecode  = clSetKernelArg(k_conj_grad[6], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_conj_grad[6], 1, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_conj_grad[6], 2, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_conj_grad[6], 3, sizeof(cl_mem), &m_colidx);
  ecode |= clSetKernelArg(k_conj_grad[6], 4, sizeof(cl_mem), &m_r);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_6");

  // arguments for conj_grad_7
  ecode  = clSetKernelArg(k_conj_grad[7], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_conj_grad[7], 1, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[7], 2, sizeof(cl_mem), &m_d);
  if (device_type != CL_DEVICE_TYPE_CPU)
    ecode |= clSetKernelArg(k_conj_grad[7], 4, sizeof(double)*CG_LWS, NULL);
  clu_CheckError(ecode, "clSetKernelArg() for conj_grad_7");

  DTIMER_STOP(T_OPENCL_API);

  init_mem();
}

static void release_opencl()
{
  int i;

  DTIMER_START(T_RELEASE);

  clReleaseMemObject(m_colidx);
  clReleaseMemObject(m_rowstr);
  clReleaseMemObject(m_q);
  clReleaseMemObject(m_z);
  clReleaseMemObject(m_r);
  clReleaseMemObject(m_p);
  clReleaseMemObject(m_x);
  clReleaseMemObject(m_a);

  clReleaseMemObject(m_norm_temp1);
  clReleaseMemObject(m_norm_temp2);
  clReleaseMemObject(m_rho);
  clReleaseMemObject(m_d);
  free(g_norm_temp1);
  free(g_norm_temp2);
  free(g_rho);
  free(g_d);

  for (i = 0; i < NUM_K_MAIN; i++) {
    clReleaseKernel(k_main[i]);
  }
  for (i = 0; i < NUM_K_CONJ_GRAD; i++) {
    clReleaseKernel(k_conj_grad[i]);
  }

  clReleaseProgram(p_makea);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);

  DTIMER_STOP(T_RELEASE);

#ifdef TIMER_DETAIL
  print_opencl_timers();
#endif
}

#ifdef TIMER_DETAIL
static void print_kernel_time(double t_kernel, int i)
{
  char *name = kernel_names[i - T_KERNEL_INIT_MEM];
  unsigned cnt = timer_count(i);
  double tt = timer_read(i);
  printf("- %-11s: %9.3lf (%u, %.3f, %.2f%%)\n",
      name, tt, cnt, tt/cnt, tt/t_kernel * 100.0);
}

static void print_opencl_timers()
{
  if (timeron) {
    int i;
    double tt;
    double t_opencl = 0.0, t_buffer = 0.0, t_kernel = 0.0;

    for (i = T_OPENCL_API; i < T_END; i++)
      t_opencl += timer_read(i);

    for (i = T_BUFFER_CREATE; i <= T_BUFFER_WRITE; i++)
      t_buffer += timer_read(i);

    for (i = T_KERNEL_INIT_MEM; i <= T_KERNEL_CONJ_GRAD_7; i++)
      t_kernel += timer_read(i);

    printf("\nOpenCL timers -\n");
    printf("Kernel       : %9.3f (%.2f%%)\n", 
        t_kernel, t_kernel/t_opencl * 100.0);
    for (i = T_KERNEL_INIT_MEM; i <= T_KERNEL_CONJ_GRAD_7; i++) {
      print_kernel_time(t_kernel, i);
    }

    printf("Buffer       : %9.3lf (%.2f%%)\n",
        t_buffer, t_buffer/t_opencl * 100.0);
    printf("- creation   : %9.3lf\n", timer_read(T_BUFFER_CREATE));
    printf("- read       : %9.3lf\n", timer_read(T_BUFFER_READ));
    printf("- write      : %9.3lf\n", timer_read(T_BUFFER_WRITE));

    tt = timer_read(T_OPENCL_API);
    printf("API          : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_BUILD);
    printf("BUILD        : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_RELEASE);
    printf("RELEASE      : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    printf("Total        : %9.3lf\n", t_opencl);
  }
}
#endif

static void init_mem()
{
  const int NUM_K_INIT = 2;
  cl_kernel k_init[NUM_K_INIT];
  size_t init_lws[1], init_gws[1];
  char kname[15];
  cl_int ecode;
  int i, n;

  // Create kernels
  DTIMER_START(T_OPENCL_API);
  for (i = 0; i < NUM_K_INIT; i++) {
    sprintf(kname, "init_mem_%d", i);
    k_init[i] = clCreateKernel(p_makea, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }
  DTIMER_STOP(T_OPENCL_API);

  DTIMER_START(T_KERNEL_INIT_MEM);

  //-------------------------------------------------------------------------
  init_lws[0] = (device_type == CL_DEVICE_TYPE_CPU) ? 512 : 256;

  // int colidx[NZ]
  n = NZ;
  ecode  = clSetKernelArg(k_init[0], 0, sizeof(cl_mem), &m_colidx);
  ecode |= clSetKernelArg(k_init[0], 1, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");
  init_gws[0] = clu_RoundWorkSize((size_t)n, init_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[0], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // int rowstr[NA+1]
  n = NA+1;
  ecode  = clSetKernelArg(k_init[0], 0, sizeof(cl_mem), &m_rowstr);
  ecode |= clSetKernelArg(k_init[0], 1, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");
  init_gws[0] = clu_RoundWorkSize((size_t)n, init_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[0], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double a[NZ]
  n = NZ;
  ecode  = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_init[1], 1, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");
  init_gws[0] = clu_RoundWorkSize((size_t)n, init_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double x[NA+2]
  n = NA+2;
  ecode  = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_init[1], 1, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");
  init_gws[0] = clu_RoundWorkSize((size_t)n, init_lws[0]);
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double z[NA+2]
  ecode = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_z);
  clu_CheckError(ecode, "clSetKernelArg()");
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double p[NA+2]
  ecode = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_p);
  clu_CheckError(ecode, "clSetKernelArg()");
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double q[NA+2]
  ecode = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_q);
  clu_CheckError(ecode, "clSetKernelArg()");
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  // double r[NA+2]
  ecode = clSetKernelArg(k_init[1], 0, sizeof(cl_mem), &m_r);
  clu_CheckError(ecode, "clSetKernelArg()");
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_init[1], 1, NULL,
                                 init_gws, init_lws, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");
  DTIMER_STOP(T_KERNEL_INIT_MEM);

  // Release kernel objects
  DTIMER_START(T_RELEASE);
  for (i = 0; i < NUM_K_INIT; i++) {
    clReleaseKernel(k_init[i]);
  }
  DTIMER_STOP(T_RELEASE);
}

