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

//---------------------------------------------------------------------
//      program mg
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include "mg_dim.h"

#include <CL/cl.h>
#include "cl_util.h"

//#define TIMER_DETAIL
//#define USE_CHECK_FINISH

#ifdef TIMER_DETAIL
enum OPENCL_TIMER {
  T_OPENCL_API = 11,
  T_BUILD,
  T_RELEASE,
  T_BUFFER_CREATE,
  T_BUFFER_READ,
  T_BUFFER_WRITE,
  T_KERNEL_PSINV,
  T_KERNEL_RESID,
  T_KERNEL_RPRJ3,
  T_KERNEL_INTERP_1,
  T_KERNEL_INTERP_2,
  T_KERNEL_INTERP_3,
  T_KERNEL_INTERP_4,
  T_KERNEL_INTERP_5,
  T_KERNEL_NORM2U3,
  T_KERNEL_COMM3_1,
  T_KERNEL_COMM3_2,
  T_KERNEL_COMM3_3,
  T_KERNEL_ZRAN3_1,
  T_KERNEL_ZRAN3_2,
  T_KERNEL_ZRAN3_3,
  T_KERNEL_ZERO3,
  T_END
};

char *kernel_names[] = {
  "psinv",
  "resid",
  "rprj3",
  "interp_1",
  "interp_2",
  "interp_3",
  "interp_4",
  "interp_5",
  "norm2u3",
  "comm3_1",
  "comm3_2",
  "comm3_3",
  "zran3_1",
  "zran3_2",
  "zran3_3",
  "zero3"
};

static void print_opencl_timers();
#define DTIMER_START(id)    if (timeron) timer_start(id)
#define DTIMER_STOP(id)     if (timeron) timer_stop(id)
#ifndef USE_CHECK_FINISH
#define USE_CHECK_FINISH
#endif
#else
#define DTIMER_START(id)
#define DTIMER_STOP(id)
#endif

#ifdef USE_CHECK_FINISH
#define CHECK_FINISH()      { cl_int ecode = clFinish(cmd_queue); \
                              clu_CheckError(ecode, "clFinish"); }
#else
#define CHECK_FINISH()
#endif

//---------------------------------------------------------------------------
// OpenCL part
//---------------------------------------------------------------------------
static cl_device_type   device_type;
static cl_device_id     device;
static char            *device_name;
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_program       program;
static size_t global[3], local[3];
static size_t max_work_group_size;

// kernels
static cl_kernel kernel_zero3;
static cl_kernel kernel_comm3_1, kernel_comm3_2, kernel_comm3_3;
static cl_kernel kernel_zran3_1, kernel_zran3_2, kernel_zran3_3;
static cl_kernel kernel_psinv, kernel_resid, kernel_rprj3;
static cl_kernel kernel_interp_1, kernel_interp_2, kernel_interp_3, kernel_interp_4, kernel_interp_5;
static cl_kernel kernel_norm2u3;

// memory objects
static cl_mem m_v, m_r, m_u, m_starts;
static cl_mem m_a;

static cl_uint PSINV_DIM;
static cl_uint RESID_DIM;
static cl_uint RPRJ3_DIM;
static cl_uint INTERP_1_DIM;
static cl_uint NORM2U3_DIM;
static cl_uint COMM3_1_DIM, COMM3_2_DIM, COMM3_3_DIM;
static cl_uint ZERO3_DIM;

static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//---------------------------------------------------------------------------


static void setup(int *n1, int *n2, int *n3);
static void mg3P(double u[], double v[], double r[],
                 double a[4], double c[4], int n1, int n2, int n3);
static void psinv(double *ou, int n1, int n2, int n3,
                  double c[4], int k, int offset);

static void resid(double *or, int n1, int n2, int n3, double a[4], int k,
                  cl_mem m_buff_u, cl_mem m_buff_v, cl_mem m_buff_r, int offset);
static void rprj3(double *or, int m1k, int m2k, int m3k,
                  double *os, int m1j, int m2j, int m3j, int k, int offset_r1, int offset_r2);
static void interp(double *oz, int mm1, int mm2, int mm3,
                   double *ou, int n1, int n2, int n3, int k, int offset_u1, int offset_u2);
static void norm2u3(int n1, int n2, int n3, double *rnm2, double *rnmu,
                    int nx, int ny, int nz, cl_mem m_buff);
static void rep_nrm(void *u, int n1, int n2, int n3, char *title, int kk);
static void comm3(int n1, int n2, int n3, int kk, cl_mem m_buff, int offset);
static void zran3(double *oz, int n1, int n2, int n3, int nx1, int ny1, int k, cl_mem m_buff, int offset);
static void showall(void *oz, int n1, int n2, int n3);
static double power(double a, int n);
static void zero3(int n1, int n2, int n3, cl_mem m_buff, int offset);


//-------------------------------------------------------------------------c
// These arrays are in common because they are quite large
// and probably shouldn't be allocated on the stack. They
// are always passed as subroutine args.
//-------------------------------------------------------------------------c
/* commcon /noautom/ */
static double u[NR];
static double v[NR];
static double r[NR];

/* common /grid/ */
static int is1, is2, is3, ie1, ie2, ie3;

/* common /rans_save/ starts */
double starts[NM];



int nextpow(int x)
{
	return pow(2, ceil(log(x)/log(2)));
}

int main(int argc, char *argv[])
{
  //-------------------------------------------------------------------------c
  // k is the current level. It is passed down through subroutine args
  // and is NOT global. it is the current iteration
  //-------------------------------------------------------------------------c
  int k, it;
  double t, tinit, mflops;

  double a[4], c[4];

  double rnm2, rnmu, old2, oldu, epsilon;
  int n1, n2, n3, nit;
  double nn, verify_value, err;
  logical verified;

  int i;
  char *t_names[T_last];
  double tmax;

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  for (i = T_init; i < T_last; i++) {
    timer_clear(i);
  }

  timer_start(T_init);

  //---------------------------------------------------------------------
  // Read in and broadcast input data
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";
    t_names[T_mg3P] = "mg3P";
    t_names[T_psinv] = "psinv";
    t_names[T_resid] = "resid";
    t_names[T_rprj3] = "rprj3";
    t_names[T_interp] = "interp";
    t_names[T_norm2] = "norm2";
    t_names[T_comm3] = "comm3";
    fclose(fp);
  } else {
    timeron = false;
  }

  setup_opencl(argc, argv);

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - MG Benchmark\n\n");

  if ((fp = fopen("mg.input", "r")) != NULL) {
    int result;
    printf(" Reading from input file mg.input\n");
    result = fscanf(fp, "%d\n", &lt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d", &nx[lt], &ny[lt], &nz[lt]);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d", &nit);
    while (fgetc(fp) != '\n');
    for (i = 0; i <= 7; i++) {
      result = fscanf(fp, "%d", &debug_vec[i]);
    }
    fclose(fp);
  } else {
    printf(" No input file. Using compiled defaults \n");
    lt = LT_DEFAULT;
    nit = NIT_DEFAULT;
    nx[lt] = NX_DEFAULT;
    ny[lt] = NY_DEFAULT;
    nz[lt] = NZ_DEFAULT;
    for (i = 0; i <= 7; i++) {
      debug_vec[i] = DEBUG_DEFAULT;
    }
  }

  if ( (nx[lt] != ny[lt]) || (nx[lt] != nz[lt]) ) {
    Class = 'U';
  } else if ( nx[lt] == 32 && nit == 4 ) {
    Class = 'S';
  } else if ( nx[lt] == 128 && nit == 4 ) {
    Class = 'W';
  } else if ( nx[lt] == 256 && nit == 4 ) {
    Class = 'A';
  } else if ( nx[lt] == 256 && nit == 20 ) {
    Class = 'B';
  } else if ( nx[lt] == 512 && nit == 20 ) {
    Class = 'C';
  } else if ( nx[lt] == 1024 && nit == 50 ) {
    Class = 'D';
  } else if ( nx[lt] == 2048 && nit == 50 ) {
    Class = 'E';
  } else {
    Class = 'U';
  }

  //---------------------------------------------------------------------
  // Use these for debug info:
  //---------------------------------------------------------------------
  //    debug_vec(0) = 1 !=> report all norms
  //    debug_vec(1) = 1 !=> some setup information
  //    debug_vec(1) = 2 !=> more setup information
  //    debug_vec(2) = k => at level k or below, show result of resid
  //    debug_vec(3) = k => at level k or below, show result of psinv
  //    debug_vec(4) = k => at level k or below, show result of rprj
  //    debug_vec(5) = k => at level k or below, show result of interp
  //    debug_vec(6) = 1 => (unused)
  //    debug_vec(7) = 1 => (unused)
  //---------------------------------------------------------------------
  a[0] = -8.0/3.0;
  a[1] =  0.0;
  a[2] =  1.0/6.0;
  a[3] =  1.0/12.0;

  if (Class == 'A' || Class == 'S' || Class =='W') {
    //---------------------------------------------------------------------
    // Coefficients for the S(a) smoother
    //---------------------------------------------------------------------
    c[0] =  -3.0/8.0;
    c[1] =  +1.0/32.0;
    c[2] =  -1.0/64.0;
    c[3] =   0.0;
  } else {
    //---------------------------------------------------------------------
    // Coefficients for the S(b) smoother
    //---------------------------------------------------------------------
    c[0] =  -3.0/17.0;
    c[1] =  +1.0/33.0;
    c[2] =  -1.0/61.0;
    c[3] =   0.0;
  }
  lb = 1;
  k  = lt;

  setup(&n1, &n2, &n3);
  zero3(n1, n2, n3, m_u, 0);
  zran3(v, n1, n2, n3, nx[lt], ny[lt], k, m_v, 0);

  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt], m_v);
  //  printf("\n");
  //  printf(" norms of random v are\n");
  //  printf("%4d%19.2f%19.2e\n", 0, rnm2, rnmu);
  //  printf(" about to evaluate resid, k=%d\n", k);

  printf(" Size: %4dx%4dx%4d  (class %c)\n", nx[lt], ny[lt], nz[lt], Class);
  printf(" Iterations:                  %5d\n", nit);
  printf("\n");

  resid(r, n1, n2, n3, a, k, m_u, m_v, m_r, 0);
  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt], m_r);
  old2 = rnm2;
  oldu = rnmu;

  //---------------------------------------------------------------------
  // One iteration for startup
  //---------------------------------------------------------------------
  mg3P(u, v, r, a, c, n1, n2, n3);
  resid(r, n1, n2, n3, a, k, m_u, m_v, m_r, 0);
  setup(&n1, &n2, &n3);
  zero3(n1, n2, n3, m_u, 0);
  zran3(v, n1, n2, n3, nx[lt], ny[lt], k, m_v, 0);

  timer_stop(T_init);
  tinit = timer_read(T_init);

  printf(" Initialization time: %15.3f seconds\n\n", tinit);

  for (i = T_bench; i < T_last; i++) {
    timer_clear(i);
  }

  timer_start(T_bench);

  if (timeron) timer_start(T_resid2);
  resid(r, n1, n2, n3, a, k, m_u, m_v, m_r, 0);
  if (timeron) timer_stop(T_resid2);
  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt], m_r);
  old2 = rnm2;
  oldu = rnmu;

  for (it = 1; it <= nit; it++) {
    if ((it == 1) || (it == nit) || ((it % 5) == 0)) {
      printf("  iter %3d\n", it);
    }
    if (timeron) timer_start(T_mg3P);
    mg3P(u, v, r, a, c, n1, n2, n3);
    if (timeron) timer_stop(T_mg3P);
    if (timeron) timer_start(T_resid2);
    resid(r, n1, n2, n3, a, k, m_u, m_v, m_r, 0);
    if (timeron) timer_stop(T_resid2);
#ifndef USE_CHECK_FINISH
    clFinish(cmd_queue);
#endif
  }

  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt], m_r);
#ifndef USE_CHECK_FINISH
  clFinish(cmd_queue);
#endif

  timer_stop(T_bench);

  t = timer_read(T_bench);

  verified = false;
  verify_value = 0.0;

  printf("\n Benchmark completed\n");

  epsilon = 1.0e-8;
  if (Class != 'U') {
    if (Class == 'S') {
      verify_value = 0.5307707005734e-04;
    } else if (Class == 'W') {
      verify_value = 0.6467329375339e-05;
    } else if (Class == 'A') {
      verify_value = 0.2433365309069e-05;
    } else if (Class == 'B') {
      verify_value = 0.1800564401355e-05;
    } else if (Class == 'C') {
      verify_value = 0.5706732285740e-06;
    } else if (Class == 'D') {
      verify_value = 0.1583275060440e-09;
    } else if (Class == 'E') {
      verify_value = 0.5630442584711e-10;
    }

    err = fabs( rnm2 - verify_value ) / verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" L2 Norm is %20.13E\n", rnm2);
      printf(" Error is   %20.13E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" L2 Norm is             %20.13E\n", rnm2);
      printf(" The correct L2 Norm is %20.13E\n", verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
    printf(" L2 Norm is %20.13E\n", rnm2);
  }

  nn = 1.0 * nx[lt] * ny[lt] * nz[lt];

  if (t != 0.0) {
    mflops = 58.0 * nit * nn * 1.0e-6 / t;
  } else {
    mflops = 0.0;
  }

  c_print_results("MG", Class, nx[lt], ny[lt], nz[lt],
                  nit, t,
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
    for (i = T_bench; i < T_last; i++) {
      t = timer_read(i);
      if (i == T_resid2) {
        t = timer_read(T_resid) - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "mg-resid", t, t*100./tmax);
      } else {
        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100./tmax);
      }
    }
  }

  release_opencl();

  fflush(stdout);

  return 0;
}


static void setup(int *n1, int *n2, int *n3)
{
  int k, j;

  int ax, mi[MAXLEVEL+1][3];
  int ng[MAXLEVEL+1][3];

  ng[lt][0] = nx[lt];
  ng[lt][1] = ny[lt];
  ng[lt][2] = nz[lt];
  for (k = lt-1; k >= 1; k--) {
    for (ax = 0; ax < 3; ax++) {
      ng[k][ax] = ng[k+1][ax]/2;
    }
  }
  for (k = lt; k >= 1; k--) {
    nx[k] = ng[k][0];
    ny[k] = ng[k][1];
    nz[k] = ng[k][2];
  }

  for (k = lt; k >= 1; k--) {
    for (ax = 0; ax < 3; ax++) {
      mi[k][ax] = 2 + ng[k][ax];
    }

    m1[k] = mi[k][0];
    m2[k] = mi[k][1];
    m3[k] = mi[k][2];
  }

  k = lt;
  is1 = 2 + ng[k][0] - ng[lt][0];
  ie1 = 1 + ng[k][0];
  *n1 = 3 + ie1 - is1;
  is2 = 2 + ng[k][1] - ng[lt][1];
  ie2 = 1 + ng[k][1];
  *n2 = 3 + ie2 - is2;
  is3 = 2 + ng[k][2] - ng[lt][2];
  ie3 = 1 + ng[k][2];
  *n3 = 3 + ie3 - is3;

  ir[lt] = 0;
  for (j = lt-1; j >= 1; j--) {
    ir[j] = ir[j+1]+ONE*m1[j+1]*m2[j+1]*m3[j+1];
  }

  if (debug_vec[1] >= 1) {
    printf(" in setup, \n");
    printf(" k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
    printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n",
        k,lt,ng[k][0],ng[k][1],ng[k][2],*n1,*n2,*n3,is1,is2,is3,ie1,ie2,ie3);
  }
}


//---------------------------------------------------------------------
// multigrid V-cycle routine
//---------------------------------------------------------------------
static void mg3P(double u[], double v[], double r[],
                 double a[4], double c[4], int n1, int n2, int n3)
{
  int j, k;

  //---------------------------------------------------------------------
  // down cycle.
  // restrict the residual from the find grid to the coarse
  //---------------------------------------------------------------------
  for (k = lt; k >= lb+1; k--) {
    j = k - 1;
    rprj3(&r[ir[k]], m1[k], m2[k], m3[k],
          &r[ir[j]], m1[j], m2[j], m3[j], k, ir[k], ir[j]);
  }

  k = lb;
  //---------------------------------------------------------------------
  // compute an approximate solution on the coarsest grid
  //---------------------------------------------------------------------
  zero3(m1[k], m2[k], m3[k], m_u, ir[k]);
  psinv(&u[ir[k]], m1[k], m2[k], m3[k], c, k, ir[k]);

  for (k = lb+1; k <= lt-1; k++) {
    j = k - 1;

    //---------------------------------------------------------------------
    // prolongate from level k-1  to k
    //---------------------------------------------------------------------
    zero3(m1[k], m2[k], m3[k], m_u, ir[k]);
    interp(&u[ir[j]], m1[j], m2[j], m3[j], &u[ir[k]], m1[k], m2[k], m3[k], k, ir[j], ir[k]);

    //---------------------------------------------------------------------
    // compute residual for level k
    //---------------------------------------------------------------------
    resid(&r[ir[k]], m1[k], m2[k], m3[k], a, k, m_u, m_r, m_r, ir[k]);

    //---------------------------------------------------------------------
    // apply smoother
    //---------------------------------------------------------------------
    psinv(&u[ir[k]], m1[k], m2[k], m3[k], c, k, ir[k]);
  }

  j = lt - 1;
  k = lt;
  interp(&u[ir[j]], m1[j], m2[j], m3[j], u, n1, n2, n3, k, ir[j], 0);
  resid(r, n1, n2, n3, a, k, m_u, m_v, m_r, 0);
  psinv(u, n1, n2, n3, c, k, 0);
}


//---------------------------------------------------------------------
// psinv applies an approximate inverse as smoother:  u = u + Cr
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.
// Presuming coefficient c(3) is zero (the NPB assumes this,
// but it is thus not a general case), 2A + 1M may be eliminated,
// resulting in 13A + 3M.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void psinv(double *ou, int n1, int n2, int n3,
                  double c[4], int k, int offset)
{
  size_t psinv_lws[3], psinv_gws[3];
  cl_int ecode;

  if (timeron) timer_start(T_psinv);

  DTIMER_START(T_BUFFER_CREATE);
	cl_mem m_c = clCreateBuffer(context, 
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              4 * sizeof(double),
                              c, &ecode);
	clu_CheckError(ecode, "clCreateBuffer()");
  DTIMER_STOP(T_BUFFER_CREATE);

  DTIMER_START(T_KERNEL_PSINV);
	if (device_type == CL_DEVICE_TYPE_GPU) {
    size_t lws0 = n1 > max_work_group_size ? max_work_group_size : n1;
    if (PSINV_DIM == 2) {
      psinv_gws[1] = n3-2;
      psinv_gws[0] = (n2-2) * lws0;
      psinv_lws[1] = 1;
      psinv_lws[0] = lws0;
    } else {
      psinv_gws[0] = (n3-2) * (n2-2) * lws0;
      psinv_lws[0] = lws0;
    }
	} else {
    if (PSINV_DIM == 2) {
      psinv_gws[0] = nextpow(n3-2);
      psinv_gws[1] = nextpow(n2-2);
      psinv_lws[0] = 16;
      psinv_lws[1] = 16;
      if (psinv_lws[0] > psinv_gws[0]) {
        psinv_lws[0] = psinv_gws[0];
        psinv_lws[1] = psinv_gws[1];
      }
    } else {
      psinv_gws[0] = n3-2;
      psinv_lws[0] = 1;
    }
	}

	ecode  = clSetKernelArg(kernel_psinv, 0, sizeof(cl_mem), (void *)&m_r);
	ecode |= clSetKernelArg(kernel_psinv, 1, sizeof(cl_mem), (void *)&m_u);
	ecode |= clSetKernelArg(kernel_psinv, 2, sizeof(cl_mem), (void *)&m_c);
	ecode |= clSetKernelArg(kernel_psinv, 3, sizeof(int), &n1);
	ecode |= clSetKernelArg(kernel_psinv, 4, sizeof(int), &n2);
	ecode |= clSetKernelArg(kernel_psinv, 5, sizeof(int), &n3);
	ecode |= clSetKernelArg(kernel_psinv, 6, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_psinv,
                                 PSINV_DIM, NULL,
                                 psinv_gws,
                                 psinv_lws,
                                 0, NULL, NULL);
	clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_PSINV);

  if (timeron) timer_stop(T_psinv);

  //---------------------------------------------------------------------
  // exchange boundary points
  //---------------------------------------------------------------------
  comm3(n1, n2, n3, k, m_u, offset);

  if (debug_vec[0] >= 1) {
    rep_nrm((double (*)[n2][n1])(void *)ou, n1, n2, n3, "   psinv", k);
  }

  if (debug_vec[3] >= k) {
    showall((double (*)[n2][n1])(void *)ou, n1, n2, n3);
  }
}


//---------------------------------------------------------------------
// resid computes the residual:  r = v - Au
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition (or Subtraction) and
// Multiplication, respectively.
// Presuming coefficient a(1) is zero (the NPB assumes this,
// but it is thus not a general case), 3A + 1M may be eliminated,
// resulting in 12A + 3M.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void resid(double *or, int n1, int n2, int n3, double a[4], int k,
                  cl_mem m_bu, cl_mem m_bv, cl_mem m_br, int offset)
{
  size_t resid_lws[3], resid_gws[3];
  cl_int ecode;

  if (timeron) timer_start(T_resid);

  DTIMER_START(T_BUFFER_WRITE);
  ecode = clEnqueueWriteBuffer(cmd_queue,
                               m_a,
                               CL_FALSE,
                               0,
                               sizeof(double) * 4,
                               a,
                               0, NULL, NULL);
  DTIMER_STOP(T_BUFFER_WRITE);

  DTIMER_START(T_KERNEL_RESID);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    size_t lws0 = n1 > max_work_group_size ? max_work_group_size : n1;
    if (RESID_DIM == 2) {
      resid_gws[1] = n3-2;
      resid_gws[0] = (n2-2) * lws0;
      resid_lws[1] = 1;
      resid_lws[0] = lws0;
    } else {
      resid_gws[0] = (n3-2) * (n2-2) * lws0;
      resid_lws[0] = lws0;
    }
  } else {
    if (RESID_DIM == 2) {
      resid_gws[0] = n3-2;
      resid_gws[1] = n2-2;

      resid_lws[0] = 16;
      resid_lws[1] = 16;

      if (resid_lws[0] > resid_gws[0]) {
        resid_lws[0] = resid_gws[0];
        resid_lws[1] = resid_gws[1];
      }
    } else {
      resid_gws[0] = n3-2;
      resid_lws[0] = 1;
    }
  }

  ecode  = clSetKernelArg(kernel_resid, 0, sizeof(cl_mem), (void *)&m_br);
  ecode |= clSetKernelArg(kernel_resid, 1, sizeof(cl_mem), (void *)&m_bu);
  ecode |= clSetKernelArg(kernel_resid, 2, sizeof(cl_mem), (void *)&m_bv);
  ecode |= clSetKernelArg(kernel_resid, 3, sizeof(cl_mem), (void *)&m_a);
  ecode |= clSetKernelArg(kernel_resid, 4, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_resid, 5, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_resid, 6, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_resid, 7, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

#if 0
  int i;
  printf("resid_gws = {");
  for (i = 0; i < RESID_DIM; i++) {
    if (i) printf(", ");
    printf("%lu", resid_gws[i]);
  }
  printf("}\n");
  printf("resid_lws = {");
  for (i = 0; i < RESID_DIM; i++) {
    if (i) printf(", ");
    printf("%lu", resid_lws[i]);
  }
  printf("}\n");
#endif

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_resid,
                                 RESID_DIM, NULL,
                                 resid_gws,
                                 resid_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RESID);

  if (timeron) timer_stop(T_resid);

  //---------------------------------------------------------------------
  // exchange boundary data
  //---------------------------------------------------------------------
  comm3(n1, n2, n3, k, m_br, offset);

  if (debug_vec[0] >= 1) {
    rep_nrm((double (*)[n2][n1])(void*)or, n1, n2, n3, "   resid", k);
  }

  if (debug_vec[2] >= k) {
    showall((double (*)[n2][n1])(void*)or, n1, n2, n3);
  }
}

//---------------------------------------------------------------------
// rprj3 projects onto the next coarser grid,
// using a trilinear Finite Element projection:  s = r' = P r
//
// This  implementation costs  20A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void rprj3(double *or, int m1k, int m2k, int m3k,
                  double *os, int m1j, int m2j, int m3j, int k, int offset_r1, int offset_r2)
{
  int d1, d2, d3, j;
  size_t rprj3_lws[3], rprj3_gws[3];
  cl_int ecode;

  if (timeron) timer_start(T_rprj3);
  if (m1k == 3) {
    d1 = 2;
  } else {
    d1 = 1;
  }

  if (m2k == 3) {
    d2 = 2;
  } else {
    d2 = 1;
  }

  if (m3k == 3) {
    d3 = 2;
  } else {
    d3 = 1;
  }

  DTIMER_START(T_KERNEL_RPRJ3);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (RPRJ3_DIM == 2) {
      rprj3_gws[1] = m3j-2;
      rprj3_gws[0] = (m2j-2) * (m1j-1);
      rprj3_lws[1] = 1;
      rprj3_lws[0] = m1j-1;
    } else {
      rprj3_gws[0] = (m3j-2) * (m2j-2) * (m1j-1);
      rprj3_lws[0] = m1j-1;
    }
  } else {
    if (RPRJ3_DIM == 2) {
      rprj3_gws[0] = nextpow(m3j-2);
      rprj3_gws[1] = nextpow(m2j-2);
      rprj3_lws[0] = 8;
      rprj3_lws[1] = 8;

      if (rprj3_lws[0] > rprj3_gws[0]) {
        rprj3_lws[0] = rprj3_gws[0];
        rprj3_lws[1] = rprj3_gws[1];
      }
    } else {
      rprj3_gws[0] = m3j-2;
      rprj3_lws[0] = 1;
    }
  }

	ecode  = clSetKernelArg(kernel_rprj3, 0, sizeof(cl_mem), (void *)&m_r);
	ecode |= clSetKernelArg(kernel_rprj3, 1, sizeof(int), &m1k);
	ecode |= clSetKernelArg(kernel_rprj3, 2, sizeof(int), &m2k);
	ecode |= clSetKernelArg(kernel_rprj3, 3, sizeof(int), &m3k);
	ecode |= clSetKernelArg(kernel_rprj3, 4, sizeof(int), &m1j);
	ecode |= clSetKernelArg(kernel_rprj3, 5, sizeof(int), &m2j);
	ecode |= clSetKernelArg(kernel_rprj3, 6, sizeof(int), &m3j);
	ecode |= clSetKernelArg(kernel_rprj3, 7, sizeof(int), &offset_r1);
	ecode |= clSetKernelArg(kernel_rprj3, 8, sizeof(int), &offset_r2);
	ecode |= clSetKernelArg(kernel_rprj3, 9, sizeof(int), &d1);
	ecode |= clSetKernelArg(kernel_rprj3, 10, sizeof(int), &d2);
	ecode |= clSetKernelArg(kernel_rprj3, 11, sizeof(int), &d3);
  clu_CheckError(ecode, "clSetKernelSetArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_rprj3, 
                                 RPRJ3_DIM, NULL,
                                 rprj3_gws,
                                 rprj3_lws,
                                 0, NULL, NULL);
	clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RPRJ3);

  if (timeron) timer_stop(T_rprj3);

  j = k-1;
  comm3(m1j, m2j, m3j, j, m_r, offset_r2);

  if (debug_vec[0] >= 1) {
    rep_nrm((double(*)[m2j][m1j])(void*)os, m1j, m2j, m3j, "   rprj3", k-1);
  }

  if (debug_vec[4] >= k) {
    showall((double(*)[m2j][m1j])(void*)os, m1j, m2j, m3j);
  }
}


//---------------------------------------------------------------------
// interp adds the trilinear interpolation of the correction
// from the coarser grid to the current approximation:  u = u + Qu'
//
// Observe that this  implementation costs  16A + 4M, where
// A and M denote the costs of Addition and Multiplication.
// Note that this vectorizes, and is also fine for cache
// based machines.  Vector machines may get slightly better
// performance however, with 8 separate "do i1" loops, rather than 4.
//---------------------------------------------------------------------
static void interp(double *oz, int mm1, int mm2, int mm3,
                   double *ou, int n1, int n2, int n3, int k, int offset_u1, int offset_u2)
{
  int d1, d2, d3, t1, t2, t3;
  cl_int ecode;

  // note that m = 1037 in globals.h but for this only need to be
  // 535 to handle up to 1024^3
  //      integer m
  //      parameter( m=535 )

  if (timeron) timer_start(T_interp);
  if (n1 != 3 && n2 != 3 && n3 != 3) {

    DTIMER_START(T_KERNEL_INTERP_1);
    if (device_type == CL_DEVICE_TYPE_GPU) {
      if (INTERP_1_DIM == 2) {
        global[1] = mm3-1;
        global[0] = (mm2-1) * mm1;
        local[1] = 1;
        local[0] = mm1;
      } else {
        global[0] = (mm3-1) * (mm2-1) * mm1;
        local[0] = mm1;
      }
    } else {
      if (INTERP_1_DIM == 2) {
        global[0] = nextpow(mm3-1);
        global[1] = nextpow(mm2-1);
        local[0] = 8;
        local[1] = 8;

        if (local[0] > global[0]) {
          local[0] = global[0];
          local[1] = global[1];
        }
      } else {
        global[0] = mm3-1;
        local[0] = 1;
      }
    }

    ecode  = clSetKernelArg(kernel_interp_1, 0, sizeof(cl_mem), (void *)&m_u);
    ecode |= clSetKernelArg(kernel_interp_1, 1, sizeof(int), &mm1);
    ecode |= clSetKernelArg(kernel_interp_1, 2, sizeof(int), &mm2);
    ecode |= clSetKernelArg(kernel_interp_1, 3, sizeof(int), &mm3);
    ecode |= clSetKernelArg(kernel_interp_1, 4, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_interp_1, 5, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_interp_1, 6, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_interp_1, 7, sizeof(int), &offset_u1);
    ecode |= clSetKernelArg(kernel_interp_1, 8, sizeof(int), &offset_u2);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_interp_1,
                                   INTERP_1_DIM, NULL,
                                   global, local,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_INTERP_1);

  } else {

    if (n1 == 3) {
      d1 = 2;
      t1 = 1;
    } else {
      d1 = 1;
      t1 = 0;
    }

    if (n2 == 3) {
      d2 = 2;
      t2 = 1;
    } else {
      d2 = 1;
      t2 = 0;
    }

    if (n3 == 3) {
      d3 = 2;
      t3 = 1;
    } else {
      d3 = 1;
      t3 = 0;
    }
    //breaking the following for into two parts

    DTIMER_START(T_KERNEL_INTERP_2);
    global[0] = nextpow(mm3);
    global[1] = nextpow(mm2);
    local[0] = 4;
    local[1] = 4;
    ecode  = clSetKernelArg(kernel_interp_2, 0, sizeof(cl_mem), (void *)&m_u);
    ecode |= clSetKernelArg(kernel_interp_2, 1, sizeof(int), &mm1);
    ecode |= clSetKernelArg(kernel_interp_2, 2, sizeof(int), &mm2);
    ecode |= clSetKernelArg(kernel_interp_2, 3, sizeof(int), &mm3);
    ecode |= clSetKernelArg(kernel_interp_2, 4, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_interp_2, 5, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_interp_2, 6, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_interp_2, 7, sizeof(int), &offset_u1);
    ecode |= clSetKernelArg(kernel_interp_2, 8, sizeof(int), &offset_u2);
    ecode |= clSetKernelArg(kernel_interp_2, 9, sizeof(int), &d1);
    ecode |= clSetKernelArg(kernel_interp_2, 10, sizeof(int), &d2);
    ecode |= clSetKernelArg(kernel_interp_2, 11, sizeof(int), &d3);
    ecode |= clSetKernelArg(kernel_interp_2, 12, sizeof(int), &t1);
    ecode |= clSetKernelArg(kernel_interp_2, 13, sizeof(int), &t2);
    ecode |= clSetKernelArg(kernel_interp_2, 14, sizeof(int), &t3);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_interp_2,
                                   2, NULL,
                                   global, local,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    clFinish(cmd_queue);
    DTIMER_STOP(T_KERNEL_INTERP_2);

    if (device_type == CL_DEVICE_TYPE_GPU) {
      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_u,
                                  CL_TRUE,
                                  offset_u2 * sizeof(double),
                                  (n1*n2*n3) * sizeof(double),
                                  ou,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    DTIMER_START(T_KERNEL_INTERP_3);
    global[0] = nextpow(mm3);
    global[1] = nextpow(mm2);
    local[0] = 4;
    local[1] = 4;
    ecode  = clSetKernelArg(kernel_interp_3, 0, sizeof(cl_mem), (void *)&m_u);
    ecode |= clSetKernelArg(kernel_interp_3, 1, sizeof(int), &mm1);
    ecode |= clSetKernelArg(kernel_interp_3, 2, sizeof(int), &mm2);
    ecode |= clSetKernelArg(kernel_interp_3, 3, sizeof(int), &mm3);
    ecode |= clSetKernelArg(kernel_interp_3, 4, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_interp_3, 5, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_interp_3, 6, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_interp_3, 7, sizeof(int), &offset_u1);
    ecode |= clSetKernelArg(kernel_interp_3, 8, sizeof(int), &offset_u2);
    ecode |= clSetKernelArg(kernel_interp_3, 9, sizeof(int), &d1);
    ecode |= clSetKernelArg(kernel_interp_3, 10, sizeof(int), &d2);
    ecode |= clSetKernelArg(kernel_interp_3, 11, sizeof(int), &d3);
    ecode |= clSetKernelArg(kernel_interp_3, 12, sizeof(int), &t1);
    ecode |= clSetKernelArg(kernel_interp_3, 13, sizeof(int), &t2);
    ecode |= clSetKernelArg(kernel_interp_3, 14, sizeof(int), &t3);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_interp_3,
                                   2, NULL,
                                   global, local,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    clFinish(cmd_queue);
    DTIMER_STOP(T_KERNEL_INTERP_3);

    if (device_type == CL_DEVICE_TYPE_GPU) {
      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_u,
                                  CL_TRUE,
                                  offset_u2 * sizeof(double),
                                  (n1*n2*n3) * sizeof(double),
                                  ou,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    DTIMER_START(T_KERNEL_INTERP_4);
    global[0] = nextpow(mm3);
    global[1] = nextpow(mm2);
    local[0] = 4;
    local[1] = 4;
    ecode  = clSetKernelArg(kernel_interp_4, 0, sizeof(cl_mem), (void *)&m_u);
    ecode |= clSetKernelArg(kernel_interp_4, 1, sizeof(int), &mm1);
    ecode |= clSetKernelArg(kernel_interp_4, 2, sizeof(int), &mm2);
    ecode |= clSetKernelArg(kernel_interp_4, 3, sizeof(int), &mm3);
    ecode |= clSetKernelArg(kernel_interp_4, 4, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_interp_4, 5, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_interp_4, 6, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_interp_4, 7, sizeof(int), &offset_u1);
    ecode |= clSetKernelArg(kernel_interp_4, 8, sizeof(int), &offset_u2);
    ecode |= clSetKernelArg(kernel_interp_4, 9, sizeof(int), &d1);
    ecode |= clSetKernelArg(kernel_interp_4, 10, sizeof(int), &d2);
    ecode |= clSetKernelArg(kernel_interp_4, 11, sizeof(int), &d3);
    ecode |= clSetKernelArg(kernel_interp_4, 12, sizeof(int), &t1);
    ecode |= clSetKernelArg(kernel_interp_4, 13, sizeof(int), &t2);
    ecode |= clSetKernelArg(kernel_interp_4, 14, sizeof(int), &t3);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_interp_4,
                                   2, NULL,
                                   global, local,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    clFinish(cmd_queue);
    DTIMER_STOP(T_KERNEL_INTERP_4);

    if (device_type == CL_DEVICE_TYPE_GPU) {
      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_u,
                                  CL_TRUE,
                                  offset_u2 * sizeof(double),
                                  (n1*n2*n3) * sizeof(double),
                                  ou,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

    DTIMER_START(T_KERNEL_INTERP_5);
    global[0] = nextpow(mm3);
    global[1] = nextpow(mm2);
    local[0] = 4;
    local[1] = 4;
    ecode  = clSetKernelArg(kernel_interp_5, 0, sizeof(cl_mem), (void *)&m_u);
    ecode |= clSetKernelArg(kernel_interp_5, 1, sizeof(int), &mm1);
    ecode |= clSetKernelArg(kernel_interp_5, 2, sizeof(int), &mm2);
    ecode |= clSetKernelArg(kernel_interp_5, 3, sizeof(int), &mm3);
    ecode |= clSetKernelArg(kernel_interp_5, 4, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_interp_5, 5, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_interp_5, 6, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_interp_5, 7, sizeof(int), &offset_u1);
    ecode |= clSetKernelArg(kernel_interp_5, 8, sizeof(int), &offset_u2);
    ecode |= clSetKernelArg(kernel_interp_5, 9, sizeof(int), &d1);
    ecode |= clSetKernelArg(kernel_interp_5, 10, sizeof(int), &d2);
    ecode |= clSetKernelArg(kernel_interp_5, 11, sizeof(int), &d3);
    ecode |= clSetKernelArg(kernel_interp_5, 12, sizeof(int), &t1);
    ecode |= clSetKernelArg(kernel_interp_5, 13, sizeof(int), &t2);
    ecode |= clSetKernelArg(kernel_interp_5, 14, sizeof(int), &t3);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_interp_5,
                                   2, NULL,
                                   global, local,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    clFinish(cmd_queue);
    DTIMER_STOP(T_KERNEL_INTERP_5);

    if (device_type == CL_DEVICE_TYPE_GPU) {
      DTIMER_START(T_BUFFER_READ);
      ecode = clEnqueueReadBuffer(cmd_queue,
                                  m_u,
                                  CL_TRUE,
                                  offset_u2 * sizeof(double),
                                  (n1*n2*n3) * sizeof(double),
                                  ou,
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
      DTIMER_STOP(T_BUFFER_READ);
    }

  }
  if (timeron) timer_stop(T_interp);

  if (debug_vec[0] >= 1) {
    rep_nrm((double (*)[mm2][mm1])(void*)oz, mm1, mm2, mm3, "z: inter", k-1);
    rep_nrm((double (*)[n1][n1])(void*)ou, n1, n2, n3, "u: inter", k);
  }

  if (debug_vec[5] >= k) {
    showall((double (*)[mm2][mm1])(void*)oz, mm1, mm2, mm3);
    showall((double (*)[n2][n1])(void*)ou, n1, n2, n3);
  }
}


//---------------------------------------------------------------------
// norm2u3 evaluates approximations to the L2 norm and the
// uniform (or L-infinity or Chebyshev) norm, under the
// assumption that the boundaries are periodic or zero.  Add the
// boundaries in with half weight (quarter weight on the edges
// and eighth weight at the corners) for inhomogeneous boundaries.
//---------------------------------------------------------------------
static void norm2u3(int n1, int n2, int n3, double *rnm2, double *rnmu,
                    int nx, int ny, int nz, cl_mem m_buff)
{
  double s;

  double dn, max_rnmu;

  size_t norm2_lws[3], norm2_gws[3];
  int temp_size;
  cl_mem m_sum, m_max;
  cl_int ecode;

  if (timeron) timer_start(T_norm2);
  dn = 1.0*nx*ny*nz;

  s = 0.0;
  max_rnmu = 0.0;

  DTIMER_START(T_BUFFER_CREATE);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (NORM2U3_DIM == 2) {
      norm2_lws[0] = 128;
      norm2_lws[1] = 1;
      norm2_gws[0] = (n2-2) * norm2_lws[0];
      norm2_gws[1] = n3-2;
      temp_size = (norm2_gws[0]*norm2_gws[1]) / (norm2_lws[0]*norm2_lws[1]);
    } else {
      norm2_lws[0] = 128;
      norm2_gws[0] = norm2_lws[0] * (n2-2) * (n3-2);
      temp_size = norm2_gws[0] / norm2_lws[0];
    }
  } else {
    norm2_gws[0] = n3-2;
    norm2_lws[0] = 1;
    temp_size = norm2_gws[0] / norm2_lws[0];
  }

  double *res_sum = (double*)malloc(temp_size * sizeof(double));
  double *res_max = (double*)malloc(temp_size * sizeof(double));

  if (device_type == CL_DEVICE_TYPE_GPU) {
    m_sum = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           temp_size * sizeof(double),
                           0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_max = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           temp_size * sizeof(double),
                           0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  } else {
    m_sum = clCreateBuffer(context,
                           CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                           temp_size * sizeof(double),
                           res_sum,
                           &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_max = clCreateBuffer(context,
                           CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                           temp_size * sizeof(double),
                           res_max,
                           &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  }
  DTIMER_STOP(T_BUFFER_CREATE);

  DTIMER_START(T_KERNEL_NORM2U3);
  ecode  = clSetKernelArg(kernel_norm2u3, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_norm2u3, 1, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_norm2u3, 2, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_norm2u3, 3, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_norm2u3, 4, sizeof(cl_mem), (void *)&m_sum);
  ecode |= clSetKernelArg(kernel_norm2u3, 5, sizeof(cl_mem), (void *)&m_max);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    size_t lsize = norm2_lws[0] * sizeof(double);
    ecode |= clSetKernelArg(kernel_norm2u3, 6, lsize, NULL);
    ecode |= clSetKernelArg(kernel_norm2u3, 7, lsize, NULL);
  }
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_norm2u3,
                                 NORM2U3_DIM, NULL,
                                 norm2_gws,
                                 norm2_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_NORM2U3);

    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_sum,
                                CL_FALSE, 0,
                                (temp_size) * sizeof(double),
                                res_sum,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");

    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_max,
                                CL_TRUE, 0,
                                (temp_size) * sizeof(double),
                                res_max,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  } else {
    ecode = clFinish(cmd_queue);
    clu_CheckError(ecode, "clFinish()");
    DTIMER_STOP(T_KERNEL_NORM2U3);
  }

  int j;
  for (j = 0; j < temp_size; j++) {
    s = s + res_sum[j];
    if (max_rnmu < res_max[j]) max_rnmu = res_max[j];
  }

  DTIMER_START(T_RELEASE);
  clReleaseMemObject(m_sum);
  clReleaseMemObject(m_max);
  free(res_sum);
  free(res_max);
  DTIMER_STOP(T_RELEASE);

  *rnmu = max_rnmu;

  *rnm2 = sqrt(s / dn);
  if (timeron) timer_stop(T_norm2);
}


//---------------------------------------------------------------------
// report on norm
//---------------------------------------------------------------------
static void rep_nrm(void *u, int n1, int n2, int n3, char *title, int kk)
{
//  double rnm2, rnmu;
//
//  norm2u3(u, n1, n2, n3, &rnm2, &rnmu, nx[kk], ny[kk], nz[kk]);
//  printf(" Level%2d in %8s: norms =%21.14E%21.14E\n", kk, title, rnm2, rnmu);
}

//---------------------------------------------------------------------
// comm3 organizes the communication on all borders
//---------------------------------------------------------------------
static void comm3(int n1, int n2, int n3, int kk, cl_mem m_buff, int offset)
{
  size_t comm3_lws[3], comm3_gws[3];
  cl_int ecode;

  if (timeron) timer_start(T_comm3);

  DTIMER_START(T_KERNEL_COMM3_1);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (COMM3_1_DIM == 2) {
      comm3_lws[1] = 1;
      comm3_lws[0] = 32;
      comm3_gws[1] = n3-2;
      comm3_gws[0] = clu_RoundWorkSize((size_t)(n2-2), comm3_lws[0]);
    } else {
      comm3_lws[0] = 32;
      comm3_gws[0] = (n3-2) * comm3_lws[0];
    }
  } else {
    comm3_gws[0] = nextpow(n3-2);
    comm3_lws[0] = (16 > comm3_gws[0]) ? comm3_gws[0] : 16;
  }

  ecode  = clSetKernelArg(kernel_comm3_1, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_comm3_1, 1, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_comm3_1, 2, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_comm3_1, 3, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_comm3_1, 4, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_comm3_1,
                                 COMM3_1_DIM, NULL,
                                 comm3_gws,
                                 comm3_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_COMM3_1);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_KERNEL_COMM3_2);
    if (COMM3_2_DIM == 2) {
      comm3_lws[1] = 1;
      comm3_lws[0] = 32;
      comm3_gws[1] = n3-2;
      comm3_gws[0] = clu_RoundWorkSize((size_t)n1, comm3_lws[0]);
    } else {
      comm3_lws[0] = 32;
      comm3_gws[0] = (n3-2) * comm3_lws[0];
    }

    ecode  = clSetKernelArg(kernel_comm3_2, 0, sizeof(cl_mem),(void*)&m_buff);
    ecode |= clSetKernelArg(kernel_comm3_2, 1, sizeof(int), &n1);
    ecode |= clSetKernelArg(kernel_comm3_2, 2, sizeof(int), &n2);
    ecode |= clSetKernelArg(kernel_comm3_2, 3, sizeof(int), &n3);
    ecode |= clSetKernelArg(kernel_comm3_2, 4, sizeof(int), &offset);
    clu_CheckError(ecode, "clSetKernelArg()");

    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_comm3_2,
                                   COMM3_2_DIM, NULL,
                                   comm3_gws,
                                   comm3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_COMM3_2);
  }

  DTIMER_START(T_KERNEL_COMM3_3);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (COMM3_3_DIM == 2) {
      comm3_lws[1] = 1;
      comm3_lws[0] = 32;
      comm3_gws[1] = n2;
      comm3_gws[0] = clu_RoundWorkSize((size_t)n1, comm3_lws[0]);
    } else {
      comm3_lws[0] = 32;
      comm3_gws[0] = n2 * comm3_lws[0];
    }
  } else {
    if (COMM3_3_DIM == 2) {
      comm3_lws[1] = 1;
      comm3_lws[0] = 64;
      comm3_gws[1] = n2;
      comm3_gws[0] = clu_RoundWorkSize((size_t)n1, comm3_lws[0]);
    } else {
      comm3_gws[0] = nextpow(n2);
      comm3_lws[0] = (16 > comm3_gws[0]) ? comm3_gws[0] : 16;
    }
  }
  ecode  = clSetKernelArg(kernel_comm3_3, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_comm3_3, 1, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_comm3_3, 2, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_comm3_3, 3, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_comm3_3, 4, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_comm3_3,
                                 COMM3_3_DIM, NULL,
                                 comm3_gws,
                                 comm3_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_COMM3_3);

  if (timeron) timer_stop(T_comm3);
}


//---------------------------------------------------------------------
// zran3  loads +1 at ten randomly chosen points,
// loads -1 at a different ten random points,
// and zero elsewhere.
//---------------------------------------------------------------------
static void zran3(double *oz, int n1, int n2, int n3, int nx1, int ny1, int k, cl_mem m_buff, int offset)
{
  cl_int ecode;
  int i0, mm0, mm1;

  int i1, i2, i3, d1, e1, e2, e3;
  double x0, a1, a2, ai;

  const int mm = 10;
  const double a = pow(5.0, 13.0);
  const double x = 314159265.0;
  double (*ten)[mm][2], best0, best1;

  int i;
  int (*j1)[mm][2];
  int (*j2)[mm][2];
  int (*j3)[mm][2];
  int jg[4][mm][2];

  ten = (double(*)[mm][2])malloc(sizeof(double) * mm * 2 * n3);
  j1 = (int(*)[mm][2])malloc(sizeof(int) * mm * 2 * n3);
  j2 = (int(*)[mm][2])malloc(sizeof(int) * mm * 2 * n3);
  j3 = (int(*)[mm][2])malloc(sizeof(int) * mm * 2 * n3);

  double rdummy;
  int myid, num_threads;

  a1 = power(a, nx1);
  a2 = power(a, nx1*ny1);

  zero3(n1, n2, n3, m_buff, offset);

  i = is1-2+nx1*(is2-2+ny1*(is3-2));

  ai = power(a, i);
  d1 = ie1 - is1 + 1;
  e1 = ie1 - is1 + 2;
  e2 = ie2 - is2 + 2;
  e3 = ie3 - is3 + 2;
  x0 = x;
  rdummy = randlc(&x0, ai);

  //---------------------------------------------------------------------
  // save the starting seeds for the following loop
  //---------------------------------------------------------------------
  for (i3 = 1; i3 < e3; i3++) {
    starts[i3] = x0;
    rdummy = randlc(&x0, a2);
  }

  //---------------------------------------------------------------------
  // fill array
  //---------------------------------------------------------------------
  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_WRITE);
    ecode = clEnqueueWriteBuffer(cmd_queue,
                                 m_buff,
                                 CL_FALSE,
                                 offset * sizeof(double),
                                 (n1*n2*n3) * sizeof(double),
                                 oz,
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer()");

    ecode = clEnqueueWriteBuffer(cmd_queue,
                                 m_starts,
                                 CL_TRUE,
                                 0,
                                 (NM) * sizeof(double), 
                                 starts,
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer()");
    DTIMER_STOP(T_BUFFER_WRITE);
  }

  DTIMER_START(T_KERNEL_ZRAN3_1);
  global[0] = nextpow(e3);
  local[0] = 4;
  ecode  = clSetKernelArg(kernel_zran3_1, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_zran3_1, 1, sizeof(cl_mem), (void *)&m_starts);
  ecode |= clSetKernelArg(kernel_zran3_1, 2, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_zran3_1, 3, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_zran3_1, 4, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_zran3_1, 5, sizeof(int), &offset);
  ecode |= clSetKernelArg(kernel_zran3_1, 6, sizeof(int), &e2);
  ecode |= clSetKernelArg(kernel_zran3_1, 7, sizeof(int), &e3);
  ecode |= clSetKernelArg(kernel_zran3_1, 8, sizeof(int), &d1);
  ecode |= clSetKernelArg(kernel_zran3_1, 9, sizeof(double), &a1);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue, kernel_zran3_1, 1, NULL, global, local, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  clFinish(cmd_queue);
  DTIMER_STOP(T_KERNEL_ZRAN3_1);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_buff,
                                CL_TRUE,
                                offset * sizeof(double),
                                (n1*n2*n3) * sizeof(double),
                                oz,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  }

  //---------------------------------------------------------------------
  // comm3(z,n1,n2,n3);
  // showall(z,n1,n2,n3);
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // each thread looks for twenty candidates
  //---------------------------------------------------------------------

  cl_mem m_ten, m_j1, m_j2, m_j3;

  DTIMER_START(T_BUFFER_CREATE);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    m_ten = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           n3*mm*2 * sizeof(double),
                           0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j1 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          n3*mm*2 * sizeof(int),
                          0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j2 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          n3*mm*2 * sizeof(int),
                          0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j3 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          n3*mm*2 * sizeof(int),
                          0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  } else {
    m_ten = clCreateBuffer(context,
                           CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                           n3*mm*2 * sizeof(double),
                           ten,
                           &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j1 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                          n3*mm*2 * sizeof(int),
                          j1,
                          &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j2 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                          n3*mm*2 * sizeof(int),
                          j2,
                          &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_j3 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                          n3*mm*2 * sizeof(int),
                          j3,
                          &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  }
  DTIMER_STOP(T_BUFFER_CREATE);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_WRITE);
    ecode = clEnqueueWriteBuffer(cmd_queue,
                                 m_buff,
                                 CL_TRUE,
                                 offset * sizeof(double),
                                 (n1*n2*n3) * sizeof(double),
                                 oz,
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer()");
    DTIMER_STOP(T_BUFFER_WRITE);
  }

  DTIMER_START(T_KERNEL_ZRAN3_2);
  global[0] = nextpow(n3);
  local[0] = 4;

  ecode  = clSetKernelArg(kernel_zran3_2, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_zran3_2, 1, sizeof(cl_mem), (void *)&m_ten);
  ecode |= clSetKernelArg(kernel_zran3_2, 2, sizeof(cl_mem), (void *)&m_j1);
  ecode |= clSetKernelArg(kernel_zran3_2, 3, sizeof(cl_mem), (void *)&m_j2);
  ecode |= clSetKernelArg(kernel_zran3_2, 4, sizeof(cl_mem), (void *)&m_j3);
  ecode |= clSetKernelArg(kernel_zran3_2, 5, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_zran3_2, 6, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_zran3_2, 7, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_zran3_2, 8, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_zran3_2,
                                 1, NULL,
                                 global, local,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  clFinish(cmd_queue);
  DTIMER_STOP(T_KERNEL_ZRAN3_2);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_ten,
                                CL_TRUE,
                                0,
                                (mm*2*n3) * sizeof(double),
                                ten,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");

    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_j1,
                                CL_TRUE,
                                0,
                                (n3*mm*2) * sizeof(int),
                                j1,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");

    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_j2,
                                CL_TRUE,
                                0,
                                (n3*mm*2) * sizeof(int),
                                j2,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");

    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_j3,
                                CL_TRUE,
                                0,
                                (n3*mm*2) * sizeof(int),
                                j3,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  }

  //---------------------------------------------------------------------
  // Now which of these are globally best?
  //---------------------------------------------------------------------
  i1 = mm - 1;
  i0 = mm - 1;

  int *a_i1 = (int*)malloc(sizeof(int)*n3);
  int *a_i0 = (int*)malloc(sizeof(int)*n3);
  for (i = 0; i < n3; i++)
  {
    a_i0[i] = mm - 1;
    a_i1[i] = mm - 1;
  }


  myid = 0;

  num_threads = n3;
  for (i = mm - 1; i >= 0; i--) {
    // ... ORDERED access is required here for sequential consistency
    // ... in case that two values are identical.
    // ... Since an "ORDERED" section is only defined in OpenMP 2,
    // ... we use a dummy loop to emulate ordered access in OpenMP 1.x.

    best1 = 0.0;
    best0 = 1.0;

    for (i2 = 1; i2 <= num_threads; i2++)
    {
      myid = i2-1;
      {
        if (ten[i2-1][a_i1[i2-1]][1] > best1) {
          best1 = ten[i2-1][a_i1[i2-1]][1];
          jg[0][i][1] = myid;
        }
        if (ten[i2-1][a_i0[i2-1]][0] < best0) {
          best0 = ten[i2-1][a_i0[i2-1]][0];
          jg[0][i][0] = myid;
        }
      }
    }

    for (i2 = 0; i2 < num_threads; i2++)//This simulates the thread parallelization
    {
      myid = i2;

      if (myid == jg[0][i][1]) {
        jg[1][i][1] = j1[i2][a_i1[i2]][1];
        jg[2][i][1] = j2[i2][a_i1[i2]][1];
        jg[3][i][1] = j3[i2][a_i1[i2]][1];
        a_i1[i2] = a_i1[i2]-1;
      }

      if (myid == jg[0][i][0]) {
        jg[1][i][0] = j1[i2][a_i0[i2]][0];
        jg[2][i][0] = j2[i2][a_i0[i2]][0];
        jg[3][i][0] = j3[i2][a_i0[i2]][0];
        a_i0[i2] = a_i0[i2]-1;
      }
    }
  }

  mm1 = 0;
  mm0 = 0;

  DTIMER_START(T_KERNEL_ZRAN3_3);
  global[0] = nextpow(n3);
  global[1] = nextpow(n2);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    global[2] = nextpow(n1);
    local[0] = 4;
    local[1] = 4;
    local[2] = 4;
  } else {
    local[0] = 8;
    local[1] = 8;
  }

  ecode  = clSetKernelArg(kernel_zran3_3, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_zran3_3, 1, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_zran3_3, 2, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_zran3_3, 3, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_zran3_3, 4, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_zran3_3,
                                   3, NULL,
                                   global, local,
                                   0, NULL, NULL);
  } else {
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   kernel_zran3_3,
                                   2, NULL,
                                   global, local,
                                   0, NULL, NULL);
  }
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");
  DTIMER_STOP(T_KERNEL_ZRAN3_3);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_buff,
                                CL_TRUE,
                                offset * sizeof(double),
                                (n1*n2*n3) * sizeof(double),
                                oz,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    DTIMER_STOP(T_BUFFER_READ);
  }

  for (i = mm-1; i >= mm0; i--) {
    oz[(jg[3][i][0])*n2*n1+(jg[2][i][0])*n1+jg[1][i][0]] = -1.0;
  }

  for (i = mm-1; i >= mm1; i--) {
    oz[(jg[3][i][1])*n2*n1+(jg[2][i][1])*n1+jg[1][i][1]] = +1.0;
  }

  if (device_type == CL_DEVICE_TYPE_GPU) {
    DTIMER_START(T_BUFFER_WRITE);
    ecode = clEnqueueWriteBuffer(cmd_queue,
                                 m_buff,
                                 CL_TRUE,
                                 offset * sizeof(double),
                                 (n1*n2*n3) * sizeof(double),
                                 oz,
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer()");
    DTIMER_STOP(T_BUFFER_WRITE);
  }

  comm3(n1, n2, n3, k, m_buff, offset);
#ifndef USE_CHECK_FINISH
  clFinish(cmd_queue);
#endif

  //---------------------------------------------------------------------
  // showall(z,n1,n2,n3);
  //---------------------------------------------------------------------
  DTIMER_START(T_RELEASE);
  free(ten);
  free(j1);
  free(j2);
  free(j3);
  DTIMER_STOP(T_RELEASE);
}


static void showall(void *oz, int n1, int n2, int n3)
{
//  double (*z)[n2][n1] = (double (*)[n2][n1])oz;
//
//  int i1, i2, i3;
//  int m1, m2, m3;
//
//  m1 = min(n1, 18);
//  m2 = min(n2, 14);
//  m3 = min(n3, 18);
//
//  printf("   \n");
//  for (i3 = 0; i3 < m3; i3++) {
//    for (i1 = 0; i1 < m1; i1++) {
//      for (i2 = 0; i2 < m2; i2++) {
//        printf("%6.3f", z[i3][i2][i1]);
//      }
//      printf("\n");
//    }
//    printf("  - - - - - - - \n");
//  }
//  printf("   \n");
}


//---------------------------------------------------------------------
// power  raises an integer, disguised as a double
// precision real, to an integer power
//---------------------------------------------------------------------
static double power(double a, int n)
{
  double aj;
  int nj;
  double rdummy;
  double power;

  power = 1.0;
  nj = n;
  aj = a;

  while (nj != 0) {
    if ((nj % 2) == 1) rdummy = randlc(&power, aj);
    rdummy = randlc(&aj, aj);
    nj = nj/2;
  }

  return power;
}

//buff and offset are the parameters used to co-ordinate to the global parameters
static void zero3(int n1, int n2, int n3, cl_mem m_buff, int offset)
{
  size_t zero3_lws[3], zero3_gws[3];
  cl_int ecode;

  DTIMER_START(T_KERNEL_ZERO3);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (ZERO3_DIM == 3) {
      zero3_lws[2] = 1;
      zero3_lws[1] = 1;
      zero3_lws[0] = 64;
      zero3_gws[2] = n3;
      zero3_gws[1] = n2;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n1, zero3_lws[0]);
    } else if (ZERO3_DIM ==2) {
      zero3_lws[1] = 1;
      zero3_lws[0] = 32;
      zero3_gws[1] = n3;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n2, zero3_lws[0]);
    } else {
      zero3_lws[0] = 32;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n3, zero3_lws[0]);
    }
  } else {
    if (ZERO3_DIM == 3) {
      zero3_lws[2] = 1;
      zero3_lws[1] = 1;
      zero3_lws[0] = NX_DEFAULT;
      zero3_gws[2] = n3;
      zero3_gws[1] = n2;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n1, zero3_lws[0]);
    } else if (ZERO3_DIM ==2) {
      zero3_lws[1] = 1;
      zero3_lws[0] = NY_DEFAULT;
      zero3_gws[1] = n3;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n2, zero3_lws[0]);
    } else {
      zero3_lws[0] = 32;
      zero3_gws[0] = clu_RoundWorkSize((size_t)n3, zero3_lws[0]);
    }
  }

  ecode  = clSetKernelArg(kernel_zero3, 0, sizeof(cl_mem), (void *)&m_buff);
  ecode |= clSetKernelArg(kernel_zero3, 1, sizeof(int), &n1);
  ecode |= clSetKernelArg(kernel_zero3, 2, sizeof(int), &n2);
  ecode |= clSetKernelArg(kernel_zero3, 3, sizeof(int), &n3);
  ecode |= clSetKernelArg(kernel_zero3, 4, sizeof(int), &offset);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel_zero3,
                                 ZERO3_DIM, NULL,
                                 zero3_gws,
                                 zero3_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_ZERO3);
}


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
static void setup_opencl(int argc, char *argv[])
{
  cl_int ecode;
  char *source_dir = "MG";
  if (argc > 1) source_dir = argv[1];

#ifdef TIMER_DETAIL
  if (timeron) {
    int i;
    for (i = T_OPENCL_API; i < T_END; i++) timer_clear(i);
  }
#endif

  DTIMER_START(T_OPENCL_API);

  //-----------------------------------------------------------------------
  // 1. Find the default device type and get a device for the device type
  //-----------------------------------------------------------------------
  device_type = clu_GetDefaultDeviceType();
  device      = clu_GetAvailableDevice(device_type);
  device_name = clu_GetDeviceName(device);

  // Device information
  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(size_t),
                          &max_work_group_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  //-----------------------------------------------------------------------
  // 2. Create a context for the specified device
  //-----------------------------------------------------------------------
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  cmd_queue = clCreateCommandQueue(context, device, 0, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  DTIMER_STOP(T_OPENCL_API);

  //-----------------------------------------------------------------------
  // 4. Build the program
  //-----------------------------------------------------------------------
  DTIMER_START(T_BUILD);
  char *source_file;
  char build_option[30];
  if (device_type == CL_DEVICE_TYPE_CPU) {
    source_file = "mg_cpu.cl";
    sprintf(build_option, "-DM=%d -I.", M);

    PSINV_DIM = PSINV_DIM_CPU;
    RESID_DIM = RESID_DIM_CPU;
    RPRJ3_DIM = RPRJ3_DIM_CPU;
    INTERP_1_DIM = INTERP_1_DIM_CPU;
    NORM2U3_DIM = NORM2U3_DIM_CPU;
    COMM3_1_DIM = COMM3_1_DIM_CPU;
    COMM3_2_DIM = COMM3_2_DIM_CPU;
    COMM3_3_DIM = COMM3_3_DIM_CPU;
    ZERO3_DIM = ZERO3_DIM_CPU;
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    source_file = "mg_gpu.cl";
    sprintf(build_option, "-DM=%d -I.", M);

    PSINV_DIM = PSINV_DIM_GPU;
    RESID_DIM = RESID_DIM_GPU;
    RPRJ3_DIM = RPRJ3_DIM_GPU;
    INTERP_1_DIM = INTERP_1_DIM_GPU;
    NORM2U3_DIM = NORM2U3_DIM_GPU;
    COMM3_1_DIM = COMM3_1_DIM_GPU;
    COMM3_2_DIM = COMM3_2_DIM_GPU;
    COMM3_3_DIM = COMM3_3_DIM_GPU;
    ZERO3_DIM = ZERO3_DIM_GPU;
  } else {
    fprintf(stderr, "%s: not supported.", clu_GetDeviceTypeName(device_type));
    exit(EXIT_FAILURE);
  }
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);
  DTIMER_STOP(T_BUILD);

  //-----------------------------------------------------------------------
  // 5. Create kernels
  //-----------------------------------------------------------------------
  DTIMER_START(T_OPENCL_API);
  kernel_zero3 = clCreateKernel(program, "kernel_zero3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_comm3_1 = clCreateKernel(program, "kernel_comm3_1", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  if (device_type == CL_DEVICE_TYPE_GPU) {
    kernel_comm3_2 = clCreateKernel(program, "kernel_comm3_2", &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }
  kernel_comm3_3 = clCreateKernel(program, "kernel_comm3_3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_zran3_1 = clCreateKernel(program, "kernel_zran3_1", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_zran3_2 = clCreateKernel(program, "kernel_zran3_2", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_zran3_3 = clCreateKernel(program, "kernel_zran3_3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_psinv = clCreateKernel(program, "kernel_psinv", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_resid = clCreateKernel(program, "kernel_resid", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_rprj3 = clCreateKernel(program, "kernel_rprj3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_interp_1 = clCreateKernel(program, "kernel_interp_1", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_interp_2 = clCreateKernel(program, "kernel_interp_2", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_interp_3 = clCreateKernel(program, "kernel_interp_3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_interp_4 = clCreateKernel(program, "kernel_interp_4", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  kernel_interp_5 = clCreateKernel(program, "kernel_interp_5", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  kernel_norm2u3 = clCreateKernel(program, "kernel_norm2u3", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  DTIMER_STOP(T_OPENCL_API);

  //-----------------------------------------------------------------------
  // 6. Creating buffers
  //-----------------------------------------------------------------------
  DTIMER_START(T_BUFFER_CREATE);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    m_v = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         NR * sizeof(double),
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_r = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         NR * sizeof(double),
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_u = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         NR * sizeof(double),
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_starts = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         NM * sizeof(double),
                         0, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  } else {
    m_v = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         NR * sizeof(double),
                         v, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_r = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         NR * sizeof(double),
                         r, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_u = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         NR * sizeof(double),
                         u, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_starts = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         NM * sizeof(double),
                         starts, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  }

  m_a = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       4 * sizeof(double),
                       0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  DTIMER_STOP(T_BUFFER_CREATE);
}

static void release_opencl()
{
  DTIMER_START(T_RELEASE);

	clReleaseMemObject(m_v);
	clReleaseMemObject(m_r);
	clReleaseMemObject(m_u);
	clReleaseMemObject(m_starts);
  clReleaseMemObject(m_a);

	clReleaseKernel(kernel_zero3);
	clReleaseKernel(kernel_comm3_1);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    clReleaseKernel(kernel_comm3_2);
  }
  clReleaseKernel(kernel_comm3_3);
	clReleaseKernel(kernel_zran3_1);
	clReleaseKernel(kernel_zran3_2);
	clReleaseKernel(kernel_zran3_3);

	clReleaseKernel(kernel_psinv);
	clReleaseKernel(kernel_resid);
	clReleaseKernel(kernel_rprj3);
	clReleaseKernel(kernel_interp_1);
	clReleaseKernel(kernel_interp_2);
	clReleaseKernel(kernel_interp_3);
	clReleaseKernel(kernel_interp_4);
	clReleaseKernel(kernel_interp_5);

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
  char *name = kernel_names[i - T_BUFFER_WRITE - 1];
  unsigned cnt = timer_count(i);
  if (cnt == 0) return;
  double tt = timer_read(i);
  printf("- %-11s: %9.3lf (%u, %.3f, %.2f%%)\n",
      name, tt, cnt, tt/cnt, tt/t_kernel * 100.0);
}

static void print_opencl_timers()
{
  int i;
  double tt;
  double t_opencl = 0.0, t_buffer = 0.0, t_kernel = 0.0;

  if (timeron) {
    for (i = T_OPENCL_API; i < T_END; i++)
      t_opencl += timer_read(i);

    for (i = T_BUFFER_CREATE; i <= T_BUFFER_WRITE; i++)
      t_buffer += timer_read(i);

    for (i = T_KERNEL_PSINV; i <= T_KERNEL_ZERO3; i++)
      t_kernel += timer_read(i);

    printf("\nOpenCL timers -\n");
    printf("Kernel       : %9.3f (%.2f%%)\n", 
        t_kernel, t_kernel/t_opencl * 100.0);
    for (i = T_KERNEL_PSINV; i <= T_KERNEL_ZERO3; i++)
      print_kernel_time(t_kernel, i);

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

