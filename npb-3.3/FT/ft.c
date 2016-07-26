//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB FT code. This OpenCL    //
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
// FT benchmark
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include "ft_dim.h"

#include <CL/cl.h>
#include "cl_util.h"

#define USE_CHECK_FINISH
//#define TIMER_DETAIL

#ifdef TIMER_DETAIL
enum OPENCL_TIMER {
  T_OPENCL_API = 20,
  T_BUILD,
  T_RELEASE,
  T_BUFFER_CREATE,
  T_BUFFER_READ,
  T_BUFFER_WRITE,
  T_END
};

char *kernel_names[] = {
  ""
};

static void print_opencl_timers();
#define DTIMER_START(id)    if (timers_enabled) timer_start(id)
#define DTIMER_STOP(id)     if (timers_enabled) timer_stop(id)
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


// OPENCL Variables
static cl_device_type   device_type;
static cl_device_id     device;
static char            *device_name;
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_program       program;
static size_t  work_item_sizes[3];
static size_t  max_work_group_size;
static cl_uint max_compute_units;

static cl_kernel k_compute_indexmap;
static cl_kernel k_compute_ics;
static cl_kernel k_cffts1;
static cl_kernel k_cffts2;
static cl_kernel k_cffts3;
static cl_kernel k_evolve;
static cl_kernel k_checksum;

//---------------------------------------------------------------------
// u0, u1, u2 are the main arrays in the problem. 
// Depending on the decomposition, these arrays will have different 
// dimensions. To accomodate all possibilities, we allocate them as 
// one-dimensional arrays and pass them to subroutines for different 
// views
//  - u0 contains the initial (transformed) initial condition
//  - u1 and u2 are working arrays
//  - twiddle contains exponents for the time evolution operator. 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// Large arrays are in common so that they are allocated on the
// heap rather than the stack. This common block is not
// referenced directly anywhere else. Padding is to avoid accidental 
// cache problems, since all array sizes are powers of two.
//---------------------------------------------------------------------
static cl_mem m_u;
static cl_mem m_u0;
static cl_mem m_u1;
static cl_mem m_twiddle;
static cl_mem m_ty1;
static cl_mem m_ty2;
static cl_mem m_chk;

static size_t cimap_lws[3], cimap_gws[3];
static size_t checksum_local_ws, checksum_global_ws, checksum_wg_num;
static dcomplex *g_chk;
static cl_uint COMPUTE_IMAP_DIM, EVOLVE_DIM;
static cl_uint CFFTS_DIM;
static size_t CFFTS_LSIZE = 32;

dcomplex u1[NTOTALP];
dcomplex u2[NTOTALP];

//---------------------------------------------------------------------------
static void init_ui(cl_mem *u0, cl_mem *u1, cl_mem *twiddle,
                    int d1, int d2, int d3);
static void evolve(cl_mem *u0, cl_mem *u1, cl_mem *twiddle,
                   int d1, int d2, int d3);
static void compute_initial_conditions(cl_mem *u0, int d1, int d2, int d3);
static double ipow46(double a, int exponent);
static void setup();
static void compute_indexmap(cl_mem *twiddle, int d1, int d2, int d3);
static void print_timers();
static void fft_init(int n);
static void fft(int dir, cl_mem *x1, cl_mem *x2);
static void cffts1(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout);
static void cffts2(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout);
static void cffts3(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout);
static int ilog2(int n);
static void checksum(int i, cl_mem *u1, int d1, int d2, int d3);
static void verify(int d1, int d2, int d3, int nt, 
                   logical *verified, char *Class);
static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//---------------------------------------------------------------------------


int main(int argc, char *argv[])
{
  int i;
  int iter;
  double total_time, mflops;
  logical verified;
  char Class;

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  //---------------------------------------------------------------------
  // Run the entire problem once to make sure all data is touched. 
  // This reduces variable startup costs, which is important for such a 
  // short benchmark. The other NPB 2 implementations are similar. 
  //---------------------------------------------------------------------
  for (i = 1; i <= T_max; i++) {
    timer_clear(i);
  }
  setup();
  setup_opencl(argc, argv);
  init_ui(&m_u0, &m_u1, &m_twiddle, dims[0], dims[1], dims[2]);
  compute_indexmap(&m_twiddle, dims[0], dims[1], dims[2]);
  compute_initial_conditions(&m_u1, dims[0], dims[1], dims[2]);
  fft_init(dims[0]);
  fft(1, &m_u1, &m_u0);

  //---------------------------------------------------------------------
  // Start over from the beginning. Note that all operations must
  // be timed, in contrast to other benchmarks. 
  //---------------------------------------------------------------------
  for (i = 1; i <= T_max; i++) {
    timer_clear(i);
  }

  timer_start(T_total);
  if (timers_enabled) timer_start(T_setup);

  DTIMER_START(T_compute_im);
  compute_indexmap(&m_twiddle, dims[0], dims[1], dims[2]);
  DTIMER_STOP(T_compute_im);

  DTIMER_START(T_compute_ics);
  compute_initial_conditions(&m_u1, dims[0], dims[1], dims[2]);
  DTIMER_STOP(T_compute_ics);

  DTIMER_START(T_fft_init);
  fft_init(dims[0]);
  DTIMER_STOP(T_fft_init);

  if (timers_enabled) timer_stop(T_setup);
  if (timers_enabled) timer_start(T_fft);
  fft(1, &m_u1, &m_u0);
  if (timers_enabled) timer_stop(T_fft);

  for (iter = 1; iter <= niter; iter++) {
    if (timers_enabled) timer_start(T_evolve);
    evolve(&m_u0, &m_u1, &m_twiddle, dims[0], dims[1], dims[2]);
    if (timers_enabled) timer_stop(T_evolve);
    if (timers_enabled) timer_start(T_fft);
    fft(-1, &m_u1, &m_u1);
    if (timers_enabled) timer_stop(T_fft);
    if (timers_enabled) timer_start(T_checksum);
    checksum(iter, &m_u1, dims[0], dims[1], dims[2]);
    if (timers_enabled) timer_stop(T_checksum);
  }

  verify(NX, NY, NZ, niter, &verified, &Class);

  timer_stop(T_total);
  total_time = timer_read(T_total);

  if (total_time != 0.0) {
    mflops = 1.0e-6 * (double)NTOTAL *
            (14.8157 + 7.19641 * log((double)NTOTAL)
            + (5.23518 + 7.21113 * log((double)NTOTAL)) * niter)
            / total_time;
  } else {
    mflops = 0.0;
  }
  c_print_results("FT", Class, NX, NY, NZ, niter,
                  total_time, mflops, "          floating point", verified, 
                  NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  clu_GetDeviceTypeName(device_type),
                  device_name);
  if (timers_enabled) print_timers();

  release_opencl();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// touch all the big data
//---------------------------------------------------------------------
static void init_ui(cl_mem *u0, cl_mem *u1, cl_mem *twiddle,
                    int d1, int d2, int d3)
{
  cl_kernel k_init_ui;
  cl_int ecode;

  DTIMER_START(T_OPENCL_API);
  // Create a kernel
  k_init_ui = clCreateKernel(program, "init_ui", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for init_ui");
  DTIMER_STOP(T_OPENCL_API);

  int n = d3 * d2 * (d1+1);
  ecode  = clSetKernelArg(k_init_ui, 0, sizeof(cl_mem), (void*)u0);
  ecode |= clSetKernelArg(k_init_ui, 1, sizeof(cl_mem), (void*)u1);
  ecode |= clSetKernelArg(k_init_ui, 2, sizeof(cl_mem), (void*)twiddle);
  ecode |= clSetKernelArg(k_init_ui, 3, sizeof(int), (void*)&n);
  clu_CheckError(ecode, "clSetKernelArg() for init_ui");

  size_t local_ws = work_item_sizes[0];
  size_t global_ws = clu_RoundWorkSize((size_t)n, local_ws);
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_init_ui,
                                 1, NULL,
                                 &global_ws,
                                 &local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for init_ui");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");

  DTIMER_START(T_RELEASE);
  clReleaseKernel(k_init_ui);
  DTIMER_STOP(T_RELEASE);
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
static void evolve(cl_mem *u0, cl_mem *u1, cl_mem *twiddle,
                   int d1, int d2, int d3)
{
  cl_int ecode;
  size_t local_ws[3], global_ws[3];

  ecode  = clSetKernelArg(k_evolve, 0, sizeof(cl_mem), u0);
  ecode |= clSetKernelArg(k_evolve, 1, sizeof(cl_mem), u1);
  ecode |= clSetKernelArg(k_evolve, 2, sizeof(cl_mem), twiddle);
  ecode |= clSetKernelArg(k_evolve, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_evolve, 4, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_evolve, 5, sizeof(int), &d3);
  clu_CheckError(ecode, "clSetKernelArg() for evolve");

  if (EVOLVE_DIM == 3) {
    local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    int temp = max_work_group_size / local_ws[0];
    local_ws[1] = d2 < temp ? d2 : temp;
    temp = temp / local_ws[1];
    local_ws[2] = d3 < temp ? d3 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);
    global_ws[2] = clu_RoundWorkSize((size_t)d3, local_ws[2]);
  } else if (EVOLVE_DIM == 2) {
    local_ws[0] = d2 < work_item_sizes[0] ? d2 : work_item_sizes[0];
    int temp = max_work_group_size / local_ws[0];
    local_ws[1] = d3 < temp ? d3 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d2, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d3, local_ws[1]);
  } else {
    int temp = d3 / max_compute_units;
    local_ws[0] = temp == 0 ? 1 : temp;
    global_ws[0] = clu_RoundWorkSize((size_t)d3, local_ws[0]);
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_evolve,
                                 EVOLVE_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for evolve");
  CHECK_FINISH();
}


//---------------------------------------------------------------------
// Fill in array u0 with initial conditions from 
// random number generator 
//---------------------------------------------------------------------
static void compute_initial_conditions(cl_mem *u0, int d1, int d2, int d3)
{
  int k;
  double start, an, dummy, starts[NZ];
  size_t local_ws, global_ws, temp;
  cl_mem m_starts;
  cl_int ecode;

  start = SEED;
  //---------------------------------------------------------------------
  // Jump to the starting element for our first plane.
  //---------------------------------------------------------------------
  an = ipow46(A, 0);
  dummy = randlc(&start, an);
  an = ipow46(A, 2*NX*NY);

  starts[0] = start;
  for (k = 1; k < dims[2]; k++) {
    dummy = randlc(&start, an);
    starts[k] = start;
  }

  if (device_type == CL_DEVICE_TYPE_CPU) {
    m_starts = clCreateBuffer(context,
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(double) * NZ,
                              starts, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_starts");

    local_ws  = 1;
    global_ws = clu_RoundWorkSize((size_t)d2, local_ws);
  } else { //GPU
    m_starts = clCreateBuffer(context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(double) * NZ,
                              starts,
                              &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_starts");

    temp = d2 / max_compute_units;
    local_ws  = temp == 0 ? 
                1 : ((temp > work_item_sizes[0]) ? work_item_sizes[0] : temp);
    global_ws = clu_RoundWorkSize((size_t)d2, local_ws);
  }

  ecode  = clSetKernelArg(k_compute_ics, 0, sizeof(cl_mem), u0);
  ecode |= clSetKernelArg(k_compute_ics, 1, sizeof(cl_mem), &m_starts);
  clu_CheckError(ecode, "clSetKernelArg() for compute_initial_conditions");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_compute_ics,
                                 1, NULL,
                                 &global_ws,
                                 &local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish()");

  DTIMER_START(T_RELEASE);
  clReleaseMemObject(m_starts);
  DTIMER_STOP(T_RELEASE);
}


//---------------------------------------------------------------------
// compute a^exponent mod 2^46
//---------------------------------------------------------------------
static double ipow46(double a, int exponent)
{
  double result, dummy, q, r;
  int n, n2;

  //---------------------------------------------------------------------
  // Use
  //   a^n = a^(n/2)*a^(n/2) if n even else
  //   a^n = a*a^(n-1)       if n odd
  //---------------------------------------------------------------------
  result = 1;
  if (exponent == 0) return result;
  q = a;
  r = 1;
  n = exponent;

  while (n > 1) {
    n2 = n / 2;
    if (n2 * 2 == n) {
      dummy = randlc(&q, q);
      n = n2;
    } else {
      dummy = randlc(&r, q);
      n = n-1;
    }
  }
  dummy = randlc(&r, q);
  result = r;
  return result;
}


static void setup()
{
  FILE *fp;
  debug = false;

  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timers_enabled = true;
    fclose(fp);
  } else {
    timers_enabled = false;
  }

  niter = NITER_DEFAULT;

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - FT Benchmark\n\n");
  printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
  printf(" Iterations          :     %10d\n", niter);
  printf("\n");

  dims[0] = NX;
  dims[1] = NY;
  dims[2] = NZ;

  //---------------------------------------------------------------------
  // Set up info for blocking of ffts and transposes.  This improves
  // performance on cache-based systems. Blocking involves
  // working on a chunk of the problem at a time, taking chunks
  // along the first, second, or third dimension. 
  //
  // - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
  // - In cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

  // Since 1st dim is always in processor, we'll assume it's long enough 
  // (default blocking factor is 16 so min size for 1st dim is 16)
  // The only case we have to worry about is cffts1 in a 2d decomposition. 
  // so the blocking factor should not be larger than the 2nd dimension. 
  //---------------------------------------------------------------------

//  fftblock = FFTBLOCK_DEFAULT;
//  fftblockpad = FFTBLOCKPAD_DEFAULT;
//
//  if (fftblock != FFTBLOCK_DEFAULT) fftblockpad = fftblock+3;
}


//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
// for time evolution exponent. 
//---------------------------------------------------------------------
static void compute_indexmap(cl_mem *twiddle, int d1, int d2, int d3)
{
  cl_int ecode;
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_compute_indexmap,
                                 COMPUTE_IMAP_DIM, NULL,
                                 cimap_gws,
                                 cimap_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for compute_indexmap");
  CHECK_FINISH();
}


static void print_timers()
{
  int i;
  double t, t_m;
  char *tstrings[T_max+1];
  tstrings[1] = "          total "; 
  tstrings[2] = "          setup "; 
  tstrings[3] = "            fft "; 
  tstrings[4] = "         evolve "; 
  tstrings[5] = "       checksum "; 
  tstrings[6] = "           fftx "; 
  tstrings[7] = "           ffty "; 
  tstrings[8] = "           fftz ";
  tstrings[9] = "   compute_imap ";
  tstrings[10]= "    compute_ics ";
  tstrings[11]= "       fft_init ";

  t_m = timer_read(T_total);
  if (t_m <= 0.0) t_m = 1.00;
  for (i = 1; i <= T_max; i++) {
    t = timer_read(i);
    printf(" timer %2d(%16s) :%9.4f (%6.2f%%)\n", 
        i, tstrings[i], t, t*100.0/t_m);
  }
}


//---------------------------------------------------------------------
// compute the roots-of-unity array that will be used for subsequent FFTs. 
//---------------------------------------------------------------------
static void fft_init(int n)
{
  int m, nu, ku, i, j, ln;
  double t, ti;

  //---------------------------------------------------------------------
  // Initialize the U array with sines and cosines in a manner that permits
  // stride one access at each FFT iteration.
  //---------------------------------------------------------------------
  nu = n;
  m = ilog2(n);
  u[0] = dcmplx(m, 0.0);
  ku = 2;
  ln = 1;

  for (j = 1; j <= m; j++) {
    t = PI / ln;

    for (i = 0; i <= ln - 1; i++) {
      ti = i * t;
      u[i+ku-1] = dcmplx(cos(ti), sin(ti));
    }

    ku = ku + ln;
    ln = 2 * ln;
  }

  int ecode;
  ecode = clEnqueueWriteBuffer(cmd_queue,
                               m_u,
                               CL_FALSE,
                               0, sizeof(dcomplex) * NXP,
                               u,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer() for m_u");
}


//---------------------------------------------------------------------
// note: args x1, x2 must be different arrays
// note: args for cfftsx are (direction, layout, xin, xout, scratch)
//       xin/xout may be the same and it can be somewhat faster
//       if they are
//---------------------------------------------------------------------
static void fft(int dir, cl_mem *x1, cl_mem *x2)
{
  if (dir == 1) {
    cffts1(1, dims[0], dims[1], dims[2], x1, x1);
    cffts2(1, dims[0], dims[1], dims[2], x1, x1);
    cffts3(1, dims[0], dims[1], dims[2], x1, x2);
  } else {
    cffts3(-1, dims[0], dims[1], dims[2], x1, x1);
    cffts2(-1, dims[0], dims[1], dims[2], x1, x1);
    cffts1(-1, dims[0], dims[1], dims[2], x1, x2);
  }
}


static void cffts1(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout)
{
  int logd1 = ilog2(d1);
  size_t local_ws[2], global_ws[2], temp;
  cl_int ecode;

  if (timers_enabled) timer_start(T_fftx);

  ecode  = clSetKernelArg(k_cffts1, 0, sizeof(cl_mem), x);
  ecode |= clSetKernelArg(k_cffts1, 1, sizeof(cl_mem), xout);
  ecode |= clSetKernelArg(k_cffts1, 3, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts1, 4, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts1, 5, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts1, 6, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts1, 7, sizeof(int), &logd1);
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts1");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    local_ws[0] = d2 < work_item_sizes[0] ? d2 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d3 < temp ? d3 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d2, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d3, local_ws[1]);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    if (CFFTS_DIM == 2) {
      local_ws[0] = CFFTS_LSIZE;
      local_ws[1] = 1;
      global_ws[0] = d2 * local_ws[0];
      global_ws[1] = d3;
    } else {
      local_ws[0] = CFFTS_LSIZE;
      global_ws[0] = d3 * local_ws[0];
    }
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_cffts1,
                                 CFFTS_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for cffts1");
  CHECK_FINISH();

  if (timers_enabled) timer_stop(T_fftx);
}


static void cffts2(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout)
{
  int logd2 = ilog2(d2);
  size_t local_ws[2], global_ws[2], temp;
  cl_int ecode;

  if (timers_enabled) timer_start(T_ffty);

  ecode  = clSetKernelArg(k_cffts2, 0, sizeof(cl_mem), x);
  ecode |= clSetKernelArg(k_cffts2, 1, sizeof(cl_mem), xout);
  ecode |= clSetKernelArg(k_cffts2, 3, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts2, 4, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts2, 5, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts2, 6, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts2, 7, sizeof(int), &logd2);
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts2");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d3 < temp ? d3 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d3, local_ws[1]);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    if (CFFTS_DIM == 2) {
      local_ws[0] = CFFTS_LSIZE;
      local_ws[1] = 1;
      global_ws[0] = d1 * local_ws[0];
      global_ws[1] = d3;
    } else {
      local_ws[0] = CFFTS_LSIZE;
      global_ws[0] = d3 * local_ws[0];
    }
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_cffts2,
                                 CFFTS_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for cffts2");
  CHECK_FINISH();

  if (timers_enabled) timer_stop(T_ffty);
}


static void cffts3(int is, int d1, int d2, int d3, cl_mem *x, cl_mem *xout)
{
  int logd3 = ilog2(d3);
  size_t local_ws[2], global_ws[2], temp;
  cl_int ecode;

  if (timers_enabled) timer_start(T_fftz);

  ecode  = clSetKernelArg(k_cffts3, 0, sizeof(cl_mem), x);
  ecode |= clSetKernelArg(k_cffts3, 1, sizeof(cl_mem), xout);
  ecode |= clSetKernelArg(k_cffts3, 3, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts3, 4, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts3, 5, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts3, 6, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts3, 7, sizeof(int), &logd3);
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts3");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    local_ws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / local_ws[0];
    local_ws[1] = d2 < temp ? d2 : temp;

    global_ws[0] = clu_RoundWorkSize((size_t)d1, local_ws[0]);
    global_ws[1] = clu_RoundWorkSize((size_t)d2, local_ws[1]);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    if (CFFTS_DIM == 2) {
      local_ws[0] = CFFTS_LSIZE;
      local_ws[1] = 1;
      global_ws[0] = d1 * local_ws[0];
      global_ws[1] = d2;
    } else {
      local_ws[0] = CFFTS_LSIZE;
      global_ws[0] = d2 * local_ws[0];
    }
  }

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_cffts3,
                                 CFFTS_DIM, NULL,
                                 global_ws,
                                 local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for cffts3");
  CHECK_FINISH();

  if (timers_enabled) timer_stop(T_fftz);
}


static int ilog2(int n)
{
  int nn, lg;
  if (n == 1) return 0;
  lg = 1;
  nn = 2;
  while (nn < n) {
    nn = nn*2;
    lg = lg+1;
  }
  return lg;
}


static void checksum(int i, cl_mem *u1, int d1, int d2, int d3)
{
  dcomplex chk = dcmplx(0.0, 0.0);
  int k;
  cl_int ecode;

  ecode = clSetKernelArg(k_checksum, 0, sizeof(cl_mem), u1);
  clu_CheckError(ecode, "clSetKernelArg() for checksum");

  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_checksum,
                                 1, NULL,
                                 &checksum_global_ws,
                                 &checksum_local_ws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();

  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_chk,
                              CL_TRUE,
                              0, checksum_wg_num * sizeof(dcomplex),
                              g_chk,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clReadBuffer()");

  // reduction
  for (k = 0; k < checksum_wg_num; k++) {
    chk = dcmplx_add(chk, g_chk[k]);
  }

  chk = dcmplx_div2(chk, (double)(NTOTAL));

  printf(" T =%5d     Checksum =%22.12E%22.12E\n", i, chk.real, chk.imag);
  sums[i] = chk;
}


static void verify(int d1, int d2, int d3, int nt, 
                   logical *verified, char *Class)
{
  int i;
  double err, epsilon;

  //---------------------------------------------------------------------
  // Reference checksums
  //---------------------------------------------------------------------
  dcomplex csum_ref[25+1];

  *Class = 'U';

  epsilon = 1.0e-12;
  *verified = false;

  if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6) {
    //---------------------------------------------------------------------
    //   Sample size reference checksums
    //---------------------------------------------------------------------
    *Class = 'S';
    csum_ref[1] = dcmplx(5.546087004964E+02, 4.845363331978E+02);
    csum_ref[2] = dcmplx(5.546385409189E+02, 4.865304269511E+02);
    csum_ref[3] = dcmplx(5.546148406171E+02, 4.883910722336E+02);
    csum_ref[4] = dcmplx(5.545423607415E+02, 4.901273169046E+02);
    csum_ref[5] = dcmplx(5.544255039624E+02, 4.917475857993E+02);
    csum_ref[6] = dcmplx(5.542683411902E+02, 4.932597244941E+02);

  } else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6) {
    //---------------------------------------------------------------------
    //   Class W size reference checksums
    //---------------------------------------------------------------------
    *Class = 'W';
    csum_ref[1] = dcmplx(5.673612178944E+02, 5.293246849175E+02);
    csum_ref[2] = dcmplx(5.631436885271E+02, 5.282149986629E+02);
    csum_ref[3] = dcmplx(5.594024089970E+02, 5.270996558037E+02);
    csum_ref[4] = dcmplx(5.560698047020E+02, 5.260027904925E+02);
    csum_ref[5] = dcmplx(5.530898991250E+02, 5.249400845633E+02);
    csum_ref[6] = dcmplx(5.504159734538E+02, 5.239212247086E+02);

  } else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6) {
    //---------------------------------------------------------------------
    //   Class A size reference checksums
    //---------------------------------------------------------------------
    *Class = 'A';
    csum_ref[1] = dcmplx(5.046735008193E+02, 5.114047905510E+02);
    csum_ref[2] = dcmplx(5.059412319734E+02, 5.098809666433E+02);
    csum_ref[3] = dcmplx(5.069376896287E+02, 5.098144042213E+02);
    csum_ref[4] = dcmplx(5.077892868474E+02, 5.101336130759E+02);
    csum_ref[5] = dcmplx(5.085233095391E+02, 5.104914655194E+02);
    csum_ref[6] = dcmplx(5.091487099959E+02, 5.107917842803E+02);

  } else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20) {
    //---------------------------------------------------------------------
    //   Class B size reference checksums
    //---------------------------------------------------------------------
    *Class = 'B';
    csum_ref[1]  = dcmplx(5.177643571579E+02, 5.077803458597E+02);
    csum_ref[2]  = dcmplx(5.154521291263E+02, 5.088249431599E+02);
    csum_ref[3]  = dcmplx(5.146409228649E+02, 5.096208912659E+02);
    csum_ref[4]  = dcmplx(5.142378756213E+02, 5.101023387619E+02);
    csum_ref[5]  = dcmplx(5.139626667737E+02, 5.103976610617E+02);
    csum_ref[6]  = dcmplx(5.137423460082E+02, 5.105948019802E+02);
    csum_ref[7]  = dcmplx(5.135547056878E+02, 5.107404165783E+02);
    csum_ref[8]  = dcmplx(5.133910925466E+02, 5.108576573661E+02);
    csum_ref[9]  = dcmplx(5.132470705390E+02, 5.109577278523E+02);
    csum_ref[10] = dcmplx(5.131197729984E+02, 5.110460304483E+02);
    csum_ref[11] = dcmplx(5.130070319283E+02, 5.111252433800E+02);
    csum_ref[12] = dcmplx(5.129070537032E+02, 5.111968077718E+02);
    csum_ref[13] = dcmplx(5.128182883502E+02, 5.112616233064E+02);
    csum_ref[14] = dcmplx(5.127393733383E+02, 5.113203605551E+02);
    csum_ref[15] = dcmplx(5.126691062020E+02, 5.113735928093E+02);
    csum_ref[16] = dcmplx(5.126064276004E+02, 5.114218460548E+02);
    csum_ref[17] = dcmplx(5.125504076570E+02, 5.114656139760E+02);
    csum_ref[18] = dcmplx(5.125002331720E+02, 5.115053595966E+02);
    csum_ref[19] = dcmplx(5.124551951846E+02, 5.115415130407E+02);
    csum_ref[20] = dcmplx(5.124146770029E+02, 5.115744692211E+02);

  } else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20) {
    //---------------------------------------------------------------------
    //   Class C size reference checksums
    //---------------------------------------------------------------------
    *Class = 'C';
    csum_ref[1]  = dcmplx(5.195078707457E+02, 5.149019699238E+02);
    csum_ref[2]  = dcmplx(5.155422171134E+02, 5.127578201997E+02);
    csum_ref[3]  = dcmplx(5.144678022222E+02, 5.122251847514E+02);
    csum_ref[4]  = dcmplx(5.140150594328E+02, 5.121090289018E+02);
    csum_ref[5]  = dcmplx(5.137550426810E+02, 5.121143685824E+02);
    csum_ref[6]  = dcmplx(5.135811056728E+02, 5.121496764568E+02);
    csum_ref[7]  = dcmplx(5.134569343165E+02, 5.121870921893E+02);
    csum_ref[8]  = dcmplx(5.133651975661E+02, 5.122193250322E+02);
    csum_ref[9]  = dcmplx(5.132955192805E+02, 5.122454735794E+02);
    csum_ref[10] = dcmplx(5.132410471738E+02, 5.122663649603E+02);
    csum_ref[11] = dcmplx(5.131971141679E+02, 5.122830879827E+02);
    csum_ref[12] = dcmplx(5.131605205716E+02, 5.122965869718E+02);
    csum_ref[13] = dcmplx(5.131290734194E+02, 5.123075927445E+02);
    csum_ref[14] = dcmplx(5.131012720314E+02, 5.123166486553E+02);
    csum_ref[15] = dcmplx(5.130760908195E+02, 5.123241541685E+02);
    csum_ref[16] = dcmplx(5.130528295923E+02, 5.123304037599E+02);
    csum_ref[17] = dcmplx(5.130310107773E+02, 5.123356167976E+02);
    csum_ref[18] = dcmplx(5.130103090133E+02, 5.123399592211E+02);
    csum_ref[19] = dcmplx(5.129905029333E+02, 5.123435588985E+02);
    csum_ref[20] = dcmplx(5.129714421109E+02, 5.123465164008E+02);

  } else if (d1 == 2048 && d2 == 1024 && d3 == 1024 && nt == 25) {
    //---------------------------------------------------------------------
    //   Class D size reference checksums
    //---------------------------------------------------------------------
    *Class = 'D';
    csum_ref[1]  = dcmplx(5.122230065252E+02, 5.118534037109E+02);
    csum_ref[2]  = dcmplx(5.120463975765E+02, 5.117061181082E+02);
    csum_ref[3]  = dcmplx(5.119865766760E+02, 5.117096364601E+02);
    csum_ref[4]  = dcmplx(5.119518799488E+02, 5.117373863950E+02);
    csum_ref[5]  = dcmplx(5.119269088223E+02, 5.117680347632E+02);
    csum_ref[6]  = dcmplx(5.119082416858E+02, 5.117967875532E+02);
    csum_ref[7]  = dcmplx(5.118943814638E+02, 5.118225281841E+02);
    csum_ref[8]  = dcmplx(5.118842385057E+02, 5.118451629348E+02);
    csum_ref[9]  = dcmplx(5.118769435632E+02, 5.118649119387E+02);
    csum_ref[10] = dcmplx(5.118718203448E+02, 5.118820803844E+02);
    csum_ref[11] = dcmplx(5.118683569061E+02, 5.118969781011E+02);
    csum_ref[12] = dcmplx(5.118661708593E+02, 5.119098918835E+02);
    csum_ref[13] = dcmplx(5.118649768950E+02, 5.119210777066E+02);
    csum_ref[14] = dcmplx(5.118645605626E+02, 5.119307604484E+02);
    csum_ref[15] = dcmplx(5.118647586618E+02, 5.119391362671E+02);
    csum_ref[16] = dcmplx(5.118654451572E+02, 5.119463757241E+02);
    csum_ref[17] = dcmplx(5.118665212451E+02, 5.119526269238E+02);
    csum_ref[18] = dcmplx(5.118679083821E+02, 5.119580184108E+02);
    csum_ref[19] = dcmplx(5.118695433664E+02, 5.119626617538E+02);
    csum_ref[20] = dcmplx(5.118713748264E+02, 5.119666538138E+02);
    csum_ref[21] = dcmplx(5.118733606701E+02, 5.119700787219E+02);
    csum_ref[22] = dcmplx(5.118754661974E+02, 5.119730095953E+02);
    csum_ref[23] = dcmplx(5.118776626738E+02, 5.119755100241E+02);
    csum_ref[24] = dcmplx(5.118799262314E+02, 5.119776353561E+02);
    csum_ref[25] = dcmplx(5.118822370068E+02, 5.119794338060E+02);

  } else if (d1 == 4096 && d2 == 2048 && d3 == 2048 && nt == 25) {
    //---------------------------------------------------------------------
    //   Class E size reference checksums
    //---------------------------------------------------------------------
    *Class = 'E';
    csum_ref[1]  = dcmplx(5.121601045346E+02, 5.117395998266E+02);
    csum_ref[2]  = dcmplx(5.120905403678E+02, 5.118614716182E+02);
    csum_ref[3]  = dcmplx(5.120623229306E+02, 5.119074203747E+02);
    csum_ref[4]  = dcmplx(5.120438418997E+02, 5.119345900733E+02);
    csum_ref[5]  = dcmplx(5.120311521872E+02, 5.119551325550E+02);
    csum_ref[6]  = dcmplx(5.120226088809E+02, 5.119720179919E+02);
    csum_ref[7]  = dcmplx(5.120169296534E+02, 5.119861371665E+02);
    csum_ref[8]  = dcmplx(5.120131225172E+02, 5.119979364402E+02);
    csum_ref[9]  = dcmplx(5.120104767108E+02, 5.120077674092E+02);
    csum_ref[10] = dcmplx(5.120085127969E+02, 5.120159443121E+02);
    csum_ref[11] = dcmplx(5.120069224127E+02, 5.120227453670E+02);
    csum_ref[12] = dcmplx(5.120055158164E+02, 5.120284096041E+02);
    csum_ref[13] = dcmplx(5.120041820159E+02, 5.120331373793E+02);
    csum_ref[14] = dcmplx(5.120028605402E+02, 5.120370938679E+02);
    csum_ref[15] = dcmplx(5.120015223011E+02, 5.120404138831E+02);
    csum_ref[16] = dcmplx(5.120001570022E+02, 5.120432068837E+02);
    csum_ref[17] = dcmplx(5.119987650555E+02, 5.120455615860E+02);
    csum_ref[18] = dcmplx(5.119973525091E+02, 5.120475499442E+02);
    csum_ref[19] = dcmplx(5.119959279472E+02, 5.120492304629E+02);
    csum_ref[20] = dcmplx(5.119945006558E+02, 5.120506508902E+02);
    csum_ref[21] = dcmplx(5.119930795911E+02, 5.120518503782E+02);
    csum_ref[22] = dcmplx(5.119916728462E+02, 5.120528612016E+02);
    csum_ref[23] = dcmplx(5.119902874185E+02, 5.120537101195E+02);
    csum_ref[24] = dcmplx(5.119889291565E+02, 5.120544194514E+02);
    csum_ref[25] = dcmplx(5.119876028049E+02, 5.120550079284E+02);
  }

  if (*Class != 'U') {
    *verified = true;
    for (i = 1; i <= nt; i++) {
      err = dcmplx_abs(dcmplx_div(dcmplx_sub(sums[i], csum_ref[i]),
                                  csum_ref[i]));
      if (!(err <= epsilon)) {
        *verified = false;
        break;
      }
    }
  }

  if (*Class != 'U') {
    if (*verified) {
      printf(" Result verification successful\n");
    } else {
      printf(" Result verification failed\n");
    }
  }
  printf(" class = %c\n", *Class);
}


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
static void setup_opencl(int argc, char *argv[])
{
  size_t temp;
  cl_int ecode;
  char *source_dir = "FT";
  if (argc > 1) source_dir = argv[1];

#ifdef TIMER_DETAIL
  if (timers_enabled) {
    int i;
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

  // FIXME: The below values are experimental.
  if (max_work_group_size > 64) {
    max_work_group_size = 64;
    int i;
    for (i = 0; i < 3; i++) {
      if (work_item_sizes[i] > 64) {
        work_item_sizes[i] = 64;
      }
    }
  }

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cl_uint),
                          &max_compute_units,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  // 2. Create a context for the specified device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  // 3. Create a command queue
  cmd_queue = clCreateCommandQueue(context, device, 0, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  DTIMER_STOP(T_OPENCL_API);

  // 4. Build the program
  DTIMER_START(T_BUILD);
  char *source_file;
  char build_option[50];
  if (device_type == CL_DEVICE_TYPE_CPU) {
    source_file = "ft_cpu.cl";
    sprintf(build_option, "-I. -DCLASS=%d -DUSE_CPU", CLASS);

    COMPUTE_IMAP_DIM = COMPUTE_IMAP_DIM_CPU;
    EVOLVE_DIM = EVOLVE_DIM_CPU;
    CFFTS_DIM = CFFTS_DIM_CPU;

  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    char vendor[50];
    ecode = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 50, vendor, NULL);
    clu_CheckError(ecode, "clGetDeviceInfo()");
    if (strncmp(vendor, DEV_VENDOR_NVIDIA, strlen(DEV_VENDOR_NVIDIA)) == 0) {
      source_file = "ft_gpu_nvidia.cl";
      CFFTS_LSIZE = 32;
    } else {
      source_file = "ft_gpu.cl";
      CFFTS_LSIZE = 64;
    }

    sprintf(build_option, "-I. -DCLASS=\'%c\' -DLSIZE=%lu",
            CLASS, CFFTS_LSIZE);

    COMPUTE_IMAP_DIM = COMPUTE_IMAP_DIM_GPU;
    EVOLVE_DIM = EVOLVE_DIM_GPU;
    CFFTS_DIM = CFFTS_DIM_GPU;

  } else {
    fprintf(stderr, "Set the environment variable OPENCL_DEVICE_TYPE!\n");
    exit(EXIT_FAILURE);
  }
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);
  DTIMER_STOP(T_BUILD);

  // 5. Create buffers
  DTIMER_START(T_BUFFER_CREATE);
  m_u = clCreateBuffer(context,
                       CL_MEM_READ_ONLY,
                       sizeof(dcomplex) * NXP,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_u");

  m_u0 = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(dcomplex) * NTOTALP,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_u0");

  m_u1 = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(dcomplex) * NTOTALP,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_u1");

  m_twiddle = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             sizeof(double) * NTOTALP,
                             NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_twiddle");

  if (device_type == CL_DEVICE_TYPE_CPU) {
    size_t ty1_size, ty2_size;
    if (CFFTS_DIM == 2) {
      ty1_size = sizeof(dcomplex) * NX * NY * NZ;
      ty2_size = sizeof(dcomplex) * NX * NY * NZ;
    } else {
      fprintf(stderr, "Wrong CFFTS_DIM: %u\n", CFFTS_DIM);
      exit(EXIT_FAILURE);
    }

    m_ty1 = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           ty1_size,
                           NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_ty1");

    m_ty2 = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           ty2_size,
                           NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_ty2");
  }

  if (device_type == CL_DEVICE_TYPE_CPU) {
    temp = 1024 / max_compute_units;
    checksum_local_ws  = temp == 0 ? 1 : temp;
    checksum_global_ws = clu_RoundWorkSize((size_t)1024, checksum_local_ws);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    checksum_local_ws  = 32;
    checksum_global_ws = clu_RoundWorkSize((size_t)1024, checksum_local_ws);
  }
  checksum_wg_num = checksum_global_ws / checksum_local_ws;
  m_chk = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sizeof(dcomplex) * checksum_wg_num,
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_chk");
  g_chk = (dcomplex *)malloc(sizeof(dcomplex) * checksum_wg_num);
  DTIMER_STOP(T_BUFFER_CREATE);

  // 6. Create kernels
  DTIMER_START(T_OPENCL_API);
  double ap = -4.0 * ALPHA * PI * PI;
  int d1 = dims[0];
  int d2 = dims[1];
  int d3 = dims[2];

  k_compute_indexmap = clCreateKernel(program, "compute_indexmap", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_indexmap");
  ecode  = clSetKernelArg(k_compute_indexmap, 0, sizeof(cl_mem), &m_twiddle);
  ecode |= clSetKernelArg(k_compute_indexmap, 1, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_compute_indexmap, 2, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_compute_indexmap, 3, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_compute_indexmap, 4, sizeof(double), &ap);
  clu_CheckError(ecode, "clSetKernelArg() for compute_indexmap");
  if (COMPUTE_IMAP_DIM == 3) {
    cimap_lws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / cimap_lws[0];
    cimap_lws[1] = d2 < temp ? d2 : temp;
    temp = temp / cimap_lws[1];
    cimap_lws[2] = d3 < temp ? d3 : temp;

    cimap_gws[0] = clu_RoundWorkSize((size_t)d1, cimap_lws[0]);
    cimap_gws[1] = clu_RoundWorkSize((size_t)d2, cimap_lws[1]);
    cimap_gws[2] = clu_RoundWorkSize((size_t)d3, cimap_lws[2]);
  } else if (COMPUTE_IMAP_DIM == 2) {
    cimap_lws[0] = d2 < work_item_sizes[0] ? d2 : work_item_sizes[0];
    temp = max_work_group_size / cimap_lws[0];
    cimap_lws[1] = d3 < temp ? d3 : temp;

    cimap_gws[0] = clu_RoundWorkSize((size_t)d2, cimap_lws[0]);
    cimap_gws[1] = clu_RoundWorkSize((size_t)d3, cimap_lws[1]);
  } else {
    //temp = d3 / max_compute_units;
    temp = 1;
    cimap_lws[0] = temp == 0 ? 1 : temp;
    cimap_gws[0] = clu_RoundWorkSize((size_t)d3, cimap_lws[0]);
  }

  k_compute_ics = clCreateKernel(program,
                                 "compute_initial_conditions", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_initial_conditions");
  ecode  = clSetKernelArg(k_compute_ics, 2, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_compute_ics, 3, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_compute_ics, 4, sizeof(int), &d3);
  clu_CheckError(ecode, "clSetKernelArg() for compute_initial_conditions");

  k_cffts1 = clCreateKernel(program, "cffts1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for cffts1");
  ecode  = clSetKernelArg(k_cffts1, 2, sizeof(cl_mem), &m_u);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_cffts1, 8, sizeof(cl_mem), &m_ty1);
    ecode |= clSetKernelArg(k_cffts1, 9, sizeof(cl_mem), &m_ty2);
  }
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts1");

  k_cffts2 = clCreateKernel(program, "cffts2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for cffts2");
  ecode  = clSetKernelArg(k_cffts2, 2, sizeof(cl_mem), &m_u);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_cffts2, 8, sizeof(cl_mem), &m_ty1);
    ecode |= clSetKernelArg(k_cffts2, 9, sizeof(cl_mem), &m_ty2);
  }
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts2");

  k_cffts3 = clCreateKernel(program, "cffts3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for cffts3");
  ecode  = clSetKernelArg(k_cffts3, 2, sizeof(cl_mem), &m_u);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_cffts3, 8, sizeof(cl_mem), &m_ty1);
    ecode |= clSetKernelArg(k_cffts3, 9, sizeof(cl_mem), &m_ty2);
  }
  clu_CheckError(ecode, "clSetKernelArg() for k_cffts3");

  k_evolve = clCreateKernel(program, "evolve", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for evolve");

  k_checksum = clCreateKernel(program, "checksum", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for checksum");
  ecode  = clSetKernelArg(k_checksum, 1, sizeof(cl_mem), &m_chk);
  ecode |= clSetKernelArg(k_checksum, 2, sizeof(dcomplex)*checksum_local_ws,
                          NULL);
  ecode |= clSetKernelArg(k_checksum, 3, sizeof(int), &dims[0]);
  ecode |= clSetKernelArg(k_checksum, 4, sizeof(int), &dims[1]);
  clu_CheckError(ecode, "clSetKernelArg() for checksum");
  DTIMER_STOP(T_OPENCL_API);
}


static void release_opencl()
{
  DTIMER_START(T_RELEASE);

  free(g_chk);

  clReleaseMemObject(m_u);
  clReleaseMemObject(m_u0);
  clReleaseMemObject(m_u1);
  clReleaseMemObject(m_twiddle);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    clReleaseMemObject(m_ty1);
    clReleaseMemObject(m_ty2);
  }
  clReleaseMemObject(m_chk);

  clReleaseKernel(k_compute_indexmap);
  clReleaseKernel(k_compute_ics);
  clReleaseKernel(k_cffts1);
  clReleaseKernel(k_cffts2);
  clReleaseKernel(k_cffts3);
  clReleaseKernel(k_evolve);
  clReleaseKernel(k_checksum);

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

  if (timers_enabled) {
    for (i = T_OPENCL_API; i < T_END; i++)
      t_opencl += timer_read(i);

    for (i = T_BUFFER_CREATE; i <= T_BUFFER_WRITE; i++)
      t_buffer += timer_read(i);

    printf("\nOpenCL timers -\n");

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
