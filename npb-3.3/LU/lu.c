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

//---------------------------------------------------------------------
//      program applu
//---------------------------------------------------------------------

//---------------------------------------------------------------------
//
//   driver for the performance evaluation of the solver for
//   five coupled parabolic/elliptic partial differential equations.
//
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "applu.incl"
#include "timers.h"
#include "print_results.h"

#include "lu_dim.h"

//---------------------------------------------------------------------
// OPENCL Variables
//---------------------------------------------------------------------
cl_device_type   device_type;
cl_device_id     device;
char            *device_name;
cl_context       context;
cl_command_queue cmd_queue;
cl_command_queue *pipe_queue;
cl_program       p_pre;
cl_program       p_main;
cl_program       p_post;
size_t  work_item_sizes[3];
size_t  max_work_group_size;
cl_uint max_compute_units;
size_t  max_pipeline;

cl_mem m_ce;
cl_mem m_u;
cl_mem m_rsd;
cl_mem m_frct;
cl_mem m_flux;
cl_mem m_qs;
cl_mem m_rho_i;
cl_mem m_sum;
cl_mem m_utmp, m_rtmp;

cl_kernel k_setbv1, k_setbv2, k_setbv3;
cl_kernel k_setiv;
cl_kernel k_l2norm;
cl_kernel k_rhs, k_rhsx, k_rhsy, k_rhsz;
cl_kernel k_ssor2, k_ssor3;
cl_kernel k_blts;
cl_kernel k_buts;

size_t setbv1_lws[3], setbv1_gws[3];
size_t setbv2_lws[3], setbv2_gws[3];
size_t setbv3_lws[3], setbv3_gws[3];
size_t setiv_lws[3], setiv_gws[3];
size_t l2norm_lws[3], l2norm_gws[3], sum_size;
size_t rhs_lws[3], rhs_gws[3];
size_t rhsx_lws[3], rhsx_gws[3];
size_t rhsy_lws[3], rhsy_gws[3];
size_t rhsz_lws[3], rhsz_gws[3];
size_t ssor1_lws[3], ssor1_gws[3];
size_t ssor2_lws[3], ssor2_gws[3];
size_t ssor3_lws[3], ssor3_gws[3];
size_t blts_lws[3], blts_gws[3];
size_t buts_lws[3], buts_gws[3];

int SETBV1_DIM, SETBV2_DIM, SETBV3_DIM;
int SETIV_DIM;
int ERHS1_DIM, ERHS2_DIM, ERHS3_DIM, ERHS4_DIM; 
int PINTGR1_DIM, PINTGR2_DIM, PINTGR3_DIM; 
int RHS_DIM, RHSX_DIM, RHSY_DIM, RHSZ_DIM;
int SSOR2_DIM, SSOR3_DIM;

static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// grid
//---------------------------------------------------------------------
/* common/cgcon/ */
double dxi, deta, dzeta;
double tx1, tx2, tx3;
double ty1, ty2, ty3;
double tz1, tz2, tz3;
int nx, ny, nz;
int nx0, ny0, nz0;
int ist, iend;
int jst, jend;
int ii1, ii2;
int ji1, ji2;
int ki1, ki2;

//---------------------------------------------------------------------
// dissipation
//---------------------------------------------------------------------
/* common/disp/ */
double dx1, dx2, dx3, dx4, dx5;
double dy1, dy2, dy3, dy4, dy5;
double dz1, dz2, dz3, dz4, dz5;
double dssp;

//---------------------------------------------------------------------
// output control parameters
//---------------------------------------------------------------------
/* common/cprcon/ */
int ipr, inorm;

//---------------------------------------------------------------------
// newton-raphson iteration control parameters
//---------------------------------------------------------------------
/* common/ctscon/ */
double dt, omega, tolrsd[5], rsdnm[5], errnm[5], frc, ttotal;
int itmax, invert;

//---------------------------------------------------------------------
// coefficients of the exact solution
//---------------------------------------------------------------------
/* common/cexact/ */
double ce[5][13];


//---------------------------------------------------------------------
// timers
//---------------------------------------------------------------------
/* common/timer/ */
double maxtime;
logical timeron;


int main(int argc, char *argv[])
{
  char Class;
  logical verified;
  double mflops;

  double t, tmax, trecs[t_last+1];
  int i;
  char *t_names[t_last+1];

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  //---------------------------------------------------------------------
  // Setup info for timers
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_jacld] = "jacld";
    t_names[t_blts] = "blts";
    t_names[t_jacu] = "jacu";
    t_names[t_buts] = "buts";
    t_names[t_add] = "add";
    t_names[t_l2norm] = "l2norm";

    t_names[t_setbv] = "setbv";
    t_names[t_setiv] = "setiv";
    t_names[t_erhs] = "erhs";
    t_names[t_error] = "error";
    t_names[t_pintgr] = "pintgr";
    t_names[t_blts1] = "blts1";
    t_names[t_buts1] = "buts1";
    fclose(fp);
  } else {
    timeron = false;
  }

  //---------------------------------------------------------------------
  // read input data
  //---------------------------------------------------------------------
  read_input();

  //---------------------------------------------------------------------
  // set up domain sizes
  //---------------------------------------------------------------------
  domain();

  //---------------------------------------------------------------------
  // set up OpenCL environment
  //---------------------------------------------------------------------
  setup_opencl(argc, argv);

  //---------------------------------------------------------------------
  // set up coefficients
  //---------------------------------------------------------------------
  setcoeff();

  //---------------------------------------------------------------------
  // set the boundary values for dependent variables
  //---------------------------------------------------------------------
  setbv();

  //---------------------------------------------------------------------
  // set the initial values for dependent variables
  //---------------------------------------------------------------------
  setiv();

  //---------------------------------------------------------------------
  // compute the forcing term based on prescribed exact solution
  //---------------------------------------------------------------------
  erhs();

  //---------------------------------------------------------------------
  // perform one SSOR iteration to touch all data pages
  //---------------------------------------------------------------------
  ssor(1);

  //---------------------------------------------------------------------
  // reset the boundary and initial values
  //---------------------------------------------------------------------
  setbv();
  setiv();

  //---------------------------------------------------------------------
  // perform the SSOR iterations
  //---------------------------------------------------------------------
  ssor(itmax);

  //---------------------------------------------------------------------
  // compute the solution error
  //---------------------------------------------------------------------
  error();

  //---------------------------------------------------------------------
  // compute the surface integral
  //---------------------------------------------------------------------
  pintgr();

  //---------------------------------------------------------------------
  // verification test
  //---------------------------------------------------------------------
  verify ( rsdnm, errnm, frc, &Class, &verified );
  mflops = (double)itmax * (1984.77 * (double)nx0
      * (double)ny0
      * (double)nz0
      - 10923.3 * pow(((double)(nx0+ny0+nz0)/3.0), 2.0) 
      + 27770.9 * (double)(nx0+ny0+nz0)/3.0
      - 144010.0)
    / (maxtime*1000000.0);

  c_print_results("LU", Class, nx0,
                  ny0, nz0, itmax,
                  maxtime, mflops, "          floating point", verified, 
                  NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, 
                  "(none)",
                  clu_GetDeviceTypeName(device_type),
                  device_name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    for (i = 1; i <= t_last; i++) {
      trecs[i] = timer_read(i);
    }
    tmax = maxtime;
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION     Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.4f  (%6.2f%%)\n",
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[i] - t;
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      }
    }
  }

  release_opencl();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
static void setup_opencl(int argc, char *argv[])
{
  int i;
  size_t temp, wg_num;
  cl_int ecode;
  char *source_dir = "LU";

  if (timeron) {
    timer_clear(TIMER_OPENCL);
    timer_clear(TIMER_BUILD);
    timer_clear(TIMER_BUFFER);
    timer_clear(TIMER_RELEASE);

    timer_start(TIMER_OPENCL);
  }

  if (argc > 1) source_dir = argv[1];

  //-----------------------------------------------------------------------
  // 1. Find the default device type and get a device for the device type
  //-----------------------------------------------------------------------
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

  ////////////////////////////////////////////////////////////////////////
  // FIXME: The below values are experimental.
  size_t default_wg_size = 64;
  if (device_type == CL_DEVICE_TYPE_CPU) {
    if (CLASS == 'B') default_wg_size = 128;
  } else {
    if (CLASS == 'B') default_wg_size = 32;
  }
  if (max_work_group_size > default_wg_size) {
    max_work_group_size = default_wg_size;
    int i;
    for (i = 0; i < 3; i++) {
      if (work_item_sizes[i] > default_wg_size) {
        work_item_sizes[i] = default_wg_size;
      }
    }
  }
  if (device_type == CL_DEVICE_TYPE_CPU) {
    SETBV1_DIM = SETBV1_DIM_CPU;
    SETBV2_DIM = SETBV2_DIM_CPU;
    SETBV3_DIM = SETBV3_DIM_CPU;
    SETIV_DIM = SETIV_DIM_CPU;
    ERHS1_DIM = ERHS1_DIM_CPU;
    ERHS2_DIM = ERHS2_DIM_CPU;
    ERHS3_DIM = ERHS3_DIM_CPU;
    ERHS4_DIM = ERHS4_DIM_CPU;
    PINTGR1_DIM = PINTGR1_DIM_CPU;
    PINTGR2_DIM = PINTGR2_DIM_CPU;
    PINTGR3_DIM = PINTGR3_DIM_CPU;
    RHS_DIM  = RHS_DIM_CPU;
    RHSX_DIM = RHSX_DIM_CPU;
    RHSY_DIM = RHSY_DIM_CPU;
    RHSZ_DIM = RHSZ_DIM_CPU;
    SSOR2_DIM = SSOR2_DIM_CPU;
    SSOR3_DIM = SSOR3_DIM_CPU;
  } else {
    SETBV1_DIM = SETBV1_DIM_GPU;
    SETBV2_DIM = SETBV2_DIM_GPU;
    SETBV3_DIM = SETBV3_DIM_GPU;
    SETIV_DIM = SETIV_DIM_GPU;
    ERHS1_DIM = ERHS1_DIM_GPU;
    ERHS2_DIM = ERHS2_DIM_GPU;
    ERHS3_DIM = ERHS3_DIM_GPU;
    ERHS4_DIM = ERHS4_DIM_GPU;
    PINTGR1_DIM = PINTGR1_DIM_GPU;
    PINTGR2_DIM = PINTGR2_DIM_GPU;
    PINTGR3_DIM = PINTGR3_DIM_GPU;
    RHS_DIM  = RHS_DIM_GPU;
    RHSX_DIM = RHSX_DIM_GPU;
    RHSY_DIM = RHSY_DIM_GPU;
    RHSZ_DIM = RHSZ_DIM_GPU;
    SSOR2_DIM = SSOR2_DIM_GPU;
    SSOR3_DIM = SSOR3_DIM_GPU;
  }
  ////////////////////////////////////////////////////////////////////////

  //-----------------------------------------------------------------------
  // 2. Create a context for the specified device
  //-----------------------------------------------------------------------
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  //-----------------------------------------------------------------------
  // 3. Create command queues
  //-----------------------------------------------------------------------
  cmd_queue = clCreateCommandQueue(context, device, 0, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  max_pipeline = (jend-jst) < max_compute_units ? (jend-jst) : max_compute_units;
  pipe_queue = (cl_command_queue *)malloc(sizeof(cl_command_queue) * max_pipeline);
  for (i = 0; i < max_pipeline; i++) {
    pipe_queue[i] = clCreateCommandQueue(context, device, 0, &ecode);
    clu_CheckError(ecode, "clCreateCommandQueue()");
  }

  //-----------------------------------------------------------------------
  // 4. Build programs
  //-----------------------------------------------------------------------
  if (timeron) timer_start(TIMER_BUILD);
  char build_option[100];

  if (device_type == CL_DEVICE_TYPE_CPU) {
    sprintf(build_option, "-I. -DCLASS=%d -DUSE_CPU", CLASS);
  } else {
    sprintf(build_option, "-I. -DCLASS=\'%c\'", CLASS);
  }

  p_pre = clu_MakeProgram(context, device, source_dir,
                          "kernel_pre.cl",
                          build_option);

  p_main = clu_MakeProgram(context, device, source_dir,
                          (device_type == CL_DEVICE_TYPE_CPU ? "kernel_main_cpu.cl" : "kernel_main_gpu.cl"),
                          build_option);

  p_post = clu_MakeProgram(context, device, source_dir,
                          "kernel_post.cl",
                          build_option);
  if (timeron) timer_stop(TIMER_BUILD);

  //-----------------------------------------------------------------------
  // 5. Create buffers
  //-----------------------------------------------------------------------
  if (timeron) timer_start(TIMER_BUFFER);
  m_ce = clCreateBuffer(context,
                        CL_MEM_READ_ONLY,
                        sizeof(double)*5*13,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_ce");

  m_u = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*(ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_u");

  m_rsd = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*(ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rsd");

  m_frct = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*(ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_frct");

  m_qs = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*(ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_qs");

  m_rho_i = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*(ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rho_i");

  // workspace for work-items
  size_t max_work_items;
  if (ERHS2_DIM == 1 && ERHS3_DIM == 1 && ERHS4_DIM == 1) {
    max_work_items = ISIZ3;
  } else {
    max_work_items = ISIZ3*ISIZ2;
  }
  m_flux = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*ISIZ1*5 * max_work_items,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_flux");

  if (RHSZ_DIM == 1) {
    max_work_items = ISIZ2;
  } else {
    max_work_items = ISIZ2*ISIZ1;
  }

  if (device_type == CL_DEVICE_TYPE_CPU) {
    m_utmp = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sizeof(double)*ISIZ3*6 * max_work_items,
                         NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_utmp");

    m_rtmp = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sizeof(double)*ISIZ3*5 * max_work_items,
                         NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_rtmp");
  }

  temp = (nz0-2) / max_compute_units;
  l2norm_lws[0] = temp == 0 ? 1 : temp;
  l2norm_gws[0] = clu_RoundWorkSize((size_t)(nz0-2), l2norm_lws[0]);
  wg_num = l2norm_gws[0] / l2norm_lws[0];
  sum_size = sizeof(double) * 5 * wg_num;
  m_sum = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sum_size, 
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  if (timeron) timer_stop(TIMER_BUFFER);

  //-----------------------------------------------------------------------
  // 6. Create kernels
  //-----------------------------------------------------------------------
  k_setbv1 = clCreateKernel(p_pre, "setbv1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for setbv1");
  ecode  = clSetKernelArg(k_setbv1, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_setbv1, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_setbv1, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_setbv1, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_setbv1, 4, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SETBV1_DIM == 3) {
    setbv1_lws[0] = 5;
    temp = max_work_group_size / setbv1_lws[0];
    setbv1_lws[1] = nx < temp ? nx : temp;
    temp = temp / setbv1_lws[1];
    setbv1_lws[2] = ny < temp ? ny : temp;
    setbv1_gws[0] = clu_RoundWorkSize((size_t)5, setbv1_lws[0]);
    setbv1_gws[1] = clu_RoundWorkSize((size_t)nx, setbv1_lws[1]);
    setbv1_gws[2] = clu_RoundWorkSize((size_t)ny, setbv1_lws[2]);
  } else if (SETBV1_DIM == 2) {
    setbv1_lws[0] = nx < work_item_sizes[0] ? nx : work_item_sizes[0];
    temp = max_work_group_size / setbv1_lws[0];
    setbv1_lws[1] = ny < temp ? ny : temp;
    setbv1_gws[0] = clu_RoundWorkSize((size_t)nx, setbv1_lws[0]);
    setbv1_gws[1] = clu_RoundWorkSize((size_t)ny, setbv1_lws[1]);
  } else {
    temp = ny / max_compute_units;
    setbv1_lws[0] = temp == 0 ? 1 : temp;
    setbv1_gws[0] = clu_RoundWorkSize((size_t)ny, setbv1_lws[0]);
  }

  k_setbv2 = clCreateKernel(p_pre, "setbv2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for setbv2");
  ecode  = clSetKernelArg(k_setbv2, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_setbv2, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_setbv2, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_setbv2, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_setbv2, 4, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SETBV2_DIM == 3) {
    setbv2_lws[0] = 5;
    temp = max_work_group_size / setbv2_lws[0];
    setbv2_lws[1] = nx < temp ? nx : temp;
    temp = temp / setbv2_lws[1];
    setbv2_lws[2] = nz < temp ? nz : temp;
    setbv2_gws[0] = clu_RoundWorkSize((size_t)5, setbv2_lws[0]);
    setbv2_gws[1] = clu_RoundWorkSize((size_t)nx, setbv2_lws[1]);
    setbv2_gws[2] = clu_RoundWorkSize((size_t)nz, setbv2_lws[2]);
  } else if (SETBV2_DIM == 2) {
    setbv2_lws[0] = nx < work_item_sizes[0] ? nx : work_item_sizes[0];
    temp = max_work_group_size / setbv2_lws[0];
    setbv2_lws[1] = nz < temp ? nz : temp;
    setbv2_gws[0] = clu_RoundWorkSize((size_t)nx, setbv2_lws[0]);
    setbv2_gws[1] = clu_RoundWorkSize((size_t)nz, setbv2_lws[1]);
  } else {
    temp = nz / max_compute_units;
    setbv2_lws[0] = temp == 0 ? 1 : temp;
    setbv2_gws[0] = clu_RoundWorkSize((size_t)nz, setbv2_lws[0]);
  }

  k_setbv3 = clCreateKernel(p_pre, "setbv3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for setbv3");
  ecode  = clSetKernelArg(k_setbv3, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_setbv3, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_setbv3, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_setbv3, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_setbv3, 4, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SETBV3_DIM == 3) {
    setbv3_lws[0] = 5;
    temp = max_work_group_size / setbv3_lws[0];
    setbv3_lws[1] = ny < temp ? ny : temp;
    temp = temp / setbv3_lws[1];
    setbv3_lws[2] = nz < temp ? nz : temp;
    setbv3_gws[0] = clu_RoundWorkSize((size_t)5, setbv3_lws[0]);
    setbv3_gws[1] = clu_RoundWorkSize((size_t)ny, setbv3_lws[1]);
    setbv3_gws[2] = clu_RoundWorkSize((size_t)nz, setbv3_lws[2]);
  } else if (SETBV3_DIM == 2) {
    setbv3_lws[0] = ny < work_item_sizes[0] ? ny : work_item_sizes[0];
    temp = max_work_group_size / setbv3_lws[0];
    setbv3_lws[1] = nz < temp ? nz : temp;
    setbv3_gws[0] = clu_RoundWorkSize((size_t)ny, setbv3_lws[0]);
    setbv3_gws[1] = clu_RoundWorkSize((size_t)nz, setbv3_lws[1]);
  } else {
    temp = nz / max_compute_units;
    setbv3_lws[0] = temp == 0 ? 1 : temp;
    setbv3_gws[0] = clu_RoundWorkSize((size_t)nz, setbv3_lws[0]);
  }

  k_setiv = clCreateKernel(p_pre, "setiv", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for setiv");
  ecode  = clSetKernelArg(k_setiv, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_setiv, 1, sizeof(cl_mem), &m_ce);
  ecode |= clSetKernelArg(k_setiv, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_setiv, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_setiv, 4, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SETIV_DIM == 3) {
    setiv_lws[0] = (nx-2) < work_item_sizes[0] ? (nx-2) : work_item_sizes[0];
    temp = max_work_group_size / setiv_lws[0];
    setiv_lws[1] = (ny-2) < temp ? (ny-2) : temp;
    temp = temp / setiv_lws[1];
    setiv_lws[2] = (nz-2) < temp ? (nz-2) : temp;
    setiv_gws[0] = clu_RoundWorkSize((size_t)(nx-2), setiv_lws[0]);
    setiv_gws[1] = clu_RoundWorkSize((size_t)(ny-2), setiv_lws[1]);
    setiv_gws[2] = clu_RoundWorkSize((size_t)(nz-2), setiv_lws[2]);
  } else if (SETIV_DIM == 2) {
    setiv_lws[0] = (ny-2) < work_item_sizes[0] ? (ny-2) : work_item_sizes[0];
    temp = max_work_group_size / setiv_lws[0];
    setiv_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    setiv_gws[0] = clu_RoundWorkSize((size_t)(ny-2), setiv_lws[0]);
    setiv_gws[1] = clu_RoundWorkSize((size_t)(nz-2), setiv_lws[1]);
  } else {
    temp = (nz-2) / max_compute_units;
    setiv_lws[0] = temp == 0 ? 1 : temp;
    setiv_gws[0] = clu_RoundWorkSize((size_t)(nz-2), setiv_lws[0]);
  }

  k_l2norm = clCreateKernel(p_main, "l2norm", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
  ecode  = clSetKernelArg(k_l2norm, 1, sizeof(cl_mem), &m_sum);
  ecode |= clSetKernelArg(k_l2norm, 2, sizeof(double)*5*l2norm_lws[0], NULL);
  clu_CheckError(ecode, "clSetKernelArg()");

  k_rhs = clCreateKernel(p_main, "rhs", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rhs");
  ecode  = clSetKernelArg(k_rhs, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_rhs, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_rhs, 2, sizeof(cl_mem), &m_frct);
  ecode |= clSetKernelArg(k_rhs, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhs, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhs, 5, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_rhs, 6, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_rhs, 7, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (RHS_DIM == 3) {
    rhs_lws[0] = nx < work_item_sizes[0] ? nx : work_item_sizes[0];
    temp = max_work_group_size / rhs_lws[0];
    rhs_lws[1] = ny < temp ? ny : temp;
    temp = temp / rhs_lws[1];
    rhs_lws[2] = nz < temp ? nz : temp;
    rhs_gws[0] = clu_RoundWorkSize((size_t)nx, rhs_lws[0]);
    rhs_gws[1] = clu_RoundWorkSize((size_t)ny, rhs_lws[1]);
    rhs_gws[2] = clu_RoundWorkSize((size_t)nz, rhs_lws[2]);
  } else if (RHS_DIM == 2) {
    rhs_lws[0] = ny < work_item_sizes[0] ? ny : work_item_sizes[0];
    temp = max_work_group_size / rhs_lws[0];
    rhs_lws[1] = nz < temp ? nz : temp;
    rhs_gws[0] = clu_RoundWorkSize((size_t)ny, rhs_lws[0]);
    rhs_gws[1] = clu_RoundWorkSize((size_t)nz, rhs_lws[1]);
  } else {
    //temp = nz / max_compute_units;
    temp = 1;
    rhs_lws[0] = temp == 0 ? 1 : temp;
    rhs_gws[0] = clu_RoundWorkSize((size_t)nz, rhs_lws[0]);
  }

  k_rhsx = clCreateKernel(p_main, "rhsx", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rhsx");
  ecode  = clSetKernelArg(k_rhsx, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_rhsx, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_rhsx, 2, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsx, 3, sizeof(cl_mem), &m_rho_i);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_rhsx, 4, sizeof(cl_mem), &m_flux);
    ecode |= clSetKernelArg(k_rhsx, 5, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsx, 6, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsx, 7, sizeof(int), &nz);
  } else {
    ecode |= clSetKernelArg(k_rhsx, 4, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsx, 5, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsx, 6, sizeof(int), &nz);
  }
  clu_CheckError(ecode, "clSetKernelArg()");
  if (RHSX_DIM == 2) {
    rhsx_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
    temp = max_work_group_size / rhsx_lws[0];
    rhsx_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    rhsx_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), rhsx_lws[0]);
    rhsx_gws[1] = clu_RoundWorkSize((size_t)(nz-2), rhsx_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    rhsx_lws[0] = temp == 0 ? 1 : temp;
    rhsx_gws[0] = clu_RoundWorkSize((size_t)(nz-2), rhsx_lws[0]);
  }

  k_rhsy = clCreateKernel(p_main, "rhsy", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rhsy");
  ecode  = clSetKernelArg(k_rhsy, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_rhsy, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_rhsy, 2, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsy, 3, sizeof(cl_mem), &m_rho_i);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_rhsy, 4, sizeof(cl_mem), &m_flux);
    ecode |= clSetKernelArg(k_rhsy, 5, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsy, 6, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsy, 7, sizeof(int), &nz);
  } else {
    ecode |= clSetKernelArg(k_rhsy, 4, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsy, 5, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsy, 6, sizeof(int), &nz);
  }
  clu_CheckError(ecode, "clSetKernelArg()");
  if (RHSY_DIM == 2) {
    rhsy_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / rhsy_lws[0];
    rhsy_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    rhsy_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), rhsy_lws[0]);
    rhsy_gws[1] = clu_RoundWorkSize((size_t)(nz-2), rhsy_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    rhsy_lws[0] = temp == 0 ? 1 : temp;
    rhsy_gws[0] = clu_RoundWorkSize((size_t)(nz-2), rhsy_lws[0]);
  }

  k_rhsz = clCreateKernel(p_main, "rhsz", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rhsz");
  ecode  = clSetKernelArg(k_rhsz, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_rhsz, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_rhsz, 2, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsz, 3, sizeof(cl_mem), &m_rho_i);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    ecode |= clSetKernelArg(k_rhsz, 4, sizeof(cl_mem), &m_flux);
    ecode |= clSetKernelArg(k_rhsz, 5, sizeof(cl_mem), &m_utmp);
    ecode |= clSetKernelArg(k_rhsz, 6, sizeof(cl_mem), &m_rtmp);
    ecode |= clSetKernelArg(k_rhsz, 7, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsz, 8, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsz, 9, sizeof(int), &nz);
  } else {
    ecode |= clSetKernelArg(k_rhsz, 4, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_rhsz, 5, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_rhsz, 6, sizeof(int), &nz);
  }
  clu_CheckError(ecode, "clSetKernelArg()");
  if (RHSZ_DIM == 2) {
    rhsz_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / rhsz_lws[0];
    rhsz_lws[1] = (jend-jst) < temp ? (jend-jst) : temp;
    rhsz_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), rhsz_lws[0]);
    rhsz_gws[1] = clu_RoundWorkSize((size_t)(jend-jst), rhsz_lws[1]);
  } else {
    //temp = (jend-jst) / max_compute_units;
    temp = 1;
    rhsz_lws[0] = temp == 0 ? 1 : temp;
    rhsz_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), rhsz_lws[0]);
  }

  k_ssor2 = clCreateKernel(p_main, "ssor2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for ssor2");
  ecode  = clSetKernelArg(k_ssor2, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_ssor2, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_ssor2, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_ssor2, 4, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SSOR2_DIM == 3) {
    ssor2_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / ssor2_lws[0];
    ssor2_lws[1] = (jend-jst) < temp ? (jend-jst) : temp;
    temp = temp / ssor2_lws[1];
    ssor2_lws[2] = (nz-2) < temp ? (nz-2) : temp;
    ssor2_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), ssor2_lws[0]);
    ssor2_gws[1] = clu_RoundWorkSize((size_t)(jend-jst), ssor2_lws[1]);
    ssor2_gws[2] = clu_RoundWorkSize((size_t)(nz-2), ssor2_lws[2]);
  } else if (SSOR2_DIM == 2) {
    ssor2_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
    temp = max_work_group_size / ssor2_lws[0];
    ssor2_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    ssor2_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), ssor2_lws[0]);
    ssor2_gws[1] = clu_RoundWorkSize((size_t)(nz-2), ssor2_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    ssor2_lws[0] = temp == 0 ? 1 : temp;
    ssor2_gws[0] = clu_RoundWorkSize((size_t)(nz-2), ssor2_lws[0]);
  }

  k_ssor3 = clCreateKernel(p_main, "ssor3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for ssor3");
  ecode  = clSetKernelArg(k_ssor3, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_ssor3, 1, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_ssor3, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_ssor3, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_ssor3, 5, sizeof(int), &nz);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (SSOR3_DIM == 3) {
    ssor3_lws[0] = (iend-ist) < work_item_sizes[0] ? (iend-ist) : work_item_sizes[0];
    temp = max_work_group_size / ssor3_lws[0];
    ssor3_lws[1] = (jend-jst) < temp ? (jend-jst) : temp;
    temp = temp / ssor3_lws[1];
    ssor3_lws[2] = (nz-2) < temp ? (nz-2) : temp;
    ssor3_gws[0] = clu_RoundWorkSize((size_t)(iend-ist), ssor3_lws[0]);
    ssor3_gws[1] = clu_RoundWorkSize((size_t)(jend-jst), ssor3_lws[1]);
    ssor3_gws[2] = clu_RoundWorkSize((size_t)(nz-2), ssor3_lws[2]);
  } else if (SSOR3_DIM == 2) {
    ssor3_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
    temp = max_work_group_size / ssor3_lws[0];
    ssor3_lws[1] = (nz-2) < temp ? (nz-2) : temp;
    ssor3_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), ssor3_lws[0]);
    ssor3_gws[1] = clu_RoundWorkSize((size_t)(nz-2), ssor3_lws[1]);
  } else {
    //temp = (nz-2) / max_compute_units;
    temp = 1;
    ssor3_lws[0] = temp == 0 ? 1 : temp;
    ssor3_gws[0] = clu_RoundWorkSize((size_t)(nz-2), ssor3_lws[0]);
  }

  k_blts = clCreateKernel(p_main, "blts", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for blts");
  ecode  = clSetKernelArg(k_blts, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_blts, 1, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_blts, 2, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_blts, 3, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_blts, 4, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_blts, 5, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_blts, 6, sizeof(int), &nx);
  clu_CheckError(ecode, "clSetKernelArg()");
  blts_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
  temp = max_work_group_size / blts_lws[0];
  blts_lws[1] = (nz-2) < temp ? (nz-2) : temp;
  blts_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), blts_lws[0]);
  blts_gws[1] = clu_RoundWorkSize((size_t)(nz-2), blts_lws[1]);

  k_buts = clCreateKernel(p_main, "buts", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for buts");
  ecode  = clSetKernelArg(k_buts, 0, sizeof(cl_mem), &m_rsd);
  ecode |= clSetKernelArg(k_buts, 1, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_buts, 2, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_buts, 3, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_buts, 4, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_buts, 5, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_buts, 6, sizeof(int), &nx);
  clu_CheckError(ecode, "clSetKernelArg()");
  buts_lws[0] = (jend-jst) < work_item_sizes[0] ? (jend-jst) : work_item_sizes[0];
  temp = max_work_group_size / buts_lws[0];
  buts_lws[1] = (nz-2) < temp ? (nz-2) : temp;
  buts_gws[0] = clu_RoundWorkSize((size_t)(jend-jst), buts_lws[0]);
  buts_gws[1] = clu_RoundWorkSize((size_t)(nz-2), buts_lws[1]);

  if (timeron) timer_stop(TIMER_OPENCL);
}


static void release_opencl()
{
  int i;

  if (timeron) {
    timer_start(TIMER_OPENCL);
    timer_start(TIMER_RELEASE);
  }

  clReleaseKernel(k_setbv1);
  clReleaseKernel(k_setbv2);
  clReleaseKernel(k_setbv3);
  clReleaseKernel(k_setiv);
  clReleaseKernel(k_l2norm);
  clReleaseKernel(k_rhs);
  clReleaseKernel(k_rhsx);
  clReleaseKernel(k_rhsy);
  clReleaseKernel(k_rhsz);
  clReleaseKernel(k_ssor2);
  clReleaseKernel(k_ssor3);
  clReleaseKernel(k_blts);
  clReleaseKernel(k_buts);

  clReleaseMemObject(m_ce);
  clReleaseMemObject(m_u);
  clReleaseMemObject(m_rsd);
  clReleaseMemObject(m_frct);
  clReleaseMemObject(m_flux);
  clReleaseMemObject(m_qs);
  clReleaseMemObject(m_rho_i);
  clReleaseMemObject(m_sum);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    clReleaseMemObject(m_utmp);
    clReleaseMemObject(m_rtmp);
  }

  clReleaseProgram(p_pre);
  clReleaseProgram(p_main);
  clReleaseProgram(p_post);
  clReleaseCommandQueue(cmd_queue);
  for (i = 0; i < max_pipeline; i++) {
    clReleaseCommandQueue(pipe_queue[i]);
  }
  free(pipe_queue);
  clReleaseContext(context);

  if (timeron) {
    timer_stop(TIMER_RELEASE);
    timer_stop(TIMER_OPENCL);

    double tt = timer_read(TIMER_OPENCL);
    printf(" OpenCL   : %9.4lf\n", tt);
    tt = timer_read(TIMER_BUILD);
    printf(" - Build  : %9.4lf\n", tt);
    tt = timer_read(TIMER_BUFFER);
    printf(" - Buffer : %9.4lf\n", tt);
    tt = timer_read(TIMER_RELEASE);
    printf(" - Release: %9.4lf\n", tt);
  }
}
