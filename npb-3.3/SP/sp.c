//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB SP code. This OpenCL    //
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
//       program SP
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "header.h"
#include "print_results.h"

#include "sp_dim.h"

//---------------------------------------------------------------------
// OPENCL Variables
//---------------------------------------------------------------------
cl_device_type   device_type;
cl_device_id     device;
char            *device_name;
cl_context       context;
cl_command_queue cmd_queue;
cl_program       p_exact_rhs;
cl_program       p_initialize;
cl_program       p_adi;
cl_program       p_error;
size_t  work_item_sizes[3];
size_t  max_work_group_size;
cl_uint max_compute_units;

cl_kernel k_compute_rhs1;
cl_kernel k_compute_rhs2;
cl_kernel k_compute_rhs3;
cl_kernel k_compute_rhs4;
cl_kernel k_compute_rhs5;
cl_kernel k_compute_rhs6;
cl_kernel k_txinvr;
cl_kernel k_x_solve;
cl_kernel k_ninvr;
cl_kernel k_y_solve;
cl_kernel k_pinvr;
cl_kernel k_z_solve;
cl_kernel k_tzetar;
cl_kernel k_add;

cl_mem m_ce;
cl_mem m_u;
cl_mem m_us;
cl_mem m_vs;
cl_mem m_ws;
cl_mem m_qs;
cl_mem m_rho_i;
cl_mem m_speed;
cl_mem m_square;
cl_mem m_rhs;
cl_mem m_forcing;

cl_mem m_cv;
cl_mem m_rhon;
cl_mem m_rhos;
cl_mem m_rhoq;

cl_mem m_lhs;
cl_mem m_lhsp;
cl_mem m_lhsm;

size_t compute_rhs1_lws[3], compute_rhs1_gws[3];
size_t compute_rhs2_lws[3], compute_rhs2_gws[3];
size_t compute_rhs3_lws[3], compute_rhs3_gws[3];
size_t compute_rhs4_lws[3], compute_rhs4_gws[3];
size_t compute_rhs5_lws[3], compute_rhs5_gws[3];
size_t compute_rhs6_lws[3], compute_rhs6_gws[3];
size_t x_solve_lws[3], x_solve_gws[3];
size_t y_solve_lws[3], y_solve_gws[3];
size_t z_solve_lws[3], z_solve_gws[3];
size_t txinvr_lws[3], txinvr_gws[3];
size_t ninvr_lws[3], ninvr_gws[3];
size_t pinvr_lws[3], pinvr_gws[3];
size_t tzetar_lws[3], tzetar_gws[3];
size_t add_lws[3], add_gws[3];

int EXACT_RHS1_DIM, EXACT_RHS5_DIM;
int INITIALIZE2_DIM;
int COMPUTE_RHS1_DIM, COMPUTE_RHS2_DIM, COMPUTE_RHS4_DIM;
int COMPUTE_RHS6_DIM;
int TXINVR_DIM, NINVR_DIM, PINVR_DIM, TZETAR_DIM, ADD_DIM;
int X_SOLVE_DIM, Y_SOLVE_DIM, Z_SOLVE_DIM;

static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//---------------------------------------------------------------------


/* common /global/ */
int grid_points[3], nx2, ny2, nz2;
logical timeron;

/* common /constants/ */
double ce[5][13], dt;


int main(int argc, char *argv[])
{
  int i, niter, step, n3;
  double mflops, t, tmax, trecs[t_last+1];
  logical verified;
  char Class;
  char *t_names[t_last+1];

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  //---------------------------------------------------------------------
  // Read input file (if it exists), else take
  // defaults from parameters
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_xsolve] = "xsolve";
    t_names[t_ysolve] = "ysolve";
    t_names[t_zsolve] = "zsolve";
    t_names[t_rdis1] = "redist1";
    t_names[t_rdis2] = "redist2";
    t_names[t_tzetar] = "tzetar";
    t_names[t_ninvr] = "ninvr";
    t_names[t_pinvr] = "pinvr";
    t_names[t_txinvr] = "txinvr";
    t_names[t_add] = "add";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - SP Benchmark\n\n");

  if ((fp = fopen("inputsp.data", "r")) != NULL) {
    int result;
    printf(" Reading from input file inputsp.data\n");
    result = fscanf(fp, "%d", &niter);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d", &grid_points[0], &grid_points[1], &grid_points[2]);
    fclose(fp);
  } else {
    printf(" No input file inputsp.data. Using compiled defaults\n");
    niter = NITER_DEFAULT;
    dt    = DT_DEFAULT;
    grid_points[0] = PROBLEM_SIZE;
    grid_points[1] = PROBLEM_SIZE;
    grid_points[2] = PROBLEM_SIZE;
  }

  printf(" Size: %4dx%4dx%4d\n", 
      grid_points[0], grid_points[1], grid_points[2]);
  printf(" Iterations: %4d    dt:  %11.7f\n", niter, dt);
  printf("\n");

  if ((grid_points[0] > IMAX) ||
      (grid_points[1] > JMAX) ||
      (grid_points[2] > KMAX) ) {
    printf(" %d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }
  nx2 = grid_points[0] - 2;
  ny2 = grid_points[1] - 2;
  nz2 = grid_points[2] - 2;

  setup_opencl(argc, argv);

  set_constants();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  exact_rhs();

  initialize();

  //---------------------------------------------------------------------
  // do one time step to touch all code, and reinitialize
  //---------------------------------------------------------------------
  adi();
  initialize();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_start(1);

  for (step = 1; step <= niter; step++) {
    if ((step % 20) == 0 || step == 1) {
      printf(" Time step %4d\n", step);
    }

    adi();
  }

  timer_stop(1);
  tmax = timer_read(1);

  verify(niter, &Class, &verified);

  if (tmax != 0.0) {
    n3 = grid_points[0]*grid_points[1]*grid_points[2];
    t = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
    mflops = (881.174 * (double)n3
             - 4683.91 * (t * t)
             + 11484.5 * t
             - 19272.4) * (double)niter / (tmax*1000000.0);
  } else {
    mflops = 0.0;
  }

  c_print_results("SP", Class, grid_points[0], 
                  grid_points[1], grid_points[2], niter, 
                  tmax, mflops, "          floating point", 
                  verified, NPBVERSION,COMPILETIME, CS1, CS2, CS3, CS4, CS5, 
                  CS6, "(none)",
                  clu_GetDeviceTypeName(device_type),
                  device_name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    for (i = 1; i <= t_last; i++) {
      trecs[i] = timer_read(i);
    }
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION   Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n", 
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[t_rhs] - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      } else if (i == t_zsolve) {
        t = trecs[t_zsolve] - trecs[t_rdis1] - trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-zsol", t, t*100./tmax);
      } else if (i == t_rdis2) {
        t = trecs[t_rdis1] + trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "redist", t, t*100./tmax);
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
  size_t temp;
  cl_int ecode;
  char *source_dir = "SP";

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
    EXACT_RHS1_DIM = EXACT_RHS1_DIM_CPU;
    EXACT_RHS5_DIM = EXACT_RHS5_DIM_CPU;
    INITIALIZE2_DIM = INITIALIZE2_DIM_CPU;
    COMPUTE_RHS1_DIM = COMPUTE_RHS1_DIM_CPU;
    COMPUTE_RHS2_DIM = COMPUTE_RHS2_DIM_CPU;
    COMPUTE_RHS4_DIM = COMPUTE_RHS4_DIM_CPU;
    COMPUTE_RHS6_DIM = COMPUTE_RHS6_DIM_CPU;
    TXINVR_DIM = TXINVR_DIM_CPU;
    NINVR_DIM = NINVR_DIM_CPU;
    PINVR_DIM = PINVR_DIM_CPU;
    TZETAR_DIM = TZETAR_DIM_CPU;
    ADD_DIM = ADD_DIM_CPU;
    X_SOLVE_DIM = X_SOLVE_DIM_CPU;
    Y_SOLVE_DIM = Y_SOLVE_DIM_CPU;
    Z_SOLVE_DIM = Z_SOLVE_DIM_CPU;
  } else {
    EXACT_RHS1_DIM = EXACT_RHS1_DIM_GPU;
    EXACT_RHS5_DIM = EXACT_RHS5_DIM_GPU;
    INITIALIZE2_DIM = INITIALIZE2_DIM_GPU;
    COMPUTE_RHS1_DIM = COMPUTE_RHS1_DIM_GPU;
    COMPUTE_RHS2_DIM = COMPUTE_RHS2_DIM_GPU;
    COMPUTE_RHS4_DIM = COMPUTE_RHS4_DIM_GPU;
    COMPUTE_RHS6_DIM = COMPUTE_RHS6_DIM_GPU;
    TXINVR_DIM = TXINVR_DIM_GPU;
    NINVR_DIM = NINVR_DIM_GPU;
    PINVR_DIM = PINVR_DIM_GPU;
    TZETAR_DIM = TZETAR_DIM_GPU;
    ADD_DIM = ADD_DIM_GPU;
    X_SOLVE_DIM = X_SOLVE_DIM_GPU;
    Y_SOLVE_DIM = Y_SOLVE_DIM_GPU;
    Z_SOLVE_DIM = Z_SOLVE_DIM_GPU;
  }
  ////////////////////////////////////////////////////////////////////////

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

  // exact_rhs()
  p_exact_rhs = clu_MakeProgram(context, device, source_dir,
                                "kernel_exact_rhs.cl",
                                build_option);

  // initialize()
  p_initialize = clu_MakeProgram(context, device, source_dir,
                                 "kernel_initialize.cl",
                                 build_option);

  // error_norm() and rhs_norm()
  p_error = clu_MakeProgram(context, device, source_dir, "kernel_error.cl",
                            build_option);

  // functions called in adi()
  if (device_type == CL_DEVICE_TYPE_CPU) {
    p_adi = clu_MakeProgram(context, device, source_dir, "kernel_adi_cpu.cl",
                            build_option);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    p_adi = clu_MakeProgram(context, device, source_dir, "kernel_adi_gpu.cl",
                            build_option);
  }
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
                       sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_u");

  m_us = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_us");

  m_vs = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_vs");

  m_ws = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_ws");

  m_qs = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_qs");

  m_rho_i = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                           NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rho_i");

  m_speed = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                           NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_speed");

  m_square = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_square");

  m_rhs = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rhs");

  m_forcing = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                             NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_forcing");

  // workspace for work-items
  size_t max_work_items, buf_size1, buf_size3;
  if ((X_SOLVE_DIM == 2) && (Y_SOLVE_DIM == 2) && (Z_SOLVE_DIM == 2)) {
    max_work_items = PROBLEM_SIZE * PROBLEM_SIZE;
    buf_size1 = sizeof(double)*PROBLEM_SIZE * max_work_items;
    buf_size3 = sizeof(double)*(IMAXP+1)*5 * max_work_items;
  } else {
    max_work_items = PROBLEM_SIZE;
    buf_size1 = sizeof(double)*PROBLEM_SIZE * max_work_items;
    buf_size3 = sizeof(double)*(IMAXP+1)*(IMAXP+1)*5 * max_work_items;
  }
  m_cv = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_cv");
  
  m_rhon = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rhon");
  
  m_rhos = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rhos");
  
  m_rhoq = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size1,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rhoq");
  
  m_lhs = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size3,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_lhs");
  
  m_lhsp = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size3,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_lhsp");
  
  m_lhsm = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        buf_size3,
                        NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_lhsm");
  if (timeron) timer_stop(TIMER_BUFFER);

  //-----------------------------------------------------------------------
  // 6. Create kernels
  //-----------------------------------------------------------------------
  int d0 = grid_points[0];
  int d1 = grid_points[1];
  int d2 = grid_points[2];

  k_compute_rhs1 = clCreateKernel(p_adi, "compute_rhs1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs1");
  ecode  = clSetKernelArg(k_compute_rhs1, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_compute_rhs1, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_compute_rhs1, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_compute_rhs1, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_compute_rhs1, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_compute_rhs1, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_compute_rhs1, 6, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_compute_rhs1, 7, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_compute_rhs1, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_compute_rhs1, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_compute_rhs1, 10, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (COMPUTE_RHS1_DIM == 3) {
    compute_rhs1_lws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs1_lws[0];
    compute_rhs1_lws[1] = d1 < temp ? d1 : temp;
    temp = temp / compute_rhs1_lws[1];
    compute_rhs1_lws[2] = d2 < temp ? d2 : temp;

    compute_rhs1_gws[0] = clu_RoundWorkSize((size_t)d0, compute_rhs1_lws[0]);
    compute_rhs1_gws[1] = clu_RoundWorkSize((size_t)d1, compute_rhs1_lws[1]);
    compute_rhs1_gws[2] = clu_RoundWorkSize((size_t)d2, compute_rhs1_lws[2]);
  } else if (COMPUTE_RHS1_DIM == 2) {
    compute_rhs1_lws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs1_lws[0];
    compute_rhs1_lws[1] = d2 < temp ? d2 : temp;

    compute_rhs1_gws[0] = clu_RoundWorkSize((size_t)d1, compute_rhs1_lws[0]);
    compute_rhs1_gws[1] = clu_RoundWorkSize((size_t)d2, compute_rhs1_lws[1]);
  } else {
    temp = d2 / max_compute_units;
    compute_rhs1_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs1_gws[0] = clu_RoundWorkSize((size_t)d2, compute_rhs1_lws[0]);
  }

  k_compute_rhs2 = clCreateKernel(p_adi, "compute_rhs2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs2");
  ecode  = clSetKernelArg(k_compute_rhs2, 0, sizeof(cl_mem), &m_forcing);
  ecode |= clSetKernelArg(k_compute_rhs2, 1, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_compute_rhs2, 2, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_compute_rhs2, 3, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_compute_rhs2, 4, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (COMPUTE_RHS2_DIM == 3) {
    compute_rhs2_lws[0] = (nx2+2) < work_item_sizes[0] ? (nx2+2) : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs2_lws[0];
    compute_rhs2_lws[1] = (ny2+2) < temp ? (ny2+2) : temp;
    temp = temp / compute_rhs2_lws[1];
    compute_rhs2_lws[2] = (nz2+2) < temp ? (nz2+2) : temp;

    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)(nx2+2), compute_rhs2_lws[0]);
    compute_rhs2_gws[1] = clu_RoundWorkSize((size_t)(ny2+2), compute_rhs2_lws[1]);
    compute_rhs2_gws[2] = clu_RoundWorkSize((size_t)(nz2+2), compute_rhs2_lws[2]);
  } else if (COMPUTE_RHS2_DIM == 2) {
    compute_rhs2_lws[0] = (ny2+2) < work_item_sizes[0] ? (ny2+2) : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs2_lws[0];
    compute_rhs2_lws[1] = (nz2+2) < temp ? (nz2+2) : temp;

    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)(ny2+2), compute_rhs2_lws[0]);
    compute_rhs2_gws[1] = clu_RoundWorkSize((size_t)(nz2+2), compute_rhs2_lws[1]);
  } else {
    temp = (nz2+2) / max_compute_units;
    compute_rhs2_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)(nz2+2), compute_rhs2_lws[0]);
  }


  k_compute_rhs3 = clCreateKernel(p_adi, "compute_rhs3", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs3");
  ecode  = clSetKernelArg(k_compute_rhs3, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_compute_rhs3, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_compute_rhs3, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_compute_rhs3, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_compute_rhs3, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_compute_rhs3, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_compute_rhs3, 6, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_compute_rhs3, 7, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_compute_rhs3, 8, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_compute_rhs3, 9, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_compute_rhs3, 10, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  compute_rhs3_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
  temp = max_work_group_size / compute_rhs3_lws[0];
  compute_rhs3_lws[1] = nz2 < temp ? nz2 : temp;
  compute_rhs3_gws[0] = clu_RoundWorkSize((size_t)ny2, compute_rhs3_lws[0]);
  compute_rhs3_gws[1] = clu_RoundWorkSize((size_t)nz2, compute_rhs3_lws[1]);

  k_compute_rhs4 = clCreateKernel(p_adi, "compute_rhs4", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs4");
  ecode  = clSetKernelArg(k_compute_rhs4, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_compute_rhs4, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_compute_rhs4, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_compute_rhs4, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_compute_rhs4, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_compute_rhs4, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_compute_rhs4, 6, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_compute_rhs4, 7, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_compute_rhs4, 8, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_compute_rhs4, 9, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_compute_rhs4, 10, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (COMPUTE_RHS4_DIM == 2) {
    compute_rhs4_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs4_lws[0];
    compute_rhs4_lws[1] = nz2 < temp ? nz2 : temp;

    compute_rhs4_gws[0] = clu_RoundWorkSize((size_t)nx2, compute_rhs4_lws[0]);
    compute_rhs4_gws[1] = clu_RoundWorkSize((size_t)nz2, compute_rhs4_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    compute_rhs4_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs4_gws[0] = clu_RoundWorkSize((size_t)nz2, compute_rhs4_lws[0]);
  }

  k_compute_rhs5 = clCreateKernel(p_adi, "compute_rhs5", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs5");
  ecode  = clSetKernelArg(k_compute_rhs5, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_compute_rhs5, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_compute_rhs5, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_compute_rhs5, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_compute_rhs5, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_compute_rhs5, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_compute_rhs5, 6, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_compute_rhs5, 7, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_compute_rhs5, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_compute_rhs5, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_compute_rhs5, 10, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "clSetKernelArg()");
  compute_rhs5_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
  temp = max_work_group_size / compute_rhs5_lws[0];
  compute_rhs5_lws[1] = (d1-2) < temp ? (d1-2) : temp;
  compute_rhs5_gws[0] = clu_RoundWorkSize((size_t)(d0-2), compute_rhs5_lws[0]);
  compute_rhs5_gws[1] = clu_RoundWorkSize((size_t)(d1-2), compute_rhs5_lws[1]);

  k_compute_rhs6 = clCreateKernel(p_adi, "compute_rhs6", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for compute_rhs6");
  ecode  = clSetKernelArg(k_compute_rhs6, 0, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_compute_rhs6, 1, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_compute_rhs6, 2, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_compute_rhs6, 3, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (COMPUTE_RHS6_DIM == 3) {
    compute_rhs6_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs6_lws[0];
    compute_rhs6_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / compute_rhs6_lws[1];
    compute_rhs6_lws[2] = nz2 < temp ? nz2 : temp;

    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)nx2, compute_rhs6_lws[0]);
    compute_rhs6_gws[1] = clu_RoundWorkSize((size_t)ny2, compute_rhs6_lws[1]);
    compute_rhs6_gws[2] = clu_RoundWorkSize((size_t)nz2, compute_rhs6_lws[2]);
  } else if (COMPUTE_RHS6_DIM == 2) {
    compute_rhs6_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs6_lws[0];
    compute_rhs6_lws[1] = nz2 < temp ? nz2 : temp;

    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)ny2, compute_rhs6_lws[0]);
    compute_rhs6_gws[1] = clu_RoundWorkSize((size_t)nz2, compute_rhs6_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    compute_rhs6_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)nz2, compute_rhs6_lws[0]);
  }

  k_txinvr = clCreateKernel(p_adi, "txinvr", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for txinvr");
  ecode  = clSetKernelArg(k_txinvr, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_txinvr, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_txinvr, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_txinvr, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_txinvr, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_txinvr, 5, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_txinvr, 6, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_txinvr, 7, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_txinvr, 8, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_txinvr, 9, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (TXINVR_DIM == 3) {
    txinvr_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / txinvr_lws[0];
    txinvr_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / txinvr_lws[1];
    txinvr_lws[2] = nz2 < temp ? nz2 : temp;
    txinvr_gws[0] = clu_RoundWorkSize((size_t)nx2, txinvr_lws[0]);
    txinvr_gws[1] = clu_RoundWorkSize((size_t)ny2, txinvr_lws[1]);
    txinvr_gws[2] = clu_RoundWorkSize((size_t)nz2, txinvr_lws[2]);
  } else if (TXINVR_DIM == 2) {
    txinvr_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / txinvr_lws[0];
    txinvr_lws[1] = nz2 < temp ? nz2 : temp;
    txinvr_gws[0] = clu_RoundWorkSize((size_t)ny2, txinvr_lws[0]);
    txinvr_gws[1] = clu_RoundWorkSize((size_t)nz2, txinvr_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    txinvr_lws[0] = temp == 0 ? 1 : temp;
    txinvr_gws[0] = clu_RoundWorkSize((size_t)nz2, txinvr_lws[0]);
  }

  k_x_solve = clCreateKernel(p_adi, "x_solve", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for x_solve");
  ecode  = clSetKernelArg(k_x_solve, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_x_solve, 1, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_x_solve, 2, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_x_solve, 3, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_x_solve, 4, sizeof(cl_mem), &m_cv);
  ecode |= clSetKernelArg(k_x_solve, 5, sizeof(cl_mem), &m_rhon);
  ecode |= clSetKernelArg(k_x_solve, 6, sizeof(cl_mem), &m_lhs);
  ecode |= clSetKernelArg(k_x_solve, 7, sizeof(cl_mem), &m_lhsp);
  ecode |= clSetKernelArg(k_x_solve, 8, sizeof(cl_mem), &m_lhsm);
  ecode |= clSetKernelArg(k_x_solve, 9, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_x_solve, 10, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_x_solve, 11, sizeof(int), &nz2);
  ecode |= clSetKernelArg(k_x_solve, 12, sizeof(int), &grid_points[0]);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (X_SOLVE_DIM == 2) {
    x_solve_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / x_solve_lws[0];
    x_solve_lws[1] = nz2 < temp ? nz2 : temp;

    x_solve_gws[0] = clu_RoundWorkSize((size_t)ny2, x_solve_lws[0]);
    x_solve_gws[1] = clu_RoundWorkSize((size_t)nz2, x_solve_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    x_solve_lws[0] = temp == 0 ? 1 : temp;
    x_solve_gws[0] = clu_RoundWorkSize((size_t)nz2, x_solve_lws[0]);
  }

  k_ninvr = clCreateKernel(p_adi, "ninvr", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for ninvr");
  ecode  = clSetKernelArg(k_ninvr, 0, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_ninvr, 1, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_ninvr, 2, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_ninvr, 3, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (NINVR_DIM == 3) {
    ninvr_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / ninvr_lws[0];
    ninvr_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / ninvr_lws[1];
    ninvr_lws[2] = nz2 < temp ? nz2 : temp;
    ninvr_gws[0] = clu_RoundWorkSize((size_t)nx2, ninvr_lws[0]);
    ninvr_gws[1] = clu_RoundWorkSize((size_t)ny2, ninvr_lws[1]);
    ninvr_gws[2] = clu_RoundWorkSize((size_t)nz2, ninvr_lws[2]);
  } else if (NINVR_DIM == 2) {
    ninvr_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / ninvr_lws[0];
    ninvr_lws[1] = nz2 < temp ? nz2 : temp;
    ninvr_gws[0] = clu_RoundWorkSize((size_t)ny2, ninvr_lws[0]);
    ninvr_gws[1] = clu_RoundWorkSize((size_t)nz2, ninvr_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    ninvr_lws[0] = temp == 0 ? 1 : temp;
    ninvr_gws[0] = clu_RoundWorkSize((size_t)nz2, ninvr_lws[0]);
  }

  k_y_solve = clCreateKernel(p_adi, "y_solve", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for y_solve");
  ecode  = clSetKernelArg(k_y_solve, 0, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_y_solve, 1, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_y_solve, 2, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_y_solve, 3, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_y_solve, 4, sizeof(cl_mem), &m_cv);
  ecode |= clSetKernelArg(k_y_solve, 5, sizeof(cl_mem), &m_rhoq);
  ecode |= clSetKernelArg(k_y_solve, 6, sizeof(cl_mem), &m_lhs);
  ecode |= clSetKernelArg(k_y_solve, 7, sizeof(cl_mem), &m_lhsp);
  ecode |= clSetKernelArg(k_y_solve, 8, sizeof(cl_mem), &m_lhsm);
  ecode |= clSetKernelArg(k_y_solve, 9, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_y_solve, 10, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_y_solve, 11, sizeof(int), &nz2);
  ecode |= clSetKernelArg(k_y_solve, 12, sizeof(int), &grid_points[1]);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (Y_SOLVE_DIM == 2) {
    y_solve_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / y_solve_lws[0];
    y_solve_lws[1] = nz2 < temp ? nz2 : temp;
    y_solve_gws[0] = clu_RoundWorkSize((size_t)nx2, y_solve_lws[0]);
    y_solve_gws[1] = clu_RoundWorkSize((size_t)nz2, y_solve_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    y_solve_lws[0] = temp == 0 ? 1 : temp;
    y_solve_gws[0] = clu_RoundWorkSize((size_t)nz2, y_solve_lws[0]);
  }

  k_pinvr = clCreateKernel(p_adi, "pinvr", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for pinvr");
  ecode  = clSetKernelArg(k_pinvr, 0, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_pinvr, 1, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_pinvr, 2, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_pinvr, 3, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (PINVR_DIM == 3) {
    pinvr_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / pinvr_lws[0];
    pinvr_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / pinvr_lws[1];
    pinvr_lws[2] = nz2 < temp ? nz2 : temp;
    pinvr_gws[0] = clu_RoundWorkSize((size_t)nx2, pinvr_lws[0]);
    pinvr_gws[1] = clu_RoundWorkSize((size_t)ny2, pinvr_lws[1]);
    pinvr_gws[2] = clu_RoundWorkSize((size_t)nz2, pinvr_lws[2]);
  } else if (PINVR_DIM == 2) {
    pinvr_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / pinvr_lws[0];
    pinvr_lws[1] = nz2 < temp ? nz2 : temp;
    pinvr_gws[0] = clu_RoundWorkSize((size_t)ny2, pinvr_lws[0]);
    pinvr_gws[1] = clu_RoundWorkSize((size_t)nz2, pinvr_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    pinvr_lws[0] = temp == 0 ? 1 : temp;
    pinvr_gws[0] = clu_RoundWorkSize((size_t)nz2, pinvr_lws[0]);
  }

  k_z_solve = clCreateKernel(p_adi, "z_solve", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for z_solve");
  ecode  = clSetKernelArg(k_z_solve, 0, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_z_solve, 1, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_z_solve, 2, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_z_solve, 3, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_z_solve, 4, sizeof(cl_mem), &m_cv);
  ecode |= clSetKernelArg(k_z_solve, 5, sizeof(cl_mem), &m_rhos);
  ecode |= clSetKernelArg(k_z_solve, 6, sizeof(cl_mem), &m_lhs);
  ecode |= clSetKernelArg(k_z_solve, 7, sizeof(cl_mem), &m_lhsp);
  ecode |= clSetKernelArg(k_z_solve, 8, sizeof(cl_mem), &m_lhsm);
  ecode |= clSetKernelArg(k_z_solve, 9, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_z_solve, 10, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_z_solve, 11, sizeof(int), &nz2);
  ecode |= clSetKernelArg(k_z_solve, 12, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (Z_SOLVE_DIM == 2) {
    z_solve_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / z_solve_lws[0];
    z_solve_lws[1] = ny2 < temp ? ny2 : temp;
    z_solve_gws[0] = clu_RoundWorkSize((size_t)nx2, z_solve_lws[0]);
    z_solve_gws[1] = clu_RoundWorkSize((size_t)ny2, z_solve_lws[1]);
  } else {
    temp = ny2 / max_compute_units;
    z_solve_lws[0] = temp == 0 ? 1 : temp;
    z_solve_gws[0] = clu_RoundWorkSize((size_t)ny2, z_solve_lws[0]);
  }

  k_tzetar = clCreateKernel(p_adi, "tzetar", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for tzetar");
  ecode  = clSetKernelArg(k_tzetar, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_tzetar, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_tzetar, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_tzetar, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_tzetar, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_tzetar, 5, sizeof(cl_mem), &m_speed);
  ecode |= clSetKernelArg(k_tzetar, 6, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_tzetar, 7, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_tzetar, 8, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_tzetar, 9, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (TZETAR_DIM == 3) {
    tzetar_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / tzetar_lws[0];
    tzetar_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / tzetar_lws[1];
    tzetar_lws[2] = nz2 < temp ? nz2 : temp;
    tzetar_gws[0] = clu_RoundWorkSize((size_t)nx2, tzetar_lws[0]);
    tzetar_gws[1] = clu_RoundWorkSize((size_t)ny2, tzetar_lws[1]);
    tzetar_gws[2] = clu_RoundWorkSize((size_t)nz2, tzetar_lws[2]);
  } else if (TZETAR_DIM == 2) {
    tzetar_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / tzetar_lws[0];
    tzetar_lws[1] = nz2 < temp ? nz2 : temp;
    tzetar_gws[0] = clu_RoundWorkSize((size_t)ny2, tzetar_lws[0]);
    tzetar_gws[1] = clu_RoundWorkSize((size_t)nz2, tzetar_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    tzetar_lws[0] = temp == 0 ? 1 : temp;
    tzetar_gws[0] = clu_RoundWorkSize((size_t)nz2, tzetar_lws[0]);
  }

  k_add = clCreateKernel(p_adi, "add", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for add");
  ecode  = clSetKernelArg(k_add, 0, sizeof(cl_mem), &m_u);
  ecode |= clSetKernelArg(k_add, 1, sizeof(cl_mem), &m_rhs);
  ecode |= clSetKernelArg(k_add, 2, sizeof(int), &nx2);
  ecode |= clSetKernelArg(k_add, 3, sizeof(int), &ny2);
  ecode |= clSetKernelArg(k_add, 4, sizeof(int), &nz2);
  clu_CheckError(ecode, "clSetKernelArg()");
  if (ADD_DIM == 3) {
    add_lws[0] = nx2 < work_item_sizes[0] ? nx2 : work_item_sizes[0];
    temp = max_work_group_size / add_lws[0];
    add_lws[1] = ny2 < temp ? ny2 : temp;
    temp = temp / add_lws[1];
    add_lws[2] = nz2 < temp ? nz2 : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)nx2, add_lws[0]);
    add_gws[1] = clu_RoundWorkSize((size_t)ny2, add_lws[1]);
    add_gws[2] = clu_RoundWorkSize((size_t)nz2, add_lws[2]);
  } else if (ADD_DIM == 2) {
    add_lws[0] = ny2 < work_item_sizes[0] ? ny2 : work_item_sizes[0];
    temp = max_work_group_size / add_lws[0];
    add_lws[1] = nz2 < temp ? nz2 : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)ny2, add_lws[0]);
    add_gws[1] = clu_RoundWorkSize((size_t)nz2, add_lws[1]);
  } else {
    temp = nz2 / max_compute_units;
    add_lws[0] = temp == 0 ? 1 : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)nz2, add_lws[0]);
  }

  if (timeron) timer_stop(TIMER_OPENCL);
}


static void release_opencl()
{
  if (timeron) {
    timer_start(TIMER_OPENCL);
    timer_start(TIMER_RELEASE);
  }

  clReleaseKernel(k_compute_rhs1);
  clReleaseKernel(k_compute_rhs2);
  clReleaseKernel(k_compute_rhs3);
  clReleaseKernel(k_compute_rhs4);
  clReleaseKernel(k_compute_rhs5);
  clReleaseKernel(k_compute_rhs6);
  clReleaseKernel(k_txinvr);
  clReleaseKernel(k_x_solve);
  clReleaseKernel(k_ninvr);
  clReleaseKernel(k_y_solve);
  clReleaseKernel(k_pinvr);
  clReleaseKernel(k_z_solve);
  clReleaseKernel(k_tzetar);
  clReleaseKernel(k_add);

  clReleaseMemObject(m_ce);
  clReleaseMemObject(m_u);
  clReleaseMemObject(m_us);
  clReleaseMemObject(m_vs);
  clReleaseMemObject(m_ws);
  clReleaseMemObject(m_qs);
  clReleaseMemObject(m_rho_i);
  clReleaseMemObject(m_speed);
  clReleaseMemObject(m_square);
  clReleaseMemObject(m_rhs);
  clReleaseMemObject(m_forcing);
  clReleaseMemObject(m_cv);
  clReleaseMemObject(m_rhon);
  clReleaseMemObject(m_rhos);
  clReleaseMemObject(m_rhoq);
  clReleaseMemObject(m_lhs);
  clReleaseMemObject(m_lhsp);
  clReleaseMemObject(m_lhsm);

  clReleaseProgram(p_exact_rhs);
  clReleaseProgram(p_initialize);
  clReleaseProgram(p_adi);
  clReleaseProgram(p_error);
  clReleaseCommandQueue(cmd_queue);
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
