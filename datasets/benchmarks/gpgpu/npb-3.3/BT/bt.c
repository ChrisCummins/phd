#include <libcecl.h>
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

//---------------------------------------------------------------------
//       program BT
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "header.h"
#include "timers.h"
#include "print_results.h"

#include "bt_dim.h"

//---------------------------------------------------------------------
// OPENCL Variables
//---------------------------------------------------------------------
cl_device_type   device_type;
cl_device_id     device;
char            *device_name;
cl_context       context;
cl_command_queue cmd_queue;
cl_program       p_initialize;
cl_program       p_exact_rhs;
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
cl_kernel k_x_solve1;
cl_kernel k_x_solve2;
cl_kernel k_x_solve3;
cl_kernel k_x_solve;
cl_kernel k_y_solve1;
cl_kernel k_y_solve2;
cl_kernel k_y_solve3;
cl_kernel k_y_solve;
cl_kernel k_z_solve1;
cl_kernel k_z_solve2;
cl_kernel k_z_solve3;
cl_kernel k_z_solve;
cl_kernel k_add;

cl_mem m_ce;
cl_mem m_us;
cl_mem m_vs;
cl_mem m_ws;
cl_mem m_qs;
cl_mem m_rho_i;
cl_mem m_square;
cl_mem m_forcing;
cl_mem m_u;
cl_mem m_rhs;

cl_mem m_fjac;
cl_mem m_njac;
cl_mem m_lhs;

size_t compute_rhs1_lws[3], compute_rhs1_gws[3];
size_t compute_rhs2_lws[3], compute_rhs2_gws[3];
size_t compute_rhs3_lws[3], compute_rhs3_gws[3];
size_t compute_rhs4_lws[3], compute_rhs4_gws[3];
size_t compute_rhs5_lws[3], compute_rhs5_gws[3];
size_t compute_rhs6_lws[3], compute_rhs6_gws[3];
size_t x_solve1_lws[3], x_solve1_gws[3];
size_t x_solve2_lws[3], x_solve2_gws[3];
size_t x_solve3_lws[3], x_solve3_gws[3];
size_t x_solve_lws[3], x_solve_gws[3];
size_t y_solve1_lws[3], y_solve1_gws[3];
size_t y_solve2_lws[3], y_solve2_gws[3];
size_t y_solve3_lws[3], y_solve3_gws[3];
size_t y_solve_lws[3], y_solve_gws[3];
size_t z_solve1_lws[3], z_solve1_gws[3];
size_t z_solve2_lws[3], z_solve2_gws[3];
size_t z_solve3_lws[3], z_solve3_gws[3];
size_t z_solve_lws[3], z_solve_gws[3];
size_t add_lws[3], add_gws[3];

int EXACT_RHS1_DIM, EXACT_RHS5_DIM;
int INITIALIZE2_DIM;
int COMPUTE_RHS1_DIM, COMPUTE_RHS2_DIM, COMPUTE_RHS6_DIM;
int X_SOLVE_DIM, Y_SOLVE_DIM, Z_SOLVE_DIM;
int ADD_DIM;

static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//---------------------------------------------------------------------


/* common /global/ */
double elapsed_time;
int grid_points[3];
logical timeron;

/* common /constants/ */
double ce[5][13], dt;


int main(int argc, char *argv[])
{
  int i, niter, step;
  double navg, mflops, n3;

  double tmax, t, trecs[t_last+1];
  logical verified;
  char Class;
  char *t_names[t_last+1];

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  //---------------------------------------------------------------------
  // Root node reads input file (if it exists) else takes
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
    t_names[t_add] = "add";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - BT Benchmark\n\n");

  if ((fp = fopen("inputbt.data", "r")) != NULL) {
    int result;
    printf(" Reading from input file inputbt.data\n");
    result = fscanf(fp, "%d", &niter);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d\n",
        &grid_points[0], &grid_points[1], &grid_points[2]);
    fclose(fp);
  } else {
    printf(" No input file inputbt.data. Using compiled defaults\n");
    niter = NITER_DEFAULT;
    dt    = DT_DEFAULT;
    grid_points[0] = PROBLEM_SIZE;
    grid_points[1] = PROBLEM_SIZE;
    grid_points[2] = PROBLEM_SIZE;
  }

  printf(" Size: %4dx%4dx%4d\n",
      grid_points[0], grid_points[1], grid_points[2]);
  printf(" Iterations: %4d       dt: %11.7f\n", niter, dt);
  printf("\n");

  if ( (grid_points[0] > IMAX) ||
       (grid_points[1] > JMAX) ||
       (grid_points[2] > KMAX) ) {
    printf(" %d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }

  setup_opencl(argc, argv);

  set_constants();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  initialize();

  exact_rhs();

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

  n3 = 1.0*grid_points[0]*grid_points[1]*grid_points[2];
  navg = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
  if(tmax != 0.0) {
    mflops = 1.0e-6 * (double)niter *
      (3478.8 * n3 - 17655.7 * (navg*navg) + 28023.7 * navg)
      / tmax;
  } else {
    mflops = 0.0;
  }
  c_print_results("BT", Class, grid_points[0],
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
      } else if (i==t_zsolve) {
        t = trecs[t_zsolve] - trecs[t_rdis1] - trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-zsol", t, t*100./tmax);
      } else if (i==t_rdis2) {
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
  char *source_dir = "BT";

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
  /* if (device_type == CL_DEVICE_TYPE_CPU) { */
  /*   if (CLASS == 'B') default_wg_size = 128; */
  /* } else { */
    if (CLASS == 'B') default_wg_size = 32;
  /* } */
  if (max_work_group_size > default_wg_size) {
    max_work_group_size = default_wg_size;
    int i;
    for (i = 0; i < 3; i++) {
      if (work_item_sizes[i] > default_wg_size) {
        work_item_sizes[i] = default_wg_size;
      }
    }
  }
  /* if (device_type == CL_DEVICE_TYPE_CPU) { */
  /*   EXACT_RHS1_DIM = EXACT_RHS1_DIM_CPU; */
  /*   EXACT_RHS5_DIM = EXACT_RHS5_DIM_CPU; */
  /*   INITIALIZE2_DIM = INITIALIZE2_DIM_CPU; */
  /*   COMPUTE_RHS1_DIM = COMPUTE_RHS1_DIM_CPU; */
  /*   COMPUTE_RHS2_DIM = COMPUTE_RHS2_DIM_CPU; */
  /*   COMPUTE_RHS6_DIM = COMPUTE_RHS6_DIM_CPU; */
  /*   X_SOLVE_DIM = X_SOLVE_DIM_CPU; */
  /*   Y_SOLVE_DIM = Y_SOLVE_DIM_CPU; */
  /*   Z_SOLVE_DIM = Z_SOLVE_DIM_CPU; */
  /*   ADD_DIM = ADD_DIM_CPU; */
  /* } else { */
    EXACT_RHS1_DIM = EXACT_RHS1_DIM_GPU;
    EXACT_RHS5_DIM = EXACT_RHS5_DIM_GPU;
    INITIALIZE2_DIM = INITIALIZE2_DIM_GPU;
    COMPUTE_RHS1_DIM = COMPUTE_RHS1_DIM_GPU;
    COMPUTE_RHS2_DIM = COMPUTE_RHS2_DIM_GPU;
    COMPUTE_RHS6_DIM = COMPUTE_RHS6_DIM_GPU;
    X_SOLVE_DIM = X_SOLVE_DIM_GPU;
    Y_SOLVE_DIM = Y_SOLVE_DIM_GPU;
    Z_SOLVE_DIM = Z_SOLVE_DIM_GPU;
    ADD_DIM = ADD_DIM_GPU;
  /* } */
  ////////////////////////////////////////////////////////////////////////

  //-----------------------------------------------------------------------
  // 2. Create a context for the specified device
  //-----------------------------------------------------------------------
  context = CECL_CREATE_CONTEXT(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "CECL_CREATE_CONTEXT()");

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  cmd_queue = CECL_CREATE_COMMAND_QUEUE(context, device, 0, &ecode);
  clu_CheckError(ecode, "CECL_CREATE_COMMAND_QUEUE()");

  //-----------------------------------------------------------------------
  // 4. Build programs
  //-----------------------------------------------------------------------
  if (timeron) timer_start(TIMER_BUILD);
  char build_option[100];

  /* if (device_type == CL_DEVICE_TYPE_CPU) { */
  /*   sprintf(build_option, "-I. -DCLASS=%d -DUSE_CPU", CLASS); */
  /* } else { */
  int classnum = -1;
  if (CLASS == 'S')
    classnum = 0;
  else if (CLASS == 'W')
    classnum = 1;
  else if (CLASS == 'A')
    classnum = 2;
  else if (CLASS == 'B')
    classnum = 3;
  else if (CLASS == 'C')
    classnum = 4;
  else if (CLASS == 'D')
    classnum = 5;
  else if (CLASS == 'E')
    classnum = 6;
  else {
    fprintf(stderr, "fatal: unrecognised CLASS '%c'!", CLASS);
  }

  sprintf(build_option, "-I. -DCLASS=%d", classnum);
  /* } */

  // initialize()
  p_initialize = clu_MakeProgram(context, device, source_dir,
                                 "kernel_initialize.cl",
                                 build_option);

  // exact_rhs()
  p_exact_rhs = clu_MakeProgram(context, device, source_dir,
                                "kernel_exact_rhs.cl",
                                build_option);

  // error_norm() and rhs_norm()
  p_error = clu_MakeProgram(context, device, source_dir, "kernel_error.cl",
                            build_option);

  // functions called in adi()
  /* if (device_type == CL_DEVICE_TYPE_CPU) { */
  /*   p_adi = clu_MakeProgram(context, device, source_dir, "kernel_adi_cpu.cl", */
  /*                           build_option); */
  /* } else if (device_type == CL_DEVICE_TYPE_GPU) { */
    p_adi = clu_MakeProgram(context, device, source_dir, "kernel_adi_gpu.cl",
                            build_option);
  /* } */
  if (timeron) timer_stop(TIMER_BUILD);

  //-----------------------------------------------------------------------
  // 5. Create buffers
  //-----------------------------------------------------------------------
  if (timeron) timer_start(TIMER_BUFFER);
  m_ce = CECL_BUFFER(context,
                        CL_MEM_READ_ONLY,
                        sizeof(double)*5*13,
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_ce");

  m_us = CECL_BUFFER(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_us");

  m_vs = CECL_BUFFER(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_vs");

  m_ws = CECL_BUFFER(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_ws");

  m_qs = CECL_BUFFER(context,
                        CL_MEM_READ_WRITE,
                        sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_qs");

  m_rho_i = CECL_BUFFER(context,
                           CL_MEM_READ_WRITE,
                           sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                           NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_rho_i");

  m_square = CECL_BUFFER(context,
                            CL_MEM_READ_WRITE,
                            sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1),
                            NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_square");

  m_forcing = CECL_BUFFER(context,
                             CL_MEM_READ_WRITE,
                             sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                             NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_forcing");

  m_u = CECL_BUFFER(context,
                       CL_MEM_READ_WRITE,
                       sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                       NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_u");

  m_rhs = CECL_BUFFER(context,
                         CL_MEM_READ_WRITE,
                         sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5,
                         NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_rhs");

  // workspace for work-items
  size_t max_work_items, buf_size1, buf_size2;
  if (X_SOLVE_DIM == 1 && Y_SOLVE_DIM == 1 && Z_SOLVE_DIM == 1) {
    max_work_items = PROBLEM_SIZE-2;
  } else {
    max_work_items = (PROBLEM_SIZE-2) * (PROBLEM_SIZE-2);
  }
  buf_size1 = sizeof(double)*(PROBLEM_SIZE+1)*5*5 * max_work_items;
  buf_size2 = sizeof(double)*(PROBLEM_SIZE+1)*3*5*5 * max_work_items;
  m_fjac = CECL_BUFFER(context,
                         CL_MEM_READ_WRITE,
                         buf_size1,
                         NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_fjac");

  m_njac = CECL_BUFFER(context,
                         CL_MEM_READ_WRITE,
                         buf_size1,
                         NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_njac");

  m_lhs = CECL_BUFFER(context,
                        CL_MEM_READ_WRITE,
                        buf_size2,
                        NULL, &ecode);
  clu_CheckError(ecode, "CECL_BUFFER() for m_lhs");
  if (timeron) timer_stop(TIMER_BUFFER);

  //-----------------------------------------------------------------------
  // 6. Create kernels
  //-----------------------------------------------------------------------
  int d0 = grid_points[0];
  int d1 = grid_points[1];
  int d2 = grid_points[2];

  k_compute_rhs1 = CECL_KERNEL(p_adi, "compute_rhs1", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs1");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs1, 0, sizeof(cl_mem), &m_u);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 1, sizeof(cl_mem), &m_us);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 2, sizeof(cl_mem), &m_vs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 3, sizeof(cl_mem), &m_ws);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 4, sizeof(cl_mem), &m_qs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 6, sizeof(cl_mem), &m_square);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 7, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 8, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs1, 9, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
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

  k_compute_rhs2 = CECL_KERNEL(p_adi, "compute_rhs2", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs2");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs2, 0, sizeof(cl_mem), &m_forcing);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs2, 1, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs2, 2, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs2, 3, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs2, 4, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  if (COMPUTE_RHS2_DIM == 3) {
    compute_rhs2_lws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs2_lws[0];
    compute_rhs2_lws[1] = d1 < temp ? d1 : temp;
    temp = temp / compute_rhs2_lws[1];
    compute_rhs2_lws[2] = d2 < temp ? d2 : temp;

    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)d0, compute_rhs2_lws[0]);
    compute_rhs2_gws[1] = clu_RoundWorkSize((size_t)d1, compute_rhs2_lws[1]);
    compute_rhs2_gws[2] = clu_RoundWorkSize((size_t)d2, compute_rhs2_lws[2]);
  } else if (COMPUTE_RHS2_DIM == 2) {
    compute_rhs2_lws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs2_lws[0];
    compute_rhs2_lws[1] = d2 < temp ? d2 : temp;

    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)d1, compute_rhs2_lws[0]);
    compute_rhs2_gws[1] = clu_RoundWorkSize((size_t)d2, compute_rhs2_lws[1]);
  } else {
    temp = d2 / max_compute_units;
    compute_rhs2_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs2_gws[0] = clu_RoundWorkSize((size_t)d2, compute_rhs2_lws[0]);
  }

  k_compute_rhs3 = CECL_KERNEL(p_adi, "compute_rhs3", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs3");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs3, 0, sizeof(cl_mem), &m_u);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 1, sizeof(cl_mem), &m_us);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 2, sizeof(cl_mem), &m_vs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 3, sizeof(cl_mem), &m_ws);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 4, sizeof(cl_mem), &m_qs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 6, sizeof(cl_mem), &m_square);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 7, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 8, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 9, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs3, 10, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  compute_rhs3_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
  temp = max_work_group_size / compute_rhs3_lws[0];
  compute_rhs3_lws[1] = (d2-2) < temp ? (d2-2) : temp;
  compute_rhs3_gws[0] = clu_RoundWorkSize((size_t)(d1-2), compute_rhs3_lws[0]);
  compute_rhs3_gws[1] = clu_RoundWorkSize((size_t)(d2-2), compute_rhs3_lws[1]);

  k_compute_rhs4 = CECL_KERNEL(p_adi, "compute_rhs4", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs4");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs4, 0, sizeof(cl_mem), &m_u);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 1, sizeof(cl_mem), &m_us);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 2, sizeof(cl_mem), &m_vs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 3, sizeof(cl_mem), &m_ws);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 4, sizeof(cl_mem), &m_qs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 6, sizeof(cl_mem), &m_square);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 7, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 8, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 9, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs4, 10, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  compute_rhs4_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
  temp = max_work_group_size / compute_rhs4_lws[0];
  compute_rhs4_lws[1] = (d2-2) < temp ? (d2-2) : temp;
  compute_rhs4_gws[0] = clu_RoundWorkSize((size_t)(d0-2), compute_rhs4_lws[0]);
  compute_rhs4_gws[1] = clu_RoundWorkSize((size_t)(d2-2), compute_rhs4_lws[1]);

  k_compute_rhs5 = CECL_KERNEL(p_adi, "compute_rhs5", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs5");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs5, 0, sizeof(cl_mem), &m_u);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 1, sizeof(cl_mem), &m_us);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 2, sizeof(cl_mem), &m_vs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 3, sizeof(cl_mem), &m_ws);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 4, sizeof(cl_mem), &m_qs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 5, sizeof(cl_mem), &m_rho_i);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 6, sizeof(cl_mem), &m_square);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 7, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 8, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 9, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs5, 10, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  compute_rhs5_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
  temp = max_work_group_size / compute_rhs5_lws[0];
  compute_rhs5_lws[1] = (d1-2) < temp ? (d1-2) : temp;
  compute_rhs5_gws[0] = clu_RoundWorkSize((size_t)(d0-2), compute_rhs5_lws[0]);
  compute_rhs5_gws[1] = clu_RoundWorkSize((size_t)(d1-2), compute_rhs5_lws[1]);

  k_compute_rhs6 = CECL_KERNEL(p_adi, "compute_rhs6", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for compute_rhs6");
  ecode  = CECL_SET_KERNEL_ARG(k_compute_rhs6, 0, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs6, 1, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs6, 2, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_compute_rhs6, 3, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  if (COMPUTE_RHS6_DIM == 3) {
    compute_rhs6_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs6_lws[0];
    compute_rhs6_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / compute_rhs6_lws[1];
    compute_rhs6_lws[2] = (d2-2) < temp ? (d2-2) : temp;

    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)(d0-2), compute_rhs6_lws[0]);
    compute_rhs6_gws[1] = clu_RoundWorkSize((size_t)(d1-2), compute_rhs6_lws[1]);
    compute_rhs6_gws[2] = clu_RoundWorkSize((size_t)(d2-2), compute_rhs6_lws[2]);
  } else if (COMPUTE_RHS6_DIM == 2) {
    compute_rhs6_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
    temp = max_work_group_size / compute_rhs6_lws[0];
    compute_rhs6_lws[1] = (d2-2) < temp ? (d2-2) : temp;

    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)(d1-2), compute_rhs6_lws[0]);
    compute_rhs6_gws[1] = clu_RoundWorkSize((size_t)(d2-2), compute_rhs6_lws[1]);
  } else {
    temp = (d2-2) / max_compute_units;
    compute_rhs6_lws[0] = temp == 0 ? 1 : temp;
    compute_rhs6_gws[0] = clu_RoundWorkSize((size_t)(d2-2), compute_rhs6_lws[0]);
  }

  if (X_SOLVE_DIM == 3) {
    k_x_solve1 = CECL_KERNEL(p_adi, "x_solve1", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for x_solve1");
    ecode  = CECL_SET_KERNEL_ARG(k_x_solve1, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 1, sizeof(cl_mem), &m_rho_i);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 2, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 3, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 4, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 5, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 6, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 7, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve1, 8, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    x_solve1_lws[0] = d0 < work_item_sizes[0] ? d0 : work_item_sizes[0];
    temp = max_work_group_size / x_solve1_lws[0];
    x_solve1_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / x_solve1_lws[1];
    x_solve1_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    x_solve1_gws[0] = clu_RoundWorkSize((size_t)d0, x_solve1_lws[0]);
    x_solve1_gws[1] = clu_RoundWorkSize((size_t)(d1-2), x_solve1_lws[1]);
    x_solve1_gws[2] = clu_RoundWorkSize((size_t)(d2-2), x_solve1_lws[2]);

    k_x_solve2 = CECL_KERNEL(p_adi, "x_solve2", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for x_solve2");
    ecode  = CECL_SET_KERNEL_ARG(k_x_solve2, 0, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve2, 1, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve2, 2, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve2, 3, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    x_solve2_lws[0] = 2;
    temp = max_work_group_size / x_solve2_lws[0];
    x_solve2_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / x_solve2_lws[1];
    x_solve2_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    x_solve2_gws[0] = clu_RoundWorkSize((size_t)2, x_solve2_lws[0]);
    x_solve2_gws[1] = clu_RoundWorkSize((size_t)(d1-2), x_solve2_lws[1]);
    x_solve2_gws[2] = clu_RoundWorkSize((size_t)(d2-2), x_solve2_lws[2]);

    k_x_solve3 = CECL_KERNEL(p_adi, "x_solve3", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for x_solve3");
    ecode  = CECL_SET_KERNEL_ARG(k_x_solve3, 0, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve3, 1, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve3, 2, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve3, 3, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve3, 4, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve3, 5, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    x_solve3_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / x_solve3_lws[0];
    x_solve3_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / x_solve3_lws[1];
    x_solve3_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    x_solve3_gws[0] = clu_RoundWorkSize((size_t)(d0-2), x_solve3_lws[0]);
    x_solve3_gws[1] = clu_RoundWorkSize((size_t)(d1-2), x_solve3_lws[1]);
    x_solve3_gws[2] = clu_RoundWorkSize((size_t)(d2-2), x_solve3_lws[2]);

    k_x_solve = CECL_KERNEL(p_adi, "x_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for x_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_x_solve, 0, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 1, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 2, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 3, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 4, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    x_solve_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
    temp = max_work_group_size / x_solve_lws[0];
    x_solve_lws[1] = (d2-2) < temp ? (d2-2) : temp;
    x_solve_gws[0] = clu_RoundWorkSize((size_t)(d1-2), x_solve_lws[0]);
    x_solve_gws[1] = clu_RoundWorkSize((size_t)(d2-2), x_solve_lws[1]);
  } else {
    k_x_solve = CECL_KERNEL(p_adi, "x_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for x_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_x_solve, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 1, sizeof(cl_mem), &m_rho_i);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 2, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 3, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 4, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 5, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 6, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 7, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 8, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 9, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_x_solve, 10, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    if (X_SOLVE_DIM ==2) {
      x_solve_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
      temp = max_work_group_size / x_solve_lws[0];
      x_solve_lws[1] = (d2-2) < temp ? (d2-2) : temp;
      x_solve_gws[0] = clu_RoundWorkSize((size_t)(d1-2), x_solve_lws[0]);
      x_solve_gws[1] = clu_RoundWorkSize((size_t)(d2-2), x_solve_lws[1]);
    } else { //X_SOLVE_DIM == 1
      //temp = (d2-2) / max_compute_units;
      temp = 1;
      x_solve_lws[0] = temp == 0 ? 1 : temp;
      x_solve_gws[0] = clu_RoundWorkSize((size_t)(d2-2), x_solve_lws[0]);
    }
  }

  if (Y_SOLVE_DIM == 3) {
    k_y_solve1 = CECL_KERNEL(p_adi, "y_solve1", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for y_solve1");
    ecode  = CECL_SET_KERNEL_ARG(k_y_solve1, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 1, sizeof(cl_mem), &m_rho_i);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 2, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 3, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 4, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 5, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 6, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 7, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve1, 8, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    y_solve1_lws[0] = d1 < work_item_sizes[0] ? d1 : work_item_sizes[0];
    temp = max_work_group_size / y_solve1_lws[0];
    y_solve1_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / y_solve1_lws[1];
    y_solve1_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    y_solve1_gws[0] = clu_RoundWorkSize((size_t)d1, y_solve1_lws[0]);
    y_solve1_gws[1] = clu_RoundWorkSize((size_t)(d0-2), y_solve1_lws[1]);
    y_solve1_gws[2] = clu_RoundWorkSize((size_t)(d2-2), y_solve1_lws[2]);

    k_y_solve2 = CECL_KERNEL(p_adi, "y_solve2", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for y_solve2");
    ecode  = CECL_SET_KERNEL_ARG(k_y_solve2, 0, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve2, 1, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve2, 2, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve2, 3, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    y_solve2_lws[0] = 2;
    temp = max_work_group_size / y_solve2_lws[0];
    y_solve2_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / y_solve2_lws[1];
    y_solve2_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    y_solve2_gws[0] = clu_RoundWorkSize((size_t)2, y_solve2_lws[0]);
    y_solve2_gws[1] = clu_RoundWorkSize((size_t)(d0-2), y_solve2_lws[1]);
    y_solve2_gws[2] = clu_RoundWorkSize((size_t)(d2-2), y_solve2_lws[2]);

    k_y_solve3 = CECL_KERNEL(p_adi, "y_solve3", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for y_solve3");
    ecode  = CECL_SET_KERNEL_ARG(k_y_solve3, 0, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve3, 1, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve3, 2, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve3, 3, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve3, 4, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve3, 5, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    y_solve3_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
    temp = max_work_group_size / y_solve3_lws[0];
    y_solve3_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / y_solve3_lws[1];
    y_solve3_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    y_solve3_gws[0] = clu_RoundWorkSize((size_t)(d1-2), y_solve3_lws[0]);
    y_solve3_gws[1] = clu_RoundWorkSize((size_t)(d0-2), y_solve3_lws[1]);
    y_solve3_gws[2] = clu_RoundWorkSize((size_t)(d2-2), y_solve3_lws[2]);

    k_y_solve = CECL_KERNEL(p_adi, "y_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for y_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_y_solve, 0, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 1, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 2, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 3, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 4, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    y_solve_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / y_solve_lws[0];
    y_solve_lws[1] = (d2-2) < temp ? (d2-2) : temp;
    y_solve_gws[0] = clu_RoundWorkSize((size_t)(d0-2), y_solve_lws[0]);
    y_solve_gws[1] = clu_RoundWorkSize((size_t)(d2-2), y_solve_lws[1]);
  } else {
    k_y_solve = CECL_KERNEL(p_adi, "y_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for y_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_y_solve, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 1, sizeof(cl_mem), &m_rho_i);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 2, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 3, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 4, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 5, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 6, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 7, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 8, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 9, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_y_solve, 10, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    if (Y_SOLVE_DIM == 2) {
      y_solve_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
      temp = max_work_group_size / y_solve_lws[0];
      y_solve_lws[1] = (d2-2) < temp ? (d2-2) : temp;
      y_solve_gws[0] = clu_RoundWorkSize((size_t)(d0-2), y_solve_lws[0]);
      y_solve_gws[1] = clu_RoundWorkSize((size_t)(d2-2), y_solve_lws[1]);
    } else { //Y_SOLVE_DIM == 1
      //temp = (d2-2) / max_compute_units;
      temp = 1;
      y_solve_lws[0] = temp == 0 ? 1 : temp;
      y_solve_gws[0] = clu_RoundWorkSize((size_t)(d2-2), y_solve_lws[0]);
    }
  }

  if (Z_SOLVE_DIM == 3) {
    k_z_solve1 = CECL_KERNEL(p_adi, "z_solve1", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for z_solve1");
    ecode  = CECL_SET_KERNEL_ARG(k_z_solve1, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 1, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 2, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 3, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 4, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 5, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 6, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve1, 7, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    z_solve1_lws[0] = d2 < work_item_sizes[0] ? d2 : work_item_sizes[0];
    temp = max_work_group_size / z_solve1_lws[0];
    z_solve1_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / z_solve1_lws[1];
    z_solve1_lws[2] = (d1-2) < temp ? (d1-2) : temp;
    z_solve1_gws[0] = clu_RoundWorkSize((size_t)d2, z_solve1_lws[0]);
    z_solve1_gws[1] = clu_RoundWorkSize((size_t)(d0-2), z_solve1_lws[1]);
    z_solve1_gws[2] = clu_RoundWorkSize((size_t)(d1-2), z_solve1_lws[2]);

    k_z_solve2 = CECL_KERNEL(p_adi, "z_solve2", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for z_solve2");
    ecode  = CECL_SET_KERNEL_ARG(k_z_solve2, 0, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve2, 1, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve2, 2, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve2, 3, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    z_solve2_lws[0] = 2;
    temp = max_work_group_size / z_solve2_lws[0];
    z_solve2_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / z_solve2_lws[1];
    z_solve2_lws[2] = (d1-2) < temp ? (d1-2) : temp;
    z_solve2_gws[0] = clu_RoundWorkSize((size_t)2, z_solve2_lws[0]);
    z_solve2_gws[1] = clu_RoundWorkSize((size_t)(d0-2), z_solve2_lws[1]);
    z_solve2_gws[2] = clu_RoundWorkSize((size_t)(d1-2), z_solve2_lws[2]);

    k_z_solve3 = CECL_KERNEL(p_adi, "z_solve3", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for z_solve3");
    ecode  = CECL_SET_KERNEL_ARG(k_z_solve3, 0, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve3, 1, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve3, 2, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve3, 3, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve3, 4, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve3, 5, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    z_solve3_lws[0] = (d2-2) < work_item_sizes[0] ? (d2-2) : work_item_sizes[0];
    temp = max_work_group_size / z_solve3_lws[0];
    z_solve3_lws[1] = (d0-2) < temp ? (d0-2) : temp;
    temp = temp / z_solve3_lws[1];
    z_solve3_lws[2] = (d1-2) < temp ? (d1-2) : temp;
    z_solve3_gws[0] = clu_RoundWorkSize((size_t)(d2-2), z_solve3_lws[0]);
    z_solve3_gws[1] = clu_RoundWorkSize((size_t)(d0-2), z_solve3_lws[1]);
    z_solve3_gws[2] = clu_RoundWorkSize((size_t)(d1-2), z_solve3_lws[2]);

    k_z_solve = CECL_KERNEL(p_adi, "z_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for z_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_z_solve, 0, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 1, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 2, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 3, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 4, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    z_solve_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / z_solve_lws[0];
    z_solve_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    z_solve_gws[0] = clu_RoundWorkSize((size_t)(d0-2), z_solve_lws[0]);
    z_solve_gws[1] = clu_RoundWorkSize((size_t)(d1-2), z_solve_lws[1]);
  } else {
    k_z_solve = CECL_KERNEL(p_adi, "z_solve", &ecode);
    clu_CheckError(ecode, "CECL_KERNEL() for z_solve");
    ecode  = CECL_SET_KERNEL_ARG(k_z_solve, 0, sizeof(cl_mem), &m_qs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 1, sizeof(cl_mem), &m_square);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 2, sizeof(cl_mem), &m_u);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 3, sizeof(cl_mem), &m_rhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 4, sizeof(cl_mem), &m_fjac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 5, sizeof(cl_mem), &m_njac);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 6, sizeof(cl_mem), &m_lhs);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 7, sizeof(int), &grid_points[0]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 8, sizeof(int), &grid_points[1]);
    ecode |= CECL_SET_KERNEL_ARG(k_z_solve, 9, sizeof(int), &grid_points[2]);
    clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
    if (Z_SOLVE_DIM == 2) {
      z_solve_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
      temp = max_work_group_size / z_solve_lws[0];
      z_solve_lws[1] = (d1-2) < temp ? (d1-2) : temp;
      z_solve_gws[0] = clu_RoundWorkSize((size_t)(d0-2), z_solve_lws[0]);
      z_solve_gws[1] = clu_RoundWorkSize((size_t)(d1-2), z_solve_lws[1]);
    } else { //Z_SOLVE_DIM == 1
      //temp = (d1-2) / max_compute_units;
      temp = 1;
      z_solve_lws[0] = temp == 0 ? 1 : temp;
      z_solve_gws[0] = clu_RoundWorkSize((size_t)(d1-2), z_solve_lws[0]);
    }
  }

  k_add = CECL_KERNEL(p_adi, "add", &ecode);
  clu_CheckError(ecode, "CECL_KERNEL() for add");
  ecode  = CECL_SET_KERNEL_ARG(k_add, 0, sizeof(cl_mem), &m_u);
  ecode |= CECL_SET_KERNEL_ARG(k_add, 1, sizeof(cl_mem), &m_rhs);
  ecode |= CECL_SET_KERNEL_ARG(k_add, 2, sizeof(int), &grid_points[0]);
  ecode |= CECL_SET_KERNEL_ARG(k_add, 3, sizeof(int), &grid_points[1]);
  ecode |= CECL_SET_KERNEL_ARG(k_add, 4, sizeof(int), &grid_points[2]);
  clu_CheckError(ecode, "CECL_SET_KERNEL_ARG()");
  if (ADD_DIM == 3) {
    add_lws[0] = (d0-2) < work_item_sizes[0] ? (d0-2) : work_item_sizes[0];
    temp = max_work_group_size / add_lws[0];
    add_lws[1] = (d1-2) < temp ? (d1-2) : temp;
    temp = temp / add_lws[1];
    add_lws[2] = (d2-2) < temp ? (d2-2) : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)(d0-2), add_lws[0]);
    add_gws[1] = clu_RoundWorkSize((size_t)(d1-2), add_lws[1]);
    add_gws[2] = clu_RoundWorkSize((size_t)(d2-2), add_lws[2]);
  } else if (ADD_DIM == 2) {
    add_lws[0] = (d1-2) < work_item_sizes[0] ? (d1-2) : work_item_sizes[0];
    temp = max_work_group_size / add_lws[0];
    add_lws[1] = (d2-2) < temp ? (d2-2) : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)(d1-2), add_lws[0]);
    add_gws[1] = clu_RoundWorkSize((size_t)(d2-2), add_lws[1]);
  } else {
    temp = (d2-2) / max_compute_units;
    add_lws[0] = temp == 0 ? 1 : temp;
    add_gws[0] = clu_RoundWorkSize((size_t)(d2-2), add_lws[0]);
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
  if (X_SOLVE_DIM == 3) {
    clReleaseKernel(k_x_solve1);
    clReleaseKernel(k_x_solve2);
    clReleaseKernel(k_x_solve3);
  }
  clReleaseKernel(k_x_solve);
  if (Y_SOLVE_DIM == 3) {
    clReleaseKernel(k_y_solve1);
    clReleaseKernel(k_y_solve2);
    clReleaseKernel(k_y_solve3);
  }
  clReleaseKernel(k_y_solve);
  if (Z_SOLVE_DIM == 3) {
    clReleaseKernel(k_z_solve1);
    clReleaseKernel(k_z_solve2);
    clReleaseKernel(k_z_solve3);
  }
  clReleaseKernel(k_z_solve);
  clReleaseKernel(k_add);

  clReleaseMemObject(m_ce);
  clReleaseMemObject(m_us);
  clReleaseMemObject(m_vs);
  clReleaseMemObject(m_ws);
  clReleaseMemObject(m_qs);
  clReleaseMemObject(m_rho_i);
  clReleaseMemObject(m_square);
  clReleaseMemObject(m_forcing);
  clReleaseMemObject(m_u);
  clReleaseMemObject(m_rhs);
  clReleaseMemObject(m_fjac);
  clReleaseMemObject(m_njac);
  clReleaseMemObject(m_lhs);

  clReleaseProgram(p_initialize);
  clReleaseProgram(p_exact_rhs);
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
