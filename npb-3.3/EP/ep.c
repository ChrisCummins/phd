//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB EP code. This OpenCL    //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB 3.3-OMP" developed by NAS.                                        //
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

//--------------------------------------------------------------------
//      program EMBAR
//--------------------------------------------------------------------
//
//  M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//  numbers.  MK is the Log_2 of the size of each batch of uniform random
//  numbers.  MK can be set for convenience on a given system, since it does
//  not affect the results.
//--------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "npbparams.h"
#include "ep.h"

#include "type.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
#include "cl_util.h"

//#define TIMER_DETAIL

#ifdef TIMER_DETAIL
enum OPENCL_TIMER {
  T_OPENCL_API = 10,
  T_BUILD,
  T_RELEASE,
  T_BUFFER_CREATE,
  T_BUFFER_READ,
  T_BUFFER_WRITE,
  T_KERNEL_EMBAR,
  T_END
};

#define DTIMER_START(id)    if (timers_enabled) timer_start(id)
#define DTIMER_STOP(id)     if (timers_enabled) timer_stop(id)
#define CHECK_FINISH()      { cl_int ecode = clFinish(cmd_queue); \
                              clu_CheckError(ecode, "clFinish"); }

#else
#define DTIMER_START(id)
#define DTIMER_STOP(id)
#define CHECK_FINISH()
#endif

//--------------------------------------------------------------------
// OpenCL part
//--------------------------------------------------------------------
static cl_device_type   device_type;
static cl_device_id     device;
static char            *device_name;
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_program       program;
static cl_kernel        kernel;
static cl_int           err_code;
static cl_mem           pgq, pgsx, pgsy;

static int GROUP_SIZE;
static int gq_size;
static int gsx_size;
static int gsy_size;

void setup_opencl(int argc, char *argv[]);
void release_opencl();
//--------------------------------------------------------------------

static int    np;
static double q[NQ];
static logical timers_enabled;


int main(int argc, char *argv[]) 
{
  double Mops, t1, t2;
  double tsx, tsy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    i, nit;
  int    k_offset, j;
  logical verified;

  char   size[16];

  FILE *fp;

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = false;
  } else {
    timers_enabled = true;
    fclose(fp);
  }

  //--------------------------------------------------------------------
  //  Because the size of the problem is too large to store in a 32-bit
  //  integer for some classes, we put it into a string (for printing).
  //  Have to strip off the decimal point put in there by the floating
  //  point print statement (internal file)
  //--------------------------------------------------------------------

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = false;

  //--------------------------------------------------------------------
  //  Compute the number of "batches" of random number pairs generated 
  //  per processor. Adjust if the number of processors does not evenly 
  //  divide the total number
  //--------------------------------------------------------------------

  np = NN; 

  setup_opencl(argc, argv);

  timer_clear(0);
  timer_start(0);

  //--------------------------------------------------------------------
  //  Compute AN = A ^ (2 * NK) (mod 2^46).
  //--------------------------------------------------------------------

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc(&t1, t1);
  }

  an = t1;
  tt = S;

  //--------------------------------------------------------------------
  //  Each instance of this loop may be performed independently. We compute
  //  the k offsets separately to take into account the fact that some nodes
  //  have more numbers to generate than others
  //--------------------------------------------------------------------

  k_offset = -1;

  DTIMER_START(T_KERNEL_EMBAR);

  // Launch the kernel
  int q_size  = GROUP_SIZE * NQ * sizeof(cl_double);
  int sx_size = GROUP_SIZE * sizeof(cl_double);
  int sy_size = GROUP_SIZE * sizeof(cl_double);
  err_code  = clSetKernelArg(kernel, 0, q_size, NULL);
  err_code |= clSetKernelArg(kernel, 1, sx_size, NULL);
  err_code |= clSetKernelArg(kernel, 2, sy_size, NULL);
  err_code |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&pgq);
  err_code |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&pgsx);
  err_code |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&pgsy);
  err_code |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&k_offset);
  err_code |= clSetKernelArg(kernel, 7, sizeof(cl_double), (void*)&an);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  size_t localWorkSize[] = { GROUP_SIZE };
  size_t globalWorkSize[] = { np };
  err_code = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
                                    globalWorkSize, 
                                    localWorkSize,
                                    0, NULL, NULL);
  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_EMBAR);

  double (*gq)[NQ] = (double (*)[NQ])malloc(gq_size);
  double *gsx = (double*)malloc(gsx_size);
  double *gsy = (double*)malloc(gsy_size);

  gc  = 0.0;
  tsx = 0.0;
  tsy = 0.0;

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }

  // 9. Get the result
  DTIMER_START(T_BUFFER_READ);
  err_code = clEnqueueReadBuffer(cmd_queue, pgq, CL_FALSE, 0, gq_size, 
                                 gq, 0, NULL, NULL);
  clu_CheckError(err_code, "clEnqueueReadbuffer()");

  err_code = clEnqueueReadBuffer(cmd_queue, pgsx, CL_FALSE, 0, gsx_size, 
                                 gsx, 0, NULL, NULL);
  clu_CheckError(err_code, "clEnqueueReadbuffer()");

  err_code = clEnqueueReadBuffer(cmd_queue, pgsy, CL_TRUE, 0, gsy_size, 
                                 gsy, 0, NULL, NULL);
  clu_CheckError(err_code, "clEnqueueReadbuffer()");
  DTIMER_STOP(T_BUFFER_READ);

  for (i = 0; i < np/localWorkSize[0]; i++) {
    for (j = 0; j < NQ; j++ ){
      q[j] = q[j] + gq[i][j];
    }
    tsx = tsx + gsx[i];
    tsy = tsy + gsy[i];
  }

  for (i = 0; i < NQ; i++) {
    gc = gc + q[i];
  }

  timer_stop(0);
  tm = timer_read(0);

  nit = 0;
  verified = true;
  if (M == 24) {
    sx_verify_value = -3.247834652034740e+3;
    sy_verify_value = -6.958407078382297e+3;
  } else if (M == 25) {
    sx_verify_value = -2.863319731645753e+3;
    sy_verify_value = -6.320053679109499e+3;
  } else if (M == 28) {
    sx_verify_value = -4.295875165629892e+3;
    sy_verify_value = -1.580732573678431e+4;
  } else if (M == 30) {
    sx_verify_value =  4.033815542441498e+4;
    sy_verify_value = -2.660669192809235e+4;
  } else if (M == 32) {
    sx_verify_value =  4.764367927995374e+4;
    sy_verify_value = -8.084072988043731e+4;
  } else if (M == 36) {
    sx_verify_value =  1.982481200946593e+5;
    sy_verify_value = -1.020596636361769e+5;
  } else if (M == 40) {
    sx_verify_value = -5.319717441530e+05;
    sy_verify_value = -3.688834557731e+05;
  } else {
    verified = false;
  }

  if (verified) {
    sx_err = fabs((tsx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((tsy - sy_verify_value) / sy_verify_value);
    verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
  }

  Mops = pow(2.0, M+1) / tm / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", tm);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", gc);
  printf("Sums = %25.15lE %25.15lE\n", tsx, tsy);
  printf("Counts: \n");
  for (i = 0; i < NQ; i++) {
    printf("%3d%15.0lf\n", i, q[i]);
  }

  c_print_results("EP", CLASS, M+1, 0, 0, nit,
      tm, Mops, 
      "Random numbers generated",
      verified, NPBVERSION, COMPILETIME, 
      CS1, CS2, CS3, CS4, CS5, CS6, CS7,
      clu_GetDeviceTypeName(device_type), device_name);

  if (timers_enabled) {
    if (tm <= 0.0) tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
  }

  free(gq);
  free(gsx);
  free(gsy);
  release_opencl();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
void setup_opencl(int argc, char *argv[])
{
  cl_int err_code;
  char *source_dir = "EP";
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

  // 2. Create a context for the specified device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err_code);
  clu_CheckError(err_code, "clCreateContext()");

  // 3. Create a command queue
  cmd_queue = clCreateCommandQueue(context, device, 0, &err_code);
  clu_CheckError(err_code, "clCreateCommandQueue()");

  DTIMER_STOP(T_OPENCL_API);

  // 4. Build the program
  DTIMER_START(T_BUILD);
  char *source_file;
  char build_option[30];
  sprintf(build_option, "-DM=%d -I.", M);
  if (device_type == CL_DEVICE_TYPE_CPU) {
    source_file = "ep_cpu.cl";
    GROUP_SIZE = 16;
  } else {
    source_file = "ep_gpu.cl";
    GROUP_SIZE = 64;
  }
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);
  DTIMER_STOP(T_BUILD);

  // 5. Create buffers
  DTIMER_START(T_BUFFER_CREATE);

  gq_size  = np / GROUP_SIZE * NQ * sizeof(double);
  gsx_size = np / GROUP_SIZE * sizeof(double);
  gsy_size = np / GROUP_SIZE * sizeof(double);

  pgq = clCreateBuffer(context, CL_MEM_READ_WRITE, gq_size, NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for pgq");

  pgsx = clCreateBuffer(context, CL_MEM_READ_WRITE, gsx_size,NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for pgsx");

  pgsy = clCreateBuffer(context, CL_MEM_READ_WRITE, gsy_size,NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for pgsy");

  DTIMER_STOP(T_BUFFER_CREATE);

  // 6. Create a kernel
  DTIMER_START(T_OPENCL_API);
  kernel = clCreateKernel(program, "embar", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  DTIMER_STOP(T_OPENCL_API);
}


void release_opencl()
{
  DTIMER_START(T_RELEASE);

  clReleaseMemObject(pgq);
  clReleaseMemObject(pgsx);
  clReleaseMemObject(pgsy);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);

  DTIMER_STOP(T_RELEASE);

#ifdef TIMER_DETAIL
  if (timers_enabled) {
    int i;
    double tt;
    double t_opencl = 0.0, t_buffer = 0.0, t_kernel = 0.0;
    unsigned cnt;

    for (i = T_OPENCL_API; i < T_END; i++)
      t_opencl += timer_read(i);

    for (i = T_BUFFER_CREATE; i <= T_BUFFER_WRITE; i++)
      t_buffer += timer_read(i);

    for (i = T_KERNEL_EMBAR; i <= T_KERNEL_EMBAR; i++)
      t_kernel += timer_read(i);

    printf("\nOpenCL timers -\n");
    printf("Kernel    : %9.3f (%.2f%%)\n", 
        t_kernel, t_kernel/t_opencl * 100.0);

    cnt = timer_count(T_KERNEL_EMBAR);
    tt = timer_read(T_KERNEL_EMBAR);
    printf("- embar   : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    printf("Buffer    : %9.3lf (%.2f%%)\n",
        t_buffer, t_buffer/t_opencl * 100.0);
    printf("- creation: %9.3lf\n", timer_read(T_BUFFER_CREATE));
    printf("- read    : %9.3lf\n", timer_read(T_BUFFER_READ));
    printf("- write   : %9.3lf\n", timer_read(T_BUFFER_WRITE));

    tt = timer_read(T_OPENCL_API);
    printf("API       : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_BUILD);
    printf("BUILD     : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_RELEASE);
    printf("RELEASE   : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    printf("Total     : %9.3lf\n", t_opencl);
  }
#endif
}

