//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB IS code. This OpenCL    //
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

#include "npbparams.h"
#include "is.h"
#include <stdlib.h>
#include <stdio.h>

#include "print_results.h"
#include "timers.h"

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
  T_KERNEL_CREATE_SEQ,
  T_KERNEL_RANK0,
  T_KERNEL_RANK1,
  T_KERNEL_RANK2,
  T_KERNEL_RANK3,
  T_KERNEL_RANK4,
  T_KERNEL_FV0,
  T_KERNEL_FV1,
  T_KERNEL_FV2,
  T_END
};

#define DTIMER_START(id)    if (timer_on) timer_start(id)
#define DTIMER_STOP(id)     if (timer_on) timer_stop(id)
#define CHECK_FINISH()      ecode = clFinish(cmd_queue); \
                            clu_CheckError(ecode, "clFinish");
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
static size_t  work_item_sizes[3];
static size_t  max_work_group_size;

// kernels
static cl_kernel k_rank0, k_rank1, k_rank2, k_rank3, k_rank4;
static cl_kernel k_rank3_0, k_rank3_1, k_rank3_2;

// memory objects
static cl_mem m_key_array, m_key_buff1, m_key_buff2;
static cl_mem m_index_array, m_rank_array, m_partial_vals;
static cl_mem m_passed_verification;
static cl_mem m_key_scan, m_sum;
static cl_mem m_bucket_ptrs, m_bucket_size;

static size_t CREATE_SEQ_GROUP_SIZE, CREATE_SEQ_GLOBAL_SIZE;
static size_t RANK1_GROUP_SIZE, RANK1_GLOBAL_SIZE;
static size_t RANK2_GROUP_SIZE, RANK2_GLOBAL_SIZE;
static size_t RANK_GROUP_SIZE, RANK_GLOBAL_SIZE;
static size_t FV2_GROUP_SIZE, FV2_GLOBAL_SIZE;

static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
//--------------------------------------------------------------------


/*****************************************************************/
/* For serial IS, buckets are not really req'd to solve NPB1 IS  */
/* spec, but their use on some machines improves performance, on */
/* other machines the use of buckets compromises performance,    */
/* probably because it is extra computation which is not req'd.  */
/* (Note: Mechanism not understood, probably cache related)      */
/* Example:  SP2-66MhzWN:  50% speedup with buckets              */
/* Example:  SGI Indy5000: 50% slowdown with buckets             */
/* Example:  SGI O2000:   400% slowdown with buckets (Wow!)      */
/*****************************************************************/
/* To disable the use of buckets, comment out the following line */
#define USE_BUCKETS


/********************/
/* Some global info */
/********************/
int      passed_verification;
int      timer_on;


/**********************/
/* Partial verif info */
/**********************/
INT_TYPE test_index_array[TEST_ARRAY_SIZE],
         test_rank_array[TEST_ARRAY_SIZE],

         S_test_index_array[TEST_ARRAY_SIZE] = 
                             {48427,17148,23627,62548,4431},
         S_test_rank_array[TEST_ARRAY_SIZE] = 
                             {0,18,346,64917,65463},

         W_test_index_array[TEST_ARRAY_SIZE] = 
                             {357773,934767,875723,898999,404505},
         W_test_rank_array[TEST_ARRAY_SIZE] = 
                             {1249,11698,1039987,1043896,1048018},

         A_test_index_array[TEST_ARRAY_SIZE] = 
                             {2112377,662041,5336171,3642833,4250760},
         A_test_rank_array[TEST_ARRAY_SIZE] = 
                             {104,17523,123928,8288932,8388264},

         B_test_index_array[TEST_ARRAY_SIZE] = 
                             {41869,812306,5102857,18232239,26860214},
         B_test_rank_array[TEST_ARRAY_SIZE] = 
                             {33422937,10244,59149,33135281,99}, 

         C_test_index_array[TEST_ARRAY_SIZE] = 
                             {44172927,72999161,74326391,129606274,21736814},
         C_test_rank_array[TEST_ARRAY_SIZE] = 
                             {61147,882988,266290,133997595,133525895},

         D_test_index_array[TEST_ARRAY_SIZE] = 
                             {1317351170,995930646,1157283250,1503301535,1453734525},
         D_test_rank_array[TEST_ARRAY_SIZE] = 
                             {1,36538729,1978098519,2145192618,2147425337};



/*****************************************************************/
/*************      C  R  E  A  T  E  _  S  E  Q      ************/
/*****************************************************************/

void create_seq( double seed, double a )
{
  cl_kernel k_cs;
  cl_int    ecode;
  size_t cs_lws[1], cs_gws[1];

  DTIMER_START(T_OPENCL_API);
  // Create a kernel
  k_cs = clCreateKernel(program, "create_seq", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for create_seq");
  DTIMER_STOP(T_OPENCL_API);

  DTIMER_START(T_KERNEL_CREATE_SEQ);
  // Set kernel arguments
  ecode  = clSetKernelArg(k_cs, 0, sizeof(cl_mem), (void*)&m_key_array);
  ecode |= clSetKernelArg(k_cs, 1, sizeof(cl_double), (void*)&seed);
  ecode |= clSetKernelArg(k_cs, 2, sizeof(cl_double), (void*)&a);
  clu_CheckError(ecode, "clSetKernelArg() for create_seq");

  // Launch the kernel
  cs_lws[0] = CREATE_SEQ_GROUP_SIZE;
  cs_gws[0] = CREATE_SEQ_GLOBAL_SIZE;
  ecode = clEnqueueNDRangeKernel(cmd_queue, k_cs, 1, NULL,
                                 cs_gws, 
                                 cs_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for create_seq");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish");
  DTIMER_STOP(T_KERNEL_CREATE_SEQ);

  DTIMER_START(T_RELEASE);
  clReleaseKernel(k_cs);
  DTIMER_STOP(T_RELEASE);
}


/*****************************************************************/
/*************    F  U  L  L  _  V  E  R  I  F  Y     ************/
/*****************************************************************/

void full_verify( void )
{
  cl_kernel k_fv1, k_fv2;
  cl_mem    m_j;
  INT_TYPE *g_j;
  INT_TYPE j = 0, i;
  size_t j_size;
  size_t fv1_lws[1], fv1_gws[1];
  size_t fv2_lws[1], fv2_gws[1];
  cl_int ecode;

  DTIMER_START(T_BUFFER_CREATE);
  // Create buffers
  j_size = sizeof(INT_TYPE) * (FV2_GLOBAL_SIZE / FV2_GROUP_SIZE);
  m_j = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       j_size,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer for m_j");
  DTIMER_STOP(T_BUFFER_CREATE);

  DTIMER_START(T_OPENCL_API);
  // Create kernels
  k_fv1 = clCreateKernel(program, "full_verify1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for full_verify1");

  k_fv2 = clCreateKernel(program, "full_verify2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for full_verify2");
  DTIMER_STOP(T_OPENCL_API);

  if (device_type == CL_DEVICE_TYPE_GPU) {
    cl_kernel k_fv0;
    size_t fv0_lws[1], fv0_gws[1];

    DTIMER_START(T_OPENCL_API);
    // Create kernels
    k_fv0 = clCreateKernel(program, "full_verify0", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for full_verify0");
    DTIMER_STOP(T_OPENCL_API);

    // Kernel execution
    DTIMER_START(T_KERNEL_FV0);
    ecode  = clSetKernelArg(k_fv0, 0, sizeof(cl_mem), (void*)&m_key_array);
    ecode |= clSetKernelArg(k_fv0, 1, sizeof(cl_mem), (void*)&m_key_buff2);
    clu_CheckError(ecode, "clSetKernelArg() for full_verify0");

    fv0_lws[0] = work_item_sizes[0];
    fv0_gws[0] = NUM_KEYS;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_fv0,
                                   1, NULL,
                                   fv0_gws, 
                                   fv0_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for full_verify0");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_FV0);

    DTIMER_START(T_KERNEL_FV1);
    ecode  = clSetKernelArg(k_fv1, 0, sizeof(cl_mem), (void*)&m_key_buff2);
    ecode |= clSetKernelArg(k_fv1, 1, sizeof(cl_mem), (void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_fv1, 2, sizeof(cl_mem), (void*)&m_key_array);
    clu_CheckError(ecode, "clSetKernelArg() for full_verify1");

    fv1_lws[0] = work_item_sizes[0];
    fv1_gws[0] = NUM_KEYS;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_fv1,
                                   1, NULL,
                                   fv1_gws, 
                                   fv1_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for full_verify1");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_FV1);

    DTIMER_START(T_KERNEL_FV2);
    ecode  = clSetKernelArg(k_fv2, 0, sizeof(cl_mem), (void*)&m_key_array);
    ecode |= clSetKernelArg(k_fv2, 1, sizeof(cl_mem), (void*)&m_j);
    ecode |= clSetKernelArg(k_fv2, 2, sizeof(INT_TYPE)*FV2_GROUP_SIZE, NULL);
    clu_CheckError(ecode, "clSetKernelArg() for full_verify2");

    fv2_lws[0] = FV2_GROUP_SIZE;
    fv2_gws[0] = FV2_GLOBAL_SIZE;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_fv2,
                                   1, NULL,
                                   fv2_gws, 
                                   fv2_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for full_verify2");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_FV2);

    g_j = (INT_TYPE *)malloc(j_size);

    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_j,
                                CL_TRUE,
                                0,
                                j_size,
                                g_j,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer() for m_j");
    DTIMER_STOP(T_BUFFER_READ);

    // reduction
    for (i = 0; i < j_size/sizeof(INT_TYPE); i++) {
      j += g_j[i];
    }

    DTIMER_START(T_RELEASE);
    clReleaseKernel(k_fv0);
    DTIMER_STOP(T_RELEASE);

  } else {

    // Kernel execution
    DTIMER_START(T_KERNEL_FV1);
    ecode  = clSetKernelArg(k_fv1, 0, sizeof(cl_mem), (void*)&m_bucket_ptrs);
    ecode |= clSetKernelArg(k_fv1, 1, sizeof(cl_mem), (void*)&m_key_buff2);
    ecode |= clSetKernelArg(k_fv1, 2, sizeof(cl_mem), (void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_fv1, 3, sizeof(cl_mem), (void*)&m_key_array);
    clu_CheckError(ecode, "clSetKernelArg() for full_verify1");

    fv1_lws[0] = RANK_GROUP_SIZE;
    fv1_gws[0] = RANK_GLOBAL_SIZE;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_fv1,
                                   1, NULL,
                                   fv1_gws, 
                                   fv1_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for full_verify1");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_FV1);

    DTIMER_START(T_KERNEL_FV2);
    ecode  = clSetKernelArg(k_fv2, 0, sizeof(cl_mem), (void*)&m_key_array);
    ecode |= clSetKernelArg(k_fv2, 1, sizeof(cl_mem), (void*)&m_j);
    ecode |= clSetKernelArg(k_fv2, 2, sizeof(INT_TYPE)*FV2_GROUP_SIZE, NULL);
    clu_CheckError(ecode, "clSetKernelArg() for full_verify2");

    fv2_lws[0] = FV2_GROUP_SIZE;
    fv2_gws[0] = FV2_GLOBAL_SIZE;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_fv2,
                                   1, NULL,
                                   fv2_gws, 
                                   fv2_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for full_verify2");
    CHECK_FINISH();
    DTIMER_STOP(T_KERNEL_FV2);

    g_j = (INT_TYPE *)malloc(j_size);

    DTIMER_START(T_BUFFER_READ);
    ecode = clEnqueueReadBuffer(cmd_queue,
                                m_j,
                                CL_TRUE,
                                0,
                                j_size,
                                g_j,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer() for m_j");
    DTIMER_STOP(T_BUFFER_READ);

    // reduction
    for (i = 0; i < j_size/sizeof(INT_TYPE); i++) {
      j += g_j[i];
    }

  }

  DTIMER_START(T_RELEASE);
  free(g_j);
  clReleaseMemObject(m_j);
  clReleaseKernel(k_fv1);
  clReleaseKernel(k_fv2);
  DTIMER_STOP(T_RELEASE);

  if (j != 0)
    printf( "Full_verify: number of keys out of sort: %ld\n", (long)j );
  else
    passed_verification++;
}



/*****************************************************************/
/*************             R  A  N  K             ****************/
/*****************************************************************/

void rank( int iteration )
{
  size_t r1_lws[1], r1_gws[1];
  size_t r2_lws[1], r2_gws[1];
  size_t r3_lws[1], r3_gws[1];
  cl_int ecode;

  DTIMER_START(T_KERNEL_RANK0);
  // rank0
  ecode = clSetKernelArg(k_rank0, 3, sizeof(cl_int), (void*)&iteration);
  clu_CheckError(ecode, "clSetKernelArg() for rank0: iteration");

  ecode = clEnqueueTask(cmd_queue, k_rank0, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueTask() for rank0");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RANK0);

  DTIMER_START(T_KERNEL_RANK1);
  // rank1
  r1_lws[0] = RANK1_GROUP_SIZE;
  r1_gws[0] = RANK1_GLOBAL_SIZE;
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rank1,
                                 1, NULL,
                                 r1_gws, 
                                 r1_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank1");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RANK1);

  DTIMER_START(T_KERNEL_RANK2);
  // rank2
  r2_lws[0] = RANK2_GROUP_SIZE;
  r2_gws[0] = RANK2_GLOBAL_SIZE;
  ecode = clEnqueueNDRangeKernel(cmd_queue,
                                 k_rank2,
                                 1, NULL,
                                 r2_gws, 
                                 r2_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank2");
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RANK2);

  DTIMER_START(T_KERNEL_RANK3);
  // rank3
  if (device_type == CL_DEVICE_TYPE_GPU) {
    r3_lws[0] = work_item_sizes[0];
    r3_gws[0] = work_item_sizes[0] * work_item_sizes[0];
    if (r3_gws[0] > MAX_KEY) r3_gws[0] = MAX_KEY;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_rank3_0,
                                   1, NULL,
                                   r3_gws, 
                                   r3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank3_0");

    r3_lws[0] = work_item_sizes[0];
    r3_gws[0] = work_item_sizes[0];
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_rank3_1,
                                   1, NULL,
                                   r3_gws, 
                                   r3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank3_1");

    r3_lws[0] = work_item_sizes[0];
    r3_gws[0] = work_item_sizes[0] * work_item_sizes[0];
    if (r3_gws[0] > MAX_KEY) r3_gws[0] = MAX_KEY;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_rank3_2,
                                   1, NULL,
                                   r3_gws, 
                                   r3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank3_2");
  } else {
    r3_lws[0] = RANK_GROUP_SIZE;
    r3_gws[0] = RANK_GLOBAL_SIZE;
    ecode = clEnqueueNDRangeKernel(cmd_queue,
                                   k_rank3,
                                   1, NULL,
                                   r3_gws, 
                                   r3_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel() for rank3");
  }
  CHECK_FINISH();
  DTIMER_STOP(T_KERNEL_RANK3);

  // rank4 - partial verification
  DTIMER_START(T_KERNEL_RANK4);
  ecode = clSetKernelArg(k_rank4, 4, sizeof(cl_int), (void*)&iteration);
  clu_CheckError(ecode, "clSetKernelArg() for rank4");

  ecode = clEnqueueTask(cmd_queue, k_rank4, 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueTask() for rank4");

  ecode = clFinish(cmd_queue);
  clu_CheckError(ecode, "clFinish");
  DTIMER_STOP(T_KERNEL_RANK4);
}


/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/

int main( int argc, char **argv )
{

  int             i, iteration;

  double          timecounter;

  FILE            *fp;

  cl_int ecode;

  if (argc == 1) {
    fprintf(stderr, "Usage: %s <kernel directory>\n", argv[0]);
    exit(-1);
  }

  /*  Initialize timers  */
  timer_on = 0;            
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    fclose(fp);
    timer_on = 1;
  }
  timer_clear( 0 );
  if (timer_on) {
    timer_clear( 1 );
    timer_clear( 2 );
    timer_clear( 3 );
  }

  if (timer_on) timer_start( 3 );

  /*  Initialize the verification arrays if a valid class */
  for( i=0; i<TEST_ARRAY_SIZE; i++ )
    switch( CLASS )
    {
      case 'S':
        test_index_array[i] = S_test_index_array[i];
        test_rank_array[i]  = S_test_rank_array[i];
        break;
      case 'A':
        test_index_array[i] = A_test_index_array[i];
        test_rank_array[i]  = A_test_rank_array[i];
        break;
      case 'W':
        test_index_array[i] = W_test_index_array[i];
        test_rank_array[i]  = W_test_rank_array[i];
        break;
      case 'B':
        test_index_array[i] = B_test_index_array[i];
        test_rank_array[i]  = B_test_rank_array[i];
        break;
      case 'C':
        test_index_array[i] = C_test_index_array[i];
        test_rank_array[i]  = C_test_rank_array[i];
        break;
      case 'D':
        test_index_array[i] = D_test_index_array[i];
        test_rank_array[i]  = D_test_rank_array[i];
        break;
    };

  /* set up the OpenCL environment. */
  setup_opencl(argc, argv);

  /*  Printout initial NPB info */
  printf( "\n\n NAS Parallel Benchmarks (NPB3.3-OCL) - IS Benchmark\n\n" );
  printf( " Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS );
  printf( " Iterations:   %d\n", MAX_ITERATIONS );

  if (timer_on) timer_start( 1 );

  /*  Generate random number sequence and subsequent keys on all procs */
  create_seq( 314159265.00,                    /* Random number gen seed */
              1220703125.00 );                 /* Random number gen mult */
  if (timer_on) timer_stop( 1 );

  /*  Do one interation for free (i.e., untimed) to guarantee initialization of  
      all data and code pages and respective tables */
  rank( 1 );  

  /*  Start verification counter */
  passed_verification = 0;

  DTIMER_START(T_BUFFER_WRITE);
  ecode = clEnqueueWriteBuffer(cmd_queue,
                               m_passed_verification,
                               CL_TRUE,
                               0,
                               sizeof(cl_int),
                               &passed_verification,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer() for m_passed_verification");
  DTIMER_STOP(T_BUFFER_WRITE);

  if( CLASS != 'S' ) printf( "\n   iteration\n" );

  /*  Start timer  */             
  timer_start( 0 );


  /*  This is the main iteration */
  for( iteration=1; iteration<=MAX_ITERATIONS; iteration++ )
  {
    if( CLASS != 'S' ) printf( "        %d\n", iteration );
    rank( iteration );
  }

  DTIMER_START(T_BUFFER_READ);
  ecode = clEnqueueReadBuffer(cmd_queue,
                              m_passed_verification,
                              CL_TRUE,
                              0,
                              sizeof(cl_int),
                              &passed_verification,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer() for m_passed_verification");
  DTIMER_STOP(T_BUFFER_READ);

  /*  End of timing, obtain maximum time of all processors */
  timer_stop( 0 );
  timecounter = timer_read( 0 );


  /*  This tests that keys are in sequence: sorting of last ranked key seq
      occurs here, but is an untimed operation                             */
  if (timer_on) timer_start( 2 );
  full_verify();
  if (timer_on) timer_stop( 2 );

  if (timer_on) timer_stop( 3 );


  /*  The final printout  */
  if( passed_verification != 5*MAX_ITERATIONS + 1 )
    passed_verification = 0;
  c_print_results( "IS",
                   CLASS,
                   (int)(TOTAL_KEYS/64),
                   64,
                   0,
                   MAX_ITERATIONS,
                   timecounter,
                   ((double) (MAX_ITERATIONS*TOTAL_KEYS))
                              /timecounter/1000000.,
                   "keys ranked", 
                   passed_verification,
                   NPBVERSION,
                   COMPILETIME,
                   CC,
                   CLINK,
                   C_LIB,
                   C_INC,
                   CFLAGS,
                   CLINKFLAGS,
                   "",
                   clu_GetDeviceTypeName(device_type),
                   device_name);

  /*  Print additional timers  */
  if (timer_on) {
    double t_total, t_percent;

    t_total = timer_read( 3 );
    printf("\nAdditional timers -\n");
    printf(" Total execution: %8.3f\n", t_total);
    if (t_total == 0.0) t_total = 1.0;
    timecounter = timer_read(1);
    t_percent = timecounter/t_total * 100.;
    printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
    timecounter = timer_read(0);
    t_percent = timecounter/t_total * 100.;
    printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
    timecounter = timer_read(2);
    t_percent = timecounter/t_total * 100.;
    printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);
  }

  release_opencl();
  
  fflush(stdout);

  return 0;
  /**************************/
} /*  E N D  P R O G R A M  */
  /**************************/


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
static void setup_opencl(int argc, char *argv[])
{
  cl_int ecode;
  char *source_dir = "IS";
  if (argc > 1) source_dir = argv[1];

#ifdef TIMER_DETAIL
  if (timer_on) {
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
  if (max_work_group_size > 256) {
    max_work_group_size = 256;
    int i;
    for (i = 0; i < 3; i++) {
      if (work_item_sizes[i] > 256) {
        work_item_sizes[i] = 256;
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
  char *source_file;
  char build_option[30];
  if (device_type == CL_DEVICE_TYPE_CPU) {
    source_file = "is_cpu.cl";
    sprintf(build_option, "-DCLASS=%d -I.", CLASS);

    CREATE_SEQ_GROUP_SIZE = 64;
    CREATE_SEQ_GLOBAL_SIZE = CREATE_SEQ_GROUP_SIZE * 256;
    RANK_GROUP_SIZE = 1;
    RANK_GLOBAL_SIZE = RANK_GROUP_SIZE * 128;
    RANK1_GROUP_SIZE = 1;
    RANK1_GLOBAL_SIZE = RANK1_GROUP_SIZE * RANK_GLOBAL_SIZE;;
    RANK2_GROUP_SIZE = RANK_GROUP_SIZE;
    RANK2_GLOBAL_SIZE = RANK_GLOBAL_SIZE;;
    FV2_GROUP_SIZE = 64;
    FV2_GLOBAL_SIZE = FV2_GROUP_SIZE * 256;
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    source_file = "is_gpu.cl";
    sprintf(build_option, "-DCLASS=\'%c\' -I.", CLASS);

    CREATE_SEQ_GROUP_SIZE = 64;
    CREATE_SEQ_GLOBAL_SIZE = CREATE_SEQ_GROUP_SIZE * 256;
    RANK1_GROUP_SIZE = work_item_sizes[0];
    RANK1_GLOBAL_SIZE = MAX_KEY;
    RANK2_GROUP_SIZE = work_item_sizes[0];
    RANK2_GLOBAL_SIZE = NUM_KEYS;
    FV2_GROUP_SIZE = work_item_sizes[0];
    FV2_GLOBAL_SIZE = NUM_KEYS;
  } else {
    fprintf(stderr, "%s: not supported.", clu_GetDeviceTypeName(device_type));
    exit(EXIT_FAILURE);
  }
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);
  DTIMER_STOP(T_BUILD);

  // 5. Create buffers
  DTIMER_START(T_BUFFER_CREATE);
  m_key_array = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               sizeof(INT_TYPE) * SIZE_OF_BUFFERS,
                               NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_key_array");

  m_key_buff1 = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               sizeof(INT_TYPE) * MAX_KEY,
                               NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_key_buff1");

  m_key_buff2 = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               sizeof(INT_TYPE) * SIZE_OF_BUFFERS,
                               NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_key_buff2");

  size_t test_array_size = sizeof(INT_TYPE) * TEST_ARRAY_SIZE;
  m_index_array = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 test_array_size,
                                 test_index_array, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_index_array");

  m_rank_array = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                test_array_size,
                                test_rank_array, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rank_array");

  m_partial_vals = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY,
                                  test_array_size,
                                  NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_partial_vals");

  m_passed_verification = clCreateBuffer(context,
                                         CL_MEM_READ_WRITE,
                                         sizeof(cl_int),
                                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_passed_verification");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    m_key_scan = clCreateBuffer(context,
                                CL_MEM_READ_WRITE,
                                sizeof(INT_TYPE) * MAX_KEY,
                                NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_key_buff1_scan");

    m_sum = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           sizeof(INT_TYPE) * work_item_sizes[0],
                           NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_sum");
  } else {
    size_t bs_size = RANK_GLOBAL_SIZE * sizeof(INT_TYPE) * NUM_BUCKETS;
    m_bucket_size = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE,
                                   bs_size,
                                   NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_bucket_size");

    m_bucket_ptrs = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE,
                                   bs_size,
                                   NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_bucket_ptrs");
  }
  DTIMER_STOP(T_BUFFER_CREATE);

  // 6. Create kernels
  DTIMER_START(T_OPENCL_API);
  k_rank0 = clCreateKernel(program, "rank0", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rank0");
  ecode  = clSetKernelArg(k_rank0, 0, sizeof(cl_mem), (void*)&m_key_array);
  ecode |= clSetKernelArg(k_rank0, 1, sizeof(cl_mem), (void*)&m_partial_vals);
  ecode |= clSetKernelArg(k_rank0, 2, sizeof(cl_mem), (void*)&m_index_array);
  clu_CheckError(ecode, "clSetKernelArg() for rank0");

  if (device_type == CL_DEVICE_TYPE_GPU) {
    k_rank1 = clCreateKernel(program, "rank1", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank1");
    ecode  = clSetKernelArg(k_rank1, 0, sizeof(cl_mem), (void*)&m_key_buff1);
    clu_CheckError(ecode, "clSetKernelArg() for rank1");

    k_rank2 = clCreateKernel(program, "rank2", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank2");
    ecode  = clSetKernelArg(k_rank2, 0, sizeof(cl_mem), (void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_rank2, 1, sizeof(cl_mem), (void*)&m_key_array);
    clu_CheckError(ecode, "clSetKernelArg() for rank2");

    k_rank3_0 = clCreateKernel(program, "rank3_0", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank3_0");
    ecode  = clSetKernelArg(k_rank3_0, 0, sizeof(cl_mem),(void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_rank3_0, 1, sizeof(cl_mem),(void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_rank3_0, 2, sizeof(cl_mem),(void*)&m_sum);
    ecode |= clSetKernelArg(k_rank3_0, 3, 
                            sizeof(INT_TYPE) * work_item_sizes[0] * 2,
                            NULL);
    clu_CheckError(ecode, "clSetKernelArg() for rank3_0");

    k_rank3_1 = clCreateKernel(program, "rank3_1", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank3_1");
    ecode  = clSetKernelArg(k_rank3_1, 0, sizeof(cl_mem), (void*)&m_sum);
    ecode  = clSetKernelArg(k_rank3_1, 1, sizeof(cl_mem), (void*)&m_sum);
    ecode |= clSetKernelArg(k_rank3_1, 2, 
                            sizeof(INT_TYPE) * work_item_sizes[0] * 2,
                            NULL);
    clu_CheckError(ecode, "clSetKernelArg() for rank3_1");

    k_rank3_2 = clCreateKernel(program, "rank3_2", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank3_2");
    ecode  = clSetKernelArg(k_rank3_2, 0, sizeof(cl_mem),(void*)&m_key_buff1);
    ecode  = clSetKernelArg(k_rank3_2, 1, sizeof(cl_mem),(void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_rank3_2, 2, sizeof(cl_mem),(void*)&m_sum);
    clu_CheckError(ecode, "clSetKernelArg() for rank3_2");
  } else {
    k_rank1 = clCreateKernel(program, "rank1", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank1");
    ecode  = clSetKernelArg(k_rank1, 0, sizeof(cl_mem),(void*)&m_key_array);
    ecode |= clSetKernelArg(k_rank1, 1, sizeof(cl_mem),(void*)&m_bucket_size);
    clu_CheckError(ecode, "clSetKernelArg() for rank1");

    k_rank2 = clCreateKernel(program, "rank2", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank2");
    ecode  = clSetKernelArg(k_rank2, 0, sizeof(cl_mem),(void*)&m_key_array);
    ecode |= clSetKernelArg(k_rank2, 1, sizeof(cl_mem),(void*)&m_bucket_size);
    ecode |= clSetKernelArg(k_rank2, 2, sizeof(cl_mem),(void*)&m_bucket_ptrs);
    ecode |= clSetKernelArg(k_rank2, 3, sizeof(cl_mem),(void*)&m_key_buff2);
    clu_CheckError(ecode, "clSetKernelArg() for rank2");

    k_rank3 = clCreateKernel(program, "rank3", &ecode);
    clu_CheckError(ecode, "clCreateKernel() for rank3");
    ecode  = clSetKernelArg(k_rank3, 0, sizeof(cl_mem),(void*)&m_bucket_size);
    ecode |= clSetKernelArg(k_rank3, 1, sizeof(cl_mem),(void*)&m_bucket_ptrs);
    ecode |= clSetKernelArg(k_rank3, 2, sizeof(cl_mem),(void*)&m_key_buff1);
    ecode |= clSetKernelArg(k_rank3, 3, sizeof(cl_mem),(void*)&m_key_buff2);
    clu_CheckError(ecode, "clSetKernelArg() for rank3");
  }

  k_rank4 = clCreateKernel(program, "rank4", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for rank4");
  ecode  = clSetKernelArg(k_rank4, 0, sizeof(cl_mem), (void*)&m_partial_vals);
  ecode |= clSetKernelArg(k_rank4, 1, sizeof(cl_mem), (void*)&m_key_buff1);
  ecode |= clSetKernelArg(k_rank4, 2, sizeof(cl_mem), (void*)&m_rank_array);
  ecode |= clSetKernelArg(k_rank4, 3, sizeof(cl_mem),
                                      (void*)&m_passed_verification);
  clu_CheckError(ecode, "clSetKernelArg() for rank4");
  DTIMER_STOP(T_OPENCL_API);
}

static void release_opencl()
{
  DTIMER_START(T_RELEASE);

  clReleaseMemObject(m_key_array);
  clReleaseMemObject(m_key_buff1);
  clReleaseMemObject(m_key_buff2);
  clReleaseMemObject(m_index_array);
  clReleaseMemObject(m_rank_array);
  clReleaseMemObject(m_partial_vals);
  clReleaseMemObject(m_passed_verification);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    clReleaseMemObject(m_key_scan);
    clReleaseMemObject(m_sum);
  } else {
    clReleaseMemObject(m_bucket_ptrs);
    clReleaseMemObject(m_bucket_size);
  }

  clReleaseKernel(k_rank0);
  clReleaseKernel(k_rank1);
  clReleaseKernel(k_rank2);
  if (device_type == CL_DEVICE_TYPE_GPU) {
    clReleaseKernel(k_rank3_0);
    clReleaseKernel(k_rank3_1);
    clReleaseKernel(k_rank3_2);
  } else {
    clReleaseKernel(k_rank3);
  }
  clReleaseKernel(k_rank4);

  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);

  DTIMER_STOP(T_RELEASE);

#ifdef TIMER_DETAIL
  if (timer_on) {
    int i;
    double tt;
    double t_opencl = 0.0, t_buffer = 0.0, t_kernel = 0.0;
    unsigned cnt;

    for (i = T_OPENCL_API; i < T_END; i++)
      t_opencl += timer_read(i);

    for (i = T_BUFFER_CREATE; i <= T_BUFFER_WRITE; i++)
      t_buffer += timer_read(i);

    for (i = T_KERNEL_CREATE_SEQ; i <= T_KERNEL_FV2; i++)
      t_kernel += timer_read(i);

    printf("\nOpenCL timers -\n");
    printf("Kernel      : %9.3f (%.2f%%)\n", 
        t_kernel, t_kernel/t_opencl * 100.0);

    cnt = timer_count(T_KERNEL_CREATE_SEQ);
    tt = timer_read(T_KERNEL_CREATE_SEQ);
    printf("- create_seq: %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_RANK0);
    tt = timer_read(T_KERNEL_RANK0);
    printf("- rank0     : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_RANK1);
    tt = timer_read(T_KERNEL_RANK1);
    printf("- rank1     : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_RANK2);
    tt = timer_read(T_KERNEL_RANK2);
    printf("- rank2     : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_RANK3);
    tt = timer_read(T_KERNEL_RANK3);
    printf("- rank3     : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_RANK4);
    tt = timer_read(T_KERNEL_RANK4);
    printf("- rank4     : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_FV0);
    tt = timer_read(T_KERNEL_FV0);
    printf("- fv0       : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_FV1);
    tt = timer_read(T_KERNEL_FV1);
    printf("- fv1       : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    cnt = timer_count(T_KERNEL_FV2);
    tt = timer_read(T_KERNEL_FV2);
    printf("- fv2       : %9.3lf (%u, %.3f, %.2f%%)\n",
        tt, cnt, tt/cnt, tt/t_kernel * 100.0);

    printf("Buffer      : %9.3lf (%.2f%%)\n",
        t_buffer, t_buffer/t_opencl * 100.0);
    printf("- creation  : %9.3lf\n", timer_read(T_BUFFER_CREATE));
    printf("- read      : %9.3lf\n", timer_read(T_BUFFER_READ));
    printf("- write     : %9.3lf\n", timer_read(T_BUFFER_WRITE));

    tt = timer_read(T_OPENCL_API);
    printf("API         : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_BUILD);
    printf("BUILD       : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    tt = timer_read(T_RELEASE);
    printf("RELEASE     : %9.3lf (%.2f%%)\n", tt, tt/t_opencl * 100.0);

    printf("Total       : %9.3lf\n", t_opencl);
  }
#endif
}

