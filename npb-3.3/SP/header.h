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
// The following include file is generated automatically by the
// "setparams" utility. It defines 
//      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
//      dt_default:    default time step for this problem size if no
//                     config file
//      niter_default: default number of iterations for this problem size
//---------------------------------------------------------------------

#include "npbparams.h"
#include "type.h"

#include "timers.h"


/* common /global/ */
extern int grid_points[3], nx2, ny2, nz2;
extern logical timeron;

/* common /constants/ */
extern double ce[5][13], dt;

#define IMAX    PROBLEM_SIZE
#define JMAX    PROBLEM_SIZE
#define KMAX    PROBLEM_SIZE
#define IMAXP   (IMAX/2*2)
#define JMAXP   (JMAX/2*2)


//-----------------------------------------------------------------------
// Timer constants
//-----------------------------------------------------------------------
#define t_total     1
#define t_rhsx      2
#define t_rhsy      3
#define t_rhsz      4
#define t_rhs       5
#define t_xsolve    6
#define t_ysolve    7
#define t_zsolve    8
#define t_rdis1     9
#define t_rdis2     10
#define t_txinvr    11
#define t_pinvr     12
#define t_ninvr     13
#define t_tzetar    14
#define t_add       15
#define t_last      15


//-----------------------------------------------------------------------
void initialize();
void exact_rhs();
void set_constants();
void adi();
void compute_rhs();
void x_solve();
void ninvr();
void y_solve();
void pinvr();
void z_solve();
void tzetar();
void add();
void txinvr();
void error_norm(double rms[5]);
void rhs_norm(double rms[5]);
void verify(int no_time_steps, char *Class, logical *verified);
//-----------------------------------------------------------------------


#include <CL/cl.h>
#include "cl_util.h"

#define TIMER_OPENCL    20
#define TIMER_BUILD     21
#define TIMER_BUFFER    22
#define TIMER_RELEASE   23

#define USE_CHECK_FINISH
//#define TIMER_DETAIL

#ifdef TIMER_DETAIL
#define DTIMER_START(id)    if (timeron) timer_start(id)
#define DTIMER_STOP(id)     if (timeron) timer_stop(id)
#else
#define DTIMER_START(id)
#define DTIMER_STOP(id)
#endif

#ifdef USE_CHECK_FINISH
#define CHECK_FINISH()      ecode = clFinish(cmd_queue); \
                            clu_CheckError(ecode, "clFinish");
#else
#define CHECK_FINISH()
#endif

//---------------------------------------------------------------------
// OPENCL Variables
//---------------------------------------------------------------------
extern cl_device_type   device_type;
extern cl_device_id     device;
extern char            *device_name;
extern cl_context       context;
extern cl_command_queue cmd_queue;
extern cl_program       p_exact_rhs;
extern cl_program       p_initialize;
extern cl_program       p_adi;
extern cl_program       p_error;
extern size_t  work_item_sizes[3];
extern size_t  max_work_group_size;
extern cl_uint max_compute_units;

extern cl_kernel k_compute_rhs1;
extern cl_kernel k_compute_rhs2;
extern cl_kernel k_compute_rhs3;
extern cl_kernel k_compute_rhs4;
extern cl_kernel k_compute_rhs5;
extern cl_kernel k_compute_rhs6;
extern cl_kernel k_txinvr;
extern cl_kernel k_x_solve;
extern cl_kernel k_ninvr;
extern cl_kernel k_y_solve;
extern cl_kernel k_pinvr;
extern cl_kernel k_z_solve;
extern cl_kernel k_tzetar;
extern cl_kernel k_add;

extern cl_mem m_ce;
extern cl_mem m_u;
extern cl_mem m_us;
extern cl_mem m_vs;
extern cl_mem m_ws;
extern cl_mem m_qs;
extern cl_mem m_rho_i;
extern cl_mem m_speed;
extern cl_mem m_square;
extern cl_mem m_rhs;
extern cl_mem m_forcing;

extern cl_mem m_cv;
extern cl_mem m_rhon;
extern cl_mem m_rhos;
extern cl_mem m_rhoq;

extern cl_mem m_lhs;
extern cl_mem m_lhsp;
extern cl_mem m_lhsm;

extern size_t compute_rhs1_lws[3], compute_rhs1_gws[3];
extern size_t compute_rhs2_lws[3], compute_rhs2_gws[3];
extern size_t compute_rhs3_lws[3], compute_rhs3_gws[3];
extern size_t compute_rhs4_lws[3], compute_rhs4_gws[3];
extern size_t compute_rhs5_lws[3], compute_rhs5_gws[3];
extern size_t compute_rhs6_lws[3], compute_rhs6_gws[3];
extern size_t x_solve_lws[3], x_solve_gws[3];
extern size_t y_solve_lws[3], y_solve_gws[3];
extern size_t z_solve_lws[3], z_solve_gws[3];
extern size_t txinvr_lws[3], txinvr_gws[3];
extern size_t ninvr_lws[3], ninvr_gws[3];
extern size_t pinvr_lws[3], pinvr_gws[3];
extern size_t tzetar_lws[3], tzetar_gws[3];
extern size_t add_lws[3], add_gws[3];

extern int EXACT_RHS1_DIM, EXACT_RHS5_DIM;
extern int INITIALIZE2_DIM;
extern int COMPUTE_RHS1_DIM, COMPUTE_RHS2_DIM, COMPUTE_RHS4_DIM;
extern int COMPUTE_RHS6_DIM;
extern int TXINVR_DIM, NINVR_DIM, PINVR_DIM, TZETAR_DIM, ADD_DIM;
extern int X_SOLVE_DIM, Y_SOLVE_DIM, Z_SOLVE_DIM;

