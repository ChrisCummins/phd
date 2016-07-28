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

#include "cg.h"
#include "cg_makea.h"

__kernel void init_mem_0(__global int *mem, int n)
{
  int i = get_global_id(0);
  if (i >= n) return;
  mem[i] = 0;
}

__kernel void init_mem_1(__global double *mem, int n)
{
  int i = get_global_id(0);
  if (i >= n) return;
  mem[i] = 0.0;
}


//////////////////////////////////////////////////////////////////////////
// Kernels for makea()
//////////////////////////////////////////////////////////////////////////
__kernel void makea_0(__global int *arow,
                      __global int *g_acol,
                      __global double *g_aelt,
                      __global int *g_ilow,
                      __global int *g_ihigh,
                      int n,
                      int nn1,
                      double tran,
                      double amult)
{
  __global int (*acol)[NONZER+1];
  __global double (*aelt)[NONZER+1];

  int iouter, ivelt, nzv;
  int ivc[NONZER+1];
  double vc[NONZER+1];
  double temp_tran = tran;

  int myid, num_threads, ilow, ihigh;
  int work;

  acol = (__global int (*)[NONZER+1])g_acol;
  aelt = (__global double (*)[NONZER+1])g_aelt;

  //---------------------------------------------------------------------
  // Generate nonzero positions and save for the use in sparse.
  //---------------------------------------------------------------------
  num_threads = get_global_size(0);
  myid = get_global_id(0);

  work  = (n + num_threads - 1)/num_threads;
  ilow  = work * myid;
  ihigh = ilow + work;
  if (ihigh > n) ihigh = n;

  g_ilow[myid] = ilow;
  g_ihigh[myid] = ihigh;

  for (iouter = 0; iouter < ihigh; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc, &temp_tran, amult);
    if (iouter >= ilow) {
      vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
      arow[iouter] = nzv;
      for (ivelt = 0; ivelt < nzv; ivelt++) {
        acol[iouter][ivelt] = ivc[ivelt] - 1;
        aelt[iouter][ivelt] = vc[ivelt];
      }
    }
  }
}


__kernel void makea_1(__global int *rowstr,
                      __global int *arow,
                      __global int *g_acol,
                      __global int *last_n,
                      __global int *g_ilow,
                      __global int *g_ihigh,
                      int n)
{
  __global int (*acol)[NONZER+1] = (__global int (*)[NONZER+1])g_acol;

  //---------------------------------------------------
  // generate a sparse matrix from a list of
  // [col, row, element] tri
  //---------------------------------------------------
  int i, j, j1, j2, nza;

  int myid = get_global_id(0);
  int num_threads = get_global_size(0);
  int ilow = g_ilow[myid];
  if (ilow >= n) return;
  int ihigh = g_ihigh[myid];

  //---------------------------------------------------------------------
  // how many rows of result
  //---------------------------------------------------------------------
  j1 = ilow + 1;
  j2 = ihigh + 1;

  //---------------------------------------------------------------------
  // ...count the number of triples in each row
  //---------------------------------------------------------------------
  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];
      if (j >= ilow && j < ihigh) {
        j = j + 1;
        rowstr[j] = rowstr[j] + arow[i];
      }
    }
  }

  if (myid == 0) {
    rowstr[0] = 0;
    j1 = 0;
  }
  for (j = j1+1; j < j2; j++) {
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }
  if (myid < num_threads) last_n[myid] = rowstr[j2-1];
}


__kernel void makea_2(__global int *rowstr,
                      __global int *last_n,
                      __global int *ilow,
                      __global int *ihigh)
{
  int myid = get_global_id(0);
  int num_threads = get_global_size(0);

  int nzrow = 0;
  if (myid < num_threads) {
    for (int i = 0; i < myid; i++) {
      nzrow = nzrow + last_n[i];
    }
  }
  if (nzrow > 0) {
    int j1 = (myid == 0) ? 0 : (ilow[myid] + 1);
    int j2 = ihigh[myid] + 1;

    for (int j = j1; j < j2; j++) {
      rowstr[j] = rowstr[j] + nzrow;
    }
  }
}


__kernel void makea_3(__global double *v,
                      __global int *iv,
                      __global int *rowstr,
                      __global int *arow,
                      __global int *g_acol,
                      __global double *g_aelt,
                      __global int *g_ilow,
                      __global int *g_ihigh,
                      int n,
                      int nz)
{
  __global int (*acol)[NONZER+1];
  __global double (*aelt)[NONZER+1];
  __global int *nzloc;

  const double rcond = RCOND;
  const double shift = SHIFT;

  int i, j, k, kk, nza, nzrow, jcol;
  double size, scale, ratio, va;
  logical cont40;

  int myid = get_global_id(0);
  int num_threads = get_global_size(0);
  int ilow = g_ilow[myid];
  if (ilow >= n) return;
  int ihigh = g_ihigh[myid];

  acol = (__global int (*)[NONZER+1])g_acol;
  aelt = (__global double (*)[NONZER+1])g_aelt;
  nzloc = &iv[nz];

  //---------------------------------------------------------------------
  // ... preload data pages
  //---------------------------------------------------------------------
  for (j = ilow; j < ihigh; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      v[k] = 0.0;
      iv[k] = -1;
    }
    nzloc[j] = 0;
  }

  //---------------------------------------------------------------------
  // ... generate actual values by summing duplicates
  //---------------------------------------------------------------------
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      if (j < ilow || j >= ihigh) continue;

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        //--------------------------------------------------------------------
        // ... add the identity * rcond to the generated matrix to bound
        //     the smallest eigenvalue from below by rcond
        //--------------------------------------------------------------------
        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        cont40 = false;
        for (k = rowstr[j]; k < rowstr[j+1]; k++) {
          if (iv[k] > jcol) {
            //----------------------------------------------------------------
            // ... insert colidx here orderly
            //----------------------------------------------------------------
            for (kk = rowstr[j+1]-2; kk >= k; kk--) {
              if (iv[kk] > -1) {
                v[kk+1]  = v[kk];
                iv[kk+1] = iv[kk];
              }
            }
            iv[k] = jcol;
            v[k]  = 0.0;
            cont40 = true;
            break;
          } else if (iv[k] == -1) {
            iv[k] = jcol;
            cont40 = true;
            break;
          } else if (iv[k] == jcol) {
            //--------------------------------------------------------------
            // ... mark the duplicated entry
            //--------------------------------------------------------------
            nzloc[j] = nzloc[j] + 1;
            cont40 = true;
            break;
          }
        }
        if (cont40 == false) {
#ifdef cl_amd_printf
          printf("internal error in sparse: i=%d\n", i);
          return;
#endif
        }
        v[k] = v[k] + va;
      }
    }
    size = size * ratio;
  }
}


__kernel void makea_4(__global int *iv,
                      __global int *last_n,
                      __global int *g_ilow,
                      __global int *g_ihigh,
                      int n,
                      int nz)
{

  int myid = get_global_id(0);
  int num_threads = get_global_size(0);
  int ilow = g_ilow[myid];
  if (ilow >= n) return;
  int ihigh = g_ihigh[myid];

  __global int *nzloc = &iv[nz];

  //---------------------------------------------------------------------
  // ... remove empty entries and generate final results
  //---------------------------------------------------------------------
  for (int j = ilow+1; j < ihigh; j++) {
    nzloc[j] = nzloc[j] + nzloc[j-1];
  }
  if (myid < num_threads) last_n[myid] = nzloc[ihigh-1];
}


__kernel void makea_5(__global int *iv,
                      __global int *last_n,
                      __global int *g_ilow,
                      __global int *g_ihigh,
                      int n,
                      int nz)
{
  int myid = get_global_id(0);
  int num_threads = get_global_size(0);
  int ilow = g_ilow[myid];
  if (ilow >= n) return;
  int ihigh = g_ihigh[myid];

  int nzrow = 0;
  if (myid < num_threads) {
    for (int i = 0; i < myid; i++) {
      nzrow = nzrow + last_n[i];
    }
  }
  if (nzrow > 0) {
    __global int *nzloc = &iv[nz];
    for (int j = ilow; j < ihigh; j++) {
      nzloc[j] = nzloc[j] + nzrow;
    }
  }
}


__kernel void makea_6(__global double *a,
                      __global double *v,
                      __global int *rowstr,
                      __global int *colidx,
                      __global int *iv,
                      int nz,
                      int nrows)
{
  int j1, j2, nza;
  int j = get_global_id(0);
  if (j >= nrows) return;

  __global int *nzloc = &iv[nz];

  if (j > 0) {
    j1 = rowstr[j] - nzloc[j-1];
  } else {
    j1 = 0;
  }
  j2 = rowstr[j+1] - nzloc[j];
  nza = rowstr[j];
  for (int k = j1; k < j2; k++) {
    a[k] = v[nza];
    colidx[k] = iv[nza];
    nza = nza + 1;
  }
}


__kernel void makea_7(__global int *rowstr,
                      __global int *iv,
                      int nrows,
                      int nz)
{
  int j = get_global_id(0) + 1;
  if (j >= (nrows+1)) return;

  __global int *nzloc = &iv[nz];

  rowstr[j] = rowstr[j] - nzloc[j-1];
}

