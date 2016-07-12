#include "macros.h"

#define NC  4
#define COARSE_GENERAL
// #define COARSE_SPEC NC

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK) {
  int indexK = get_global_id(0);
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

__kernel void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     __global float* x, __global float* y, __global float* z,
	     __global float* Qr, __global float* Qi, __global struct kValues* ck) 
{
#ifdef COARSE_GENERAL

  float sX[NC];
  float sY[NC];
  float sZ[NC];
  float sQr[NC];
  float sQi[NC];

  #pragma unroll
  for (int tx = 0; tx < NC; tx++) {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0) + tx;

    sX[tx] = x[xIndex];
    sY[tx] = y[xIndex];
    sZ[tx] = z[xIndex];
    sQr[tx] = Qr[xIndex];
    sQi[tx] = Qi[xIndex];
  }

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex ++, kGlobalIndex ++) {
    float kx = ck[kIndex].Kx;
    float ky = ck[kIndex].Ky;
    float kz = ck[kIndex].Kz;
    float pm = ck[kIndex].PhiMag;

    #pragma unroll
    for (int tx = 0; tx < NC; tx++) {
      float expArg = PIx2 *
                   (kx * sX[tx] +
                    ky * sY[tx] +
                    kz * sZ[tx]);
      sQr[tx] += pm * cos(expArg);
      sQi[tx] += pm * sin(expArg);
    }
  }

  #pragma unroll
  for (int tx = 0; tx < NC; tx++) {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0) + tx;
    Qr[xIndex] = sQr[tx];
    Qi[xIndex] = sQi[tx];
  }

#elif (COARSE_SPEC==2)

  float2 sX;
  float2 sY;
  float2 sZ;
  float2 sQr;
  float2 sQi;

  {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC*get_local_id(0);

    sX = *(((__global float2*)(x + xIndex)));
    sY = *(((__global float2*)(y + xIndex)));
    sZ = *(((__global float2*)(z + xIndex)));
    sQr = *(((__global float2*)(Qr + xIndex)));
    sQi = *(((__global float2*)(Qi + xIndex)));
  }

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 1, kGlobalIndex += 1) {
    float kx = ck[kIndex].Kx;
    float ky = ck[kIndex].Ky;
    float kz = ck[kIndex].Kz;
    float pm = ck[kIndex].PhiMag;

    // #pragma unroll
    float expArg;
    expArg = PIx2 *
                   (kx * sX.x +
                    ky * sY.x +
                    kz * sZ.x);
    sQr.x += pm * cos(expArg);
    sQi.x += pm * sin(expArg);
    expArg = PIx2 *
                   (kx * sX.y +
                    ky * sY.y +
                    kz * sZ.y);
    sQr.y += pm * cos(expArg);
    sQi.y += pm * sin(expArg);
  }

  {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0);
    *((__global float2*)(Qr + xIndex)) = sQr;
    *((__global float2*)(Qi + xIndex)) = sQi;
  }

#elif (COARSE_SPEC==4)

  float4 sX;
  float4 sY;
  float4 sZ;
  float4 sQr;
  float4 sQi;

  {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC*get_local_id(0);

    sX = *((__global float4*)(x + xIndex));
    sY = *((__global float4*)(y + xIndex));
    sZ = *((__global float4*)(z + xIndex));
    sQr = *((__global float4*)(Qr + xIndex));
    sQi = *((__global float4*)(Qi + xIndex));
  }

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 1, kGlobalIndex += 1) {
    float kx = ck[kIndex].Kx;
    float ky = ck[kIndex].Ky;
    float kz = ck[kIndex].Kz;
    float pm = ck[kIndex].PhiMag;

    // #pragma unroll
    float4 expArg;
    expArg.x = PIx2 *
                   (kx * sX.x +
                    ky * sY.x +
                    kz * sZ.x);
    sQr.x += pm * cos(expArg.x);
    sQi.x += pm * sin(expArg.x);
    expArg.y = PIx2 *
                   (kx * sX.y +
                    ky * sY.y +
                    kz * sZ.y);
    sQr.y += pm * cos(expArg.y);
    sQi.y += pm * sin(expArg.y);
    expArg.z = PIx2 *
                   (kx * sX.z +
                    ky * sY.z +
                    kz * sZ.z);
    sQr.z += pm * cos(expArg.z);
    sQi.z += pm * sin(expArg.z);
    expArg.w = PIx2 *
                   (kx * sX.w +
                    ky * sY.w +
                    kz * sZ.w);
    sQr.w += pm * cos(expArg.w);
    sQi.w += pm * sin(expArg.w);
  }

  {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0);
    *((__global float4*)(Qr + xIndex)) = sQr;
    *((__global float4*)(Qi + xIndex)) = sQi;
  }

#else

// Uncoarse

#ifdef UNROLL_2X

  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + get_local_id(0);

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  if (numK % 2) {
    float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
    sQr += ck[0].PhiMag * cos(expArg);
    sQi += ck[0].PhiMag * sin(expArg);
    kIndex++;
    kGlobalIndex++;
  }

  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 2, kGlobalIndex += 2) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr += ck[kIndex].PhiMag * cos(expArg);
    sQi += ck[kIndex].PhiMag * sin(expArg);

    int kIndex1 = kIndex + 1;
    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX +
			    ck[kIndex1].Ky * sY +
			    ck[kIndex1].Kz * sZ);
    sQr += ck[kIndex1].PhiMag * cos(expArg1);
    sQi += ck[kIndex1].PhiMag * sin(expArg1);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;

#else

  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + get_local_id(0);

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex ++, kGlobalIndex ++) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr += ck[kIndex].PhiMag * cos(expArg);
    sQi += ck[kIndex].PhiMag * sin(expArg);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;

#endif  /* UNROLL_2X */

#endif
}
