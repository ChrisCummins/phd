/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define TILE 64
#define LOG_TILE 6

__constant__ float cutoff2_c;
__constant__ float cutoff_c;
__constant__ int gridSize_c[3];
__constant__ int size_xy_c;
__constant__ float _1overCutoff2_c;

__global__ void binning_kernel (unsigned int n, ReconstructionSample* sample_g, unsigned int* idxKey_g,
                                unsigned int* idxValue_g, unsigned int* binCount_g, unsigned int binsize, unsigned int gridNumElems){
  unsigned int key;
  unsigned int sampleIdx = blockIdx.x*blockDim.x+threadIdx.x;
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int count;

  if (sampleIdx < n){
    pt = sample_g[sampleIdx];

    binIdx = (unsigned int)(pt.kZ)*size_xy_c + (unsigned int)(pt.kY)*gridSize_c[0] + (unsigned int)(pt.kX);

    count = atomicAdd(binCount_g+binIdx, 1);
    if (count < binsize){
      key = binIdx;
    } else {
      atomicSub(binCount_g+binIdx, 1);
      key = gridNumElems;
    }

    idxKey_g[sampleIdx] = key;
    idxValue_g[sampleIdx] = sampleIdx;
  }
}

__global__ void reorder_kernel(int n, unsigned int* idxValue_g, ReconstructionSample* samples_g, ReconstructionSample* sortedSample_g){
  unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int old_index;
  ReconstructionSample pt;

  if (index < n){
    old_index = idxValue_g[index];
    pt = samples_g[old_index];
    sortedSample_g[index] = pt;
  }
}

__device__ float kernel_value(float v){

  float rValue = 0;

  float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
                (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
                 0.479440257548300e-16f) + 0.435125971262668e-13f ) +
                 0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
                 0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
                 0.463076284721000e0f)   + 0.754337328948189e2f   ) +
                 0.830792541809429e4f)   + 0.571661130563785e6f   ) +
                 0.216415572361227e8f)   + 0.356644482244025e9f   ) +
                 0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  rValue = __fdividef(-num,den);

  return rValue;
}

__global__ void gridding_GPU (ReconstructionSample* sample_g, unsigned int* binStartAddr_g, float2* gridData_g, float* sampleDensity_g, float beta){
  __shared__ ReconstructionSample sharedBin[TILE];

  const int flatIdx = threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;

  // figure out starting point of the tile
  const int z0 = blockDim.z*(blockIdx.y/(gridSize_c[1]/blockDim.y));
  const int y0 = blockDim.y*(blockIdx.y%(gridSize_c[1]/blockDim.y));
  const int x0 = blockIdx.x*blockDim.x;

  const int X  = x0+threadIdx.x;
  const int Y  = y0+threadIdx.y;
  const int Z  = z0+threadIdx.z;

  const int xl = x0-ceil(cutoff_c);
  const int xL = (xl < 0) ? 0 : xl;
  const int xh = x0+blockDim.x+cutoff_c;
  const int xH = (xh >= gridSize_c[0]) ? gridSize_c[0]-1 : xh;

  const int yl = y0-ceil(cutoff_c);
  const int yL = (yl < 0) ? 0 : yl;
  const int yh = y0+blockDim.y+cutoff_c;
  const int yH = (yh >= gridSize_c[1]) ? gridSize_c[1]-1 : yh;

  const int zl = z0-ceil(cutoff_c);
  const int zL = (zl < 0) ? 0 : zl;
  const int zh = z0+blockDim.z+cutoff_c;
  const int zH = (zh >= gridSize_c[2]) ? gridSize_c[2]-1 : zh;

  const int idx = Z*size_xy_c + Y*gridSize_c[0] + X;

  float2 pt;
  pt.x = 0.0;
  pt.y = 0.0;
  float density = 0.0;

  for (int z = zL; z <= zH; z++){
    for (int y = yL; y <= yH; y++){
      const unsigned int *addr = binStartAddr_g+z*size_xy_c+ y*gridSize_c[0];
      const unsigned int start = *(addr+xL);
      const unsigned int end   = *(addr+xH+1);
      const unsigned int delta = end-start;
      for (int x = 0; x < ((delta+TILE-1)>>LOG_TILE); x++){
        int tileSize = ((delta-(x<<LOG_TILE)) > TILE) ? TILE : (delta-(x<<LOG_TILE));
        int globalIdx = flatIdx+(x<<LOG_TILE);
        __syncthreads();
        if(flatIdx < tileSize){
          sharedBin[flatIdx] = sample_g[start+globalIdx];
        }
        __syncthreads();

        for (int j=0; j< tileSize; j++){
          const float real = sharedBin[j].real;
          const float imag = sharedBin[j].imag;
          const float sdc = sharedBin[j].sdc;

          if((real != 0.0 || imag != 0.0) && sdc != 0.0){
            float v = (sharedBin[j].kX-X)*(sharedBin[j].kX-X);
            v += (sharedBin[j].kY-Y)*(sharedBin[j].kY-Y);
            v += (sharedBin[j].kZ-Z)*(sharedBin[j].kZ-Z);
            if(v<cutoff2_c){
              const float w = kernel_value(beta*sqrtf(1.0-(v*_1overCutoff2_c))) *sdc;
              pt.x += w*real;
              pt.y += w*imag;
              density += 1.0;
            }
          }
        }
      }
    }
  }

  gridData_g[idx] = pt;
  sampleDensity_g[idx] = density;
}

