/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <malloc.h>
#include <CL/cl.h>
#include "ocl.h"
#include "macros.h"
#include "computeQ.h"

void computePhiMag_GPU(int numK,cl_mem phiR_d,cl_mem phiI_d,cl_mem phiMag_d,clPrmtr* clPrm)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;
  
  size_t DimPhiMagBlock = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  size_t DimPhiMagGrid = phiMagBlocks*KERNEL_PHI_MAG_THREADS_PER_BLOCK;

  cl_int clStatus;
  clStatus = clSetKernelArg(clPrm->clKernel,0,sizeof(cl_mem),&phiR_d);
  clStatus = clSetKernelArg(clPrm->clKernel,1,sizeof(cl_mem),&phiI_d);
  clStatus = clSetKernelArg(clPrm->clKernel,2,sizeof(cl_mem),&phiMag_d);
  clStatus = clSetKernelArg(clPrm->clKernel,3,sizeof(int),&numK);
  CHECK_ERROR("clSetKernelArg")

  clStatus = clEnqueueNDRangeKernel(clPrm->clCommandQueue,clPrm->clKernel,1,NULL,&DimPhiMagGrid,&DimPhiMagBlock,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")
}

void computeQ_GPU (int numK,int numX,
		   cl_mem x_d, cl_mem y_d, cl_mem z_d,
		   struct kValues* kVals,
		   cl_mem Qr_d, cl_mem Qi_d,
		   clPrmtr* clPrm)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
    QGrids++;
  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK)
    QBlocks++;

  size_t DimQBlock = KERNEL_Q_THREADS_PER_BLOCK;
  size_t DimQGrid = QBlocks*KERNEL_Q_THREADS_PER_BLOCK;

  cl_int clStatus;
  cl_mem ck;
  ck = clCreateBuffer(clPrm->clContext,CL_MEM_READ_ONLY,KERNEL_Q_K_ELEMS_PER_GRID*sizeof(struct kValues),NULL,&clStatus);

  int QGrid;
  for (QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    struct kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    clStatus = clEnqueueWriteBuffer(clPrm->clCommandQueue,ck,CL_TRUE,0,numElems*sizeof(struct kValues),kValsTile,0,NULL,NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    
    clStatus = clSetKernelArg(clPrm->clKernel,0,sizeof(int),&numK);
    clStatus = clSetKernelArg(clPrm->clKernel,1,sizeof(int),&QGridBase);
    clStatus = clSetKernelArg(clPrm->clKernel,2,sizeof(cl_mem),&x_d);
    clStatus = clSetKernelArg(clPrm->clKernel,3,sizeof(cl_mem),&y_d);
    clStatus = clSetKernelArg(clPrm->clKernel,4,sizeof(cl_mem),&z_d);
    clStatus = clSetKernelArg(clPrm->clKernel,5,sizeof(cl_mem),&Qr_d);
    clStatus = clSetKernelArg(clPrm->clKernel,6,sizeof(cl_mem),&Qi_d);
    clStatus = clSetKernelArg(clPrm->clKernel,7,sizeof(cl_mem),&ck);
    CHECK_ERROR("clSetKernelArg")

    clStatus = clEnqueueNDRangeKernel(clPrm->clCommandQueue,clPrm->clKernel,1,NULL,&DimQGrid,&DimQBlock,0,NULL,NULL);
    CHECK_ERROR("clEnqueueNDRangeKernel")
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}

