/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"


__global__ void block2D_reg_tiling(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{

	int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    
	float bottom=A0[Index3D (nx, ny, i, j, 0)] ;
	float current=A0[Index3D (nx, ny, i, j, 1)] ;
	if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
	{
		for(int k=1;k<nz-1;k++)
		{
			float top =A0[Index3D (nx, ny, i, j, k+1)] ;
			
			Anext[Index3D (nx, ny, i, j, k)] = 
			(top +
			bottom +
			A0[Index3D (nx, ny, i, j + 1, k)] +
			A0[Index3D (nx, ny, i, j - 1, k)] +
			A0[Index3D (nx, ny, i + 1, j, k)] +
			A0[Index3D (nx, ny, i - 1, j, k)])*c1
			- current*c0;
			bottom=current;
			current=top;
		}
	}

}


