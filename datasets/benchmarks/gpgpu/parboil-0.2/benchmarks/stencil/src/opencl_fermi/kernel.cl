/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

__kernel void block2D_reg_tiling(float c0,float c1, __global float *A0,
				 __global float *Anext, int nx, int ny, int nz)
{
    	int i = get_global_id(0);
	int j = get_global_id(1);

	float bottom=A0[Index3D (nx, ny, i, j, 0)];
	float current=A0[Index3D (nx, ny, i, j, 1)];

	if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
	{
		for(int k=1;k<nz-1;k++)
		{
			float top =A0[Index3D (nx, ny, i, j, k+1)] ;

			Anext[Index3D (nx, ny, i, j, k)] = c1 *
			( top				    +
			  bottom		 	    +
		 	  A0[Index3D (nx, ny, i, j + 1, k)] +
		       	  A0[Index3D (nx, ny, i, j - 1, k)] +
			  A0[Index3D (nx, ny, i + 1, j, k)] +
			  A0[Index3D (nx, ny, i - 1, j, k)] )
			- current * c0;
			
			bottom=current;
			current=top;
		}
	}

}
