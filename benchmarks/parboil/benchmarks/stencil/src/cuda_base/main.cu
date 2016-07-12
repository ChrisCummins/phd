
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "common.h"
#include "cuerr.h"
#include "kernels.cu"
static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
	int s=0;
	for(int i=0;i<nz;i++)
	{
		for(int j=0;j<ny;j++)
		{
			for(int k=0;k<nx;k++)
			{
                                fread(A0+s,sizeof(float),1,fp);
				s++;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	

	
	printf("CUDA accelerated 7 points stencil codes****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
  int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	//host data
	float *h_A0;
	float *h_Anext;
	//device
	float *d_A0;
	float *d_Anext;

	
	size=nx*ny*nz;
	
  h_A0=(float*)malloc(sizeof(float)*size);
  h_Anext=(float*)malloc(sizeof(float)*size);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
  read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_A0, size*sizeof(float));
	cudaMalloc((void **)&d_Anext, size*sizeof(float));
	cudaMemset(d_Anext,0,size*sizeof(float));

	//memory copy
	cudaMemcpy(d_A0, h_A0, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Anext, d_A0, size*sizeof(float), cudaMemcpyDeviceToDevice);

	
	cudaThreadSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	//only use 1D thread block
	dim3 block (nx-1, 1, 1);
	dim3 grid (ny-2, nz-2,1);

	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
	for(int t=0;t<iteration;t++)
	{
		naive_kernel<<<grid, block>>>(c0,c1, d_A0, d_Anext, nx, ny,  nz);
    float *d_temp = d_A0;
    d_A0 = d_Anext;
    d_Anext = d_temp;

	}
  CUERR // check and clear any existing errorsi
  float *d_temp = d_A0;
  d_A0 = d_Anext;
  d_Anext = d_temp;
	
	cudaThreadSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	cudaMemcpy(h_Anext, d_Anext,size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaFree(d_A0);
  cudaFree(d_Anext);
 
	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
	free (h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
