/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "parboil.h"

#include "UDTypes.h"
#include "CUDA_interface.h"
#include "CPU_kernels.h"

#define PI 3.14159265
#define CUERR \
  do { \
    cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
      printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
      return 0; \
    } \
  } while (0)

/************************************************************ 
 * This function reads the parameters from the file provided
 * as a comman line argument.
 ************************************************************/
void setParameters(FILE* file, parameters* p){
  fscanf(file,"aquisition.numsamples=%d\n",&(p->numSamples));
  fscanf(file,"aquisition.kmax=%f %f %f\n",&(p->kMax[0]), &(p->kMax[1]), &(p->kMax[2]));
  fscanf(file,"aquisition.matrixSize=%d %d %d\n", &(p->aquisitionMatrixSize[0]), &(p->aquisitionMatrixSize[1]), &(p->aquisitionMatrixSize[2]));
  fscanf(file,"reconstruction.matrixSize=%d %d %d\n", &(p->reconstructionMatrixSize[0]), &(p->reconstructionMatrixSize[1]), &(p->reconstructionMatrixSize[2]));
  fscanf(file,"gridding.matrixSize=%d %d %d\n", &(p->gridSize[0]), &(p->gridSize[1]), &(p->gridSize[2]));
  fscanf(file,"gridding.oversampling=%f\n", &(p->oversample));
  fscanf(file,"kernel.width=%f\n", &(p->kernelWidth));
  fscanf(file,"kernel.useLUT=%d\n", &(p->useLUT));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("  Total amount of GPU memory: %llu bytes\n", (unsigned long long) deviceProp.totalGlobalMem);
  printf("  Number of samples = %d\n", p->numSamples);
  if (p->numSamples > 10000000 && deviceProp.totalGlobalMem/1024/1024 < 3000) {
    printf("  Need at least 3GB of GPU memory for large dataset\n");
    exit(1);
  }
  printf("  Grid Size = %dx%dx%d\n", p->gridSize[0], p->gridSize[1], p->gridSize[2]);
  printf("  Input Matrix Size = %dx%dx%d\n", p->aquisitionMatrixSize[0], p->aquisitionMatrixSize[1], p->aquisitionMatrixSize[2]);
  printf("  Recon Matrix Size = %dx%dx%d\n", p->reconstructionMatrixSize[0], p->reconstructionMatrixSize[1], p->reconstructionMatrixSize[2]);
  printf("  Kernel Width = %f\n", p->kernelWidth);
  printf("  KMax = %.2f %.2f %.2f\n", p->kMax[0], p->kMax[1], p->kMax[2]);
  printf("  Oversampling = %f\n", p->oversample);
  printf("  GPU Binsize = %d\n", p->binsize);
  printf("  Use LUT = %s\n", (p->useLUT)?"Yes":"No");
}

/************************************************************ 
 * This function reads the sample point data from the kspace
 * and klocation files (and sdc file if provided) into the
 * sample array.
 * Returns the number of samples read successfully.
 ************************************************************/
unsigned int readSampleData(parameters params, FILE* uksdata_f, ReconstructionSample* samples){
  unsigned int i;

  for(i=0; i<params.numSamples; i++){
    if (feof(uksdata_f)){
      break;
    }
    fread((void*) &(samples[i]), sizeof(ReconstructionSample), 1, uksdata_f);
  }

  float kScale[3];
  kScale[0] = float(params.aquisitionMatrixSize[0])/(float(params.reconstructionMatrixSize[0])*float(params.kMax[0]));
  kScale[1] = float(params.aquisitionMatrixSize[1])/(float(params.reconstructionMatrixSize[1])*float(params.kMax[1]));
  kScale[2] = float(params.aquisitionMatrixSize[2])/(float(params.reconstructionMatrixSize[2])*float(params.kMax[2]));

  int size_x = params.gridSize[0];
  int size_y = params.gridSize[1];
  int size_z = params.gridSize[2];

  float ax = (kScale[0]*(size_x-1))/2.0;
  float bx = (float)(size_x-1)/2.0;

  float ay = (kScale[1]*(size_y-1))/2.0;
  float by = (float)(size_y-1)/2.0;

  float az = (kScale[2]*(size_z-1))/2.0;
  float bz = (float)(size_z-1)/2.0;

  for(int n=0; n<i; n++){
    samples[n].kX = floor((samples[n].kX*ax)+bx);
    samples[n].kY = floor((samples[n].kY*ay)+by);
    samples[n].kZ = floor((samples[n].kZ*az)+bz);
  }

  return i;
}


int main (int argc, char* argv[]){
  struct pb_Parameters* prms;
  struct pb_TimerSet timers;

  prms = pb_ReadParameters(&argc,argv);
  pb_InitializeTimerSet(&timers);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  char uksdata[250];
  parameters params;

  FILE* uksfile_f = NULL;
  FILE* uksdata_f = NULL;

  strcpy(uksdata,prms->inpFiles[0]);
  strcat(uksdata,".data");

  uksfile_f = fopen(prms->inpFiles[0],"r");
  if (uksfile_f == NULL){
    printf("ERROR: Could not open %s\n",prms->inpFiles[0]);
    exit(1);
  }

  printf("\nReading parameters\n");

  if (argc >= 2){
    params.binsize = atoi(argv[1]);
  } else { //default binsize value;
    params.binsize = 128;
  }

  setParameters(uksfile_f, &params);

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  ReconstructionSample* samples; //Input Data
  float* LUT; //use look-up table for faster execution on CPU (intermediate data)
  unsigned int sizeLUT; //set in the function calculateLUT (intermediate data)

  cmplx* gridData; //Output Data
  float* sampleDensity; //Output Data

  cmplx* gridData_gold; //Gold Output Data
  float* sampleDensity_gold; //Gold Output Data

  cudaMallocHost((void**)&samples, params.numSamples*sizeof(ReconstructionSample));
  CUERR;
  if (samples == NULL){
    printf("ERROR: Unable to allocate memory for input data\n");
    exit(1);
  }

  uksdata_f = fopen(uksdata,"rb");

  if(uksdata_f == NULL){
    printf("ERROR: Could not open data file\n");
    exit(1);
  }

  printf("Reading input data from files\n");

  unsigned int n = readSampleData(params, uksdata_f, samples);
  fclose(uksdata_f);

  if (params.useLUT){
    printf("Generating Look-Up Table\n");
    float beta = PI * sqrt(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-.5)*(params.oversample-.5)-.8);
    calculateLUT(beta, params.kernelWidth, &LUT, &sizeLUT);
  }

  int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  gridData_gold = (cmplx*) calloc (gridNumElems, sizeof(cmplx));
  sampleDensity_gold = (float*) calloc (gridNumElems, sizeof(float));
  if (sampleDensity_gold == NULL || gridData_gold == NULL){
    printf("ERROR: Unable to allocate memory for output data\n");
    exit(1);
  }

  printf("Running gold version\n");

  gridding_Gold(n, params, samples, LUT, sizeLUT, gridData_gold, sampleDensity_gold);

  cudaMallocHost((void**)&gridData, gridNumElems*sizeof(cmplx));
  cudaMallocHost((void**)&sampleDensity, gridNumElems*sizeof(float));
  CUERR;
  if (sampleDensity == NULL || gridData == NULL){
    printf("ERROR: Unable to allocate memory for output data\n");
    exit(1);
  }

  printf("Running CUDA version\n");

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  //Interface function to GPU implementation of gridding
  CUDA_interface(&timers, n, params, samples, LUT, sizeLUT, gridData, sampleDensity);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  int passed=1;
  for (int i=0; i<gridNumElems; i++){
    if(sampleDensity[i] != sampleDensity_gold[i]) {
      passed=0;
      break;
    }
  }
  //(passed) ? printf("Comparing GPU and Gold results... PASSED\n"):printf("Comparing GPU and Gold results... FAILED\n");

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  FILE* outfile;
  if(!(outfile=fopen(prms->outFile,"w")))
  {
        printf("Cannot open output file!\n");
  } else {
        fwrite(&passed,sizeof(int),1,outfile);
        fclose(outfile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  if (params.useLUT){
    free(LUT);
  }
  cudaFreeHost(samples);
  cudaFreeHost(gridData);
  cudaFreeHost(sampleDensity);
  free(gridData_gold);
  free(sampleDensity_gold);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(prms);

  return 0;
}
