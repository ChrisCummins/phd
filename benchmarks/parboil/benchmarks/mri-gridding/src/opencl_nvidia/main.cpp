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
#include <CL/cl.h>
#include "parboil.h"

#include "UDTypes.h"
#include "OpenCL_interface.h"
#include "OpenCL_common.h"
#include "CPU_kernels.h"

#define PI 3.14159265

char *oclOverhead = "OpenCL Overhead";

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

  cl_int ciErrNum;
  cl_platform_id clPlatform;
  cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
  cl_device_id clDevice;

  int deviceFound = getOpenCLDevice(&clPlatform, &clDevice, &deviceType, 0);
  if (deviceFound < 0) {
    fprintf(stderr, "No suitable device was found\n");
    exit(1);
  }
  cl_ulong mem_size;
  clGetDeviceInfo(clDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);

  printf("  Number of samples = %d\n", p->numSamples);
  printf("  Total amount of GPU memory: %llu bytes\n", (unsigned long long) mem_size);
  if (p->numSamples > 10000000 && mem_size/1024/1024 < 3000) {
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
  
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

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
//  cl_mem samplesPin; 
  float* LUT; //use look-up table for faster execution on CPU (intermediate data)
  unsigned int sizeLUT; //set in the function calculateLUT (intermediate data)

  cmplx* gridData; //Output Data
  float* sampleDensity; //Output Data
//  cl_mem gridDataPin;
//  cl_mem sampleDensityPin;

  cmplx* gridData_gold; //Gold Output Data
  float* sampleDensity_gold; //Gold Output Data
  
  cl_int ciErrNum;
  cl_platform_id clPlatform;
  cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
  cl_device_id clDevice;
  cl_context clContext;

  int deviceFound = getOpenCLDevice(&clPlatform, &clDevice, &deviceType, 0);

  size_t max_alloc_size = 0;
  (void) clGetDeviceInfo(clDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_alloc_size, 0);
  size_t global_mem_size = 0;
  (void) clGetDeviceInfo(clDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &global_mem_size, 0);

  size_t samples_size = params.numSamples*sizeof(ReconstructionSample);
  int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];
  size_t output_size = gridNumElems*sizeof(cmplx);

  if ( (deviceFound < 0) ||
       ((samples_size+output_size) > global_mem_size) ||
       (samples_size > max_alloc_size) || 
       (output_size > max_alloc_size ) ) {
    fprintf(stderr, "No suitable device was found\n");
    if(deviceFound >= 0) {
      fprintf(stderr, "Memory requirements for this dataset exceed device capabilities\n");
    }
    exit(1);
  }
  
  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform, 0};
  clContext = clCreateContextFromType(cps, deviceType, NULL, NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  cl_uint workItemDimensions;
  OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &workItemDimensions, NULL) );
  
  size_t workItemSizes[workItemDimensions];
  OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, workItemDimensions*sizeof(size_t), workItemSizes, NULL) );
  
  pb_SetOpenCL(&clContext, &clCommandQueue);
    
    /*
  samplesPin = clCreateBuffer(clContext, CL_MEM_ALLOC_HOST_PTR, 
      params.numSamples*sizeof(ReconstructionSample),
      NULL, &ciErrNum);
*/
  samples = (ReconstructionSample *) malloc ( params.numSamples*sizeof(ReconstructionSample) );
  
  /*(ReconstructionSample *) clEnqueueMapBuffer(clCommandQueue, samplesPin, CL_TRUE, CL_MAP_WRITE, 0, params.numSamples*sizeof(ReconstructionSample), 0, NULL, NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
*/
  if (samples == NULL){
    printf("ERROR: Unable to allocate and map memory for input data\n");
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

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  gridData_gold = (cmplx*) calloc (gridNumElems, sizeof(cmplx));
  sampleDensity_gold = (float*) calloc (gridNumElems, sizeof(float));
  if (sampleDensity_gold == NULL || gridData_gold == NULL){
    printf("ERROR: Unable to allocate memory for output data\n");
    exit(1);
  }

  printf("Running gold version\n");

  gridding_Gold(n, params, samples, LUT, sizeLUT, gridData_gold, sampleDensity_gold);

  printf("Running OpenCL version\n");

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

/*
  OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, samplesPin, CL_TRUE, 
                          0, // Offset in bytes
                          n*sizeof(ReconstructionSample), // Size of data to write
                          samples, // Host Source
  
                          0, NULL, NULL) );*/
 // OCL_ERRCK_RETVAL ( clFinish(clCommandQueue) );
 
 /*
  gridDataPin = clCreateBuffer(clContext, CL_MEM_ALLOC_HOST_PTR, 
      gridNumElems*sizeof(cmplx), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  */
  gridData = (cmplx *) malloc ( gridNumElems*sizeof(cmplx) );
  if (gridData == NULL) { fprintf(stderr, "Could not allocate memory on host! (%s: %d)\n", __FILE__, __LINE__); exit(1); }
  
  /*(cmplx *) clEnqueueMapBuffer(clCommandQueue, gridDataPin, CL_TRUE, CL_MAP_READ, 0, gridNumElems*sizeof(cmplx), 0, NULL, NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  */
  
  /*
  sampleDensityPin = clCreateBuffer(clContext, CL_MEM_ALLOC_HOST_PTR, 
      gridNumElems*sizeof(float), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  */
  
  sampleDensity = (float *) malloc ( gridNumElems*sizeof(float) );
  if (sampleDensity == NULL) { fprintf(stderr, "Could not allocate memory on host! (%s: %d)\n", __FILE__, __LINE__); exit(1); }
  
  /*(float *) clEnqueueMapBuffer(clCommandQueue, sampleDensityPin, CL_TRUE, CL_MAP_READ, 0, gridNumElems*sizeof(float), 0, NULL, NULL, &ciErrNum);
  */
  
  OCL_ERRCK_VAR(ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  if (sampleDensity == NULL || gridData == NULL){
    printf("ERROR: Unable to allocate memory for output data\n");
    exit(1);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  
  //Interface function to GPU implementation of gridding
  OpenCL_interface(&timers, n, params, samples, LUT, sizeLUT, gridData, sampleDensity, clContext, clCommandQueue, clDevice, workItemSizes);

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
  
  /*
  OCL_ERRCK_RETVAL ( clEnqueueUnmapMemObject(clCommandQueue, samplesPin, samples, 0, NULL, NULL) );
  OCL_ERRCK_RETVAL ( clEnqueueUnmapMemObject(clCommandQueue, gridDataPin, gridData, 0, NULL, NULL) );
  OCL_ERRCK_RETVAL ( clEnqueueUnmapMemObject(clCommandQueue, sampleDensityPin, sampleDensity, 0, NULL, NULL) );
  
  clReleaseMemObject(samplesPin);
  clReleaseMemObject(gridDataPin);
  clReleaseMemObject(sampleDensityPin);
  */
  
  free(samples);
  free(gridData);
  free(sampleDensity);
  
  
  free(gridData_gold);
  free(sampleDensity_gold);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(prms);

  return 0;
}

