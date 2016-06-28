/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include "scanLargeArray.h"
#include "OpenCL_common.h"

#define UINT32_MAX 4294967295
#define BITS 4
#define LNB 4

#define SORT_BS 256

void sort (int numElems, unsigned int max_value, cl_mem* &dkeysPtr, cl_mem* &dvaluesPtr, cl_mem* &dkeys_oPtr, cl_mem* &dvalues_oPtr, cl_context clContext, cl_command_queue clCommandQueue, const cl_device_id clDevice, size_t *workItemSizes){
  
  size_t block[1] = { SORT_BS };
  size_t grid[1] = { ((numElems+4*SORT_BS-1)/(4*SORT_BS)) * block[0] };

  unsigned int iterations = 0;
  while(max_value > 0){
    max_value >>= BITS;
    iterations++;
  }

  cl_int ciErrNum;
  
  cl_program sort_program;
  cl_kernel splitSort;
  cl_kernel splitRearrange;
  
  cl_mem dhisto;
  cl_mem* original = dkeysPtr;

  unsigned int *zeroData;
  zeroData = (unsigned int *) calloc( (1<<BITS)*grid[0], sizeof(unsigned int) );
  if (zeroData == NULL) { fprintf(stderr, "Could not allocate host memory! (%s: %d)\n", __FILE__, __LINE__); exit(1); }

  dhisto = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, (1<<BITS)*((numElems+4*SORT_BS-1)/(4*SORT_BS))*sizeof(unsigned int), zeroData, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  
  free(zeroData);
  
  //char compileOptions[256];
  //                -cl-nv-verbose // Provides register info for NVIDIA devices
  // Set all Macros referenced by kernels
  /*  sprintf(compileOptions, "\
                -D CUTOFF2_VAL=%f -D CUTOFF_VAL=%f\
                -D GRIDSIZE_VAL1=%d -D GRIDSIZE_VAL2=%d -D GRIDSIZE_VAL3=%d\
                -D SIZE_XY_VAL=%d -D ONE_OVER_CUTOFF2_VAL=%f",
                cutoff2, cutoff,
                params.gridSize[0], params.gridSize[1], params.gridSize[2],
                size_xy, _1overCutoff2
            );*/ 
  
  size_t program_length;
  const char *source_path = "src/opencl_nvidia/sort.cl";
  char *source;

  // Dynamically allocate buffer for source
  source = oclLoadProgSource(source_path, "", &program_length);
  if(!source) {
    fprintf(stderr, "Could not load program source (%s)\n", __FILE__); exit(1);
  }
  	
  sort_program = clCreateProgramWithSource(clContext, 1, (const char **)&source, &program_length, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  	  	
  free(source);
  
  OCL_ERRCK_RETVAL ( clBuildProgram(sort_program, 1, &clDevice, NULL /*compileOptions*/, NULL, NULL) );  
  
  /*
  // Uncomment to get build log from compiler for debugging
  char *build_log;
       size_t ret_val_size;
       ciErrNum = clGetProgramBuildInfo(sort_program, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK_VAR(ciErrNum);
       build_log = (char *)malloc(ret_val_size+1);
       ciErrNum = clGetProgramBuildInfo(sort_program, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
       	OCL_ERRCK_VAR(ciErrNum);
       	

       // to be carefully, terminate with \0
       // there's no information in the reference whether the string is 0 terminated or not
       build_log[ret_val_size] = '\0';

       fprintf(stderr, "%s\n", build_log );
  */
  
  splitSort = clCreateKernel(sort_program, "splitSort", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  splitRearrange = clCreateKernel(sort_program, "splitRearrange", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);      
  
  OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 0, sizeof(int), &numElems) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 2, sizeof(cl_mem), (void *)dkeysPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 3, sizeof(cl_mem), (void *)dvaluesPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 4, sizeof(cl_mem), (void *)&dhisto) );
  
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 0, sizeof(int), &numElems) );
  
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 2, sizeof(cl_mem), (void *)dkeysPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 3, sizeof(cl_mem), (void *)dkeys_oPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 4, sizeof(cl_mem), (void *)dvaluesPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 5, sizeof(cl_mem), (void *)dvalues_oPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 6, sizeof(cl_mem), (void *)&dhisto) );

  for (int i=0; i<iterations; i++){
  
    OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 1, sizeof(int), &i) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 2, sizeof(cl_mem), (void *)dkeysPtr) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitSort, 3, sizeof(cl_mem), (void *)dvaluesPtr) );    
    OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, splitSort, 1, 0,
                            grid, block, 0, 0, 0) );
    
    scanLargeArray(((numElems+4*SORT_BS-1)/(4*SORT_BS))*(1<<BITS), dhisto, clContext, clCommandQueue, clDevice, workItemSizes);

    OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 1, sizeof(int), &i ) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 2, sizeof(cl_mem), (void *)dkeysPtr) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 3, sizeof(cl_mem), (void *)dkeys_oPtr) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 4, sizeof(cl_mem), (void *)dvaluesPtr) );
    OCL_ERRCK_RETVAL( clSetKernelArg(splitRearrange, 5, sizeof(cl_mem), (void *)dvalues_oPtr) );

    OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, splitRearrange, 1, 0,
                            grid, block, 0, 0, 0) );

    cl_mem* temp = dkeysPtr;
    dkeysPtr = dkeys_oPtr;
    dkeys_oPtr = temp;

    temp = dvaluesPtr;
    dvaluesPtr = dvalues_oPtr;
    dvalues_oPtr = temp;
  }
  
  OCL_ERRCK_RETVAL ( clReleaseKernel(splitSort) );
  OCL_ERRCK_RETVAL ( clReleaseKernel(splitRearrange) );
  
  OCL_ERRCK_RETVAL ( clReleaseMemObject(*dkeys_oPtr) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(*dvalues_oPtr) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(dhisto) );
  
  OCL_ERRCK_RETVAL ( clReleaseProgram(sort_program) );

}
