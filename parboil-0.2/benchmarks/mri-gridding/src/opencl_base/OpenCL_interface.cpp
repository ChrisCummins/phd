/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "parboil.h"

#include "UDTypes.h"
#include "scanLargeArray.h"
#include "CPU_kernels.h"

#include "sort.h"
#include "scanLargeArray.h"
#include "OpenCL_common.h"


#define BLOCKSIZE 512
#define PI 3.14159265359

/***********************************************************************
 * CUDA_interface is the main function for GPU execution. This
 * implementation uses compact binning to distribute input elements
 * into unit-cubed sized bins. The bins are then visited by GPU
 * threads, where every thread computes the value of one (or small set)
 * of output elements by computing the contributions of elements in 
 * neighboring bins to these output elements.
 *
 * The bins have a limited bin size and everything beyond that bin size
 * is offloaded to the CPU to be computed in parallel with the GPU
 * gridding.
 ***********************************************************************/
void OpenCL_interface (
  struct pb_TimerSet* timers,
  unsigned int n,       // Number of input elements
  parameters params,    // Parameter struct which defines output gridSize, cutoff distance, etc.
  ReconstructionSample* sample, // Array of input elements
  float* LUT,           // Precomputed LUT table of Kaiser-Bessel function. 
                          // Used for computation on CPU instead of using the function every time
  int sizeLUT,          // Size of LUT
  cmplx* gridData,      // Array of output grid points. Each element has a real and imaginary component
  float* sampleDensity,  // Array of same size as gridData couting the number of contributions
                          // to each grid point in the gridData array
  cl_context clContext,
  cl_command_queue clCommandQueue, //const cl_device clDevice
  const cl_device_id clDevice,
  size_t *workItemSizes // maximum size of work-items for each dimension
){

  /* Initializing all variables */
  size_t blockSize = workItemSizes[0];
  int dims[3] = {8,4,2}; //size of a gridding block on the GPU

  /* x, y, z dimensions of the output grid (gridData) */
  int size_x = params.gridSize[0];
  int size_y = params.gridSize[1];
  int size_z = params.gridSize[2];
  int size_xy = size_y*size_x;

  int gridNumElems = size_x * size_y * size_z;  // Total number of grid points
  
  float beta = PI * sqrt(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-.5)*(params.oversample-.5)-.8);

  float cutoff = float(params.kernelWidth)/2.0; // cutoff radius
  float cutoff2 = cutoff*cutoff;                // square of cutoff radius
  float _1overCutoff2 = 1/cutoff2;              // 1 over square of cutoff radius

  /* Declarations of device data structures */
  cl_int ciErrNum;
  cl_mem sample_d;    // Device array for original input array
  cl_mem sortedSample_d;             // Device array of the sorted (into bins) input elements.
  
                                            // This array is accessed by sortedSampleSoA_d in a structure
                                            //   of arrays manner.
  cl_mem gridData_d;                // Device array for output grid
  cl_mem sampleDensity_d;            // Device array for output sample density
  cl_mem idxKey_d;            // Array of bin indeces generated in the binning kernel
                                            //   and used to sort the input elements into their
                                            //   corresponding bins
  cl_mem idxValue_d;          // This array holds the indices of input elements in the
                                            //   the original array. This array is sorted using the
                                            //   the idxKey_d array, and once sorted, it is used in
                                            //   the reorder kernel to move the actual elements into
                                            //   their corresponding bins.
  //cl_mem binCount_d;          // Zero-initialized array which counts the number of elements
                                            //   put in each bin. Based on this array, we determine which
                                            //   elements get offloaded to the CPU
  cl_mem binStartAddr_d;      // Array of start offset of each of the compact bins
  
  cl_mem *idxValue_dPtr;
  cl_mem *idxKey_dPtr;
  
  cl_program gpu_kernels;
  cl_kernel binning_kernel;
  cl_kernel reorder_kernel;
  cl_kernel gridding_GPU;

  /* Allocating device memory */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  
  unsigned int *zeroData = NULL, *maxIntData = NULL;
  
  size_t sizeZeroData = sizeof(float)* 2 * gridNumElems;
  if ( n*sizeof(ReconstructionSample) > sizeZeroData) {
    sizeZeroData = n*sizeof(ReconstructionSample);
  }    
  if ( (sizeof(unsigned int) * (gridNumElems+1)) > sizeZeroData) {
    // Not going to be taken, but included just in case since this is used for multiple variables
    sizeZeroData = sizeof(unsigned int) * (gridNumElems+1);
  }
  if ( (((n+3)/4)*4)*sizeof(unsigned int) > sizeZeroData) {
    sizeZeroData = (((n+3)/4)*4)*sizeof(unsigned int);
  }
  
  zeroData = (unsigned int *) malloc(sizeZeroData);
  if (zeroData == NULL) { fprintf(stderr, "Could not allocate dummy memset memory\n"); exit(1); }
  maxIntData = (unsigned int *) malloc((((n+3)/4)*4)*sizeof(unsigned int));
  if (maxIntData == NULL) { fprintf(stderr, "Could not allocate dummy memset memory\n"); exit(1); }
  
  memset(zeroData, 0, sizeZeroData);
  // Initialize padding to max integer value, so that when sorted,
  // these elements get pushed to the end of the array.
  memset(maxIntData+n, 0xFF, (((n+3)&~(3))-n)*sizeof(unsigned int));

  sortedSample_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n*sizeof(ReconstructionSample), zeroData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum);
  binStartAddr_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, (gridNumElems+1)*sizeof(unsigned int), zeroData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum);
  sample_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n*sizeof(ReconstructionSample), sample, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum);
  idxKey_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, (((n+3)/4)*4)*sizeof(unsigned int), maxIntData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum); //Pad to nearest multiple of 4 to 
  idxValue_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, (((n+3)/4)*4)*sizeof(unsigned int), zeroData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum); //satisfy a property of the sorting kernel.
  
  idxKey_dPtr = &idxKey_d;
  idxValue_dPtr = &idxValue_d;
  
  pb_SwitchToTimer(timers, pb_TimerID_DRIVER);

  char compileOptions[1024];
  //                -cl-nv-verbose // Provides register info for NVIDIA devices
  // Set all Macros referenced by kernels
  sprintf(compileOptions, "\
                -D CUTOFF2_VAL=%f -D CUTOFF_VAL=%f -D CEIL_CUTOFF_VAL=%f\
                -D GRIDSIZE_VAL1=%d -D GRIDSIZE_VAL2=%d -D GRIDSIZE_VAL3=%d\
                -D SIZE_XY_VAL=%d -D ONE_OVER_CUTOFF2_VAL=%f",
                cutoff2, cutoff, ceil(cutoff),
                params.gridSize[0], params.gridSize[1], params.gridSize[2],
                size_xy, _1overCutoff2
            );
  
  size_t program_length;
  const char *source_path = "src/opencl_base/GPU_kernels.cl";
  char *source;

  // Dynamically allocate buffer for source
  source = oclLoadProgSource(source_path, "", &program_length);
  if(!source) {
    fprintf(stderr, "Could not load program source (%s) \n", __FILE__); exit(1);
  }
  	
  gpu_kernels = clCreateProgramWithSource(clContext, 1, (const char **)&source, &program_length, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  	  	
  free(source);
  
  OCL_ERRCK_RETVAL ( clBuildProgram(gpu_kernels, 1, &clDevice, compileOptions, NULL, NULL) );
  
  /*
  // Uncomment to view build log from compiler for debugging
  char *build_log;
  size_t ret_val_size;
  ciErrNum = clGetProgramBuildInfo(gpu_kernels, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK_VAR(ciErrNum);
  build_log = (char *)malloc(ret_val_size+1);
  ciErrNum = clGetProgramBuildInfo(gpu_kernels, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
  OCL_ERRCK_VAR(ciErrNum);
       	
  // to be carefully, terminate with \0
  // there's no information in the reference whether the string is 0 terminated or not
  build_log[ret_val_size] = '\0';

  fprintf(stderr, "%s\n", build_log );
  */
  
  
  binning_kernel = clCreateKernel(gpu_kernels, "binning_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  reorder_kernel = clCreateKernel(gpu_kernels, "reorder_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  gridding_GPU = clCreateKernel(gpu_kernels, "gridding_GPU", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
                     
  free(maxIntData);  
  
  pb_SwitchToTimer(timers, pb_TimerID_DRIVER);
  
  size_t block1[1] = { blockSize };
  size_t grid1[1] = { ((n+blockSize-1)/blockSize)*block1[0] };
  
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 0, sizeof(unsigned int), &n) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 1, sizeof(cl_mem), (void *)&sample_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 2, sizeof(cl_mem), (void *)idxKey_dPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 3, sizeof(cl_mem), (void *)idxValue_dPtr) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 4, sizeof(cl_mem), (void *)&binStartAddr_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 5, sizeof(int), &(params.binsize)) );
  OCL_ERRCK_RETVAL( clSetKernelArg(binning_kernel, 6, sizeof(unsigned int), &gridNumElems) );
  
  OCL_ERRCK_RETVAL( clSetKernelArg(reorder_kernel, 0, sizeof(unsigned int), &n) );
  OCL_ERRCK_RETVAL( clSetKernelArg(reorder_kernel, 2, sizeof(cl_mem), (void *)&sample_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(reorder_kernel, 3, sizeof(cl_mem), (void *)&sortedSample_d) );
  
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);

  /* STEP 1: Perform binning. This kernel determines which output bin each input element
   * goes into. Any excess (beyond binsize) is put in the CPU bin
   */
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, binning_kernel, 1, 0,
                            grid1, block1, 0, 0, 0) );

  /* STEP 2: Sort the index-value pair generate in the binning kernel */
  cl_mem dkeys_o = clCreateBuffer(clContext, CL_MEM_READ_WRITE, n*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  cl_mem dvalues_o = clCreateBuffer(clContext, CL_MEM_READ_WRITE, n*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  
  cl_mem *dkeys_oPtr = &dkeys_o;
  cl_mem *dvalues_oPtr = &dvalues_o;
  
  cl_mem *beforePointer = idxKey_dPtr;

  sort(n, gridNumElems+1, idxKey_dPtr, idxValue_dPtr, dkeys_oPtr, dvalues_oPtr, &clContext, clCommandQueue, clDevice, workItemSizes);

  /* STEP 3: Reorder the input data, based on the sorted values from Step 2.
   * this step also involves changing the data from array of structs to a struct
   * of arrays. Also in this kernel, we populate an array with the starting index
   * of every output bin features in the input array, based on the sorted indices 
   * from Step 2.
   * At the end of this step, we copy the start address and list of input elements
   * that will be computed on the CPU.
   */
  OCL_ERRCK_RETVAL( clSetKernelArg(reorder_kernel, 1, sizeof(cl_mem), (void *)idxValue_dPtr) );
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, reorder_kernel, 1, 0,
                            grid1, block1, 0, 0, 0) );

  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  OCL_ERRCK_RETVAL ( clReleaseMemObject(*idxValue_dPtr) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(*idxKey_dPtr) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(sample_d) );

  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);

  /* STEP 4: In this step we generate the ADD scan of the array of starting indices
   * of the output bins. The result is an array that contains the starting address of
   * every output bin.
   */
  scanLargeArray(gridNumElems+1, binStartAddr_d, clContext, clCommandQueue, clDevice, workItemSizes);
  
  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  // Copy back to the CPU the indices of the input elements that will be processed on the CPU
  unsigned int cpuStart;
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, binStartAddr_d, CL_TRUE, 
                          gridNumElems*sizeof(unsigned int), // Offset in bytes
                          sizeof(unsigned int), // Size of data to read
                          &cpuStart, // Host Source
                          0, NULL, NULL) );

  int CPUbin_size = int(n)-int(cpuStart);
  
  ReconstructionSample* CPUbin;

  CPUbin = (ReconstructionSample *) malloc ( CPUbin_size*sizeof(ReconstructionSample) );
  if (CPUbin == NULL) { fprintf(stderr, "Could not allocate memory on host! (%s: %d)\n", __FILE__, __LINE__); exit(1); }
  
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, sortedSample_d, CL_TRUE, 
                          cpuStart*sizeof(ReconstructionSample), // Offset in bytes
                          CPUbin_size*sizeof(ReconstructionSample), // Size of data to read
                          CPUbin, // Host Source
                          0, NULL, NULL) );

  /* STEP 5: Perform the binning on the GPU. The results are computed in a gather fashion
   * where each thread computes the value of one output element by reading the relevant
   * bins.
   */
  gridData_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, gridNumElems*sizeof(cmplx), zeroData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum);
  sampleDensity_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, gridNumElems*sizeof(float), zeroData, &ciErrNum);  OCL_ERRCK_VAR(ciErrNum);
  
  free(zeroData);

  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  
  size_t block2[3] = {dims[0], dims[1], dims[2]};
  size_t grid2[3] = { (size_x/dims[0]) * block2[0], ((size_y*size_z)/(dims[1]*dims[2])) * block2[1], 1 * block2[2] };

  OCL_ERRCK_RETVAL( clSetKernelArg(gridding_GPU, 0, sizeof(cl_mem), (void *)&sortedSample_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(gridding_GPU, 1, sizeof(cl_mem), (void *)&binStartAddr_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(gridding_GPU, 2, sizeof(cl_mem), (void *)&gridData_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(gridding_GPU, 3, sizeof(cl_mem), (void *)&sampleDensity_d) );
  OCL_ERRCK_RETVAL( clSetKernelArg(gridding_GPU, 4, sizeof(float), &beta) );
  
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, gridding_GPU, 3, 0,
                            grid2, block2, 0, 0, 0) );
                                
  OCL_ERRCK_RETVAL ( clReleaseMemObject(binStartAddr_d) );
  
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
                       
  /* Copying the results from the Device to the Host */
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, sampleDensity_d, CL_FALSE, 
                          0, // Offset in bytes
                          gridNumElems*sizeof(float), // Size of data to write
                          sampleDensity, // Host Source
                          0, NULL, NULL) );
                          
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, gridData_d, CL_TRUE, 
                          0, // Offset in bytes
                          gridNumElems*sizeof(cmplx), // Size of data to write
                          gridData, // Host Source
                          0, NULL, NULL) );                          

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);

  /* STEP 6: Computing the contributions of the sample points handled by the Host
   * and adding those to the GPU results.
   */
  gridding_Gold(CPUbin_size, params, CPUbin, LUT, sizeLUT, gridData, sampleDensity);

  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  free(CPUbin);

  OCL_ERRCK_RETVAL ( clReleaseMemObject(gridData_d) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(sampleDensity_d) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(sortedSample_d) );
  
  pb_SwitchToTimer(timers, pb_TimerID_NONE);

  return;
}
