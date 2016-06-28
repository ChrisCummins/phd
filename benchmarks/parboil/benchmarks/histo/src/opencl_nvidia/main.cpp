/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/


#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#include "util.h"
#include "OpenCL_common.h"

#define BLOCK_X         14

#define PRESCAN_THREADS     512
#define PRESCAN_BLOCKS_X    64

#define UNROLL 16

/******************************************************************************
* Implementation: GPU
* Details:
* in the GPU implementation of histogram, we begin by computing the span of the
* input values into the histogram. Then the histogramming computation is carried
* out by a (BLOCK_X, BLOCK_Y) sized grid, where every group of Y (same X)
* computes its own partial histogram for a part of the input, and every Y in the
* group exclusively writes to a portion of the span computed in the beginning.
* Finally, a reduction is performed to combine all the partial histograms into
* the final result.
******************************************************************************/

int main(int argc, char* argv[]) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }
  
  char oclOverhead[] = "OCL Overhead";
  char prescans[] = "PreScanKernel";
  char postpremems[] = "PostPreMems";
  char intermediates[] = "IntermediatesKernel";
  char mains[] = "MainKernel";
  char finals[] = "FinalKernel";

  pb_InitializeTimerSet(&timers);
  
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, prescans, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, postpremems, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, mains, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, finals, pb_TimerID_KERNEL);

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;
  unsigned int lmemKB;
  unsigned int nThreads;
  unsigned int bins_per_block;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  cl_int ciErrNum;
  cl_platform_id clPlatform;
  int deviceType = CL_DEVICE_TYPE_GPU;
  cl_device_id clDevice;
  cl_context clContext;
  cl_command_queue clCommandQueue;
  
  cl_program clProgram[4];
  
  cl_kernel histo_prescan_kernel;
  cl_kernel histo_intermediates_kernel;
  cl_kernel histo_main_kernel;
  cl_kernel histo_final_kernel;

  int even_width = ((img_width+1)/2)*2;

  cl_mem input;
  cl_mem ranges;
  cl_mem sm_mappings;
  cl_mem global_subhisto;
  cl_mem global_histo;
  cl_mem global_overflow;
  cl_mem final_histo;
  
  OCL_ERRCK_RETVAL ( clGetPlatformIDs(1, &clPlatform, NULL) );
  OCL_ERRCK_RETVAL ( clGetDeviceIDs(clPlatform, deviceType, 1, &clDevice, NULL) );
  
  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform, 0};
  clContext = clCreateContextFromType(cps, deviceType, NULL, NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  pb_SetOpenCL(&clContext, &clCommandQueue);  
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  
  long unsigned int lmemSize = 0;
  OCL_ERRCK_RETVAL ( clGetDeviceInfo(clDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lmemSize, NULL) );
  
  // lmemKB = lmemSize / 1024; // Should be valid, but not taken into consideration for initial programming
  
  if (lmemSize >= 48*1024) {
    lmemKB = 48;
  } else if (lmemSize >= 24*1024) {
    lmemKB = 24;
  } else {
    lmemKB = 8;
  }
  
  lmemKB = 24;
  
  bins_per_block = lmemKB * 1024;
  
  switch (lmemKB) {
    case 48: nThreads = 1024; break;
    case 24: nThreads = 768; break;
    default: nThreads = 512; break;
  }
  
  
  
  size_t program_length[4];
  const char *source_path[4] = { "src/opencl_nvidia/histo_prescan.cl",
    "src/opencl_nvidia/histo_intermediates.cl", "src/opencl_nvidia/histo_main.cl","src/opencl_nvidia/histo_final.cl"};
  char *source[4];

  for (int i = 0; i < 4; ++i) {
    // Dynamically allocate buffer for source
    source[i] = oclLoadProgSource(source_path[i], "", &program_length[i]);
    if(!source[i]) {
      fprintf(stderr, "Could not load program source\n"); exit(1);
    }
  	
  	clProgram[i] = clCreateProgramWithSource(clContext, 1, (const char **)&source[i], &program_length[i], &ciErrNum);
  	OCL_ERRCK_VAR(ciErrNum);
  	  	
  	free(source[i]);
  }
  	
  	  	  	  	  	  	  	
  char compileOptions[1024];
  //                -cl-nv-verbose // Provides register info for NVIDIA devices
  // Set all Macros referenced by kernels
  sprintf(compileOptions, "\
                -D PRESCAN_THREADS=%u\
                -D KB=%u -D UNROLL=%u\
                -D BINS_PER_BLOCK=%u -D BLOCK_X=%u",
                
                PRESCAN_THREADS,
                lmemKB, UNROLL,
                bins_per_block, BLOCK_X
            ); 
  
  for (int i = 0; i < 4; ++i) {
//fprintf(stderr, "Building Program #%d...\n", i);
    OCL_ERRCK_RETVAL ( clBuildProgram(clProgram[i], 1, &clDevice, compileOptions, NULL, NULL) );
       
          /*
       char *build_log;
       size_t ret_val_size;
       ciErrNum = clGetProgramBuildInfo(clProgram[i], clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK_VAR(ciErrNum);
       build_log = (char *)malloc(ret_val_size+1);
       ciErrNum = clGetProgramBuildInfo(clProgram[i], clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
       	OCL_ERRCK_VAR(ciErrNum);
       	

       // to be carefully, terminate with \0
       // there's no information in the reference whether the string is 0 terminated or not
       build_log[ret_val_size] = '\0';

       fprintf(stderr, "%s\n", build_log );
       */
  }
  	
  histo_prescan_kernel = clCreateKernel(clProgram[0], "histo_prescan_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  histo_intermediates_kernel = clCreateKernel(clProgram[1], "histo_intermediates_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  histo_main_kernel = clCreateKernel(clProgram[2], "histo_main_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  histo_final_kernel = clCreateKernel(clProgram[3], "histo_final_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);  	

  	
  pb_SwitchToTimer(&timers, pb_TimerID_IO);  

  input = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  ranges = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      2*sizeof(unsigned int), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);  
  sm_mappings = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*img_height*4*sizeof(unsigned char), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  global_subhisto = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*histo_height*sizeof(unsigned int), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  global_histo = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*histo_height*sizeof(unsigned short), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  global_overflow = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*histo_height*sizeof(unsigned int), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  final_histo = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*histo_height*sizeof(unsigned char), NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  

  // Must dynamically allocate. Too large for stack
  unsigned int *zeroData;
  zeroData = (unsigned int *) malloc(sizeof(unsigned int) *img_width*histo_height);
  if (zeroData == NULL) {
    fprintf(stderr, "Failed to allocate %ld bytes of memory!\n", sizeof(unsigned int) * img_width * histo_height);
    exit(1);
  }
  memset(zeroData, 0, img_width*histo_height*sizeof(unsigned int));
   
  for (int y=0; y < img_height; y++){
    OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, input, CL_FALSE, 
                          y*even_width*sizeof(unsigned int), // Offset in bytes
                          img_width*sizeof(unsigned int), // Size of data to write
                          &img[y*img_width], // Host Source
                          0, NULL, NULL) );
  }
 
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  unsigned int img_dim = img_height*img_width;
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_prescan_kernel, 0, sizeof(cl_mem), (void *)&input) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_prescan_kernel, 1, sizeof(unsigned int), &img_dim) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_prescan_kernel, 2, sizeof(cl_mem), (void *)&ranges) );

  unsigned int half_width = (img_width+1)/2;
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 0, sizeof(cl_mem), (void *)&input) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 1, sizeof(unsigned int), &img_height) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 2, sizeof(unsigned int), &img_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 3, sizeof(unsigned int), &half_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 4, sizeof(cl_mem), (void *)&sm_mappings) );

  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 0, sizeof(cl_mem), (void *)&sm_mappings) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 1, sizeof(unsigned int), &img_dim) );

  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 4, sizeof(unsigned int), &histo_height) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 5, sizeof(unsigned int), &histo_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 6, sizeof(cl_mem), (void *)&global_subhisto) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 7, sizeof(cl_mem), (void *)&global_histo) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 8, sizeof(cl_mem), (void *)&global_overflow) );
  

  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 2, sizeof(unsigned int), &histo_height) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 3, sizeof(unsigned int), &histo_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 4, sizeof(cl_mem), (void *)&global_subhisto) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 5, sizeof(cl_mem), (void *)&global_histo) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 6, sizeof(cl_mem), (void *)&global_overflow) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 7, sizeof(cl_mem), (void *)&final_histo) );

  size_t prescan_localWS[1] = {PRESCAN_THREADS};
  size_t prescan_globalWS[1] = {PRESCAN_BLOCKS_X*prescan_localWS[0]};
  size_t inter_localWS[1] = {(img_width+1)/2};
  size_t inter_globalWS[1] = {((img_height + UNROLL-1)/UNROLL) * inter_localWS[0]};
  size_t main_localWS[2] = {nThreads, 1};
  size_t main_globalWS[2];  main_globalWS[0] = BLOCK_X * main_localWS[0];
  size_t final_localWS[1] = {512};
  size_t final_globalWS[1] = {BLOCK_X*3 * final_localWS[0]};
    

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  for (int iter = 0; iter < numIterations; iter++) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    
    // how about something like
    // __global__ unsigned int ranges[2];
    // ...kernel
    // __shared__ unsigned int s_ranges[2];
    // if (threadIdx.x == 0) {s_ranges[0] = ranges[0]; s_ranges[1] = ranges[1];}
    // __syncthreads();
    
    // Although then removing the blocking cudaMemcpy's might cause something about
    // concurrent kernel execution.
    // If kernel launches are synchronous, then how can 2 kernels run concurrently? different host threads?

  OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, ranges, CL_TRUE, 
                          0, // Offset in bytes
                          2*sizeof(unsigned int), // Size of data to write
                          ranges_h, // Host Source
                          0, NULL, NULL) );
                                                    
  pb_SwitchToSubTimer(&timers, prescans , pb_TimerID_KERNEL);
                         
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_prescan_kernel, 1, 0,
                            prescan_globalWS, prescan_localWS, 0, 0, 0) );

  pb_SwitchToSubTimer(&timers, postpremems , pb_TimerID_KERNEL);
    
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, ranges, CL_FALSE, 
                          0, // Offset in bytes
                          2*sizeof(unsigned int), // Size of data to read
                          ranges_h, // Host Source
                          0, NULL, NULL) );

  OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, global_subhisto, CL_TRUE, 
                          0, // Offset in bytes
                          img_width*histo_height*sizeof(unsigned int), // Size of data to write
                          zeroData, // Host Source
                          0, NULL, NULL) );

  pb_SwitchToSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
                     
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_intermediates_kernel, 1, 0,
                            inter_globalWS, inter_localWS, 0, 0, 0) );                          

  main_globalWS[1] = ranges_h[1]-ranges_h[0]+1;
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 2, sizeof(unsigned int), &ranges_h[0]) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_main_kernel, 3, sizeof(unsigned int), &ranges_h[1]) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 0, sizeof(unsigned int), &ranges_h[0]) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 1, sizeof(unsigned int), &ranges_h[1]) );

  pb_SwitchToSubTimer(&timers, mains, pb_TimerID_KERNEL);


  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_main_kernel, 2, 0,
                            main_globalWS, main_localWS, 0, 0, 0) );

  pb_SwitchToSubTimer(&timers, finals, pb_TimerID_KERNEL);

  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_final_kernel, 1, 0,
                            final_globalWS, final_localWS, 0, 0, 0) );
  
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);


  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, final_histo, CL_TRUE, 
                          0, // Offset in bytes
                          histo_height*histo_width*sizeof(unsigned char), // Size of data to read
                          histo, // Host Source
                          0, NULL, NULL) );

  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_prescan_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_intermediates_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_main_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_final_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[0]) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[1]) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[2]) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[3]) );

  OCL_ERRCK_RETVAL ( clReleaseMemObject(input) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(ranges) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(sm_mappings) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(global_subhisto) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(global_histo) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(global_overflow) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(final_histo) );
  


  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }


  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);


  free(zeroData);
  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);
  
  OCL_ERRCK_RETVAL ( clReleaseCommandQueue(clCommandQueue) );
  OCL_ERRCK_RETVAL ( clReleaseContext(clContext) );
  
  pb_DestroyTimerSet(&timers);

  return 0;
}
