
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "OpenCL_common.h"

#define GRID_SIZE 65535
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define EXPANDED_SIZE(__x) (__x+(__x>>LOG_NUM_BANKS)+(__x>>(2*LOG_NUM_BANKS)))

void scanLargeArray( unsigned int gridNumElems, cl_mem data_d, cl_context clContext, cl_command_queue clCommandQueue, const cl_device_id clDevice, size_t *workItemSizes) {

    size_t blockSize = (workItemSizes[0]*2 < 1024) ? workItemSizes[0]*2 : 1024;

    // Run the prescan
    unsigned int size = (gridNumElems+blockSize-1)/blockSize;
    
    unsigned int dim_block;
    unsigned int current_max = size*blockSize;
    for (int block_size_lcv = 128; block_size_lcv <= blockSize; block_size_lcv *= 2){
      unsigned int array_size = block_size_lcv;
      while(array_size < size){
        array_size *= block_size_lcv;
      }
      if (array_size <= current_max){
        current_max = array_size;
        dim_block = block_size_lcv;
      }
    }    

    cl_mem inter_d;
    cl_int ciErrNum;
    cl_program scanLargeArray_program;

    cl_kernel scan_L1_kernel;
    cl_kernel scan_inter1_kernel;
    cl_kernel scan_inter2_kernel;
    cl_kernel uniformAdd;
    
    // allocate device memory input and output arrays
    unsigned int *zeroData;
    zeroData = (unsigned int *)calloc( current_max, sizeof(unsigned int) );
    if (zeroData == NULL) { fprintf(stderr, "Could not allocate host memory! (%s)\n", __FILE__); exit(1); }

    inter_d = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, current_max*sizeof(unsigned int), zeroData, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
    
    free(zeroData);
    
    char compileOptions[128];
    //                -cl-nv-verbose // Provides register info for NVIDIA devices
    // Set all Macros referenced by kernels
    sprintf(compileOptions, "\
                -D DYN_LOCAL_MEM_SIZE=%lu",
                EXPANDED_SIZE(dim_block)*sizeof(unsigned int)
            );
  
    size_t program_length;
    const char *source_path = "src/opencl_base/scanLargeArray.cl";
    char *source;

    // Dynamically allocate buffer for source
    source = oclLoadProgSource(source_path, "", &program_length);
    if(!source) {
      fprintf(stderr, "Could not load program source! (%s)\n", __FILE__); exit(1);
    }
  	
    scanLargeArray_program = clCreateProgramWithSource(clContext, 1, (const char **)&source, &program_length, &ciErrNum);
    OCL_ERRCK_VAR(ciErrNum);

    free(source);
    OCL_ERRCK_RETVAL ( clBuildProgram(scanLargeArray_program, 1, &clDevice, compileOptions, NULL, NULL) ); 
      
  /*
    // Uncomment for build log from compiler for debugging
    char *build_log;
    size_t ret_val_size;
    ciErrNum = clGetProgramBuildInfo(scanLargeArray_program, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK_VAR(ciErrNum);
    build_log = (char *)malloc(ret_val_size+1);
    ciErrNum = clGetProgramBuildInfo(scanLargeArray_program, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    OCL_ERRCK_VAR(ciErrNum);
    
    // to be carefully, terminate with \0
    // there's no information in the reference whether the string is 0 terminated or not
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "%s\n", build_log );
    */   
        
    scan_L1_kernel = clCreateKernel(scanLargeArray_program, "scan_L1_kernel", &ciErrNum);
    OCL_ERRCK_VAR(ciErrNum);
      
    scan_inter1_kernel = clCreateKernel(scanLargeArray_program, "scan_inter1_kernel", &ciErrNum);
    OCL_ERRCK_VAR(ciErrNum);
    scan_inter2_kernel = clCreateKernel(scanLargeArray_program, "scan_inter2_kernel", &ciErrNum);
    OCL_ERRCK_VAR(ciErrNum);  
      
    uniformAdd = clCreateKernel(scanLargeArray_program, "uniformAdd", &ciErrNum);
    OCL_ERRCK_VAR(ciErrNum);
    
    OCL_ERRCK_RETVAL( clSetKernelArg(scan_L1_kernel, 1, sizeof(cl_mem), (void *)&data_d) );
    OCL_ERRCK_RETVAL( clSetKernelArg(scan_L1_kernel, 3, sizeof(cl_mem), (void *)&inter_d) );
    
    OCL_ERRCK_RETVAL( clSetKernelArg(scan_inter1_kernel, 0, sizeof(cl_mem), (void *)&inter_d) );
    OCL_ERRCK_RETVAL( clSetKernelArg(scan_inter2_kernel, 0, sizeof(cl_mem), (void *)&inter_d) );
    
    OCL_ERRCK_RETVAL( clSetKernelArg(uniformAdd, 1, sizeof(cl_mem), (void *)&data_d) );
    OCL_ERRCK_RETVAL( clSetKernelArg(uniformAdd, 3, sizeof(cl_mem), (void *)&inter_d) );

    for (unsigned int i=0; i < (size+GRID_SIZE-1)/GRID_SIZE; i++) {
        unsigned int gridSize = ((size-(i*GRID_SIZE)) > GRID_SIZE) ? GRID_SIZE : (size-i*GRID_SIZE);
        unsigned int numElems = ((gridNumElems-(i*GRID_SIZE*blockSize)) > (GRID_SIZE*blockSize)) ? (GRID_SIZE*blockSize) : (gridNumElems-(i*GRID_SIZE*blockSize));
        
        unsigned int data_offset = i*GRID_SIZE*blockSize;
        unsigned int inter_offset = i*GRID_SIZE;
        OCL_ERRCK_RETVAL( clSetKernelArg(scan_L1_kernel, 0, sizeof(unsigned int), &numElems) );
        OCL_ERRCK_RETVAL( clSetKernelArg(scan_L1_kernel, 2, sizeof(unsigned int), &data_offset) );
        OCL_ERRCK_RETVAL( clSetKernelArg(scan_L1_kernel, 4, sizeof(unsigned int), &inter_offset) );
               
        size_t block[1] = { blockSize/2 };
        size_t grid[1] = { gridSize * block[0] };
        
        OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, scan_L1_kernel, 1, 0,
                            grid, block, 0, 0, 0) );
    }

    unsigned int stride = 1;
    for (unsigned int d = current_max; d > 1; d /= dim_block) {        
        size_t block[1] = { dim_block/2 };
        size_t grid[1] = { (d/dim_block) * block[0] };
        
        OCL_ERRCK_RETVAL( clSetKernelArg(scan_inter1_kernel, 1, sizeof(unsigned int), &stride) );
        OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, scan_inter1_kernel, 1, 0,
                            grid, block, 0, 0, 0) );
        
        stride *= dim_block;
    }
    
    unsigned int singleZero = 0;
    OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, inter_d, CL_TRUE, 
                          (current_max-1)*sizeof(unsigned int), // Offset in bytes
                          sizeof(unsigned int), // Size of data to write
                          &singleZero, // Host Source
                          0, NULL, NULL) );

    for (unsigned int d = dim_block; d <= current_max; d *= dim_block) {
        stride /= dim_block;
        
        size_t block[1] = { dim_block/2 };
        size_t grid[1] = { (d/dim_block) * block[0] };
        
        OCL_ERRCK_RETVAL( clSetKernelArg(scan_inter2_kernel, 1, sizeof(unsigned int), &stride) );
        
        OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, scan_inter2_kernel, 1, 0,
                            grid, block, 0, 0, 0) );                       
    }

    for (unsigned int i=0; i < (size+GRID_SIZE-1)/GRID_SIZE; i++) {
        unsigned int gridSize = ((size-(i*GRID_SIZE)) > GRID_SIZE) ? GRID_SIZE : (size-i*GRID_SIZE);
        unsigned int numElems = ((gridNumElems-(i*GRID_SIZE*blockSize)) > (GRID_SIZE*blockSize)) ? (GRID_SIZE*blockSize) : (gridNumElems-(i*GRID_SIZE*blockSize));
        
        unsigned int data_offset = i*GRID_SIZE*blockSize;
        unsigned int inter_offset = i*GRID_SIZE;
        OCL_ERRCK_RETVAL( clSetKernelArg(uniformAdd, 0, sizeof(unsigned int), &numElems) );
        OCL_ERRCK_RETVAL( clSetKernelArg(uniformAdd, 2, sizeof(unsigned int), &data_offset) );
        OCL_ERRCK_RETVAL( clSetKernelArg(uniformAdd, 4, sizeof(unsigned int), &inter_offset) );
        
        size_t block[1] = { blockSize/2 };
        size_t grid[1] = { gridSize * block[0] };
        
        OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, uniformAdd, 1, 0,
                            grid, block, 0, 0, 0) ); 
    }

    OCL_ERRCK_RETVAL ( clReleaseMemObject(inter_d) );
    OCL_ERRCK_RETVAL ( clReleaseKernel(scan_L1_kernel) );
    OCL_ERRCK_RETVAL ( clReleaseKernel(scan_inter1_kernel) );
    OCL_ERRCK_RETVAL ( clReleaseKernel(scan_inter2_kernel) );
    OCL_ERRCK_RETVAL ( clReleaseKernel(uniformAdd) );

    OCL_ERRCK_RETVAL ( clReleaseProgram(scanLargeArray_program) );
}

