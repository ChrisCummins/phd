/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <parboil.h>

#include "args.h"
#include "model.h"

extern unsigned int NUM_SETS;
extern unsigned int NUM_ELEMENTS;

// create the bin boundaries
void initBinB( struct pb_TimerSet *timers, cl_mem dev_binb, cl_command_queue clCommandQueue)
{
  float *binb = (float*)malloc((NUM_BINS+1)*sizeof(float));
  for (int k = 0; k < NUM_BINS+1; k++)
    {
      binb[k] = cos(pow(10.0, (log10(min_arcmin) + k*1.0/bins_per_dec)) 
		    / 60.0*D2R);
    }

  pb_SwitchToTimer( timers, pb_TimerID_COPY );

  cl_int clStatus;
  clStatus = clEnqueueWriteBuffer(clCommandQueue,dev_binb,CL_TRUE,0,(NUM_BINS+1)*sizeof(float),binb,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  pb_SwitchToTimer( timers, pb_TimerID_COMPUTE );
  free(binb);
}

void TPACF(cl_mem histograms, cl_mem d_x_data, 
	   cl_mem dev_binb, 
	   cl_command_queue clCommandQueue, cl_kernel clKernel)
{
  size_t dimBlock = BLOCK_SIZE;
  size_t dimGrid = (NUM_SETS*2 + 1)*dimBlock;
  
  cl_int clStatus;
  clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),&histograms);
  clStatus = clSetKernelArg(clKernel,1,sizeof(cl_mem),&d_x_data);
  clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),&dev_binb);
  clStatus = clSetKernelArg(clKernel,3,sizeof(int),&NUM_SETS);
  clStatus = clSetKernelArg(clKernel,4,sizeof(int),&NUM_ELEMENTS);
  
  CHECK_ERROR("clSetKernelArg")

  clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,1,NULL,&dimGrid,&dimBlock,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")

  clStatus = clFinish(clCommandQueue);
  CHECK_ERROR("clFinish")
}

int 
main( int argc, char** argv) 
{
  struct pb_TimerSet timers;
  struct pb_Parameters *params;

  pb_InitializeTimerSet( &timers );
  params = pb_ReadParameters( &argc, argv );

  options args;
  parse_args(argc, argv, &args);
  
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  NUM_ELEMENTS = args.npoints;
  NUM_SETS = args.random_count;
  int num_elements = NUM_ELEMENTS; 
  
  printf("Min distance: %f arcmin\n", min_arcmin);
  printf("Max distance: %f arcmin\n", max_arcmin);
  printf("Bins per dec: %i\n", bins_per_dec);
  printf("Total bins  : %i\n", NUM_BINS);

  //read in files 
  unsigned mem_size = (1+NUM_SETS)*num_elements*sizeof(struct cartesian);
  unsigned f_mem_size = (1+NUM_SETS)*num_elements*sizeof(float);

  // container for all the points read from files
  struct cartesian *h_all_data;
  h_all_data = (struct cartesian*) malloc(mem_size); 
  // Until I can get libs fixed
    
  // iterator for data files
  struct cartesian *working = h_all_data;
    
  // go through and read all data and random points into h_all_data
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  readdatafile(params->inpFiles[0], working, num_elements);
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  working += num_elements;
  for(int i = 0; i < (NUM_SETS); i++)
    {
      pb_SwitchToTimer( &timers, pb_TimerID_IO );
      char fileName[50];
      readdatafile(params->inpFiles[i+1], working, num_elements);
      pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

      working += num_elements;
    }

  // split into x, y, and z arrays
  float * h_x_data = (float*) malloc (3*f_mem_size);
  float * h_y_data = h_x_data + NUM_ELEMENTS*(NUM_SETS+1);
  float * h_z_data = h_y_data + NUM_ELEMENTS*(NUM_SETS+1);
  for(int i = 0; i < (NUM_SETS+1); ++i)
    {
      for(int j = 0; j < NUM_ELEMENTS; ++j)
	{
	  h_x_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].x;
	  h_y_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].y;
	  h_z_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].z;
	}
    }

  // from on use x, y, and z arrays, free h_all_data
  free(h_all_data);
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );

  cl_int clStatus;

  cl_platform_id clPlatform;
  clStatus = clGetPlatformIDs(1,&clPlatform,NULL);
  CHECK_ERROR("clGetPlatformIDs")

  cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
  cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_CPU,NULL,NULL,&clStatus);
  CHECK_ERROR("clCreateContextFromType")
   
  cl_device_id clDevice;
  clStatus = clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_CPU,1,&clDevice,NULL);
  CHECK_ERROR("clGetDeviceIDs")

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_base");

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"gen_hists",&clStatus);
  CHECK_ERROR("clCreateKernel")
  
  // allocate OpenCL memory to hold all points
  //Sub-buffers are not defined in OpenCL 1.0
  cl_mem d_x_data;
  d_x_data = clCreateBuffer(clContext,CL_MEM_READ_ONLY,3*f_mem_size,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  // allocate OpenCL memory to hold final histograms
  // (1 for dd, and NUM_SETS for dr and rr apiece)
  cl_mem d_hists;
  d_hists = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,NUM_BINS*(NUM_SETS*2+1)*sizeof(hist_t),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  cl_mem dev_binb;
  dev_binb = clCreateBuffer(clContext,CL_MEM_READ_ONLY,(NUM_BINS+1)*sizeof(float),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  // allocate system memory for final histograms
  hist_t *new_hists = (hist_t *) malloc(NUM_BINS*(NUM_SETS*2+1)*
					sizeof(hist_t));

  // Initialize the boundary constants for bin search
  initBinB(&timers, dev_binb, clCommandQueue);

  // **===------------------ Kick off TPACF on OpenCL------------------===**
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  clStatus = clEnqueueWriteBuffer(clCommandQueue,d_x_data,CL_TRUE,0,3*f_mem_size,h_x_data,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  TPACF(d_hists,d_x_data,dev_binb,clCommandQueue,clKernel);

  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  clStatus = clEnqueueReadBuffer(clCommandQueue,d_hists,CL_TRUE,0,NUM_BINS*(NUM_SETS*2+1)*sizeof(hist_t),new_hists,0,NULL,NULL);
  CHECK_ERROR("clEnqueueReadBuffer")

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  // **===-------------------------------------------------------------===**

  // references into output histograms
  hist_t *dd_hist = new_hists;
  hist_t *rr_hist = dd_hist + NUM_BINS;
  hist_t *dr_hist = rr_hist + NUM_BINS*NUM_SETS;

  // add up values within dr and rr
  int rr[NUM_BINS];
  for(int i=0; i<NUM_BINS; i++)
    {
      rr[i] = 0;
    }
  for(int i=0; i<NUM_SETS; i++)
    {
      for(int j=0; j<NUM_BINS; j++)
	{
	  rr[j] += rr_hist[i*NUM_BINS + j];
	}
    }
  int dr[NUM_BINS];
  for(int i=0; i<NUM_BINS; i++)
    {
      dr[i] = 0;
    }
  for(int i=0; i<NUM_SETS; i++)
    {
      for(int j=0; j<NUM_BINS; j++)
	{
	  dr[j] += dr_hist[i*NUM_BINS + j];
	}
    }

  FILE *outfile;
  if ((outfile = fopen(params->outFile, "w")) == NULL)
    {
      fprintf(stderr, "Unable to open output file %s for writing, "
	      "assuming stdout\n", params->outFile);
      outfile = stdout;
    }
  
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  // print out final histograms + omega (while calculating omega)
  for(int i=0; i<NUM_BINS; i++)
    {
      fprintf(outfile, "%d\n%d\n%d\n", dd_hist[i], dr[i], rr[i]);
    }

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  if(outfile != stdout)
    fclose(outfile);

  // cleanup memory
  free(new_hists);
  free( h_x_data);

  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  clStatus = clReleaseMemObject(d_hists);
  clStatus = clReleaseMemObject(d_x_data);
  clStatus = clReleaseMemObject(dev_binb);
  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext);
  CHECK_ERROR("clReleaseContext")

  free((void*)clSource[0]);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
}

