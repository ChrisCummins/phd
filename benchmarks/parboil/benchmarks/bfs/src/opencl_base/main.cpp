/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
  Implementing Breadth first search on CUDA using algorithm given in DAC'10
  paper "An Effective GPU Implementation of Breadth-First Search"

  Copyright (c) 2010 University of Illinois at Urbana-Champaign. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Author: Lijiuan Luo (lluo3@uiuc.edu)
  Revised for Parboil 2.5 Benchmark Suite by: Geng Daniel Liu (gengliu2@illinois.edu)
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "parboil.h"
#include "OpenCL_common.h"
#include "config.h"


#define CHECK_ERROR(errorMessage)        \
if(clStatus != CL_SUCCESS)               \
{                                        \
  printf("Error: %s!\n",errorMessage);   \
  printf("Line: %d\n",__LINE__);         \
  exit(1);                               \
}

FILE *fp;
char* readFile(const char* fileName)                                                                   
{                                                                                                    
  FILE* fp;                                                                                      
  fp = fopen(fileName,"r");                                                                      
  if(fp == NULL)                                                                                 
  {                                                                                              
    printf("Error 1!\n");                                                                  
    exit(1);                                                                               
  }                                                                                              

  fseek(fp,0,SEEK_END);                                                                          
  long size = ftell(fp);                                                                         
  rewind(fp);                                                                                    

  char* buffer = (char*)malloc(sizeof(char)*size);                                               
  if(buffer  == NULL)                                                                            
  {                                                                                              
    printf("Error 2!\n");                                                                  
    fclose(fp);                                                                            
    exit(1);                                                                               
  }                                                                                              

  size_t res = fread(buffer,1,size,fp);                                                          
  if(res != size)                                                                                
  {                                                                                              
    printf("Error 3!\n");                                                                  
    fclose(fp);                                                                            
    exit(1);                                                                               
  }                                                                                              

  fclose(fp);                                                                                    
  return buffer;                                                                                 
}     
const int h_top = 1;
const int zero = 0;
void runGPU(int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

  //the number of nodes in the graph
  int num_of_nodes = 0; 
  //the number of edges in the graph
  int num_of_edges = 0;
  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
  {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  //Read in Graph from a file
  fp = fopen(params->inpFiles[0],"r");
  if(!fp)
  {
    printf("Error Reading graph file\n");
    return 0;
  }
  int source;

  fscanf(fp,"%d",&num_of_nodes);
  // allocate host memory
  struct Node* h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*num_of_nodes);
  int *color = (int*) malloc(sizeof(int)*num_of_nodes);
  int start, edgeno;   
  // initalize the memory
  int i;
  for( i = 0; i < num_of_nodes; i++) 
  {
    fscanf(fp,"%d %d",&start,&edgeno);
    h_graph_nodes[i].x = start;
    h_graph_nodes[i].y = edgeno;
    color[i]=WHITE;
  }
  //read the source node from the file
  fscanf(fp,"%d",&source);
  fscanf(fp,"%d",&num_of_edges);
  int id,cost;
  struct Edge* h_graph_edges = (struct Edge*) malloc(sizeof(struct Edge)*num_of_edges);
  for(i=0; i < num_of_edges ; i++)
  {
    fscanf(fp,"%d",&id);
    fscanf(fp,"%d",&cost);
    h_graph_edges[i].x = id;
    h_graph_edges[i].y = cost;
  }
  if(fp)
    fclose(fp);    

  // allocate mem for the result on host side
  int* h_cost = (int*) malloc( sizeof(int)*num_of_nodes);
  for(i = 0; i < num_of_nodes; i++){
    h_cost[i] = INF;
  }
  h_cost[source] = 0;

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  cl_int clStatus;
  cl_device_id clDevice;
  cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
  cl_platform_id clPlatform;
  OCL_ERRCK_RETVAL(clGetPlatformIDs(1,&clPlatform,NULL));
  cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
  int deviceFound = getOpenCLDevice(&clPlatform, &clDevice, &deviceType, 0);
  if (deviceFound < 0) {
    fprintf(stderr, "No suitable device was found\n");
    exit(1);
  }

  cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  OCL_ERRCK_VAR(clStatus);

  pb_SetOpenCL(&clContext, &clCommandQueue);

  char *clSource;
  size_t program_length;
  const char *clSource_path = "src/opencl_base/kernel.cl";
  clSource = oclLoadProgSource(clSource_path, "", &program_length);
  cl_program clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&clSource, &program_length, &clStatus);
  OCL_ERRCK_VAR(clStatus);

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_base");
  OCL_ERRCK_RETVAL(clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL));

  // Uncomment to view build log from compiler for debugging
  /* 
  char *build_log;
  size_t ret_val_size;
  clStatus = clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);  
  build_log = (char *)malloc(ret_val_size+1);
  clStatus = clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
  // there's no information in the reference whether the string is 0 terminated or not
  build_log[ret_val_size] = '\0';
  printf("%s\n", build_log );
  */

  cl_kernel BFS_kernel = clCreateKernel(clProgram,"BFS_kernel",&clStatus);
  OCL_ERRCK_VAR(clStatus);

  //Copy the Node list to device memory
  cl_mem d_graph_nodes;
  d_graph_nodes = clCreateBuffer(clContext,CL_MEM_READ_ONLY,num_of_nodes*sizeof(struct Node),NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_graph_nodes,CL_TRUE,0,num_of_nodes*sizeof(struct Node),h_graph_nodes,0,NULL,NULL));
  //Copy the Edge List to device Memory
  cl_mem d_graph_edges;
  d_graph_edges = clCreateBuffer(clContext,CL_MEM_READ_ONLY,num_of_edges*sizeof(struct Edge),NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_graph_edges,CL_TRUE,0,num_of_edges*sizeof(struct Edge),h_graph_edges,0,NULL,NULL));

  cl_mem d_color, d_cost, d_q1, d_q2, tail;
  d_color = clCreateBuffer(clContext,CL_MEM_READ_WRITE,num_of_nodes*sizeof(int),NULL,&clStatus);
  d_cost = clCreateBuffer(clContext,CL_MEM_READ_WRITE,num_of_nodes*sizeof(int),NULL,&clStatus);
  d_q1 = clCreateBuffer(clContext,CL_MEM_READ_WRITE,num_of_nodes*sizeof(int),NULL,&clStatus);
  d_q2 = clCreateBuffer(clContext,CL_MEM_READ_WRITE,num_of_nodes*sizeof(int),NULL,&clStatus);
  tail = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_color,CL_TRUE,0,num_of_nodes*sizeof(int),color,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_cost,CL_TRUE,0,num_of_nodes*sizeof(int),h_cost,0,NULL,NULL));

  printf("Starting GPU kernel\n");
  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  int num_of_blocks; 
  int num_of_threads_per_block;

  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,tail,CL_TRUE,0,sizeof(int),&h_top,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_cost,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,d_q1,CL_TRUE,0,sizeof(int),&source,0,NULL,NULL));

  int num_t;//number of threads
  int k=0;//BFS level index

  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,2,sizeof(cl_mem),(void*)&d_graph_nodes));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,3,sizeof(cl_mem),(void*)&d_graph_edges));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,4,sizeof(cl_mem),(void*)&d_color));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,5,sizeof(cl_mem),(void*)&d_cost));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,6,sizeof(cl_mem),(void*)&tail));

  do
  {
    OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,tail,CL_TRUE,0,sizeof(int),&num_t,0,NULL,NULL));
    OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,tail,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));

    if(num_t == 0){//frontier is empty
      break;
    }

    num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK); 
    num_of_threads_per_block = num_t > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : num_t;

    size_t grid[1] = {num_of_blocks*num_of_threads_per_block};
    size_t block[1] = {num_of_threads_per_block};


    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,7,sizeof(int),(void*)&num_t));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,9,sizeof(int),(void*)&k));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,10,sizeof(int),NULL));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,11,LOCAL_MEM_SIZE*sizeof(int),NULL));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,12,sizeof(int),NULL));
    if(k%2 == 0){
      int gray = GRAY0;
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,0,sizeof(cl_mem),(void*)&d_q1));
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,1,sizeof(cl_mem),(void*)&d_q2));
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,8,sizeof(int),(void*)&gray));
    }
    else{
      int gray = GRAY1;
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,0,sizeof(cl_mem),(void*)&d_q2));
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,1,sizeof(cl_mem),(void*)&d_q1));
      OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel,8,sizeof(int),(void*)&gray));
    }
    OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel,1,0,grid,block,0,0,0));
    OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
    k++;
  } while(1);
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  printf("GPU kernel done\n");

  // copy result from device to host
  OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,d_cost,CL_TRUE,0,num_of_nodes*sizeof(int),h_cost,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,d_color,CL_TRUE,0,num_of_nodes*sizeof(int),color,0,NULL,NULL));

  OCL_ERRCK_RETVAL(clReleaseMemObject(d_graph_nodes));
  OCL_ERRCK_RETVAL(clReleaseMemObject(d_graph_edges));
  OCL_ERRCK_RETVAL(clReleaseMemObject(d_color));
  OCL_ERRCK_RETVAL(clReleaseMemObject(d_cost));
  OCL_ERRCK_RETVAL(clReleaseMemObject(tail));
  //Store the result into a file
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  FILE *fp = fopen(params->outFile,"w");
  fprintf(fp, "%d\n", num_of_nodes);
  int j = 0;
  for(j=0;j<num_of_nodes;j++)
    fprintf(fp,"%d %d\n",j,h_cost[j]);
  fclose(fp);
  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(color);
  free(h_cost);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}
