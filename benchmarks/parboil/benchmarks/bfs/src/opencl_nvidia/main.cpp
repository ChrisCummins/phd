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
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <parboil.h>
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
struct Node {
  int x;
  int y;
};
struct Edge {
  int x;
  int y;
};
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
//#include "kernel.cl"
//Somehow "cudaMemset" does not work. So I use cudaMemcpy of constant variables for initialization
const int h_top = 1;
const int zero = 0;

int BFS_GPU(cl_mem d_graph_nodes,cl_mem d_graph_edges, cl_mem d_color, cl_mem d_cost, cl_mem d_q1, cl_mem d_q2, cl_mem tail, int * source, cl_int clStatus, cl_command_queue clCommandQueue, cl_kernel BFS_kernel_S, cl_kernel BFS_kernel_M, cl_kernel BFS_kernel_L, cl_device_id clDevice, cl_context clContext){
}
void runGPU(int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
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
  //printf("Reading File\n");
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
  cl_platform_id clPlatform;
  OCL_ERRCK_RETVAL(clGetPlatformIDs(1,&clPlatform,NULL));
  cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
  OCL_ERRCK_RETVAL(clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_GPU,1,&clDevice,NULL));
  size_t MAX_THREADS_PER_BLOCK = 0; 
  clStatus = clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(MAX_THREADS_PER_BLOCK), &MAX_THREADS_PER_BLOCK, NULL);
  if(MAX_THREADS_PER_BLOCK > 512)
    MAX_THREADS_PER_BLOCK = 512;
  OCL_ERRCK_VAR(clStatus);

  int NUM_SM = 0;
  clStatus = clGetDeviceInfo(clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(NUM_SM), &NUM_SM, NULL);
  OCL_ERRCK_VAR(clStatus);

  cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  OCL_ERRCK_VAR(clStatus);

  pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("src/opencl_nvidia/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_nvidia -DMAX_THREADS_PER_BLOCK=%d -DNUM_SM=%d", MAX_THREADS_PER_BLOCK, NUM_SM);
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

  //Small kernel: only 1 block
  cl_kernel BFS_kernel_S = clCreateKernel(clProgram,"BFS_in_GPU_kernel",&clStatus);
  //Medium kernel: 1 block per SM
  cl_kernel BFS_kernel_M = clCreateKernel(clProgram,"BFS_kernel_multi_blk_inGPU",&clStatus);
  //Large kernel: No restriction
  cl_kernel BFS_kernel_L = clCreateKernel(clProgram,"BFS_kernel",&clStatus);
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

  cl_mem switch_kd, num_td, global_kt_d;
  switch_kd = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  num_td = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  global_kt_d = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  int switch_k;
  int global_kt = 0;
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,global_kt_d,CL_TRUE,0,sizeof(int),&global_kt,0,NULL,NULL));

  cl_mem count;
  cl_mem num_of_nodes_vol;
  cl_mem stay_vol;
  count = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  num_of_nodes_vol = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  stay_vol = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(int),NULL,&clStatus);
  OCL_ERRCK_VAR(clStatus);
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,count,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,num_of_nodes_vol,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,stay_vol,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));

  //BFS_kernel_S arguments setup
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,2,sizeof(cl_mem),(void*)&d_graph_nodes));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,3,sizeof(cl_mem),(void*)&d_graph_edges));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,4,sizeof(cl_mem),(void*)&d_color));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,5,sizeof(cl_mem),(void*)&d_cost));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,6,sizeof(cl_mem),(void*)&tail));

  //BFS_kernel_M arguments setup
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,2,sizeof(cl_mem),(void*)&d_graph_nodes));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,3,sizeof(cl_mem),(void*)&d_graph_edges));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,4,sizeof(cl_mem),(void*)&d_color));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,5,sizeof(cl_mem),(void*)&d_cost));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,6,sizeof(cl_mem),(void*)&num_td));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,7,sizeof(cl_mem),(void*)&tail));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,10,sizeof(cl_mem),(void*)&switch_kd));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,11,sizeof(cl_mem),(void*)&global_kt_d));
  //volatile mem
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,19,sizeof(cl_mem),(void*)&count));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,20,sizeof(cl_mem),(void*)&num_of_nodes_vol));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,21,sizeof(cl_mem),(void*)&stay_vol));

  //BFS_kernel_L arguments setup
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,2,sizeof(cl_mem),(void*)&d_graph_nodes));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,3,sizeof(cl_mem),(void*)&d_graph_edges));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,4,sizeof(cl_mem),(void*)&d_color));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,5,sizeof(cl_mem),(void*)&d_cost));
  OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,6,sizeof(cl_mem),(void*)&tail));
  do
  {
    OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,tail,CL_TRUE,0,sizeof(int),&num_t,0,NULL,NULL));
    OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,tail,CL_TRUE,0,sizeof(int),&zero,0,NULL,NULL));

    if(num_t == 0){//frontier is empty
      break;
    }

    num_of_blocks = 1;
    num_of_threads_per_block = num_t;
    if(num_of_threads_per_block <NUM_BIN)
      num_of_threads_per_block = NUM_BIN;
    if(num_t>MAX_THREADS_PER_BLOCK)
    {
      num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK); 
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }
    if(num_of_blocks == 1)//will call "BFS_in_GPU_kernel" 
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    if(num_of_blocks >1 && num_of_blocks <= NUM_SM)// will call "BFS_kernel_multi_blk_inGPU"
      num_of_blocks = NUM_SM;

    //assume "num_of_blocks" can not be very large
    size_t grid[1] = {num_of_blocks*num_of_threads_per_block};
    size_t block[1] = {num_of_threads_per_block};

    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,7,sizeof(int),(void*)&num_t));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,9,sizeof(int),(void*)&k));

    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,9,sizeof(int),(void*)&k));

    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,7,sizeof(int),(void*)&num_t));
    OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,9,sizeof(int),(void*)&k));

    if(k%2 == 0){
      int gray = GRAY0;
      if(num_of_blocks == 1) {
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,0,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,1,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,10,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,11,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,13,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,14,MAX_THREADS_PER_BLOCK*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,15,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_S,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
      } else if (num_of_blocks <= NUM_SM) {
        OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,num_td,CL_TRUE,0,sizeof(int),&num_t,0,NULL,NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,0,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,1,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,13,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,14,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,15,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,16,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,17,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,18,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_M,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
        OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,switch_kd,CL_TRUE,0,sizeof(int),&switch_k,0,NULL,NULL));
        if(!switch_k){
          k--;
        }
      } else {
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,0,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,1,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,10,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,11,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,13,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,14,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_L,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
      }
    }
    else {
      int gray = GRAY1;
      if(num_of_blocks == 1) {
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,0,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,1,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,10,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,11,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,13,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,14,MAX_THREADS_PER_BLOCK*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_S,15,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_S,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
      } else if (num_of_blocks <= NUM_SM) {
        OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue,num_td,CL_TRUE,0,sizeof(int),&num_t,0,NULL,NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,0,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,1,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,13,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,14,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,15,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,16,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,17,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_M,18,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_M,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
        OCL_ERRCK_RETVAL(clEnqueueReadBuffer(clCommandQueue,switch_kd,CL_TRUE,0,sizeof(int),&switch_k,0,NULL,NULL));
        if(!switch_k){
          k--;
        }
      } else {
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,0,sizeof(cl_mem),(void*)&d_q2));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,1,sizeof(cl_mem),(void*)&d_q1));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,8,sizeof(int),(void*)&gray));
        //shared_mem
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,10,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,11,NUM_BIN*W_QUEUE_SIZE*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,12,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,13,NUM_BIN*sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clSetKernelArg(BFS_kernel_L,14,sizeof(int),NULL));
        OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue,BFS_kernel_L,1,0,grid,block,0,0,0));
        OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
      }
    }
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
