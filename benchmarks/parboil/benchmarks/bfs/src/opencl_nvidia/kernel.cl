/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
  Implementing Breadth first search in OpenCL using algorithm given in DAC'10
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
#ifndef _KERNEL_H_
#define _KERNEL_H_
/*
Define colors for BFS
1) the definition of White, gray and black comes from the text book "Introduction to Algorithms"
2) For path search problems, people may choose to use different colors to record the found paths.
Therefore we reserve numbers (0-16677216) for this purpose. Only nodes with colors bigger than
UP_LIMIT are free to visit 
3) We define two gray shades to differentiate between the new frontier nodes and the old frontier nodes that
 have not been marked BLACK 
*/

#include "config.h"
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable

//Inter-block sychronization
//This only works when there is only one block per SM
void start_global_barrier(int fold, __global volatile int* count){
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  if(get_local_id(0) == 0){
    atom_add(count, 1);
    int count_val = atom_or(count,0);
    while(count_val < NUM_SM*fold){
      count_val = atom_or(count,0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}
//-------------------------------------------------
//This is the version for one-block situation. The propagation idea is basically the same as
//BFS_kernel.
//The major differences are:
// 1) This kernel can propagate though multiple BFS levels (while loop) using __synchThreads() between levels 
// 2) the intermediate queues are stored in shared memory (next_wf) 
//\param q1: the current frontier queue when the kernel is launched
//\param q2: the new frontier queue when the  kernel returns
//--------------------------------------------------
__kernel void
BFS_in_GPU_kernel(__global int *q1,
                  __global int *q2,
                  __global int2 *g_graph_nodes,
                  __global int2 *g_graph_edges,
                  __global int *g_color,
                  __global int *g_cost, 
                  __global int *tail, 
                  int no_of_nodes, 
                  int gray_shade, 
                  int k,
                  __local int *local_q_tail,
                  __local int *local_q, 
                  __local int *prefix_q,
                  __local int *thread_n_q,
                  __local int *next_wf,
                  __local int *tot_sum) 
{
  //next/new wave front
  if(get_local_id(0) == 0)	
    *tot_sum = 0;//total number of new frontier nodes
  while(1){//propage through multiple BFS levels until the wavfront overgrows one-block limit
    if(get_local_id(0) < NUM_BIN){
      local_q_tail[get_local_id(0)] = 0;
      thread_n_q[get_local_id(0)] = get_local_size(0)>>EXP;
      if((get_local_size(0)&MOD_OP) > get_local_id(0)){
        thread_n_q[get_local_id(0)]++;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    int tid = get_group_id(0)*MAX_THREADS_PER_BLOCK + get_local_id(0);
    if( tid<no_of_nodes)
    {
      int pid;
      if(*tot_sum == 0)//this is the first BFS level of current kernel call
        pid = q1[tid];  
      else
        pid = next_wf[tid];//read the current frontier info from last level's propagation
      g_color[pid] = BLACK;
      int cur_cost = g_cost[pid];
      int q_i = get_local_id(0)&MOD_OP; 
      int2 cur_node = g_graph_nodes[pid];
      for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++) {
        int2 cur_edge = g_graph_edges[i];
        int id = cur_edge.x;
        int cost = cur_edge.y;
        cost += cur_cost;
        int orig_cost = atom_min(&g_cost[id],cost);
        if(orig_cost > cost){
          int old_color = atom_xchg(&g_color[id],gray_shade);
          if(old_color != gray_shade) {
            //push to the queue
            int index = atom_add(&local_q_tail[q_i],1);
            local_q[q_i*W_QUEUE_SIZE+index] = id;
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if(get_local_id(0) == 0){
      prefix_q[0] = 0;
      for(int i = 1; i < NUM_BIN; i++){
        prefix_q[i] = prefix_q[i-1]+local_q_tail[i-1];
      }
      *tot_sum = prefix_q[NUM_BIN-1] + local_q_tail[NUM_BIN-1];
      *tail = *tot_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    int q_i = get_local_id(0)&MOD_OP;
    int local_shift = get_local_id(0)>>EXP;
    if(*tot_sum == 0)//the new frontier becomes empty; BFS is over
      return;
    if(*tot_sum <= MAX_THREADS_PER_BLOCK){//the new frontier is still within one-block limit;
      //stay in current kernel
      while (local_shift < local_q_tail[q_i]){
        next_wf[prefix_q[q_i]+local_shift] = local_q[q_i*W_QUEUE_SIZE+local_shift];
        local_shift += thread_n_q[q_i];
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      no_of_nodes = *tot_sum;
      if(get_local_id(0) == 0){
        if(gray_shade == GRAY0)
          gray_shade = GRAY1;
        else
          gray_shade = GRAY0;
      }
    }
    else{//the new frontier outgrows one-block limit; terminate current kernel
      while(local_shift < local_q_tail[q_i]){
        q2[prefix_q[q_i]+local_shift] = local_q[q_i*W_QUEUE_SIZE+local_shift];
        local_shift += thread_n_q[q_i];
      }
      return;
    }
  }//while
}	
//----------------------------------------------------------------
//This BFS kernel propagates through multiple levels using global synchronization 
//The basic propagation idea is the same as "BFS_kernel"
//The major differences are:
// 1) propagate through multiple levels by using GPU global sync ("start_global_barrier")
// 2) use q1 and q2 alternately for the intermediate queues
//\param q1: the current frontier when the kernel is called
//\param q2: possibly the new frontier when the kernel returns depending on how many levels of propagation
//           has been done in current kernel; the new frontier could also be stored in q1
//\param switch_k: whether or not to adjust the "k" value on the host side
//                Normally on the host side, when "k" is even, q1 is the current frontier; when "k" is
//                odd, q2 is the current frontier; since this kernel can propagate through multiple levels,
//                the k value may need to be adjusted when this kernel returns.
//\param global_kt: the total number of global synchronizations, 
//                   or the number of times to call "start_global_barrier" 
//--------------------------------------------------------------
__kernel void
BFS_kernel_multi_blk_inGPU(__global int *q1,
                           __global int *q2,
                           __global int2 *g_graph_nodes,
                           __global int2 *g_graph_edges,
                           __global int *g_color,
                           __global int *g_cost,
                           __global int *no_of_nodes, 
                           __global int *tail, 
                           int gray_shade, 
                           int k,
                           __global int *switch_k,
                           __global int *global_kt,
                           __local int*local_q_tail,
                           __local int*local_q, 
                           __local int*prefix_q,
                           __local int*thread_n_q,
                           __local int*shift,
                           __local int*no_of_nodes_sm,
                           __local int*odd_time,
                           __global volatile int *count,
                           __global volatile int *no_of_nodes_vol,
                           __global volatile int *stay_vol) 
{
  if(get_local_id(0) == 0){
    *odd_time = 1;//true;
    if(get_group_id(0) == 0)
      *no_of_nodes_vol = *no_of_nodes;
  }
  int kt = atom_or(global_kt, 0);// the total count of GPU global synchronization 
  while (1){//propagate through multiple levels
    if(get_local_id(0) < NUM_BIN){
      local_q_tail[get_local_id(0)] = 0;
      thread_n_q[get_local_id(0)] = get_local_size(0)>>EXP;
      if((get_local_size(0)&MOD_OP) > get_local_id(0)){
        thread_n_q[get_local_id(0)]++;
      }
    }
    if(get_local_id(0) == 0)
      *no_of_nodes_sm =  *no_of_nodes_vol; 
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    int tid = get_group_id(0)*MAX_THREADS_PER_BLOCK + get_local_id(0);
    if( tid<*no_of_nodes_sm)
    {
      int pid;
      if(*odd_time == 1)
        pid = atom_or(&q1[tid], 0);  
      else
        pid = atom_or(&q2[tid], 0);
      g_color[pid] = BLACK;
      int cur_cost = atom_or(&g_cost[pid], 0);
      int q_i = get_local_id(0)&MOD_OP; 
      int2 cur_node = g_graph_nodes[pid];
      for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++) {
        int2 cur_edge = g_graph_edges[i];
        int id = cur_edge.x;
        int cost = cur_edge.y;
        cost += cur_cost;
        int orig_cost = atom_min(&g_cost[id],cost);
        if(orig_cost > cost){
          if(g_color[id] > UP_LIMIT){
            int old_color = atom_xchg(&g_color[id],gray_shade);
            if(old_color != gray_shade)
            {
              //push to the queue
              int index = atom_add(&local_q_tail[q_i],1);
              //if(q_i*W_QUEUE_SIZE + index >= NUM_BIN*W_QUEUE_SIZE)
              //  local_q_tail[-1] = 0;
              local_q[q_i*W_QUEUE_SIZE+index] = id;
            }
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if(get_local_id(0) == 0){
      prefix_q[0] = 0;
      for(int i = 1; i < NUM_BIN; i++){
        prefix_q[i] = prefix_q[i-1] + local_q_tail[i-1];
      }
      int tot_sum = prefix_q[NUM_BIN-1] + local_q_tail[NUM_BIN-1];
      *shift = atom_add(tail,tot_sum);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    int q_i = get_local_id(0)&MOD_OP;
    int local_shift = get_local_id(0)>>EXP;
    while (local_shift < local_q_tail[q_i]){
      if(*odd_time)
        q2[*shift+prefix_q[q_i]+local_shift] = local_q[q_i*W_QUEUE_SIZE+local_shift];
      else
        q1[*shift+prefix_q[q_i]+local_shift] = local_q[q_i*W_QUEUE_SIZE+local_shift];
      local_shift += thread_n_q[q_i];
    }
    if(get_local_id(0) == 0){
      *odd_time = (*odd_time+1)%2;
      if(gray_shade == GRAY0)
        gray_shade = GRAY1;
      else
        gray_shade = GRAY0;
    }

    //synchronize among all the blks
    start_global_barrier(kt+1, count);
    if(get_group_id(0) == 0 && get_local_id(0) == 0){
      *stay_vol = 0;
      if(*tail< NUM_SM*MAX_THREADS_PER_BLOCK && *tail > MAX_THREADS_PER_BLOCK){
        *stay_vol = 1;
        *no_of_nodes_vol = *tail;
        *tail = 0;
      }
    }
    start_global_barrier(kt+2, count);
    kt+= 2;
    if(*stay_vol == 0)
    {
      if(get_group_id(0) == 0 && get_local_id(0) == 0)
      {
        *global_kt = kt;
        *switch_k = (*odd_time+1)%2;
        *no_of_nodes = *no_of_nodes_vol;
      }
      return;
    }
  }
}

/*****************************************************************************
  This is the  most general version of BFS kernel, i.e. no assumption about #block in the grid  
  \param q1: the array to hold the current frontier
  \param q2: the array to hold the new frontier
  \param g_graph_nodes: the nodes in the input graph
  \param g_graph_edges: the edges i nthe input graph
  \param g_color: the colors of nodes
  \param g_cost: the costs of nodes
  \param no_of_nodes: the number of nodes in the current frontier
  \param tail: pointer to the location of the tail of the new frontier. *tail is the size of the new frontier 
  \param gray_shade: the shade of the gray in current BFS propagation. See GRAY0, GRAY1 macro definitions for more details
  \param k: the level of current propagation in the BFS tree. k= 0 for the first propagation.
 ***********************************************************************/
  __kernel void
BFS_kernel(__global int *q1,
           __global int *q2,
           __global int2 *g_graph_nodes,
           __global int2 *g_graph_edges,
           __global int *g_color,
           __global int *g_cost, 
           __global int *tail, 
           int no_of_nodes, 
           int gray_shade, 
           int k,
           __local int *local_q_tail,
           __local int *local_q,
           __local int *prefix_q,
           __local int *thread_n_q,
           __local int *shift)
{
  if(get_local_id(0) < NUM_BIN){
    local_q_tail[get_local_id(0)] = 0;//initialize the tail of w-queue
    thread_n_q[get_local_id(0)] = get_local_size(0)>>EXP;//#thread/NUM_BIN
    if((get_local_size(0)&MOD_OP) > get_local_id(0)){//#thread%NUM_BIN > get_local_id(0)
      thread_n_q[get_local_id(0)]++;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  //first, propagate and add the new frontier elements into w-queues
  int tid = get_group_id(0)*MAX_THREADS_PER_BLOCK + get_local_id(0);
  if( tid<no_of_nodes)
  {
    int pid = q1[tid]; //the current frontier node, or the parent node of the new frontier nodes 
    g_color[pid] = BLACK;
    int cur_cost = g_cost[pid];
    int q_i = get_local_id(0)&MOD_OP; //the id of the queue which new frontier nodes will be pushed
    //into
    int2 cur_node = g_graph_nodes[pid];
    for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++)//visit each neighbor of the
      //current frontier node.
    {
      int2 cur_edge = g_graph_edges[i];
      int id = cur_edge.x;
      int cost = cur_edge.y;
      cost += cur_cost;
      int orig_cost = atom_min(&g_cost[id],cost);
      if(orig_cost > cost){//the node should be visited
        if(g_color[id] > UP_LIMIT){
          int old_color = atom_xchg(&g_color[id],gray_shade);
          //this guarantees that only one thread will push this node
          //into a queue
          if(old_color != gray_shade) {

            //atomic operation guarantees the correctness
            //even if multiple warps are executing simultaneously
            int index = atom_add(&local_q_tail[q_i],1);
            local_q[q_i*W_QUEUE_SIZE+index] = id;
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  if(get_local_id(0) == 0){
    //now calculate the prefix sum
    prefix_q[0] = 0;
    for(int i = 1; i < NUM_BIN; i++){
      //the prefix sum of one queue is equal to the prefix sum of its predecessor queue
      //plus the number of elements in the predecessor queue
      prefix_q[i] = prefix_q[i-1]+local_q_tail[i-1];
    }
    //the total number of elements in the block-level queue is the prefix sum of the last w-queue
    //plus the number of elements in the last w-queue
    int tot_sum = prefix_q[NUM_BIN-1] + local_q_tail[NUM_BIN-1];

    //the offset or "shift" of the block-level queue within the grid-level queue
    //is determined by atomic operation
    *shift = atom_add(tail,tot_sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  //now copy the elements from w-queues into grid-level queues.
  //Note that we have bypassed the copy to/from block-level queues for efficiency reason

  int q_i = get_local_id(0)&MOD_OP;//w-queue index
  int local_shift = get_local_id(0)>>EXP;//shift within a w-queue

  //loop unrolling was originally used for better performance, but removed for better readability
  while(local_shift < local_q_tail[q_i]){
    q2[*shift+prefix_q[q_i]+local_shift] = local_q[q_i*W_QUEUE_SIZE+local_shift];
    local_shift+= thread_n_q[q_i];//multiple threads are copying elements at the same time,
    //so we shift by multiple elements for next iteration  
  }
}
#endif 
