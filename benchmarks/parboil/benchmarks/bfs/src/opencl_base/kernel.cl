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
/**********
Define colors for BFS
1) the definition of White, gray and black comes from the text book "Introduction to Algorithms"
2) For path search problems, people may choose to use different colors to record the found paths.
Therefore we reserve numbers (0-16677216) for this purpose. Only nodes with colors bigger than
UP_LIMIT are free to visit 
3) We define two gray shades to differentiate between the new frontier nodes and the old frontier nodes that
 have not been marked BLACK 
*************/

#include "config.h"
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
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
           __global struct Node *g_graph_nodes,
           __global struct Edge *g_graph_edges,
           __global int *g_color,
           __global int *g_cost, 
           __global int *tail, 
           int no_of_nodes,
           int gray_shade, 
           int k , 
           __local int *local_q_tail, 
           __local int *local_q, 
           __local int *shift) 
{

  if(get_local_id(0) == 0){
    *local_q_tail = 0;//initialize the tail of w-queue
  }
  barrier(CLK_LOCAL_MEM_FENCE); 

  //first, propagate and add the new frontier elements into w-queues
  //int tid = get_group_id(0)*MAX_THREADS_PER_BLOCK + get_local_id(0);
  int tid = get_global_id(0);

  if( tid<no_of_nodes)
  {
    int pid = q1[tid]; //the current frontier node, or the parent node of the new frontier nodes 
    g_color[pid] = BLACK;
    int cur_cost = g_cost[pid];
    //into
    struct Node cur_node = g_graph_nodes[pid];

    for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++)//visit each neighbor of the
      //current frontier node.
    {
      struct Edge cur_edge = g_graph_edges[i];
      int id = cur_edge.x;
      int cost = cur_edge.y;
      cost += cur_cost;
      int orig_cost = atom_min (&g_cost[id],cost);
      if(orig_cost > cost){//the node should be visited
        if(g_color[id] > UP_LIMIT){
         int old_color = atom_xchg (&g_color[id],gray_shade);
          //this guarantees that only one thread will push this node
          //into a queue
          if(old_color != gray_shade) {
            //atomic operation guarantees the correctness
            //even if multiple warps are executing simultaneously
            int index = atom_add (local_q_tail,1);
            local_q[index] = id;
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(get_local_id(0) == 0){
    int tot_sum = *local_q_tail; 
    //the offset or "shift" of the block-level queue within the grid-level queue
    //is determined by atomic operation
    *shift = atom_add (tail,tot_sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //shift within a w-queue
  int local_shift = get_local_id(0);

  while(local_shift < *local_q_tail){
    q2[*shift + local_shift] = local_q[local_shift];
    //multiple threads are copying elements at the same time,
    //so we shift by multiple elements for next iteration  
    local_shift += get_local_size(0);
  }
}
