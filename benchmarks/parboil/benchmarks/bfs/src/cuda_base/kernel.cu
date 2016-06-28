/***********************************************************************************
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
 ************************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_
/**********
Define colors for BFS
1) the definition of White, gray and black comes from the text book "Introduction to Algorithms"
2) For path search problems, people may choose to use different colors to record the found paths.
Therefore we reserve numbers (0-16677216) for this purpose. Only nodes with colors bigger than
UP_LIMIT are free to visit 
3) We define two gray shades to differentiate between the new frontier nodes and the old frontier nodes that
 have not been marked BLACK 
*************/

#define UP_LIMIT 16677216//2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221

#include "config.h"
texture<Node> g_graph_node_ref;
texture<Edge> g_graph_edge_ref;

volatile __device__ int count = 0;
volatile __device__ int no_of_nodes_vol = 0;
volatile __device__ int stay_vol = 0;

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
__global__ void
BFS_kernel(int * q1, 
           int * q2, 
           Node* g_graph_nodes, 
           Edge* g_graph_edges, 
           int* g_color, 
           int * g_cost, 
           int no_of_nodes, 
           int * tail, 
           int gray_shade, 
           int k) 
{
  __shared__ int local_q_tail;//the tails of each local warp-level queue
  __shared__ int local_q[NUM_BIN*W_QUEUE_SIZE];//the local warp-level queues 
  //current w-queue, a.k.a prefix sum
  __shared__ int shift;

  if(threadIdx.x == 0){
    local_q_tail = 0;//initialize the tail of w-queue
  }
  __syncthreads();

  //first, propagate and add the new frontier elements into w-queues
  int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
  if( tid<no_of_nodes)
  {
    int pid = q1[tid]; //the current frontier node, or the parent node of the new frontier nodes 
    g_color[pid] = BLACK;
    int cur_cost = g_cost[pid];
    //into
    Node cur_node = tex1Dfetch(g_graph_node_ref,pid);
    for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++)//visit each neighbor of the
      //current frontier node.
    {
      Edge cur_edge = tex1Dfetch(g_graph_edge_ref,i);
      int id = cur_edge.x;
      int cost = cur_edge.y;
      cost += cur_cost;
      int orig_cost = atomicMin(&g_cost[id],cost);
      if(orig_cost > cost){//the node should be visited
        if(g_color[id] > UP_LIMIT){
          int old_color = atomicExch(&g_color[id],gray_shade);
          //this guarantees that only one thread will push this node
          //into a queue
          if(old_color != gray_shade) {

            //atomic operation guarantees the correctness
            //even if multiple warps are executing simultaneously
            int index = atomicAdd(&local_q_tail,1);
            local_q[index] = id;
          }
        }
      }
    }
  }
  __syncthreads();

  if(threadIdx.x == 0){
    int tot_sum = local_q_tail; 

    //the offset or "shift" of the block-level queue within the grid-level queue
    //is determined by atomic operation
    shift = atomicAdd(tail,tot_sum);
  }
  __syncthreads();


  int local_shift = threadIdx.x;//shift within a w-queue

  //loop unrolling was originally used for better performance, but removed for better readability
  while(local_shift < local_q_tail){
    q2[shift + local_shift] = local_q[local_shift];
    local_shift += blockDim.x;//multiple threads are copying elements at the same time,
    //so we shift by multiple elements for next iteration  
  }
}
#endif 
