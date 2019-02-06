/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
 
#define BLOCK_SIZE 1024
#define GRID_SIZE 65535
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

//#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#define LNB LOG_NUM_BANKS
#define CONFLICT_FREE_OFFSET(index) (((unsigned int)(index) >> min((unsigned int)(LNB)+(index), (unsigned int)(32-(2*LNB))))>>(2*LNB))
#define EXPANDED_SIZE(__x) (__x+(__x>>LOG_NUM_BANKS)+(__x>>(2*LOG_NUM_BANKS)))

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
__kernel void scan_L1_kernel(unsigned int n, __global unsigned int* dataBase, unsigned int data_offset, __global unsigned int* interBase, unsigned int inter_offset)
{
    __local unsigned int s_data[EXPANDED_SIZE(BLOCK_SIZE)]; 
    
    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;

    unsigned int thid = get_local_id(0);
    unsigned int g_ai = get_group_id(0)*2*get_local_size(0) + get_local_id(0);
    unsigned int g_bi = g_ai + get_local_size(0);

    unsigned int s_ai = thid;
    unsigned int s_bi = thid + get_local_size(0);

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = (g_ai < n) ? data[g_ai] : 0;
    s_data[s_bi] = (g_bi < n) ? data[g_bi] : 0;

    unsigned int stride = 1;
    for (unsigned int d = get_local_size(0); d > 0; d >>= 1) {

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        unsigned int i  = 2*stride*thid;
        unsigned int ai = i + stride - 1;
        unsigned int bi = ai + stride;

        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        s_data[bi] += s_data[ai];
      }

        stride *= 2;
    }

    if (thid == 0) {
      unsigned int last = get_local_size(0)*2 -1;
      last += CONFLICT_FREE_OFFSET(last);
      inter[get_group_id(0)] = s_data[last];
      s_data[last] = 0;
    }

    for (unsigned int d = 1; d <= get_local_size(0); d *= 2) {
      stride >>= 1;

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        unsigned int i  = 2*stride*thid;
        unsigned int ai = i + stride - 1;
        unsigned int bi = ai + stride;

        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        unsigned int t  = s_data[ai];
        s_data[ai] = s_data[bi];
        s_data[bi] += t;
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    if (g_ai < n) { data[g_ai] = s_data[s_ai]; }
    if (g_bi < n) { data[g_bi] = s_data[s_bi]; }
}



__kernel void scan_inter1_kernel(__global unsigned int* data, unsigned int iter)
{
    __local unsigned int s_data[DYN_LOCAL_MEM_SIZE];

    unsigned int thid = get_local_id(0);
    unsigned int gthid = get_global_id(0);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

    unsigned int stride = 1;
    for (unsigned int d = get_local_size(0); d > 0; d >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        unsigned int i  = 2*stride*thid;
        unsigned int ai = i + stride - 1;
        unsigned int bi = ai + stride;

        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        s_data[bi] += s_data[ai];
      }

      stride *= 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}

__kernel void scan_inter2_kernel(__global unsigned int* data, unsigned int iter)
{
    __local unsigned int s_data[DYN_LOCAL_MEM_SIZE];

    unsigned int thid = get_local_id(0);
    unsigned int gthid = get_global_id(0);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

    unsigned int stride = get_local_size(0)*2;

    for (unsigned int d = 1; d <= get_local_size(0); d *= 2) {
      stride >>= 1;

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        unsigned int i  = 2*stride*thid;
        unsigned int ai = i + stride - 1;
        unsigned int bi = ai + stride;

        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        unsigned int t  = s_data[ai];
        s_data[ai] = s_data[bi];
        s_data[bi] += t;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}


__kernel void uniformAdd(unsigned int n, __global unsigned int *dataBase, unsigned int data_offset, __global unsigned int *interBase, unsigned int inter_offset)
{
    __local unsigned int uni;
    
    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;
       
    if (get_local_id(0) == 0) { uni = inter[get_group_id(0)]; }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    unsigned int g_ai = get_group_id(0)*2*get_local_size(0) + get_local_id(0);
    unsigned int g_bi = g_ai + get_local_size(0);

    if (g_ai < n) { data[g_ai] += uni; }
    if (g_bi < n) { data[g_bi] += uni; }
}
