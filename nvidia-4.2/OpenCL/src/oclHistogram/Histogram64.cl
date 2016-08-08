/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


////////////////////////////////////////////////////////////////////////////////
// Common definition
////////////////////////////////////////////////////////////////////////////////
#define HISTOGRAM64_BIN_COUNT 64

//Data type used for input data fetches
typedef uint4 data_t;

//Both map to a single instruction on G8x / G9x / G10x
#define UMUL(a, b)    ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

//Passed on clBuildProgram
//#define LOCAL_MEMORY_BANKS 16 (default)

//Passed on clBuildProgram
//must be a multiple of (4 * LOCAL_MEMORY_BANKS) because of the bit permutation of local ID
//#define HISTOGRAM64_WORKGROUP_SIZE (4 * LOCAL_MEMORY_BANKS)



////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial histograms
////////////////////////////////////////////////////////////////////////////////
inline void addByte(__local uchar *l_WorkitemBase, uint data){
    l_WorkitemBase[UMUL(data, HISTOGRAM64_WORKGROUP_SIZE)]++;
}

inline void addWord(__local uchar *l_WorkitemBase, uint data){
    //Only higher 6 bits of each byte matter, as this is a 64-bin histogram
    addByte(l_WorkitemBase, (data >>  2) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 10) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 18) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 26) & 0x3FU);
}

__kernel __attribute__((reqd_work_group_size(HISTOGRAM64_WORKGROUP_SIZE, 1, 1)))
void histogram64(
    __global uint *d_PartialHistograms,
    __global data_t *d_Data,
    uint dataCount
){
    //Encode local id in order to avoid bank conflicts at l_Hist[] accesses:
    //each group of LOCAL_MEMORY_BANKS work-items accesses consecutive local memory banks
    //and the same bytes [0..3] within the banks
    //Because of this permutation workgroup size should be a multiple of 4 * LOCAL_MEMORY_BANKS
    const uint lPos = 
        ( (get_local_id(0) & ~(LOCAL_MEMORY_BANKS * 4 - 1)) << 0 ) |
        ( (get_local_id(0) &  (LOCAL_MEMORY_BANKS     - 1)) << 2 ) |
        ( (get_local_id(0) &  (LOCAL_MEMORY_BANKS * 3    )) >> 4 );

    //Work-item subhistogram storage
    __local uchar l_Hist[HISTOGRAM64_WORKGROUP_SIZE * HISTOGRAM64_BIN_COUNT];
    __local uchar *l_WorkitemBase = l_Hist + lPos;

    //Initialize local memory (writing 32-bit words)
    for(uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++)
        ((__local uint *)l_Hist)[lPos + i * HISTOGRAM64_WORKGROUP_SIZE] = 0;

    //Read data from global memory and submit to the local-memory subhistogram storage
    //Since histogram counters are byte-sized, every single work-item can't do more than 255 submission
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint pos = get_global_id(0); pos < dataCount; pos += get_global_size(0)){
        data_t data = d_Data[pos];
        addWord(l_WorkitemBase, data.x);
        addWord(l_WorkitemBase, data.y);
        addWord(l_WorkitemBase, data.z);
        addWord(l_WorkitemBase, data.w);
    }

    //Merge work-item subhistograms into work-group partial histogram
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) < HISTOGRAM64_BIN_COUNT){
        __local uchar *l_HistBase = l_Hist + UMUL(get_local_id(0), HISTOGRAM64_WORKGROUP_SIZE);

        uint sum = 0;
        uint pos = 4 * (get_local_id(0) & (LOCAL_MEMORY_BANKS - 1));
        for(uint i = 0; i < (HISTOGRAM64_WORKGROUP_SIZE / 4); i++){
            sum += 
                l_HistBase[pos + 0] + 
                l_HistBase[pos + 1] + 
                l_HistBase[pos + 2] + 
                l_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_WORKGROUP_SIZE - 1);
        }

        d_PartialHistograms[get_group_id(0) * HISTOGRAM64_BIN_COUNT + get_local_id(0)] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge histogram64() output
// Run one workgroup per bin; each workgroup adds up the same bin counter 
// from every partial histogram. Reads are uncoalesced, but mergeHistogram64
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
//Passed down on clBuildProgram
//#define MERGE_WORKGROUP_SIZE 256

__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void mergeHistogram64(
    __global uint *d_Histogram,
    __global uint *d_PartialHistograms,
    uint histogramCount
){
    __local uint l_Data[MERGE_WORKGROUP_SIZE];

    uint sum = 0;
    for(uint i = get_local_id(0); i < histogramCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialHistograms[get_group_id(0) + i * HISTOGRAM64_BIN_COUNT];
    l_Data[get_local_id(0)] = sum;

    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) < stride)
            l_Data[get_local_id(0)] += l_Data[get_local_id(0) + stride];
    }

    if(get_local_id(0) == 0)
        d_Histogram[get_group_id(0)] = l_Data[0];
}
