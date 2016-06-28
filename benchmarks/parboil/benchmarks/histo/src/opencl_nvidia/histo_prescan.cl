/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void histo_prescan_kernel (__global unsigned int* input, int size, __global unsigned int* minmax)
{

    __local float Avg[PRESCAN_THREADS];
    __local float StdDev[PRESCAN_THREADS];

    int threadIdxx = get_local_id(0);
    int blockDimx = get_local_size(0);
    int blockIdxx = get_group_id(0);
    int stride = size/(get_num_groups(0));
    int addr = blockIdxx*stride+threadIdxx;
    int end = blockIdxx*stride + stride/8; // Only sample 1/8th of the input data

    // Compute the average per thread
    float avg = 0.0;
    unsigned int count = 0;
    while (addr < end){
        avg += input[addr];
        count++;
	addr += blockDimx;
    }
    avg /= count;
    Avg[threadIdxx] = avg;

    // Compute the standard deviation per thread
    int addr2 = blockIdxx*stride+threadIdxx;
    float stddev = 0;
    while (addr2 < end){
        stddev += (input[addr2]-avg)*(input[addr2]-avg);
        addr2 += blockDimx;
    }
    stddev /= count;
    StdDev[threadIdxx] = sqrt(stddev);

#define SUM(stride__)\
if(threadIdxx < stride__){\
    Avg[threadIdxx] += Avg[threadIdxx+stride__];\
    StdDev[threadIdxx] += StdDev[threadIdxx+stride__];\
}

    // Add all the averages and standard deviations from all the threads
    // and take their arithmetic average (as a simplified approximation of the
    // real average and standard deviation.
#if (PRESCAN_THREADS >= 32)    
    for (int stride = PRESCAN_THREADS/2; stride >= 32; stride = stride >> 1){
	barrier(CLK_LOCAL_MEM_FENCE);
	SUM(stride);
    }
#endif
#if (PRESCAN_THREADS >= 16)
    SUM(16);
#endif
#if (PRESCAN_THREADS >= 8)
    SUM(8);
#endif
#if (PRESCAN_THREADS >= 4)
    SUM(4);
#endif
#if (PRESCAN_THREADS >= 2)
    SUM(2);
#endif

    if (threadIdxx == 0){
        float avg = Avg[0]+Avg[1];
	avg /= PRESCAN_THREADS;
	float stddev = StdDev[0]+StdDev[1];
	stddev /= PRESCAN_THREADS;

        // Take the maximum and minimum range from all the blocks. This will
        // be the final answer. The standard deviation is taken out to 10 sigma
        // away from the average. The value 10 was obtained empirically.
	    atom_min(minmax,((unsigned int)(avg-10*stddev))/(KB*1024));
        atom_max(minmax+1,((unsigned int)(avg+10*stddev))/(KB*1024));
    }  
}
