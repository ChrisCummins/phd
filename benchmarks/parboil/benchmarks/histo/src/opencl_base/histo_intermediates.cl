/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

__kernel void calculateBin (
        __const unsigned int bin,
        __global uchar4 *sm_mapping)
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        uchar4 sm;
        sm.x = block;
        sm.y = indexhi;
        sm.z = indexlo;
        sm.w = offset;

        *sm_mapping = sm;
}

__kernel void histo_intermediates_kernel (
        __global uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        __global uchar4 *sm_mappings)
{
        int threadIdxx = get_local_id(0);
        int blockDimx = get_local_size(0);
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint2 *load_bin = input + line * input_pitch + threadIdxx;

        unsigned int store = line * width + threadIdxx;
        bool skip = (width % 2) && (threadIdxx == (blockDimx - 1));

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;

                calculateBin (
                        bin_value.x,
                        &sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        &sm_mappings[store + blockDimx]
                );

                load_bin += input_pitch;
                store += width;
        }
}

__kernel void histo_intermediates_kernel_compat (
        __global uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        __global uchar4 *sm_mappings)
{
        int threadIdxx = get_local_id(0);
        int blockDimx = input_pitch; //get_local_size(0);
        
        int tid2 = get_local_id(0) + get_local_size(0);
        
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint2 *load_bin = input + line * input_pitch + threadIdxx;
        __global uint2 *load_bin2 = input + line * input_pitch + tid2;

        unsigned int store = line * width + threadIdxx;
        unsigned int store2 = line * width + tid2;
        
        bool skip = (width % 2) && (threadIdxx == (input_pitch - 1));
        bool skip2 = (width % 2) && (tid2 == (input_pitch - 1));

        bool does2 = tid2 < input_pitch;

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;


                calculateBin (
                        bin_value.x,
                        &sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        &sm_mappings[store + blockDimx]
                );

                load_bin += input_pitch;
                store += width;
                
                if (does2) {
                  uint2 bin_val2 = *load_bin2;
                                
                  calculateBin (
                        bin_val2.x,
                        &sm_mappings[store2]
                  );

                  if (!skip) calculateBin (
                        bin_val2.y,
                        &sm_mappings[store2 + blockDimx]
                  );

                  load_bin2 += input_pitch;
                  store2 += width;
                }
        }
        
        /*
        if (does2) {
          #pragma unroll
          for (int i = 0; i < UNROLL; i++) {
            uint2 bin_val2 = *load_bin2;
                                
            calculateBin (
                bin_val2.x,
                &sm_mappings[store2]
            );

            if (!skip) calculateBin (
                           bin_val2.y,
                           &sm_mappings[store2 + blockDimx]
                        );

            load_bin2 += input_pitch;
            store2 += width;
          }
       }
     */
        
}


/*
__kernel void histo_intermediates_kernel_variable (
        __global uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch, // = half_width in orig
        unsigned int stride,
        __global uchar4 *sm_mappings)
{
        int threadIdxx = get_local_id(0);
        int blockDimx = get_local_size(0);
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint2 *load_bin = input + line * input_pitch + threadIdxx;

        unsigned int store = line * width + threadIdxx;
        bool skip = (width % 2) && (threadIdxx == (blockDimx - 1));

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;

                calculateBin (
                        bin_value.x,
                        &sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        &sm_mappings[store + blockDimx]
                );

                load_bin += input_pitch;
                store += width;
        }
}
*/
