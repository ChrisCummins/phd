/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/* Combine all the sub-histogram results into one final histogram */
__kernel void histo_final_kernel (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    __global unsigned int *global_subhisto,
    __global unsigned int *global_histo,
    __global unsigned int *global_overflow,
    __global unsigned int *final_histo) //final output
{
    unsigned int blockDimx = get_local_size(0);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int start_offset = get_local_id(0) + get_group_id(0) * blockDimx;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDimx * blockDimx)
    {
        ushort4 global_histo_data = ((__global ushort4*)global_histo)[i];
        ((__global ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, (ushort) 255);
        global_histo_data.y = min (global_histo_data.y, (ushort) 255);
        global_histo_data.z = min (global_histo_data.z, (ushort) 255);
        global_histo_data.w = min (global_histo_data.w, (ushort) 255);

        uchar4 final_histo_data = (uchar4) (
            (unsigned char) global_histo_data.x,
            (unsigned char) global_histo_data.y,
            (unsigned char) global_histo_data.z,
            (unsigned char) global_histo_data.w
        );

        ((__global uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDimx * blockDimx)
    {
        uint4 global_histo_data = ((__global uint4*)global_overflow)[i];
        ((__global uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = (uint4)(
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        );

        unsigned int bin4in0 = ((__global unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((__global unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((__global unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((__global unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, (unsigned int) 255);
        internal_histo_data.y = min (bin4in1, (unsigned int) 255);
        internal_histo_data.z = min (bin4in2, (unsigned int) 255);
        internal_histo_data.w = min (bin4in3, (unsigned int) 255);

        uchar4 final_histo_data = (uchar4) (
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        );

        ((__global uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDimx * blockDimx)
    {
        ushort4 global_histo_data = ((__global ushort4*)global_histo)[i];
        ((__global ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, (ushort) 255);
        global_histo_data.y = min (global_histo_data.y, (ushort) 255);
        global_histo_data.z = min (global_histo_data.z, (ushort) 255);
        global_histo_data.w = min (global_histo_data.w, (ushort) 255);

        uchar4 final_histo_data = (uchar4) (
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        );

        ((__global uchar4*)final_histo)[i] = final_histo_data;
    }
}
