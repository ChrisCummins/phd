/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Optimized sparse matrix vector multiplication
 *
 * Launch a work-item for each row
 * Each work-item will calculate a single row
 */

__kernel void spmv_jds_vec(
    __global float *dst_vector,  // 1D destination vector
    __global float *d_data,      // JD sparse matrix data, padded in groups of
                                    // 64 to multiples of 256b (4 floats)
    __global int *d_index,       // JD sparse matrix mapping to real column index
    __global int *d_perm,        // destination index into dst_vector given row
    __global float *x_vec,       // full vector data
    const int dim,               // number of rows in sparse matrix
    __constant int *jds_ptr_int, // index into flattened sparse matrix data in d_data
                                    // column-major, so it takes the current col 
    __constant int *sh_zcnt_int, // non-zero col count in a group of 64 rows
                                    // indexed by [col/64]
    const int warp_size)        // warp size cols are grouped by
{
    int mat_row = get_global_id(0); // row index
    
    int j;
    //__local int jds_ptr_local[256];
    
    // vload4 both sparse matrix values and indexes
    float4 sm_value;
    int4 sm_index;
    float4 dat_value; // use sm_index to populate this vector individually
    if (mat_row < dim) { // check to see that this thread is actually a valid row
        int bound=sh_zcnt_int[mat_row/warp_size]; // get padded # of items in this row
        // load jds ptr into local memory
        //int offset = get_local_id(0);
        //while (offset < 265 && offset < bound) {
        //    jds_ptr_local[offset] = jds_ptr_int[offset];
        //    offset += get_local_size(0);
        //}

        //barrier(CLK_LOCAL_MEM_FENCE);
        float4 sum = (float4) 0.0f;

        int col = 0;
        while (col<bound) { // for each col in row. does 4 cols per iteration     
            // memory access clause
            // vector lookups vload4(offset, ptr) are from ptr + offset*4
            j = jds_ptr_int[col];      // get array index into other structures
            // the pointer offset (by 1 per float)
            sm_index = vload4(mat_row, d_index+j);      // vec4 load sparse index
    
            sm_value = vload4(mat_row, d_data+j);     // vec4 load sparse data
            // is this a valid column? invalid ones have -1 for index values
            dat_value.x = (sm_index.x >= 0) ? x_vec[sm_index.x] : 0.0f;        // load data vector values
            dat_value.y = (sm_index.y >= 0) ? x_vec[sm_index.y] : 0.0f;
            dat_value.z = (sm_index.z >= 0) ? x_vec[sm_index.z] : 0.0f;
            dat_value.w = (sm_index.w >= 0) ? x_vec[sm_index.w] : 0.0f;
    
            // calculation clause
            col += 1; // increase col index 
            sum += dat_value*sm_value; // multiply all 4 pairs
        }  

        dst_vector[d_perm[mat_row]] = sum.x+sum.y+sum.z+sum.w;
        //dst_vector[d_perm[mat_row]] = sm_value.x;
        //dst_vector[mat_row] = dat_value.x;
    }
}
