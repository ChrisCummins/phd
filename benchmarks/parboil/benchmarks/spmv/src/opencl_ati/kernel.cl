/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Base sparse matrix vector multiplication
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
    const int warp_size)        // warp size
{
    int mat_row = get_global_id(0); // row index
    
    int j;
    
    float sm_value;
    int sm_index;
    float dat_value;
    if (mat_row < dim) { // check to see that this thread is actually a valid row
        int bound=sh_zcnt_int[mat_row/warp_size]; // get padded # of items in this row

        float sum = 0.0f;

        int col = 0;
        while (col<bound) { // for each col in row  
            // memory access clause
            j = jds_ptr_int[col] + mat_row;      // get array index into other structures
            sm_index = d_index[j];
            sm_value = d_data[j];
            dat_value = x_vec[sm_index];
    
            // calculation clause
            col += 1; // increase col index 
            sum += dat_value*sm_value; // multiply
        }  

        dst_vector[d_perm[mat_row]] = sum;
    }
}
