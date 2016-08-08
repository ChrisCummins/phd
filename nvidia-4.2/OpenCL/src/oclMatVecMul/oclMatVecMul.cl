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
 
/* Matrix-vector multiplication: W = M * V.
 * Device code.
 *
 * This sample implements matrix-vector multiplication.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles and optimizatoins, not with the goal of providing
 * the most performant generic kernel for matrix-vector multiplication.
 *
 * CUBLAS provides high-performance matrix-vector multiplication on GPU.
 */
 
// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulUncoalesced0(const __global float* M,
                                    const __global float* V,
                                    uint width, uint height,
                                    __global float* W)
{
    // Row index
    uint y = get_global_id(0);
    if (y < height) {
    
        // Row pointer
        const __global float* row = M + y * width;

        // Compute dot product  
        float dotProduct = 0;
        for (int x = 0; x < width; ++x)
            dotProduct += row[x] * V[x];

        // Write result to global memory
        W[y] = dotProduct;
    }
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulUncoalesced1(const __global float* M,
                                    const __global float* V,
                                    uint width, uint height,
                                    __global float* W)
{        
    // Each work-item handles as many matrix rows as necessary
    for (uint y = get_global_id(0);
         y < height;
         y += get_global_size(0))
    {

        // Row pointer
        const __global float* row = M + y * width;

        // Compute dot product  
        float dotProduct = 0;
        for (uint x = 0; x < width; ++x)
            dotProduct += row[x] * V[x];

        // Write result to global memory
        W[y] = dotProduct;
    }
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulCoalesced0(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{    
    // Each work-group handles as many matrix rows as necessary
    for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

        // Row pointer
        const __global float* row = M + y * width;
        
        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        float sum = 0;
        for (uint x = get_local_id(0); x < width; x += get_local_size(0))
            sum += row[x] * V[x];

        // Each partial dot product is stored in shared memory
        partialDotProduct[get_local_id(0)] = sum;

        // Synchronize to make sure each work-item is done updating
        // shared memory; this is necessary because in the next step,
        // the first work-item needs to read from shared memory
        // the partial dot products written by the other work-items
        barrier(CLK_LOCAL_MEM_FENCE);

        // The first work-item in the work-group adds all partial
        // dot products together and writes the result to global memory
        if (get_local_id(0) == 0) {
            float dotProduct = 0;
            for (uint t = 0; t < get_local_size(0); ++t)
                dotProduct += partialDotProduct[t];
            W[y] = dotProduct;
	    }

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
	}
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulCoalesced1(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{    
    // Each work-group handles as many matrix rows as necessary
    for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

        // Row pointer
        const __global float* row = M + y * width;
        
        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        float sum = 0;
        for (uint x = get_local_id(0); x < width; x += get_local_size(0))
            sum += row[x] * V[x];

        // Each partial dot product is stored in shared memory
        partialDotProduct[get_local_id(0)] = sum;
        
        // Perform parallel reduction to add each work-item's
        // partial dot product together
        for (uint stride = 1; stride < get_local_size(0); stride *= 2) {

            // Synchronize to make sure each work-item is done updating
            // shared memory; this is necessary because work-items read
            // results that have been written by other work-items
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Index into the "partialDotProduct" array where
            // the work-item will write during this step
            uint index = 2 * stride * get_local_id(0);
            
            // Check for valid indices
            if (index < get_local_size(0)) {
            
                // Add two elements from the "partialDotProduct" array
                // and store the result in partialDotProduct[index]
                partialDotProduct[index] += partialDotProduct[index + stride];
            }
        }

        // Write the result of the reduction to global memory
        if (get_local_id(0) == 0)
            W[y] = partialDotProduct[0];

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulCoalesced2(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{    
    // Each work-group handles as many matrix rows as necessary
    for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

        // Row pointer
        const __global float* row = M + y * width;
        
        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        float sum = 0;
        for (uint x = get_local_id(0); x < width; x += get_local_size(0))
            sum += row[x] * V[x];

        // Each partial dot product is stored in shared memory
        partialDotProduct[get_local_id(0)] = sum;
        
        // Perform parallel reduction to add each work-item's
        // partial dot product together
        for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {

            // Synchronize to make sure each work-item is done updating
            // shared memory; this is necessary because work-items read
            // results that have been written by other work-items
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Only the first work-items in the work-group add elements together
            if (get_local_id(0) < stride) {
            
                // Add two elements from the "partialDotProduct" array
                // and store the result in partialDotProduct[index]
                partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
            }
        }

        // Write the result of the reduction to global memory
        if (get_local_id(0) == 0)
            W[y] = partialDotProduct[0];

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#define WARP_SIZE 32
__kernel void MatVecMulCoalesced3(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{
   // Each work-group computes multiple elements of W
   for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {
      const __global float* row = M + y * width;

      // Each work-item accumulates as many products as necessary
      // into local variable "sum"
      float sum = 0;
      for (uint x = get_local_id(0); x < width; x += get_local_size(0))
         sum += row[x] * V[x];

      // Each partial dot product is stored in shared memory
      partialDotProduct[get_local_id(0)] = sum;

      // Perform parallel reduction to add each work-item's
      // partial dot product together

      // Synchronize to make sure each work-item is done writing to
      // partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);

      // Thread local ID within a warp
      uint id = get_local_id(0) & (WARP_SIZE - 1); 

      // Each warp reduces 64 consecutive elements
      float warpResult = 0.0f;
      if (get_local_id(0) < get_local_size(0)/2 )
      {
          volatile __local float* p = partialDotProduct + 2 * get_local_id(0) - id;
          p[0] += p[32];
          p[0] += p[16];
          p[0] += p[8];
          p[0] += p[4];
          p[0] += p[2];
          p[0] += p[1];
          warpResult = p[0];
      }

      // Synchronize to make sure each warp is done reading
      // partialDotProduct before it is overwritten in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // The first thread of each warp stores the result of the reduction
      // at the beginning of partialDotProduct
      if (id == 0)
         partialDotProduct[get_local_id(0) / WARP_SIZE] = warpResult;

      // Synchronize to make sure each warp is done writing to
      // partialDotProduct before it is read in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // Number of remaining elements after the first reduction
      uint size = get_local_size(0) / (2 * WARP_SIZE);

      // get_local_size(0) is less or equal to 512 on NVIDIA GPUs, so
      // only a single warp is needed for the following last reduction
      // step
      if (get_local_id(0) < size / 2) {
         volatile __local float* p = partialDotProduct + get_local_id(0);
         if (size >= 8)
            p[0] += p[4];
         if (size >= 4)
            p[0] += p[2];
         if (size >= 2)
            p[0] += p[1];
      }

      // Write the result of the reduction to global memory
      if (get_local_id(0) == 0)
         W[y] = partialDotProduct[0];

      // Synchronize to make sure the first work-item is done with
      // reading partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}
