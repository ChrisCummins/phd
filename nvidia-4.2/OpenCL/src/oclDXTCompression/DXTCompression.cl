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

#define NUM_THREADS   64      // Number of threads per work group.


//MATH functions

// Use power method to find the first eigenvector.
// http://www.miislita.com/information-retrieval-tutorial/matrix-tutorial-3-eigenvalues-eig// envectors.html

float4 firstEigenVector( __local float* matrix )
{
    // 8 iterations seems to be more than enough.

    float4 v = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
    #pragma unroll
    for(int i = 0; i < 8; i++) {
      float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
      float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
      float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
      float m = max(max(x, y), z);        
      float iv = 1.0f / m;
      
      v.x = x * iv;
      v.y = y * iv;
      v.z = z * iv;      
    }

    return v;
}

void colorSums(__local const float4 * colors, __local float4 * sums)
{
    const int idx = get_local_id(0);

    sums[idx] = colors[idx];
    sums[idx] += sums[idx^8];
    sums[idx] += sums[idx^4];
    sums[idx] += sums[idx^2];
    sums[idx] += sums[idx^1];
}

float4 bestFitLine(__local const float4 * colors, float4 color_sum, __local float* covariance)
{
    // Compute covariance matrix of the given colors.
    const int idx = get_local_id(0);

    float4 diff = colors[idx] - color_sum * 0.0625f; // * 1.0f / 16.0f

    covariance[6 * idx + 0] = diff.x * diff.x;    // 0, 6, 12, 2, 8, 14, 4, 10, 0
    covariance[6 * idx + 1] = diff.x * diff.y;
    covariance[6 * idx + 2] = diff.x * diff.z;
    covariance[6 * idx + 3] = diff.y * diff.y;
    covariance[6 * idx + 4] = diff.y * diff.z;
    covariance[6 * idx + 5] = diff.z * diff.z;

    #pragma unroll
    for(int d = 8; d > 0; d >>= 1)
    {
        if (idx < d)
        {
            covariance[6 * idx + 0] += covariance[6 * (idx+d) + 0];
            covariance[6 * idx + 1] += covariance[6 * (idx+d) + 1];
            covariance[6 * idx + 2] += covariance[6 * (idx+d) + 2];
            covariance[6 * idx + 3] += covariance[6 * (idx+d) + 3];
            covariance[6 * idx + 4] += covariance[6 * (idx+d) + 4];
            covariance[6 * idx + 5] += covariance[6 * (idx+d) + 5];
        }
    }

    // Compute first eigen vector.
    return firstEigenVector(covariance);
}

// ////////////////////////////////////////////////////////////////////////////////
// // Sort colors
// ////////////////////////////////////////////////////////////////////////////////
void sortColors(__local const float * values, __local int * ranks)
{
    const int tid = get_local_id(0);

    int rank = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        rank += (values[i] < values[tid]);
    }
    
    ranks[tid] = rank;

    // Resolve elements with the same index.
    #pragma unroll
    for (int i = 0; i < 15; i++)
    {
        if (tid > i && ranks[tid] == ranks[i]) ++ranks[tid];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
void loadColorBlock(__global const uint * image, __local float4 * colors, __local float4 * sums, __local int * xrefs, __local float* temp, int groupOffset)
{
    const int bid = get_group_id(0) + groupOffset;
    const int idx = get_local_id(0);

    float4 tmp;

    if (idx < 16)
    {
        // Read color and copy to shared mem.
        uint c = image[(bid) * 16 + idx];
    
        colors[idx].x = ((c >> 0) & 0xFF) * 0.003921568627f;    // * (1.0f / 255.0f);
        colors[idx].y = ((c >> 8) & 0xFF) * 0.003921568627f;    // * (1.0f / 255.0f);
        colors[idx].z = ((c >> 16) & 0xFF) * 0.003921568627f;   //* (1.0f / 255.0f);

        // No need to synchronize, 16 < warp size.	

        // Sort colors along the best fit line.
	    colorSums(colors, sums);
	    float4 axis = bestFitLine(colors, sums[idx], temp);
            
        temp[idx] = colors[idx].x * axis.x + colors[idx].y * axis.y + colors[idx].z * axis.z;
        
        sortColors(temp, xrefs);
        
        tmp = colors[idx];

        colors[xrefs[idx]] = tmp;
    }
}

// ////////////////////////////////////////////////////////////////////////////////
// // Round color to RGB565 and expand
// ////////////////////////////////////////////////////////////////////////////////
float4 roundAndExpand(float4 v, ushort * w)
{
    ushort x = rint(clamp(v.x, 0.0f, 1.0f) * 31.0f);
    ushort y = rint(clamp(v.y, 0.0f, 1.0f) * 63.0f);
    ushort z = rint(clamp(v.z, 0.0f, 1.0f) * 31.0f);

    *w = ((x << 11) | (y << 5) | z);
    v.x = x * 0.03227752766457f; // approximate integer bit expansion.
    v.y = y * 0.01583151765563f;
    v.z = z * 0.03227752766457f;
    return v;
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
float evalPermutation(__local const float4* colors, uint permutation, ushort* start, ushort* end, float4 color_sum,
                       __constant float* alphaTable4, __constant int* prods4, float weight)
{
    float4 alphax_sum = (float4)(0.0f, 0.0f, 0.0f,0.0f);
    int akku = 0;

    // Compute alpha & beta for this permutation.
    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable4[bits & 3] * colors[i];
        akku += prods4[bits & 3];
    }

    float alpha2_sum = (akku >> 16);
    float beta2_sum = ((akku >> 8) & 0xff);
    float alphabeta_sum = ((akku >> 0) & 0xff);
    float4 betax_sum = weight * color_sum - alphax_sum;

    //// Compute endpoints using least squares.
 
    // alpha2, beta2, alphabeta and factor could be precomputed for each permutation, but it's faster to recompute them.
    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float4 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float4 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;
    
    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float4 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (1.0f/weight) * (e.x + e.y + e.z);
}

float evalPermutation3(__local const float4 * colors, uint permutation, ushort * start, ushort * end, float4 color_sum,
                       __constant float* alphaTable3, __constant int* prods3)
{
    float4 alphax_sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int akku = 0;

    // Compute alpha & beta for this permutation.
    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable3[bits & 3] * colors[i];
        akku += prods3[bits & 3];
    }

    float alpha2_sum = (akku >> 16);
    float beta2_sum = ((akku >> 8) & 0xff);
    float alphabeta_sum = ((akku >> 0) & 0xff);
    float4 betax_sum = 4.0f * color_sum - alphax_sum;

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float4 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float4 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;
    
    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float4 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.25f) * (e.x + e.y + e.z);
}

uint4 evalAllPermutations(__local const float4 * colors, __global const unsigned int * permutations,			 
			  __local float *errors, float4 color_sum, __local uint * s_permutations, 
              __constant float* alphaTable4, __constant int* prods4,
              __constant float* alphaTable3, __constant int* prods3)
{
    const int idx = get_local_id(0);

    uint bestStart;
    uint bestEnd;
    uint bestPermutation;
    uint temp;

  
    float bestError = FLT_MAX;
    
    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
      int pidx = idx + NUM_THREADS * i;
        if (pidx >= 992) break;
        
        ushort start, end;
        uint permutation = permutations[pidx];
        if (pidx < 160) s_permutations[pidx] = permutation;
                
        float error = evalPermutation(colors, permutation, &start, &end, color_sum, alphaTable4, prods4, 9.0f);        
        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;
        }
    }

    if (bestStart < bestEnd)
    {
        temp = bestEnd;
        bestEnd = bestStart;
        bestStart = temp;
        
        bestPermutation ^= 0x55555555;    // Flip indices.
    }

    #pragma unroll
    for(int i = 0; i < 3; i++)
    {
        int pidx = idx + NUM_THREADS * i;
        if (pidx >= 160) break;
        
        ushort start, end;
        uint permutation = s_permutations[pidx];
        float error = evalPermutation(colors, permutation, &start, &end, color_sum, alphaTable3, prods3, 4.0f);
        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;
            
            if (bestStart > bestEnd)
            {
                temp = bestEnd;
                bestEnd = bestStart;
                bestStart = temp;

                bestPermutation ^= (~bestPermutation >> 1) & 0x55555555;    // Flip indices.
            }
        }
    }

    errors[idx] = bestError;
    
    uint4 result = (uint4)(bestStart, bestEnd, bestPermutation, 0);
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
int findMinError(__local float * errors, __local int * indices)
{
    const int idx = get_local_id(0);

    indices[idx] = idx;

    #pragma unroll
    for(int d = NUM_THREADS/2; d > 32; d >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (idx < d)
        {
            float err0 = errors[idx];
            float err1 = errors[idx + d];
            
            if (err1 < err0) {
                errors[idx] = err1;
                indices[idx] = indices[idx + d];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // unroll last 6 iterations
    if (idx < 32)
    {
        if (errors[idx + 32] < errors[idx]) {
            errors[idx] = errors[idx + 32];
            indices[idx] = indices[idx + 32];
        }
        if (errors[idx + 16] < errors[idx]) {
            errors[idx] = errors[idx + 16];
            indices[idx] = indices[idx + 16];
        }
        if (errors[idx + 8] < errors[idx]) {
            errors[idx] = errors[idx + 8];
            indices[idx] = indices[idx + 8];
        }
        if (errors[idx + 4] < errors[idx]) {
            errors[idx] = errors[idx + 4];
            indices[idx] = indices[idx + 4];
        }
        if (errors[idx + 2] < errors[idx]) {
            errors[idx] = errors[idx + 2];
            indices[idx] = indices[idx + 2];
        }
        if (errors[idx + 1] < errors[idx]) {
            errors[idx] = errors[idx + 1];
            indices[idx] = indices[idx + 1];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    return indices[0];
}


//Save DXT block
void saveBlockDXT1(uint start, uint end, uint permutation, __local int* xrefs, __global uint2 * result, int groupOffset)
{
    const int bid = get_group_id(0) + groupOffset;

    if (start == end)
    {
        permutation = 0;
    }
    
    // Reorder permutation.
    uint indices = 0;
    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
        int ref = xrefs[i];
        indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
    }
    
    // Write endpoints.
    result[bid].x = (end << 16) | start;
    
    // Write palette indices.
    result[bid].y = indices;
}

////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
__kernel void compress(__global const uint * permutations, __global const uint * image, 
		       __global uint2 * result, 
               __constant float* alphaTable4, __constant int* prods4,
               __constant float* alphaTable3, __constant int* prods3,
			   int groupOffset)
{
	__local float4 colors[16];
	__local float4 sums[16];
	__local int s_int[64];
	__local float s_float[16*6];
	__local uint s_permutations[160];
	__local int xrefs[16];

	const int idx = get_local_id(0);
    
    loadColorBlock(image, colors, sums, xrefs, s_float, groupOffset);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint4 best = evalAllPermutations(colors, permutations,s_float, sums[0], s_permutations, alphaTable4, prods4, alphaTable3, prods3);

    // Use a parallel reduction to find minimum error.
    const int minIdx = findMinError(s_float, s_int);    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Only write the result of the winner thread.
    if (idx == minIdx)
    {
        saveBlockDXT1(best.x, best.y, best.z, xrefs, result, groupOffset);
    }
}
