/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/



/**
*******************************************************************************
*  @fn     separableFilter
*  @brief  This is the separable filter. It applies
*           the coloumn vector convolution first and then applies the row vector convolution.
*
*  @param[in] input          : Buffer containing padded input image 
*  @param[in] row_filter     : Buffer containing the row convoluion kernel
*  @param[in] col_filter     : Buffer containing the col convoluion kernel
*  @param[out] output        : Buffer containing the output of filter
*  @param[in] nWidth         : Image width in pixels
*  @param[in] nHeight        : Image height in pixels
*  @param[in] nExWidth       : Padded image width in pixels
* 
*******************************************************************************
*/

__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, LOCAL_YRES, 1)))
void advancedSeparableConvolution(
								__global uchar4 *input,
								__global float *row_filter,
								__global float *col_filter,
								__global uchar4 *output,
								uint nWidth,
								uint nHeight,
								uint nExWidth)                   
{
    __local float4 local_output[LOCAL_XRES * (LOCAL_YRES + FILTERSIZE - 1)];

    int col = get_global_id(0);
    int row = get_global_id(1);
    
    if (col >= nWidth || row >= nHeight) return;

    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);

    int start_col, start_row;

    int cnt = 0;

    /***************************************************************************************
    * If using LDS, get the data to local memory. Else, get the global memory indices ready 
    ***************************************************************************************/
#if USE_LDS == 1
    __local uchar4 local_input[(LOCAL_XRES + FILTERSIZE - 1) * (LOCAL_YRES + FILTERSIZE - 1)];
    
    int tile_xres = (LOCAL_XRES + FILTERSIZE - 1);
    int tile_yres = (LOCAL_YRES + FILTERSIZE - 1);

    start_col = get_group_id(0) * LOCAL_XRES; //Image is padded
    start_row = get_group_id(1) * LOCAL_YRES; 
 
    int lid = lid_y * LOCAL_XRES + lid_x; 
    int gx, gy;
    
     /*********************************************************************
     * Read input from global buffer and put in local buffer 
     * Read 256 global memory locations at a time (256 WI). 
     * Conitnue in a loop till all pixels in the tile are read.
     **********************************************************************/

    do {
        gy = lid / tile_xres;
        gx = lid - gy * tile_xres; 
        
        local_input[lid] = input[(start_row + gy) * nExWidth + (start_col + gx)];
        lid += (LOCAL_XRES * LOCAL_YRES);
    } while (lid < (tile_xres * tile_yres));

    barrier(CLK_LOCAL_MEM_FENCE);

    start_col = lid_x;
    
#else   
    /************************************************************************
    * Non - LDS implementation
    * Read pixels directly from global memory
    ************************************************************************/
	start_col = col; 
	
#endif

    /***********************************************************************************
    * Row-wise convolution - Inputs will be read from local or global memory         
    ************************************************************************************/
    float4 sum = (float4)0.0f;
    cnt = 0;

#pragma unroll FILTERSIZE
    for (int i = start_col; i < start_col + FILTERSIZE; i++) {
#if USE_LDS == 1
        sum = mad(convert_float4(local_input[lid_y * tile_xres + i]), (float4)row_filter[cnt++], sum);    
#else
        sum = mad(convert_float4(input[row * nExWidth + i]), (float4)row_filter[cnt++], sum);                 
#endif
    }
    
    /***********************************************************************************
    * Output is stored in local memory
    ************************************************************************************/
    local_output[lid_y * LOCAL_XRES + lid_x] = sum;

    /***************************************************************************************
    * Row-wise convolution of pixels in the remaining rows
    ***************************************************************************************/
    if (lid_y < FILTERSIZE - 1) {
        cnt = 0;
        sum = (float4)0.0f;

#pragma unroll FILTERSIZE
        for (int i = start_col; i < start_col + FILTERSIZE; i++) {
#if USE_LDS == 1
            sum = mad(convert_float4(local_input[(lid_y + LOCAL_YRES) * tile_xres + i]), (float4)row_filter[cnt++], sum);  
#else
            sum = mad(convert_float4(input[(row + LOCAL_YRES) * nExWidth + i]), (float4)row_filter[cnt++], sum);                    
#endif
        }
        /***********************************************************************************
        * Again the output is stored in local memory
        ************************************************************************************/
        local_output[(lid_y + LOCAL_YRES) * LOCAL_XRES + lid_x] = sum;
    }

    /***********************************************************************************
    * Wait for all the local WIs to finish row-wise convolution.
    ************************************************************************************/
    barrier(CLK_LOCAL_MEM_FENCE); 

   /************************************************************************************
    * Column-wise convolution - Input is the output of row-wise convolution
    * Inputs are always read from local memory. 
    * The output is written to global memory.
    ***********************************************************************************/
    start_row = lid_y;
    
    sum = (float4)0.0f;
    cnt = 0;

#pragma unroll FILTERSIZE
    for (int i = start_row; i < start_row + FILTERSIZE; i++) {
        sum = mad(local_output[i * LOCAL_XRES + lid_x], (float4)col_filter[cnt++], sum);        
    }
    
    /* Save Output */
	sum = (sum < 0.0f) ? 0.0f : sum;
    output[row * nWidth + col] = (convert_uchar4_sat_rte)(sum);
}



/**
*******************************************************************************
*  @fn     non-separableFilter
*  @brief  This is the non-separable filter. It applies
*           both coloumn vector as well as row vector convolution 
*			in a single pass.
*
*  @param[in] input          : Buffer containing padded input image 
*  @param[in] mask		     : Buffer containing convolution filter
*  @param[out] output        : Buffer containing the output of filter
*  @param[in] nWidth         : Image width in pixels
*  @param[in] nHeight        : Image height in pixels
*  @param[in] nExWidth       : Padded image width in pixels
* 
*******************************************************************************
*/

__kernel
__attribute__((reqd_work_group_size(LOCAL_XRES, LOCAL_YRES, 1)))
void advancedNonSeparableConvolution(
									__global uchar4 *input,
									__global float  *mask,  
									__global uchar4 *output,
									uint nWidth,
									uint nHeight,
									uint nExWidth)                   
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    if (col >= nWidth || row >= nHeight) return; 

    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);

    int start_col, start_row;

    int cnt = 0;

    /***************************************************************************************
    * If using LDS, get the data to local memory. Else, get the global memory indices ready 
    ***************************************************************************************/
#if USE_LDS == 1
    __local uchar4 local_input[(LOCAL_XRES + FILTERSIZE - 1) * (LOCAL_YRES + FILTERSIZE - 1)];
    
    int tile_xres = (LOCAL_XRES + FILTERSIZE - 1);
    int tile_yres = (LOCAL_YRES + FILTERSIZE - 1);

    start_col = get_group_id(0) * LOCAL_XRES; //Image is padded
    start_row = get_group_id(1) * LOCAL_YRES; 
 
    int lid = lid_y * LOCAL_XRES + lid_x; 
    int gx, gy;
    
     /*********************************************************************
     * Read input from global buffer and put in local buffer 
     * Read 256 global memory locations at a time (256 WI). 
     * Conitnue in a loop till all pixels in the tile are read.
     **********************************************************************/

    do {
        gy = lid / tile_xres;
        gx = lid - gy * tile_xres; 
        
        local_input[lid] = input[(start_row + gy) * nExWidth + (start_col + gx)];
        lid += (LOCAL_XRES * LOCAL_YRES);
    } while (lid < (tile_xres * tile_yres));

    barrier(CLK_LOCAL_MEM_FENCE);

    start_col = lid_x;
	start_row = lid_y;
    
#else   
    /************************************************************************
    * Non - LDS implementation
    * Read pixels directly from global memory
    ************************************************************************/
	start_col = col;
	start_row = row; 
#endif
    /***********************************************************************************
    * convolution - Inputs will be read from local or global memory         
    ************************************************************************************/
    float4 sum = (float4)0.0f;
	int m = 0, n = 0;

#pragma unroll FILTERSIZE
	for (int j = start_row; j < start_row + FILTERSIZE; j++,m++)
	{
		n = 0;
		for (int i = start_col; i < start_col + FILTERSIZE; i++,n++) {
#if USE_LDS == 1
			sum = mad(convert_float4(local_input[j * tile_xres + i]), (float4)mask[m * FILTERSIZE + n], sum);    
#else
			sum = mad(convert_float4(input[(j)*nExWidth  + (i)]), (float4)mask[m * FILTERSIZE + n], sum);
#endif
		}
    }

    /* Save Output */
    output[row * nWidth + col] = (convert_uchar4_sat)(sum);
}
