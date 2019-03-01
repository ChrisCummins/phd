#include <libcecl.h>
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

#include <oclUtils.h>
#include "oclConvolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for convolutionRows / convolutionColumns kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL convolutionSeparable program
static cl_program
    cpConvolutionSeparable;

//OpenCL convolutionSeparable kernels
static cl_kernel
    ckConvolutionRows, ckConvolutionColumns;

static cl_command_queue
    cqDefaultCommandQueue;

static const cl_uint
    ROWS_BLOCKDIM_X   = 16, COLUMNS_BLOCKDIM_X = 16,
    ROWS_BLOCKDIM_Y   = 4,  COLUMNS_BLOCKDIM_Y = 8,
    ROWS_RESULT_STEPS = 8,  COLUMNS_RESULT_STEPS = 8,
    ROWS_HALO_STEPS   = 1,  COLUMNS_HALO_STEPS = 1;

extern "C" void initConvolutionSeparable(cl_context cxGPUContext, cl_command_queue cqParamCommandQueue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("Loading ConvolutionSeparable.cl...\n");
        char *cPathAndName = shrFindFilePath("ConvolutionSeparable.cl", argv[0]);
        oclCheckError(cPathAndName != NULL, shrTRUE);
        char *cConvolutionSeparable = oclLoadProgSource(cPathAndName, "// My comment\n", &kernelLength);
        oclCheckError(cConvolutionSeparable != NULL, shrTRUE);

    shrLog("Creating convolutionSeparable program...\n");
        cpConvolutionSeparable = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cConvolutionSeparable, &kernelLength, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Building convolutionSeparable program...\n");
        char compileOptions[2048];
        #ifdef _WIN32
            sprintf_s(compileOptions, 2048, "\
                -cl-fast-relaxed-math                                  \
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u -D COLUMNS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u -D COLUMNS_BLOCKDIM_Y=%u\
                -D ROWS_RESULT_STEPS=%u -D COLUMNS_RESULT_STEPS=%u\
                -D ROWS_HALO_STEPS=%u -D COLUMNS_HALO_STEPS=%u\
                ",
                KERNEL_RADIUS,
                ROWS_BLOCKDIM_X,   COLUMNS_BLOCKDIM_X,
                ROWS_BLOCKDIM_Y,   COLUMNS_BLOCKDIM_Y,
                ROWS_RESULT_STEPS, COLUMNS_RESULT_STEPS,
                ROWS_HALO_STEPS,   COLUMNS_HALO_STEPS
            );
        #else
            sprintf(compileOptions, "\
                -cl-fast-relaxed-math                                  \
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u -D COLUMNS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u -D COLUMNS_BLOCKDIM_Y=%u\
                -D ROWS_RESULT_STEPS=%u -D COLUMNS_RESULT_STEPS=%u\
                -D ROWS_HALO_STEPS=%u -D COLUMNS_HALO_STEPS=%u\
                ",
                KERNEL_RADIUS,
                ROWS_BLOCKDIM_X,   COLUMNS_BLOCKDIM_X,
                ROWS_BLOCKDIM_Y,   COLUMNS_BLOCKDIM_Y,
                ROWS_RESULT_STEPS, COLUMNS_RESULT_STEPS,
                ROWS_HALO_STEPS,   COLUMNS_HALO_STEPS
            );
        #endif
        ciErrNum = CECL_PROGRAM(cpConvolutionSeparable, 0, NULL, compileOptions, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckConvolutionRows = CECL_KERNEL(cpConvolutionSeparable, "convolutionRows", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckConvolutionColumns = CECL_KERNEL(cpConvolutionSeparable, "convolutionColumns", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    cqDefaultCommandQueue = cqParamCommandQueue;
    free(cConvolutionSeparable);
}

extern "C" void closeConvolutionSeparable(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseKernel(ckConvolutionColumns);
    ciErrNum |= clReleaseKernel(ckConvolutionRows);
    ciErrNum |= clReleaseProgram(cpConvolutionSeparable);
}

extern "C" void convolutionRows(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
){
    cl_int ciErrNum;
    size_t localWorkSize[2], globalWorkSize[2];

    oclCheckError( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS, shrTRUE );
    oclCheckError( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0, shrTRUE );
    oclCheckError( imageH % ROWS_BLOCKDIM_Y == 0, shrTRUE );

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQueue;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckConvolutionRows, 0, sizeof(cl_mem),       (void*)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionRows, 1, sizeof(cl_mem),       (void*)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionRows, 2, sizeof(cl_mem),       (void*)&c_Kernel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionRows, 3, sizeof(unsigned int), (void*)&imageW);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionRows, 4, sizeof(unsigned int), (void*)&imageH);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionRows, 5, sizeof(unsigned int), (void*)&imageW);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize[0] = ROWS_BLOCKDIM_X;
    localWorkSize[1] = ROWS_BLOCKDIM_Y;
    globalWorkSize[0] = imageW / ROWS_RESULT_STEPS;
    globalWorkSize[1] = imageH;

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckConvolutionRows, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void convolutionColumns(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
){
    cl_int ciErrNum;
    size_t localWorkSize[2], globalWorkSize[2];

    oclCheckError( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS, shrTRUE );
    oclCheckError( imageW % COLUMNS_BLOCKDIM_X == 0, shrTRUE );
    oclCheckError( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0, shrTRUE );

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQueue;

    ciErrNum  = CECL_SET_KERNEL_ARG(ckConvolutionColumns, 0, sizeof(cl_mem),       (void*)&d_Dst);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionColumns, 1, sizeof(cl_mem),       (void*)&d_Src);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionColumns, 2, sizeof(cl_mem),       (void*)&c_Kernel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionColumns, 3, sizeof(unsigned int), (void*)&imageW);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionColumns, 4, sizeof(unsigned int), (void*)&imageH);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckConvolutionColumns, 5, sizeof(unsigned int), (void*)&imageW);
    oclCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize[0] = COLUMNS_BLOCKDIM_X;
    localWorkSize[1] = COLUMNS_BLOCKDIM_Y;
    globalWorkSize[0] = imageW;
    globalWorkSize[1] = imageH / COLUMNS_RESULT_STEPS;

    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckConvolutionColumns, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}
