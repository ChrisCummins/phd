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

// *********************************************************************
// Demo application for realtime DXT1 compression using OpenCL
// Based on the CUDA-C DXTC sample
// *********************************************************************

// standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>

#include "dds.h"
#include "permutations.h"
#include "block.h"

const char *image_filename = "lena_std.ppm";
const char *refimage_filename = "lena_ref.dds";

unsigned int width, height;
cl_uint* h_img = NULL;

#define ERROR_THRESHOLD 0.02f

#define NUM_THREADS   64      // Number of threads per work group.

// constants for this demo
const cl_float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
const cl_float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
const cl_int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
const cl_int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

// Main function
// *********************************************************************
int main(int argc, char** argv) 
{
    shrQAStart(argc, argv);

    // start logs
    shrSetLogFileName ("oclDXTCompression.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    cl_platform_id cpPlatform = NULL;
    cl_uint uiNumDevices = 0;
    cl_device_id *cdDevices = NULL;
    cl_context cxGPUContext;
    cl_command_queue cqCommandQueue;
    cl_program cpProgram;
    cl_kernel ckKernel;
    cl_mem cmMemObjs[3];
    cl_mem cmAlphaTable4, cmProds4;
    cl_mem cmAlphaTable3, cmProds3;
    size_t szGlobalWorkSize[1];
    size_t szLocalWorkSize[1];
    cl_int ciErrNum;

    // Get the path of the filename
    char *filename;
    if (shrGetCmdLineArgumentstr(argc, (const char **)argv, "image", &filename)) {
        image_filename = filename;
    }
    // load image
    const char* image_path = shrFindFilePath(image_filename, argv[0]);
    oclCheckError(image_path != NULL, shrTRUE);
    shrLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);
    oclCheckError(h_img != NULL, shrTRUE);
    shrLog("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);

    // Convert linear image to block linear. 
    const uint memSize = width * height * sizeof(cl_uint);
    uint* block_image = (uint*)malloc(memSize);

    // Convert linear image to block linear. 
    for(uint by = 0; by < height/4; by++) {
        for(uint bx = 0; bx < width/4; bx++) {
            for (int i = 0; i < 16; i++) {
                const int x = i & 3;
                const int y = i / 4;
                block_image[(by * width/4 + bx) * 16 + i] = 
                    ((uint *)h_img)[(by * 4 + y) * 4 * (width/4) + bx * 4 + x];
            }
        }
    }

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get the platform's GPU devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Create the context
    cxGPUContext = CECL_CREATE_CONTEXT(0, uiNumDevices, cdDevices, NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // get and log device
    cl_device_id device;
    if( shrCheckCmdLineFlag(argc, (const char **)argv, "device") ) {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, (const char **)argv, "device", &device_nr);
      device = oclGetDev(cxGPUContext, device_nr);
      if( device == (cl_device_id)-1 ) {
          shrLog(" Invalid GPU Device: devID=%d.  %d valid GPU devices detected\n\n", device_nr, uiNumDevices);
		  shrLog(" exiting...\n");
          return -1;
      }
    } else {
      device = oclGetMaxFlopsDev(cxGPUContext);
    }

    oclPrintDevName(LOGBOTH, device);
    shrLog("\n");

    // create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, device, 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Memory Setup

    // Constants
    cmAlphaTable4 = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(cl_float), (void*)&alphaTable4[0], &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cmProds4 = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(cl_int), (void*)&prods4[0], &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cmAlphaTable3 = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(cl_float), (void*)&alphaTable3[0], &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cmProds3 = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(cl_int), (void*)&prods3[0], &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Compute permutations.
    cl_uint permutations[1024];
    computePermutations(permutations);

    // Upload permutations.
    cmMemObjs[0] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_uint) * 1024, permutations, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Image
    cmMemObjs[1] = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, memSize, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    // Result
    const uint compressedSize = (width / 4) * (height / 4) * 8;
    cmMemObjs[2] = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, compressedSize, NULL , &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    unsigned int * h_result = (uint*)malloc(compressedSize);

    // Program Setup
    size_t program_length;
    const char* source_path = shrFindFilePath("DXTCompression.cl", argv[0]);
    oclCheckError(source_path != NULL, shrTRUE);
    char *source = oclLoadProgSource(source_path, "", &program_length);
    oclCheckError(source != NULL, shrTRUE);

    // create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1,
        (const char **) &source, &program_length, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // build the program
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclDXTCompression.ptx");
        oclCheckError(ciErrNum, CL_SUCCESS); 
    }

    // create the kernel
    ckKernel = CECL_KERNEL(cpProgram, "compress", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // set the args values
    ciErrNum  = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), (void *) &cmMemObjs[0]);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_mem), (void *) &cmMemObjs[1]);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(cl_mem), (void *) &cmMemObjs[2]);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(cl_mem), (void*)&cmAlphaTable4);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 4, sizeof(cl_mem), (void*)&cmProds4);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 5, sizeof(cl_mem), (void*)&cmAlphaTable3);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 6, sizeof(cl_mem), (void*)&cmProds3);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Copy input data host to device
    CECL_WRITE_BUFFER(cqCommandQueue, cmMemObjs[1], CL_FALSE, 0, sizeof(cl_uint) * width * height, block_image, 0,0,0);

    // Determine launch configuration and run timed computation numIterations times
	int blocks = ((width + 3) / 4) * ((height + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

	// Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
	cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
	int blocksPerLaunch = MIN(blocks, 768 * (int)compute_units);

    // set work-item dimensions
    szGlobalWorkSize[0] = blocksPerLaunch * NUM_THREADS;
    szLocalWorkSize[0]= NUM_THREADS;

#ifdef GPU_PROFILING
    shrLog("\nRunning DXT Compression on %u x %u image...\n", width, height);
    shrLog("\n%u Workgroups, %u Work Items per Workgroup, %u Work Items in NDRange...\n\n", 
           blocks, NUM_THREADS, blocks * NUM_THREADS);

    int numIterations = 50;
    for (int i = -1; i < numIterations; ++i) {
        if (i == 0) { // start timing only after the first warmup iteration
            clFinish(cqCommandQueue); // flush command queue
            shrDeltaT(0); // start timer
        }
#endif
        // execute kernel
		for( int j=0; j<blocks; j+= blocksPerLaunch ) {
			CECL_SET_KERNEL_ARG(ckKernel, 7, sizeof(int), &j);
			szGlobalWorkSize[0] = MIN( blocksPerLaunch, blocks-j ) * NUM_THREADS;
			ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 1, NULL,
				                              szGlobalWorkSize, szLocalWorkSize, 
					                          0, NULL, NULL);
			oclCheckError(ciErrNum, CL_SUCCESS);
		}

#ifdef GPU_PROFILING
    }
    clFinish(cqCommandQueue);
    double dAvgTime = shrDeltaT(0) / (double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclDXTCompression, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %d\n", 
           (1.0e-6 * (double)(width * height)/ dAvgTime), dAvgTime, (width * height), 1, szLocalWorkSize[0]); 
#endif

    // blocking read output
    ciErrNum = CECL_READ_BUFFER(cqCommandQueue, cmMemObjs[2], CL_TRUE, 0,
                                   compressedSize, h_result, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Write DDS file.
    FILE* fp = NULL;
    char output_filename[1024];
    #ifdef WIN32
        strcpy_s(output_filename, 1024, image_path);
        strcpy_s(output_filename + strlen(image_path) - 3, 1024 - strlen(image_path) + 3, "dds");
        fopen_s(&fp, output_filename, "wb");
    #else
        strcpy(output_filename, image_path);
        strcpy(output_filename + strlen(image_path) - 3, "dds");
        fp = fopen(output_filename, "wb");
    #endif
    oclCheckError(fp != NULL, shrTRUE);

    DDSHeader header;
    header.fourcc = FOURCC_DDS;
    header.size = 124;
    header.flags  = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
    header.height = height;
    header.width = width;
    header.pitch = compressedSize;
    header.depth = 0;
    header.mipmapcount = 0;
    memset(header.reserved, 0, sizeof(header.reserved));
    header.pf.size = 32;
    header.pf.flags = DDPF_FOURCC;
    header.pf.fourcc = FOURCC_DXT1;
    header.pf.bitcount = 0;
    header.pf.rmask = 0;
    header.pf.gmask = 0;
    header.pf.bmask = 0;
    header.pf.amask = 0;
    header.caps.caps1 = DDSCAPS_TEXTURE;
    header.caps.caps2 = 0;
    header.caps.caps3 = 0;
    header.caps.caps4 = 0;
    header.notused = 0;

    fwrite(&header, sizeof(DDSHeader), 1, fp);
    fwrite(h_result, compressedSize, 1, fp);

    fclose(fp);

    // Make sure the generated image matches the reference image (regression check)
    shrLog("\nComparing against Host/C++ computation...\n");     
    const char* reference_image_path = shrFindFilePath(refimage_filename, argv[0]);
    oclCheckError(reference_image_path != NULL, shrTRUE);

    // read in the reference image from file
    #ifdef WIN32
        fopen_s(&fp, reference_image_path, "rb");
    #else
        fp = fopen(reference_image_path, "rb");
    #endif
    oclCheckError(fp != NULL, shrTRUE);
    fseek(fp, sizeof(DDSHeader), SEEK_SET);
    uint referenceSize = (width / 4) * (height / 4) * 8;
    uint * reference = (uint *)malloc(referenceSize);
    fread(reference, referenceSize, 1, fp);
    fclose(fp);

    // compare the reference image data to the sample/generated image
    float rms = 0;
    for (uint y = 0; y < height; y += 4)
    {
        for (uint x = 0; x < width; x += 4)
        {
            // binary comparison of data
            uint referenceBlockIdx = ((y/4) * (width/4) + (x/4));
            uint resultBlockIdx = ((y/4) * (width/4) + (x/4));
            int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);

            // log deviations, if any
            if (cmp != 0.0f) 
            {
                compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);
                shrLog("Deviation at (%d, %d):\t%f rms\n", x/4, y/4, float(cmp)/16/3);
            }
            rms += cmp;
        }
    }
    rms /= width * height * 3;
    shrLog("RMS(reference, result) = %f\n\n", rms);

    // Free OpenCL resources
    oclDeleteMemObjs(cmMemObjs, 3);
    clReleaseMemObject(cmAlphaTable4);
    clReleaseMemObject(cmProds4);
    clReleaseMemObject(cmAlphaTable3);
    clReleaseMemObject(cmProds3);
    clReleaseKernel(ckKernel);
    clReleaseProgram(cpProgram);
    clReleaseCommandQueue(cqCommandQueue);
    clReleaseContext(cxGPUContext);

    // Free host memory
    free(source);
    free(h_img);

    // finish
    shrQAFinishExit(argc, (const char **)argv, (rms <= ERROR_THRESHOLD) ? QA_PASSED : QA_FAILED);
}
