/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>
#include <parboil.h>
#include <CL/cl.h>

#include "sad.h"
#include "sad_kernel.h"
#include "file.h"
#include "image.h"

const char* oclErrorString(cl_int error)
{
// From NVIDIA SDK
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;

	return (index >= 0 && index < errorCount) ? errorString[index] : "";
}
  
#define OCL_ERRCK(s) \
  { if (s != CL_SUCCESS) fprintf(stderr, "OpenCL Error (Line %d): %s\n", __LINE__, oclErrorString(s)); }
  
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}


static unsigned short *
load_sads(char *filename);
static void
write_sads(char *filename,
	   int image_size_macroblocks,
	   unsigned short *sads);
static void
write_sads_directly(char *filename,
		    int width,
		    int height,
		    unsigned short *sads);

/* FILE I/O */

unsigned short *
load_sads(char *filename)
{
  FILE *infile;
  unsigned short *sads;
  int w;
  int h;
  int sads_per_block;

  infile = fopen(filename, "r");

  if (!infile)
    {
      fprintf(stderr, "Cannot find file '%s'\n", filename);
      exit(-1);
    }

  /* Read image dimensions (measured in macroblocks) */
  w = read16u(infile);
  h = read16u(infile);

  /* Read SAD values.  Only interested in the 4x4 SAD values, which are
   * at the end of the file. */
  sads_per_block = MAX_POS_PADDED * (w * h);
  fseek(infile, 25 * sads_per_block * sizeof(unsigned short), SEEK_CUR);

  sads = (unsigned short *)malloc(sads_per_block * 16 * sizeof(unsigned short));
  fread(sads, sizeof(unsigned short), sads_per_block * 16, infile);
  fclose(infile);

  return sads;
}

/* Compare the reference SADs to the expected SADs.
 */
void
check_sads(unsigned short *sads_reference,
	   unsigned short *sads_computed,
	   int image_size_macroblocks)
{
  int block;

  /* Check the 4x4 SAD values.  These are in sads_reference.
   * Ignore the data at the beginning of sads_computed. */
  sads_computed += 25 * MAX_POS_PADDED * image_size_macroblocks;

  for (block = 0; block < image_size_macroblocks; block++)
    {
      int subblock;

      for (subblock = 0; subblock < 16; subblock++)
	{
	  int sad_index;

	  for (sad_index = 0; sad_index < MAX_POS; sad_index++)
	    {
	      int index =
		(block * 16 + subblock) * MAX_POS_PADDED + sad_index;

	      if (sads_reference[index] != sads_computed[index])
		{
#if 0
		  /* Print exactly where the mismatch was seen */
		  printf("M %3d %2d %4d (%d = %d)\n", block, subblock, sad_index, sads_reference[index], sads_computed[index]);
#else
		  goto mismatch;
#endif
		}
	    }
	}
    }

  printf("Success.\n");
  return;

 mismatch:
  printf("Computed SADs do not match expected values.\n");
}

/* Extract the SAD data for a particular block type for a particular
 * macroblock from the array of SADs of that block type. */
static inline void
write_subblocks(FILE *outfile, unsigned short *subblock_array, int macroblock,
		int count)
{
  int block;
  int pos;

  for (block = 0; block < count; block++)
    {
      unsigned short *vec = subblock_array +
	(block + macroblock * count) * MAX_POS_PADDED;

      /* Write all SADs for this sub-block */
      for (pos = 0; pos < MAX_POS; pos++)
	write16u(outfile, *vec++);
    }
}

void
write_sads(char *filename,
	   int image_size_macroblocks,
	   unsigned short *sads)
{
  FILE *outfile = fopen(filename, "w");
  int block;

  if (outfile == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }

  /* Write size in macroblocks */
  write32u(outfile, image_size_macroblocks);

  /* Write zeros */
  write32u(outfile, 0);

  /* Each macroblock */
  for (block = 0; block < image_size_macroblocks; block++)
    {
      int blocktype;

      /* Write SADs for all sub-block types */
      for (blocktype = 1; blocktype <= 7; blocktype++)
	write_subblocks(outfile,
			sads + SAD_TYPE_IX(blocktype, image_size_macroblocks),
			block,
			SAD_TYPE_CT(blocktype));
    }

  fclose(outfile);
}

/* FILE I/O for debugging */

static void
write_sads_directly(char *filename,
		    int width,
		    int height,
		    unsigned short *sads)
{
  FILE *f = fopen(filename, "w");
  int n;

  write16u(f, width);
  write16u(f, height);
  for (n = 0; n < 41 * MAX_POS_PADDED * (width * height); n++) {
    write16u(f, sads[n]);
  }
  fclose(f);
}

static void
print_test_sad_vector(unsigned short *base, int macroblock, int count)
{
  int n;
  int searchpos = 17*33+17;
  for (n = 0; n < count; n++)
    printf(" %d", base[(count * macroblock + n) * MAX_POS_PADDED + searchpos]);
}

static void
print_test_sads(unsigned short *sads_computed,
		int mbs)
{
  int macroblock = 5;
  int blocktype;

  for (blocktype = 1; blocktype <= 7; blocktype++)
    {
      printf("%d:", blocktype);
      print_test_sad_vector(sads_computed + SAD_TYPE_IX(blocktype, mbs),
			    macroblock, SAD_TYPE_CT(blocktype));
      puts("\n");
    }
}

/* MAIN */

int
main(int argc, char **argv)
{
  struct image_i16 *ref_image;
  struct image_i16 *cur_image;
  unsigned short *sads_computed; /* SADs generated by the program */

  int image_size_bytes;
  int image_width_macroblocks, image_height_macroblocks;
  int image_size_macroblocks;

  struct pb_TimerSet timers;
  struct pb_Parameters *params;

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);

  if (pb_Parameters_CountInputs(params) != 2)
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }

  /* Read input files */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  ref_image = load_image(params->inpFiles[0]);
  cur_image = load_image(params->inpFiles[1]);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if ((ref_image->width != cur_image->width) ||
      (ref_image->height != cur_image->height))
    {
      fprintf(stderr, "Input images must be the same size\n");
      exit(-1);
    }
  if ((ref_image->width % 16) || (ref_image->height % 16))
    {
      fprintf(stderr, "Input image size must be an integral multiple of 16\n");
      exit(-1);
    }

  /* Compute parameters, allocate memory */
  image_size_bytes = ref_image->width * ref_image->height * sizeof(short);
  image_width_macroblocks = ref_image->width >> 4;
  image_height_macroblocks = ref_image->height >> 4;
  image_size_macroblocks = image_width_macroblocks * image_height_macroblocks;
  
  sads_computed = (unsigned short *)
    malloc(41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(short));

  /* Run the kernel code */
  {
  	cl_int ciErrNum;
  	cl_platform_id clPlatform;
  	int deviceType = CL_DEVICE_TYPE_DEFAULT;
	cl_device_id clDevice;
	cl_context clContext;
	cl_command_queue clCommandQueue;

	cl_kernel mb_sad_calc;
	cl_kernel larger_sad_calc_8;
	cl_kernel larger_sad_calc_16;
	
	cl_mem imgRef;		/* Reference image on the device */
	cl_mem d_cur_image;	/* Current image on the device */
	cl_mem d_sads;		/* SADs on the device */

    // x : image_width_macroblocks
    // y : image_height_macroblocks

/*
	if (argc > 1) {
		if (strcmp(argv[1],"-gpu") == 0) {
			deviceType = CL_DEVICE_TYPE_GPU;
		} else if (strcmp(argv[1],"-cpu")==0) {
			deviceType = CL_DEVICE_TYPE_CPU;
		}
	}
*/

	
    pb_SwitchToTimer(&timers, pb_TimerID_DRIVER);

	ciErrNum = clGetPlatformIDs(1, &clPlatform, NULL); 
	OCL_ERRCK(ciErrNum);
	ciErrNum = clGetDeviceIDs(clPlatform, deviceType, 1, &clDevice, NULL);
	OCL_ERRCK(ciErrNum);

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform, 0};
    clContext = clCreateContextFromType(cps, deviceType, NULL, NULL, &ciErrNum);
  	OCL_ERRCK(ciErrNum);
  	
  	clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, &ciErrNum);
  	OCL_ERRCK(ciErrNum);
  	
  	size_t program_length;
    const char* source_path = "/home/twentz/parboil_nonAC/benchmarks/sad/src/opencl_nvidia/sad_kernel.cl";
    //const char* source_path = "/afs/crhc.illinois.edu/user/wentz1/ECE598HK/parboil_nonAC/benchmarks/sad/src/opencl_nvidia/sad_kernel.cl";
    char* source = oclLoadProgSource(source_path, "", &program_length);
    if(!source) {
        fprintf(stderr, "Could not load program source\n"); exit(1);
    }
  	
  	cl_program clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&source, &program_length, &ciErrNum);
  	OCL_ERRCK(ciErrNum);
  	
  	free(source);
    
    char compileOptions[1024];
    //                -cl-nv-verbose
    sprintf(compileOptions, "\
                -D SAD_LOC_SIZE_BYTES=%u\
                -D MAX_POS=%u -D CEIL_POS=%u\
                -D POS_PER_THREAD=%u -D MAX_POS_PADDED=%u\
                -D THREADS_W=%u -D THREADS_H=%u\
                -D SEARCH_RANGE=%u -D SEARCH_DIMENSION=%u\
                ",
                SAD_LOC_SIZE_BYTES,
                MAX_POS, CEIL(MAX_POS, POS_PER_THREAD),
                POS_PER_THREAD,   MAX_POS_PADDED,
                THREADS_W,   THREADS_H,
                SEARCH_RANGE, SEARCH_DIMENSION
            ); 
    
    ciErrNum = clBuildProgram(clProgram, 1, &clDevice, compileOptions, NULL, NULL);
   	OCL_ERRCK(ciErrNum);
   	
   	/*
   	   char *build_log;
       size_t ret_val_size;
       ciErrNum = clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK(ciErrNum);
       build_log = (char *)malloc(ret_val_size+1);
       ciErrNum = clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
       	OCL_ERRCK(ciErrNum);

       // to be carefully, terminate with \0
       // there's no information in the reference whether the string is 0 terminated or not
       build_log[ret_val_size] = '\0';

       fprintf(stderr, "%s\n", build_log );
       */

    mb_sad_calc = clCreateKernel(clProgram, "mb_sad_calc", &ciErrNum);
   	OCL_ERRCK(ciErrNum);    
   	larger_sad_calc_8 = clCreateKernel(clProgram, "larger_sad_calc_8", &ciErrNum);
   	OCL_ERRCK(ciErrNum);
   	larger_sad_calc_16 = clCreateKernel(clProgram, "larger_sad_calc_16", &ciErrNum);
   	OCL_ERRCK(ciErrNum);

    size_t wgSize;
    size_t comp_wgSize[3];
    cl_ulong localMemSize;
    size_t prefwgSizeMult;
    cl_ulong privateMemSize;

/*    
	ciErrNum = clGetKernelWorkGroupInfo(larger_sad_calc_8, NULL, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgSize), &wgSize, NULL);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clGetKernelWorkGroupInfo(larger_sad_calc_8, NULL, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3*sizeof(size_t), comp_wgSize, NULL);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clGetKernelWorkGroupInfo(larger_sad_calc_8, NULL, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
	OCL_ERRCK(ciErrNum);
    ciErrNum = clGetKernelWorkGroupInfo(larger_sad_calc_8, NULL, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &prefwgSizeMult, NULL);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clGetKernelWorkGroupInfo(larger_sad_calc_8, NULL, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &privateMemSize, NULL);
	OCL_ERRCK(ciErrNum);
	*/
/*
fprintf(stderr, "Work Group Size: %lu\n", wgSize);
fprintf(stderr, "Compile Work Group Size: %lu, %lu, %lu\n", comp_wgSize[0], comp_wgSize[1], comp_wgSize[2]);
fprintf(stderr, "Local Memory Size: %lu\n", localMemSize);
fprintf(stderr, "Preferred Work Group Size Multiple: %lu\n", prefwgSizeMult);
fprintf(stderr, "Private Memory Size: %lu\n", privateMemSize);	
*/
 
    pb_SwitchToTimer(&timers, pb_TimerID_COPY); 
    

    cl_image_format img_format;
    img_format.image_channel_order = CL_R;
    img_format.image_channel_data_type = CL_UNSIGNED_INT16;

/* Transfer reference image to device */
	imgRef = clCreateImage2D(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, 
                                      ref_image->width /** sizeof(unsigned short)*/, // width
                                      ref_image->height, // height
                                      ref_image->width * sizeof(unsigned short), // row_pitch
                                      ref_image->data, &ciErrNum);
    OCL_ERRCK(ciErrNum);                                      
	
    /* Allocate SAD data on the device */

    unsigned short *tmpZero = (unsigned short *)malloc(41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
    memset(tmpZero, 0, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
    d_sads = clCreateBuffer(clContext,/* CL_MEM_READ_WRITE |*/ CL_MEM_COPY_HOST_PTR, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), tmpZero, &ciErrNum);
    OCL_ERRCK(ciErrNum);
    free(tmpZero);
    
    d_cur_image = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size_bytes, cur_image->data, &ciErrNum);
    OCL_ERRCK(ciErrNum);



    pb_SwitchToTimer(&timers, pb_TimerID_DRIVER);


	/* Set Kernel Parameters */	
	ciErrNum = clSetKernelArg(mb_sad_calc, 0, sizeof(cl_mem), (void *)&d_sads);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_8, 0, sizeof(cl_mem), (void *)&d_sads);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_16, 0, sizeof(cl_mem), (void *)&d_sads);
	OCL_ERRCK(ciErrNum);
	
	ciErrNum = clSetKernelArg(mb_sad_calc, 1, sizeof(cl_mem), (void *)&d_cur_image);
	OCL_ERRCK(ciErrNum);
	
	ciErrNum = clSetKernelArg(mb_sad_calc, 2, sizeof(int), &image_width_macroblocks);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_8, 1, sizeof(int), &image_width_macroblocks);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_16, 1, sizeof(int), &image_width_macroblocks);
	OCL_ERRCK(ciErrNum);
	
	ciErrNum = clSetKernelArg(mb_sad_calc, 3, sizeof(int), &image_height_macroblocks);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_8, 2, sizeof(int), &image_height_macroblocks);
	OCL_ERRCK(ciErrNum);
	ciErrNum = clSetKernelArg(larger_sad_calc_16, 2, sizeof(int), &image_height_macroblocks);
	OCL_ERRCK(ciErrNum);
	
	ciErrNum = clSetKernelArg(mb_sad_calc, 4, sizeof(cl_mem), (void *)&imgRef);
	OCL_ERRCK(ciErrNum);
	
	/*
	printf("MaxPos: %d\tPos/Thr: %d\tThr(w,h): (%d,%d)\n", MAX_POS, POS_PER_THREAD, THREADS_W, THREADS_H);
	printf("Local worksize/ThreadsPerBlock: %d\n", CEIL(MAX_POS, POS_PER_THREAD));
	printf("ref_image->w/h:\t(%u,%u)\nDiv by 4: \t(%u,%u)\nCeil with W/H: \t(%u,%u)\nRemult, Grid: (%u,%u)\n",
	ref_image->width, ref_image->height,
	ref_image->width/4, ref_image->height/4,
	CEIL(ref_image->width / 4, THREADS_W), CEIL(ref_image->height / 4, THREADS_H),
	CEIL(ref_image->width / 4, THREADS_W) * THREADS_W,  CEIL(ref_image->height / 4, THREADS_H) * THREADS_H);
	*/
	
	size_t mb_sad_calc_localWorkSize[2] = {
	    CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H,
	    1 };
	size_t mb_sad_calc_globalWorkSize[2] = {
        CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H *
            CEIL(ref_image->width / 4, THREADS_W),
	    CEIL(ref_image->height / 4, THREADS_H) };
	
	size_t larger_sad_calc_8_localWorkSize[2] = {32,4};
	size_t larger_sad_calc_8_globalWorkSize[2] = {image_width_macroblocks * 32, image_height_macroblocks * 4};
	
	size_t larger_sad_calc_16_localWorkSize[2] = {32, 1};
	size_t larger_sad_calc_16_globalWorkSize[2] = {image_width_macroblocks * 32, image_height_macroblocks};
	
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
	
/*
printf("Launching:\n"
"mb_sad_calc:        Global (%lu,%d) ; Local (%lu,%d)\n"
"larger_sad_calc_8:  Global (%lu,%lu) ; Local (%lu,%lu)\n"
"larger_sad_calc_16: Global (%lu,%lu) ; Local (%lu,%lu)\n",
mb_sad_calc_globalWorkSize[0],1,mb_sad_calc_localWorkSize[0],1,
larger_sad_calc_8_globalWorkSize[0],larger_sad_calc_8_globalWorkSize[1],
larger_sad_calc_8_localWorkSize[0],larger_sad_calc_8_localWorkSize[1],
larger_sad_calc_16_globalWorkSize[0],larger_sad_calc_16_globalWorkSize[1],
larger_sad_calc_16_localWorkSize[0],larger_sad_calc_16_localWorkSize[1]);
*/
    /* Run the 4x4 kernel */	
	ciErrNum = clEnqueueNDRangeKernel(clCommandQueue, mb_sad_calc, 2, 0, mb_sad_calc_globalWorkSize, mb_sad_calc_localWorkSize, 0, 0, 0);
	OCL_ERRCK(ciErrNum);
		//cuFuncSetSharedSize(mb_sad_calc, SAD_LOC_SIZE_BYTES);
		
	/* Run the larger-blocks kernels */
	ciErrNum = clEnqueueNDRangeKernel(clCommandQueue, larger_sad_calc_8, 2, 0, larger_sad_calc_8_globalWorkSize, larger_sad_calc_8_localWorkSize, 0, 0, 0);
	OCL_ERRCK(ciErrNum);
		
	ciErrNum = clEnqueueNDRangeKernel(clCommandQueue, larger_sad_calc_16, 2, 0, larger_sad_calc_16_globalWorkSize, larger_sad_calc_16_localWorkSize, 0, 0, 0);
	OCL_ERRCK(ciErrNum);	

    //clFinish(clCommandQueue);
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    /* Transfer SAD data to the host */
    
    ciErrNum = clEnqueueReadBuffer(clCommandQueue, d_sads, CL_TRUE, 0, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), sads_computed, 0, NULL, NULL);
    OCL_ERRCK(ciErrNum);

/*
ciErrNum = clFinish(clCommandQueue);
    OCL_ERRCK(ciErrNum);

  print_test_sads(sads_computed, image_size_macroblocks);
for (int i = 26*MAX_POS_PADDED; i < 26*MAX_POS_PADDED+41; i += 3) {
printf("Host %2d:%5u %2d:%5u %2d:%5u\n", i, sads_computed[i],i+1, sads_computed[i+1],i+2, sads_computed[i+2]);
}*/

    /* Free GPU memory */
    

    ciErrNum = clReleaseKernel(larger_sad_calc_8);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseKernel(larger_sad_calc_16);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseProgram(clProgram);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseCommandQueue(clCommandQueue);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseContext(clContext);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseMemObject(d_sads);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseMemObject(imgRef);
    OCL_ERRCK(ciErrNum);
    ciErrNum = clReleaseMemObject(d_cur_image);
    OCL_ERRCK(ciErrNum);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    
  }

  /* Print output */
  if (params->outFile)
    {
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      write_sads(params->outFile, image_size_macroblocks, sads_computed);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

#if 0  /* Debugging */
  print_test_sads(sads_computed, image_size_macroblocks);
  write_sads_directly("sad-debug.bin",
		      ref_image->width / 16, ref_image->height / 16,
		      sads_computed);
#endif

  /* Free memory */
  free(sads_computed);
  free_image(ref_image);
  free_image(cur_image);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}
