#include <libcecl.h>
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

#include "Shared.h"

extern int status;

#define SUCCESS 0
#define FAILURE 1
#define EXPECTED_FAILURE 2


cl_command_queue queue;
cl_context       context;
cl_program       program;
cl_kernel        read_kernel;
cl_kernel        write_kernel;
int              devnum;
char             devname[256];

cl_uint deviceMaxComputeUnits;
char extensions[1000];

cl_mem_flags inFlags = 0;
cl_mem_flags outFlags = 0;
cl_mem_flags copyFlags = 0;

struct _flags flags[] = {

             { CL_MEM_READ_ONLY,              "CL_MEM_READ_ONLY" },
             { CL_MEM_WRITE_ONLY,             "CL_MEM_WRITE_ONLY" },
             { CL_MEM_READ_WRITE,             "CL_MEM_READ_WRITE" },
             { CL_MEM_USE_HOST_PTR,           "CL_MEM_USE_HOST_PTR" },
             { CL_MEM_COPY_HOST_PTR,          "CL_MEM_COPY_HOST_PTR" },
             { CL_MEM_ALLOC_HOST_PTR,         "CL_MEM_ALLOC_HOST_PTR" },
             { CL_MEM_USE_PERSISTENT_MEM_AMD, "CL_MEM_USE_PERSISTENT_MEM_AMD"} };

int nFlags = sizeof(flags) / sizeof(flags[0]);

const char *cluErrorString(cl_int err) {

   switch(err) {

      case CL_SUCCESS: return "CL_SUCCESS";
      case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
      case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
      case CL_COMPILER_NOT_AVAILABLE: return
                                       "CL_COMPILER_NOT_AVAILABLE";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE: return
                                       "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
      case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
      case CL_PROFILING_INFO_NOT_AVAILABLE: return
                                       "CL_PROFILING_INFO_NOT_AVAILABLE";
      case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
      case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED: return
                                       "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
      case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
      case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
      case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
      case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
      case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
      case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
      case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
      case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
      case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
      case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return
                                       "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
      case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
      case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
      case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
      case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
      case CL_INVALID_PROGRAM_EXECUTABLE: return
                                       "CL_INVALID_PROGRAM_EXECUTABLE";
      case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
      case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
      case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
      case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
      case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
      case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
      case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
      case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
      case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
      case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
      case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
      case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
      case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
      case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
      case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
      case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
      case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
      case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";

      default: return "UNKNOWN CL ERROR CODE";
   }
}

cl_int spinForEventsComplete( cl_uint num_events, cl_event *event_list )
{
    cl_int ret = 0;
#if 0
    ret = clWaitForEvents( num_events, event_list );
#else
    cl_int param_value;
    size_t param_value_size_ret;

    for( cl_uint e=0; e < num_events; e++ )
    {
        while(1)
        {
            ret |= clGetEventInfo( event_list[ e ], 
                                   CL_EVENT_COMMAND_EXECUTION_STATUS,
                                   sizeof( cl_int ),
                                   &param_value,
                                   &param_value_size_ret );

            if( param_value == CL_COMPLETE )
                break;
        }
    }
#endif

    for( cl_uint e=0; e < num_events; e++ )
        clReleaseEvent( event_list[e] );

    return ret;
}

void initCL( char *kernel_file )
{
    // Get a platform, device, context and queue
	cl_platform_id   platform;
    cl_device_id     devices[128];
    cl_device_id     device;
    cl_uint          num_devices;
    cl_int           ret;

    cl_device_type   devs[] = { CL_DEVICE_TYPE_GPU,
                                CL_DEVICE_TYPE_GPU };

    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if(status != 0)
    {
		printf("clGetPlatformIDs failed.\n");
		exit(FAILURE);
	}
    if (0 < numPlatforms) 
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if(status != 0)
		{
			printf("clGetPlatformIDs failed.\n");
			exit(FAILURE);
		}

            char platformName[100];
            for (unsigned i = 0; i < numPlatforms; ++i) 
            {
                status = clGetPlatformInfo(platforms[i],
                                           CL_PLATFORM_VENDOR,
                                           sizeof(platformName),
                                           platformName,
                                           NULL);
				if(status != 0)
				{
					printf("clGetPlatformIDs failed.\n");
					exit(FAILURE);
				}

                platform = platforms[i];
                if (!strcmp(platformName, "Advanced Micro Devices, Inc.")) 
                {
                    break;
                }
            }
            std::cout << "Platform found : " << platformName << "\n";
        delete[] platforms;
    }

    if(NULL == platform)
    {
        printf("NULL platform found so Exiting Application.");
        exit(FAILURE);
    }

    ret = clGetDeviceIDs( platform,
                          devs[1],
                          128,
                          devices,
                          &num_devices );
    if(ret == CL_DEVICE_NOT_FOUND)
    {
        std::cout << "GPU not found. Exiting application" << std::endl;
        exit(FAILURE);
    }

    ASSERT_CL_RETURN( ret );

    device = devices[devnum];

    ret = clGetDeviceInfo( device,
                           CL_DEVICE_MAX_COMPUTE_UNITS,
                           sizeof(cl_uint),
                           &deviceMaxComputeUnits,
                           NULL);

    ASSERT_CL_RETURN( ret );

    ret = clGetDeviceInfo( device,
                           CL_DEVICE_NAME,
                           256,
                           devname,
                           NULL);

    ASSERT_CL_RETURN( ret );

    // Code Added to check if cl_khr_local_int32_base_atomics extension 
    // is supported by the selected device
    ret = clGetDeviceInfo( device,
                            CL_DEVICE_EXTENSIONS,
                            1000,
                            extensions,
                            NULL);

    ASSERT_CL_RETURN( ret );

    if(strstr(extensions,"cl_khr_local_int32_base_atomics") == NULL)
    {
        printf("Expected Error: cl_khr_local_int32_base_atomics is not supported by the device/n");
        status = EXPECTED_FAILURE;
		return;
    }

    context = CECL_CREATE_CONTEXT( NULL,
                               1,
                               &device,
                               NULL, NULL, NULL );

    queue = CECL_CREATE_COMMAND_QUEUE( context,
                                 device,
                                 0,
                                 NULL );

    // Minimal error check.

    if( queue == NULL ) {
        fprintf( stderr, "Compute device setup failed\n");
        exit(FAILURE);
    }

    // Perform runtime source compilation, and obtain kernel entry points.

    FILE *fp = fopen( kernel_file, "rb" );

    if( fp == NULL )
    {
        fprintf( stderr, "%s:%d: can't open kernel file: %s\n", \
             __FILE__, __LINE__, strerror( errno ));\
        exit(FAILURE);
    }

    fseek( fp, 0, SEEK_END );
    const size_t size = ftell( fp );
    const char *kernel_source = (const char *) malloc( size );

    rewind( fp );
    if ( size != (fread( (void *) kernel_source, 1, size, fp )) )
    {
        printf("fread failed\n");
   	    exit(FAILURE); 
    }
    

    program = CECL_PROGRAM_WITH_SOURCE( context,
                                         1,
                                         &kernel_source,
                                         &size, 
                                         NULL );

    ret = CECL_PROGRAM( program, 1, &device, NULL, NULL, NULL );

    static char buf[0x10000]={0};

    clGetProgramBuildInfo( program,
                           device,
                           CL_PROGRAM_BUILD_LOG,
                           0x10000,
                           buf,
                           NULL );

    printf( "%s\n", buf );

    ASSERT_CL_RETURN( ret );

    read_kernel = CECL_KERNEL( program, "read_kernel", &ret );

    ASSERT_CL_RETURN( ret );

    write_kernel = CECL_KERNEL( program, "write_kernel", &ret );

    ASSERT_CL_RETURN( ret );
}
