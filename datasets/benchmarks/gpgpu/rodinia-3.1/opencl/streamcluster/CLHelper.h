#include <libcecl.h>
/********************************************************************
//--cambine:helper function for OpenCL
//--programmer:	Jianbin Fang
//--date:	27/12/2010
********************************************************************/
#ifndef _CL_HELPER_
#define _CL_HELPER_

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>


using std::string;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

#define PROFILE_
#ifdef PROFILE_
double TE;	//: total execution time;
double CC;	//: Context creation time;
double CR;	//: Context release time;
double MA;	//: GPU memory allocation time;
double MF;	//: GPU memory free time;
double H2D;	//: the time to transfer data from host to device;
double D2H;	//: the time to transfer data from device to host;
double D2D; //: the time to transfer data from device to device;
double KE;	//: the kernel execution time
double KC;  //: the kernel compilation time
#endif

//#pragma OPENCL EXTENSION cl_nv_compiler_options:enable
#define WORK_DIM 2	//work-items dimensions
/*------------------------------------------------------------
	@struct:	the structure of device properties
	@date:		24/03/2011
------------------------------------------------------------*/
struct _clDeviceProp{
/*CL_DEVICE_ADDRESS_BITS                       
CL_DEVICE_AVAILABLE                          
CL_DEVICE_COMPILER_AVAILABLE                 
CL_DEVICE_ENDIAN_LITTLE                      
CL_DEVICE_ERROR_CORRECTION_SUPPORT           
CL_DEVICE_EXECUTION_CAPABILITIES             
CL_DEVICE_EXTENSIONS
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE              
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE              
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE          
CL_DEVICE_GLOBAL_MEM_SIZE                    
CL_DEVICE_IMAGE_SUPPORT                      
CL_DEVICE_IMAGE2D_MAX_HEIGHT                 
CL_DEVICE_IMAGE2D_MAX_WIDTH                  
CL_DEVICE_IMAGE3D_MAX_DEPTH                  
CL_DEVICE_IMAGE3D_MAX_HEIGHT                 
CL_DEVICE_IMAGE3D_MAX_WIDTH                  
CL_DEVICE_LOCAL_MEM_SIZE                     
CL_DEVICE_LOCAL_MEM_TYPE                     
CL_DEVICE_MAX_CLOCK_FREQUENCY                
CL_DEVICE_MAX_COMPUTE_UNITS                  
CL_DEVICE_MAX_CONSTANT_ARGS                  
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE           
CL_DEVICE_MAX_MEM_ALLOC_SIZE                 
CL_DEVICE_MAX_PARAMETER_SIZE                 
CL_DEVICE_MAX_READ_IMAGE_ARGS                
CL_DEVICE_MAX_SAMPLERS                       
CL_DEVICE_MAX_WORK_GROUP_SIZE                
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS           
CL_DEVICE_MAX_WORK_ITEM_SIZES                
CL_DEVICE_MAX_WRITE_IMAGE_ARGS               
CL_DEVICE_MEM_BASE_ADDR_ALIGN                
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE           
CL_DEVICE_NAME                               
CL_DEVICE_PLATFORM
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR        
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE      
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT       
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT         
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG        
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT       
CL_DEVICE_PROFILE                           
CL_DEVICE_PROFILING_TIMER_RESOLUTION         
CL_DEVICE_QUEUE_PROPERTIES                   
CL_DEVICE_SINGLE_FP_CONFIG                   
CL_DEVICE_TYPE                              
CL_DEVICE_VENDOR_ID                          
CL_DEVICE_VENDOR                            
CL_DEVICE_VERSION                      
CL_DRIVER_VERSION*/
char device_name[100];
};
struct oclHandleStruct{
    cl_context              context;
    cl_device_id            *devices;
    cl_command_queue        queue;
    cl_program              program;
    cl_int		cl_status;
    std::string error_str;
    std::vector<cl_kernel>  kernel;
    cl_mem pinned_mem_out;
    cl_mem pinned_mem_in;
};

struct oclHandleStruct oclHandles;

char kernel_file[100]  = "Kernels.cl";
int total_kernels = 2;
string kernel_names[2] = {"memset_kernel", "pgain_kernel"};
int work_group_size = 256;
int device_id_inused = 0; //deviced id used (default : 0)
int number_devices = 0;


double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}
/*------------------------------------------------------------
	@function:	select device to use
	@params:	
		size: 	the index of device to be used
	@return:	NULL
	@date:		24/03/2011
------------------------------------------------------------*/
void _clSetDevice(int idx) throw(string){
	if(idx>(number_devices-1))
	  throw(string(":invalid device ID:"));
	device_id_inused = idx;
	
}

/*------------------------------------------------------------
	@function:	get device properties indexed by 'idx'
	@params:	
		idx:	device index
		prop:	output properties
	@return:	prop
	@date:		24/03/2011
------------------------------------------------------------*/
void _clGetDeviceProperties(int idx, _clDeviceProp *prop) throw(string){
   
   oclHandles.cl_status= clGetDeviceInfo(oclHandles.devices[idx], CL_DEVICE_NAME, 100, prop->device_name, NULL);
   
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
	  oclHandles.error_str = "exception in _clGetDeviceProperties-> ";
	  switch(oclHandles.cl_status){
	  	case CL_INVALID_DEVICE:
	  		oclHandles.error_str += "CL_INVALID_DEVICE";
	  		break;
	  	case CL_INVALID_VALUE:
	  		oclHandles.error_str += "CL_INVALID_VALUE";
	  		break;	   
	  	default:
	  		oclHandles.error_str += "unknown reasons";
	  		break;	  		
	  }
	throw(oclHandles.error_str);
	}	
#endif
}

/*
 * Converts the contents of a file into a string
 */
string FileToString(const string fileName){
    ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

    try{
        size_t size;
        char*  str;
        string s;

        if(f.is_open()){
            size_t fileSize;
            f.seekg(0, ifstream::end);
            size = fileSize = f.tellg();
            f.seekg(0, ifstream::beg);

            str = new char[size+1];
            if (!str) throw(string("Could not allocate memory"));

            f.read(str, fileSize);
            f.close();
            str[size] = '\0';
        
            s = str;
            delete [] str;
            return s;
        }
    }
    catch(std::string msg){
        cerr << "Exception caught in FileToString(): " << msg << endl;
        if(f.is_open())
            f.close();
    }
    catch(...){
        cerr << "Exception caught in FileToString()" << endl;
        if(f.is_open())
            f.close();
    }
    string errorMsg = "FileToString()::Error: Unable to open file "
                            + fileName;
    throw(errorMsg);
}

/*------------------------------------------------------------
	@function:	Read command line parameters
	@params:	NULL
	@return:
	@date:		24/03/2011
------------------------------------------------------------*/
char device_type[3];
int device_id = 0;
void _clCmdParams(int argc, char* argv[]){
	for (int i = 0; i < argc; ++i){
		switch (argv[i][1]){
			case 't':	//--t stands for device type
				if (++i < argc){
					sscanf(argv[i], "%s", device_type);
				}
				else{
					std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
					throw;
				}
				break;
			  case 'd':	 //--d stands for device id
				if (++i < argc){
					sscanf(argv[i], "%d", &device_id);
				}
				else{
					std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
					throw;
				}
				break;
			default:
				;
		}
	}
}

/*------------------------------------------------------------
	@function:	Initlize CL objects
	@params:	
		device_id: device id
		device_type: the types of devices, e.g. CPU, GPU, ACCERLERATOR,...	
		(1) -t cpu/gpu/acc -d 0/1/2/...
		(2) -t cpu/gpu/acc [-d 0]
		(3) [-t default] -d 0/1/2/...
		(4) NULL [-d 0]
	@return:
	@description:
		there are 5 steps to initialize all the OpenCL objects needed,
	@revised: 
		get the number of devices and devices have no relationship with context
	@date:		24/03/2011
------------------------------------------------------------*/
void _clInit(string device_type, int device_id)throw(string){

#ifdef PROFILE_
	TE = 0;
	CC = 0;
	CR = 0;
	MA = 0;
	MF = 0;
	H2D = 0;
	D2H = 0;
	D2D = 0;
	KE = 0;
	KC = 0;
#endif
	int DEVICE_ID_INUSED = 0;	
	_clDeviceProp prop;
#ifdef PROFILE_
	double t1 = gettime();
#endif

    cl_int resultCL;    
    oclHandles.context = NULL;
    oclHandles.devices = NULL;
    oclHandles.queue = NULL;
    oclHandles.program = NULL;

    cl_uint deviceListSize;
    //-----------------------------------------------
    //--cambine-1: find the available platforms and select one

    cl_uint numPlatforms;
    cl_platform_id targetPlatform = NULL;

    resultCL = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting number of platforms (clGetPlatformIDs)"));
    //printf("number of platforms:%d\n",numPlatforms);	//by cambine
#ifdef	DEV_INFO
	std::cout<<"--cambine: number of platforms: "<<numPlatforms<<std::endl;
#endif

    if (!(numPlatforms > 0))
        throw (string("InitCL()::Error: No platforms found (clGetPlatformIDs)"));

    cl_platform_id* allPlatforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

    resultCL = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting platform ids (clGetPlatformIDs)"));

    // Select the target platform. Default: first platform 
    targetPlatform = allPlatforms[0];
    for (int i = 0; i < numPlatforms; i++)
    {
        char pbuff[128];
        resultCL = clGetPlatformInfo( allPlatforms[i],
                                        CL_PLATFORM_VENDOR,
                                        sizeof(pbuff),
                                        pbuff,
                                        NULL);
        if (resultCL != CL_SUCCESS)
            throw (string("InitCL()::Error: Getting platform info (clGetPlatformInfo)"));

		//printf("vedor is %s\n",pbuff);
#ifdef	DEV_INFO
	std::cout<<"--cambine: vedor is: "<<pbuff<<std::endl;
#endif

    }
    free(allPlatforms);
    //-----------------------------------------------
    //--cambine-2: detect OpenCL devices	
    // First, get the size of device list 
    if(device_type.compare("")!=0){
		if(device_type.compare("cpu")==0){
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceListSize);
		   if(oclHandles.cl_status!=CL_SUCCESS){
			   throw(string("exception in _clInit -> clGetDeviceIDs -> CPU"));   				
	       }
		}	
		if(device_type.compare("gpu")==0){		   
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceListSize);
		   if(oclHandles.cl_status!=CL_SUCCESS){
			   throw(string("exception in _clInit -> clGetDeviceIDs -> GPU"));   	
	       }
		}
		if(device_type.compare("acc")==0){
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceListSize);
		   if(oclHandles.cl_status!=CL_SUCCESS){
			   throw(string("exception in _clInit -> clGetDeviceIDs -> ACCELERATOR"));   	
	       }
		}	 	
    }
    else{
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceListSize);
		   if(oclHandles.cl_status!=CL_SUCCESS){
			   throw(string("exception in _clInit -> clGetDeviceIDs -> ALL"));   	
	       }
    }
    
   if (deviceListSize == 0)
        throw(string("InitCL()::Error: No devices found."));
        
#ifdef	DEV_INFO
	std::cout<<"--cambine: number of device="<<deviceListSize<<std::endl;
#endif
	number_devices = deviceListSize;
    // Now, allocate the device list 
	//    oclHandles.devices = (cl_device_id *)malloc(deviceListSize);
    oclHandles.devices = (cl_device_id *)malloc(sizeof(cl_device_id)*deviceListSize);

    if (oclHandles.devices == 0)
        throw(string("InitCL()::Error: Could not allocate memory."));

    // Next, get the device list data 
	if(device_type.compare("")!=0){
	   if(device_type.compare("cpu")==0){
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, deviceListSize, oclHandles.devices, NULL);
		   if(oclHandles.cl_status!=CL_SUCCESS){
		   		throw(string("exception in _clInit -> clGetDeviceIDs -> CPU ->2"));   		   	
	   	   }
	   }
   	   if(device_type.compare("gpu")==0){
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, deviceListSize, oclHandles.devices, NULL);
		   if(oclHandles.cl_status!=CL_SUCCESS){
		   	throw(string("exception in _clInit -> clGetDeviceIDs -> GPU -> 2"));   		   	
	   	   }
   	 	}
   	   if(device_type.compare("acc")==0){
		   oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, deviceListSize, oclHandles.devices, NULL);
		   if(oclHandles.cl_status!=CL_SUCCESS){
		   	throw(string("exception in _clInit -> clGetDeviceIDs -> ACCELERATOR -> 2"));   		   	
	   	   }
   	   }  	 	
   	}
   	else{
	   	 oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_GPU, deviceListSize, oclHandles.devices, NULL);
	   	 if(oclHandles.cl_status!=CL_SUCCESS){
	   	 	throw(string("exception in _clInit -> clGetDeviceIDs -> ALL -> 2"));   		   	
   	   	}
   	}
   if(device_id!=0){
   	if(device_id>(deviceListSize-1))
   		throw(string("Invalidate device id"));
   	DEVICE_ID_INUSED = device_id;
   }
     
   	_clGetDeviceProperties(DEVICE_ID_INUSED, &prop);
	std::cout<<"--cambine: device name="<<prop.device_name<<std::endl;
	   	
#ifdef	DEV_INFO
	std::cout<<"--cambine: return device list successfully!"<<std::endl;
#endif  
 
    //-----------------------------------------------
    //--cambine-3: create an OpenCL context
#ifdef	DEV_INFO
	std::cout<<"--cambine: before creating context"<<std::endl;
#endif
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)targetPlatform, 0 };
    oclHandles.context = CECL_CREATE_CONTEXT(0, 
                                        deviceListSize, 
                                        oclHandles.devices, 
                                        NULL,
										NULL,
                                        &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL))
        throw (string("InitCL()::Error: Creating Context (CECL_CREATE_CONTEXT_FROM_TYPE)"));
#ifdef	DEV_INFO
	std::cout<<"--cambine: create OCL context successfully!"<<std::endl;
#endif

   //-----------------------------------------------
   //--cambine-4: Create an OpenCL command queue    
    oclHandles.queue = CECL_CREATE_COMMAND_QUEUE(oclHandles.context, 
                                            oclHandles.devices[DEVICE_ID_INUSED], 
                                            0, 
                                            &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.queue == NULL))
        throw(string("InitCL()::Creating Command Queue. (CECL_CREATE_COMMAND_QUEUE)"));
#ifdef PROFILE_
	double t2 = gettime();
	CC += t2 - t1;
#endif
    //-----------------------------------------------
    //--cambine-5: Load CL file, build CL program object, create CL kernel object
    std::string  source_str = FileToString(kernel_file);
    const char * source    = source_str.c_str();
    size_t sourceSize[]    = { source_str.length() };

    oclHandles.program = CECL_PROGRAM_WITH_SOURCE(oclHandles.context, 
                                                    1, 
                                                    &source,
                                                    sourceSize,
                                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
        throw(string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));    
    //insert debug information
    std::string options= "";
    //options += " -cl-nv-opt-level=3";
    resultCL = CECL_PROGRAM(oclHandles.program, deviceListSize, oclHandles.devices, options.c_str(), NULL,  NULL);
	
    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)){
        cerr << "InitCL()::Error: In CECL_PROGRAM" << endl;

		size_t length;
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[DEVICE_ID_INUSED], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        0, 
                                        NULL, 
                                        &length);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		char* buffer = (char*)malloc(length);
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[DEVICE_ID_INUSED], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        length, 
                                        buffer, 
                                        NULL);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		cerr << buffer << endl;
		FILE * fp = fopen("errinfo", "w");
		fprintf(fp, "%s\n", buffer);
		fclose(fp);
        free(buffer);

        throw(string("InitCL()::Error: Building Program (CECL_PROGRAM)"));
    } 
#ifdef PROFILE_
	double t3 = gettime();
	KC += t3 - t2;
#endif
    //get program information in intermediate representation
#ifdef PTX_MSG    
    size_t binary_sizes[deviceListSize];
    char * binaries[deviceListSize];
    //figure out number of devices and the sizes of the binary for each device. 
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*deviceListSize, &binary_sizes, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-2"));
    }

    std::cout<<"--cambine:"<<binary_sizes<<std::endl;
    //copy over all of the generated binaries. 
    for(int i=0;i<deviceListSize;i++)
	binaries[i] = (char *)malloc( sizeof(char)*(binary_sizes[i]+1));
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARIES, sizeof(char *)*deviceListSize, binaries, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-3"));
    }
    for(int i=0;i<deviceListSize;i++)
      binaries[i][binary_sizes[i]] = '\0';
    std::cout<<"--cambine:writing ptd information..."<<std::endl;
    FILE * ptx_file = fopen("cl.ptx","w");
    if(ptx_file==NULL){
	throw(string("exceptions in allocate ptx file."));
    }
    fprintf(ptx_file,"%s",binaries[DEVICE_ID_INUSED]);
    fclose(ptx_file);
    std::cout<<"--cambine:writing ptd information done."<<std::endl;
    for(int i=0;i<deviceListSize;i++)
	free(binaries[i]);
#endif

    for (int nKernel = 0; nKernel < total_kernels; nKernel++)
    {
        // get a kernel object handle for a kernel with the given name 
        cl_kernel kernel = CECL_KERNEL(oclHandles.program,
                                            (kernel_names[nKernel]).c_str(),
                                            &resultCL);

        if ((resultCL != CL_SUCCESS) || (kernel == NULL))
        {
            string errorMsg = "InitCL()::Error: Creating Kernel (CECL_KERNEL) \"" + kernel_names[nKernel] + "\"";
            throw(errorMsg);
        }

        oclHandles.kernel.push_back(kernel);
    }
  //get resource alocation information
#ifdef RES_MSG
    char * build_log;
    size_t ret_val_size;
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    if(oclHandles.cl_status!=CL_SUCCESS){
		throw(string("exceptions in _InitCL -> getting resource information"));
    }    

    build_log = (char *)malloc(ret_val_size+1);
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    if(oclHandles.cl_status!=CL_SUCCESS){
		throw(string("exceptions in _InitCL -> getting resources allocation information-2"));
    }
    build_log[ret_val_size] = '\0';
    std::cout<<"--cambine:"<<build_log<<std::endl;
    free(build_log);
#endif
#ifdef PROFILE_
	double t4 = gettime();
	CC += t4 - t3;
#endif
}


/*------------------------------------------------------------
	@function:	release CL objects
	@params:	NULL
	@return:
	@date:		24/03/2011
------------------------------------------------------------*/
void _clRelease()
{
#ifdef PROFILE_
	double t1 = gettime();
#endif
    bool errorFlag = false;

    for (int nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++){
        if (oclHandles.kernel[nKernel] != NULL){
            cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
            if (resultCL != CL_SUCCESS){
                cerr << "ReleaseCL()::Error: In clReleaseKernel" << endl;
                errorFlag = true;
            }
            oclHandles.kernel[nKernel] = NULL;
        }
        oclHandles.kernel.clear();
    }

    if (oclHandles.program != NULL){
        cl_int resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS){
            cerr << "ReleaseCL()::Error: In clReleaseProgram" << endl;
            errorFlag = true;
        }
        oclHandles.program = NULL;
    }

    if (oclHandles.queue != NULL){
        cl_int resultCL = clReleaseCommandQueue(oclHandles.queue);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseCommandQueue" << endl;
            errorFlag = true;
        }
        oclHandles.queue = NULL;
    }

    free(oclHandles.devices);

    if (oclHandles.context != NULL){
        cl_int resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS){
            cerr << "ReleaseCL()::Error: In clReleaseContext" << endl;
            errorFlag = true;
        }
        oclHandles.context = NULL;
    }

    if (errorFlag) throw(string("ReleaseCL()::Error encountered."));
#ifdef PROFILE_
	double t2 = gettime();
	CR += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	create read and write buffer for devices
	@params:	
		size: 	the size of device memory to be allocated
	@return:	mem_d
	@date:		24/03/2011
------------------------------------------------------------*/
cl_mem _clMalloc(int size) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
 	cl_mem d_mem;
	d_mem = CECL_BUFFER(oclHandles.context, CL_MEM_READ_WRITE, size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
	  oclHandles.error_str = "excpetion in _clMalloc -> ";
	  switch(oclHandles.cl_status){
	  	case CL_INVALID_CONTEXT:
	  		oclHandles.error_str += "CL_INVALID_CONTEXT";
	  		break;
	  	case CL_INVALID_VALUE:
	  		oclHandles.error_str += "CL_INVALID_VALUE";
	  		break;
	   	case CL_INVALID_BUFFER_SIZE:
	  		oclHandles.error_str += "CL_INVALID_BUFFER_SIZE";
	  		break;
	  	case CL_INVALID_HOST_PTR:
	  		oclHandles.error_str += "CL_INVALID_HOST_PTR";
	  		break;
	  	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
	  		oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	  		break;
	  	case CL_OUT_OF_HOST_MEMORY:
	  		oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
	  		break;
	  	default:
	  		oclHandles.error_str += "unknown reasons";
	  		break;	  		
	  }
	throw(oclHandles.error_str);
	}	
#endif
#ifdef PROFILE_
	double t2 = gettime();
	MA += t2 - t1;
#endif
	return d_mem;
}
/*------------------------------------------------------------
	@function:	malloc pinned memoty
	@params:
		size: 	the size of data to be transferred in bytes
	@return:	the pointer of host adress
	@date:		06/04/2011
------------------------------------------------------------*/

void* _clMallocHost(int size)throw(string){
	void * mem_h;
	oclHandles.pinned_mem_out = CECL_BUFFER(oclHandles.context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
	  oclHandles.error_str = "excpetion in _clMallocHost -> CECL_BUFFER";
	  switch(oclHandles.cl_status){
	  	case CL_INVALID_CONTEXT:
	  		oclHandles.error_str += "CL_INVALID_CONTEXT";
	  		break;
	  	case CL_INVALID_VALUE:
	  		oclHandles.error_str += "CL_INVALID_VALUE";
	  		break;
	   	case CL_INVALID_BUFFER_SIZE:
	  		oclHandles.error_str += "CL_INVALID_BUFFER_SIZE";
	  		break;
	  	case CL_INVALID_HOST_PTR:
	  		oclHandles.error_str += "CL_INVALID_HOST_PTR";
	  		break;
	  	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
	  		oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	  		break;
	  	case CL_OUT_OF_HOST_MEMORY:
	  		oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
	  		break;
	  	default:
	  		oclHandles.error_str += "unknown reasons";
	  		break;	  		
	  }
	throw(oclHandles.error_str);
	}	
#endif

	mem_h = CECL_MAP_BUFFER(oclHandles.queue, oclHandles.pinned_mem_out, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &oclHandles.cl_status);
	
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS||mem_h==NULL){
	  oclHandles.error_str = "excpetion in _clMallocHost -> CECL_MAP_BUFFER";
	  switch(oclHandles.cl_status){
	  	case CL_INVALID_COMMAND_QUEUE:
	  		oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
	  		break;
	  	case CL_INVALID_CONTEXT:
	  		oclHandles.error_str += "CL_INVALID_CONTEXT";
	  		break;
	  	case CL_INVALID_MEM_OBJECT:
	  		oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
	  		break;
	  	case CL_INVALID_VALUE:
	  		oclHandles.error_str += "CL_INVALID_VALUE";
	  		break;
	   	case CL_INVALID_EVENT_WAIT_LIST:
	  		oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
	  		break;
	  	case CL_MAP_FAILURE:
	  		oclHandles.error_str += "CL_MAP_FAILURE";
	  		break;
	  	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
	  		oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	  		break;
	  	case CL_OUT_OF_HOST_MEMORY:
	  		oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
	  		break;
	  	default:
	  		oclHandles.error_str += "unknown reasons";
	  		break;	  		
	  }
	throw(oclHandles.error_str);
	}	
#endif
	return mem_h;
}
/*------------------------------------------------------------
	@function:	free pinned memory
	@params:
		io:		to free pinned-in or pinned-out memory
		mem_h: 	the host address
	@return:	NULL
	@date:		06/04/2011
------------------------------------------------------------*/
void _clFreeHost(int io, void * mem_h){
	if(io==0){	//in
		if(mem_h){
			oclHandles.cl_status = clEnqueueUnmapMemObject(oclHandles.queue, oclHandles.pinned_mem_in, (void*)mem_h, 0, NULL, NULL);
#ifdef ERRMSG
			if(oclHandles.cl_status != CL_SUCCESS){
			  oclHandles.error_str = "excpetion in _clFreeHost -> clEnqueueUnmapMemObject(in)";
			  switch(oclHandles.cl_status){
			  	case CL_INVALID_COMMAND_QUEUE:
			  		oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			  		break;
			  	case CL_INVALID_MEM_OBJECT:
			  		oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
			  		break;
			  	case CL_INVALID_VALUE:
			  		oclHandles.error_str += "CL_INVALID_VALUE";
			  		break;
			   	case CL_OUT_OF_RESOURCES:
			  		oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			  		break;
			  	case CL_OUT_OF_HOST_MEMORY:
			  		oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			  		break;
			  	case CL_INVALID_CONTEXT:
			  		oclHandles.error_str += "CL_INVALID_CONTEXT";
			  		break;
			  	default:
			  		oclHandles.error_str += "unknown reasons";
			  		break;	  		
			  }
				throw(oclHandles.error_str);
			 }
#endif
	 	}
	}
	else if(io==1){	//out
			if(mem_h){
				oclHandles.cl_status = clEnqueueUnmapMemObject(oclHandles.queue, oclHandles.pinned_mem_out, (void*)mem_h, 0, NULL, NULL);
	#ifdef ERRMSG
				if(oclHandles.cl_status != CL_SUCCESS){
				  oclHandles.error_str = "excpetion in _clFreeHost -> clEnqueueUnmapMemObject(in)";
				  switch(oclHandles.cl_status){
				  	case CL_INVALID_COMMAND_QUEUE:
				  		oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				  		break;
				  	case CL_INVALID_MEM_OBJECT:
				  		oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				  		break;
				  	case CL_INVALID_VALUE:
				  		oclHandles.error_str += "CL_INVALID_VALUE";
				  		break;
				   	case CL_OUT_OF_RESOURCES:
				  		oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				  		break;
				  	case CL_OUT_OF_HOST_MEMORY:
				  		oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				  		break;
				  	case CL_INVALID_CONTEXT:
				  		oclHandles.error_str += "CL_INVALID_CONTEXT";
				  		break;
				  	default:
				  		oclHandles.error_str += "unknown reasons";
				  		break;	  		
				  }
					throw(oclHandles.error_str);
			   }
#endif
	 	 }		
	}
	else
		throw(string("encounter invalid choice when freeing pinned memmory"));
}
/*------------------------------------------------------------
	@function:	transfer data from host to device
	@params:
		dest:	the destination device memory
		src:	the source host memory	
		size: 	the size of data to be transferred in bytes
	@return:	NULL
	@date:		17/01/2011
------------------------------------------------------------*/
void _clMemcpyH2D(cl_mem dst, const void *src, int size) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	oclHandles.cl_status = CECL_WRITE_BUFFER(oclHandles.queue, dst, CL_TRUE, 0, size, src, 0, NULL, NULL);
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clMemcpyH2D -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;	
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;
			case CL_INVALID_VALUE:
				oclHandles.error_str += "CL_INVALID_VALUE";
				break;	
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;	
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;		
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}		
		throw(oclHandles.error_str);
	}		
#endif
#ifdef PROFILE_
	double t2 = gettime();
	H2D += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	transfer data from device to host
	@params:
		dest:	the destination device memory
		src:	the source host memory	
		size: 	the size of data to be transferred in bytes
	@return:	NULL
	@date:		17/01/2011
------------------------------------------------------------*/
void _clMemcpyD2H(void * dst, cl_mem src, int size) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	oclHandles.cl_status = CECL_READ_BUFFER(oclHandles.queue, src, CL_TRUE, 0, size, dst, 0,0,0);
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clMemCpyD2H -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;	
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;
			case CL_INVALID_VALUE:
				oclHandles.error_str += "CL_INVALID_VALUE";
				break;	
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;	
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;		
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}		
		throw(oclHandles.error_str);
	}		
#endif
#ifdef PROFILE_
	double t2 = gettime();
	D2H += t2 - t1;
#endif
}
/*------------------------------------------------------------
	@function:	transfer data from device to device
	@params:
		dest:	the destination device memory
		src:	the source device memory	
		size: 	the size of data to be transferred in bytes
	@return:	NULL
	@date:		27/03/2011
------------------------------------------------------------*/
void _clMemcpyD2D(cl_mem dst, cl_mem src, int size) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	oclHandles.cl_status = clEnqueueCopyBuffer(oclHandles.queue, src, dst, 0, 0, size, 0, NULL, NULL);
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clCpyMemD2D -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;	
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;
			case CL_INVALID_VALUE:
				oclHandles.error_str += "CL_INVALID_VALUE";
				break;	
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_MISALIGNED_SUB_BUFFER_OFFSET:
				oclHandles.error_str += "CL_MISALIGNED_SUB_BUFFER_OFFSET";
				break;	
			case CL_MEM_COPY_OVERLAP:
				oclHandles.error_str += "CL_MEM_COPY_OVERLAP";
				break;	
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;	
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;	
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}		
		throw(oclHandles.error_str);
	}		
#endif
#ifdef PROFILE_
	double t2 = gettime();
	D2D += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	set kernel arguments
	@params:
		kernel_id:	the index of kernel to set
		arg_idx:	the index of argument
		d_mem: 		the variable of device memory
		size:		the size of device memory
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/
void _clSetArgs(int kernel_id, int arg_idx, void * d_mem, int size = 0) throw(string){
	if(!size){ // normal device memory object
		oclHandles.cl_status = CECL_SET_KERNEL_ARG(oclHandles.kernel[kernel_id], arg_idx, sizeof(d_mem), &d_mem);
#ifdef ERRMSG
		if(oclHandles.cl_status != CL_SUCCESS){
			oclHandles.error_str = "excpetion in _CECL_SET_KERNEL_ARG() ";
			switch(oclHandles.cl_status){
				case CL_INVALID_KERNEL:
					oclHandles.error_str += "CL_INVALID_KERNEL";
					break;
				case CL_INVALID_ARG_INDEX:
					oclHandles.error_str += "CL_INVALID_ARG_INDEX";
					break;	
				case CL_INVALID_ARG_VALUE:
					oclHandles.error_str += "CL_INVALID_ARG_VALUE";
					break;
				case CL_INVALID_MEM_OBJECT:
					oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
					break;	
				case CL_INVALID_SAMPLER:
					oclHandles.error_str += "CL_INVALID_SAMPLER";
					break;
				case CL_INVALID_ARG_SIZE:
					oclHandles.error_str += "CL_INVALID_ARG_SIZE";
					break;	
				case CL_OUT_OF_RESOURCES:
					oclHandles.error_str += "CL_OUT_OF_RESOURCES";
					break;
				case CL_OUT_OF_HOST_MEMORY:
					oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
					break;
				default:
					oclHandles.error_str += "Unknown reason";
					break;
			}
			throw(oclHandles.error_str);
		}
#endif
	}
	else{	//special device object:(1) local memory; (2) single word
		oclHandles.cl_status = CECL_SET_KERNEL_ARG(oclHandles.kernel[kernel_id], arg_idx, size, d_mem);
#ifdef ERRMSG
		if(oclHandles.cl_status != CL_SUCCESS){
				oclHandles.error_str = "excpetion in _CECL_SET_KERNEL_ARG() ";
			switch(oclHandles.cl_status){
				case CL_INVALID_KERNEL:
					oclHandles.error_str += "CL_INVALID_KERNEL";
					break;
				case CL_INVALID_ARG_INDEX:
					oclHandles.error_str += "CL_INVALID_ARG_INDEX";
					break;	
				case CL_INVALID_ARG_VALUE:
					oclHandles.error_str += "CL_INVALID_ARG_VALUE";
					break;
				case CL_INVALID_MEM_OBJECT:
					oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
					break;	
				case CL_INVALID_SAMPLER:
					oclHandles.error_str += "CL_INVALID_SAMPLER";
					break;
				case CL_INVALID_ARG_SIZE:
					oclHandles.error_str += "CL_INVALID_ARG_SIZE";
					break;	
				case CL_OUT_OF_RESOURCES:
					oclHandles.error_str += "CL_OUT_OF_RESOURCES";
					break;
				case CL_OUT_OF_HOST_MEMORY:
					oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
					break;
				default:
					oclHandles.error_str += "Unknown reason";
					break;
			}
			throw(oclHandles.error_str);
		}
#endif
	}
}
void _clFinish() throw(string){
	oclHandles.cl_status = clFinish(oclHandles.queue);	
#ifdef ERRMSG
	if(oclHandles.cl_status!=CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clFinish";
		switch(oclHandles.cl_status){
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;		
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reasons";
				break;
		}
		throw(oclHandles.error_str);
	}
#endif
}
/*------------------------------------------------------------
	@function:	entry of invoke the kernel function
	@params:
		kernel_id:	the index of kernel to set
		work_items:	the number of working items
		work_group_size: the size of each work group
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/
void _clInvokeKernel(int kernel_id, int work_items, int work_group_size) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	cl_uint work_dim = WORK_DIM;
	cl_event e[1];
	if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));
  	size_t local_work_size[] = {work_group_size, 1};
	size_t global_work_size[] = {work_items, 1};
	oclHandles.cl_status = CECL_ND_RANGE_KERNEL(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, \
											global_work_size, local_work_size, 0 , 0, &(e[0]) );	
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_PROGRAM_EXECUTABLE:
				oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
				break;
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;
			case CL_INVALID_KERNEL_ARGS:
				oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
				break;
			case CL_INVALID_WORK_DIMENSION:
				oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
				break;
			case CL_INVALID_GLOBAL_WORK_SIZE:
				oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
				break;
			case CL_INVALID_WORK_GROUP_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
				break;
			case CL_INVALID_WORK_ITEM_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
				break;
			case CL_INVALID_GLOBAL_OFFSET:
				oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
				break;
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default: 
				oclHandles.error_str += "Unkown reseason";
				break;		
		}
	
		throw(oclHandles.error_str);	
	}
	#endif
	//_clFinish();
	oclHandles.cl_status = clWaitForEvents(1, &e[0]);
	#ifdef ERRMSG
    if (oclHandles.cl_status!= CL_SUCCESS){
    	oclHandles.error_str = "excpetion in _clEnqueueNDRange() -> clWaitForEvents ->";
   		switch(oclHandles.cl_status){
    	case CL_INVALID_VALUE:
			oclHandles.error_str += "CL_INVALID_VALUE";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_EVENT:
			oclHandles.error_str += "CL_INVALID_EVENT";
			break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			oclHandles.error_str += "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			oclHandles.error_str += "Unkown Reason";
			break;
    	}
    	throw(oclHandles.error_str);
    }
    #endif
#ifdef PROFILE_
	double t2 = gettime();
	KE += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	set device memory in an easy manner
	@params:	
		mem_d: the device memory to be set;
		val:   set the selected memory to 'val';
		number_elements:	the number of elements in the selected memory
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/

void _clMemset(cl_mem mem_d, short val, int number_bytes)throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	int kernel_id = 0;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, mem_d);
	_clSetArgs(kernel_id, arg_idx++, &val, sizeof(short));
	_clSetArgs(kernel_id, arg_idx++, &number_bytes, sizeof(int));	
	cl_uint work_dim = WORK_DIM;
	int work_items = number_bytes;
	cl_event e[1];
	if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));
  	size_t local_work_size[] = {work_group_size, 1};
	size_t global_work_size[] = {work_items, 1};
	oclHandles.cl_status = CECL_ND_RANGE_KERNEL(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, \
											global_work_size, local_work_size, 0 , 0, &(e[0]) );	
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clMemset() -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_PROGRAM_EXECUTABLE:
				oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
				break;
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;
			case CL_INVALID_KERNEL_ARGS:
				oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
				break;
			case CL_INVALID_WORK_DIMENSION:
				oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
				break;
			case CL_INVALID_GLOBAL_WORK_SIZE:
				oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
				break;
			case CL_INVALID_WORK_GROUP_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
				break;
			case CL_INVALID_WORK_ITEM_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
				break;
			case CL_INVALID_GLOBAL_OFFSET:
				oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
				break;
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default: 
				oclHandles.error_str += "Unkown reseason";
				break;		
		}
	
		throw(oclHandles.error_str);	
	}
#endif
	//_clFinish();
	oclHandles.cl_status = clWaitForEvents(1, &e[0]);
#ifdef ERRMSG
    if (oclHandles.cl_status!= CL_SUCCESS){
    	oclHandles.error_str = "excpetion in _clMemset() -> clWaitForEvents ->";
   		switch(oclHandles.cl_status){
    	case CL_INVALID_VALUE:
			oclHandles.error_str += "CL_INVALID_VALUE";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_EVENT:
			oclHandles.error_str += "CL_INVALID_EVENT";
			break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			oclHandles.error_str += "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			oclHandles.error_str += "Unkown Reason";
			break;
    	}
    	throw(oclHandles.error_str);
    }
#endif
#ifdef PROFILE_
	double t2 = gettime();
	H2D += t2 - t1;
#endif
}
/*------------------------------------------------------------
	@function:	entry of invoke the kernel function using 2d working items
	@params:
		kernel_id:	the index of kernel to set
		range_x:	the number of working items in x direction
		range_y:	the number of working items in y direction
		group_x:	the number of working items in each work group in x direction
		group_y:	the number of working items	in each work group in y direction	 
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/
void _clInvokeKernel2D(int kernel_id, int range_x, int range_y, int group_x, int group_y) throw(string){
#ifdef PROFILE_
	double t1 = gettime();
#endif
	cl_uint work_dim = WORK_DIM;
	size_t local_work_size[] = {group_x, group_y};
	size_t global_work_size[] = {range_x, range_y};
	cl_event e[1];
	/*if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));*/
	oclHandles.cl_status = CECL_ND_RANGE_KERNEL(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, \
											global_work_size, local_work_size, 0 , 0, &(e[0]) );	
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_PROGRAM_EXECUTABLE:
				oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
				break;
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;
			case CL_INVALID_KERNEL_ARGS:
				oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
				break;
			case CL_INVALID_WORK_DIMENSION:
				oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
				break;
			case CL_INVALID_GLOBAL_WORK_SIZE:
				oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
				break;
			case CL_INVALID_WORK_GROUP_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
				break;
			case CL_INVALID_WORK_ITEM_SIZE:
				oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
				break;
			case CL_INVALID_GLOBAL_OFFSET:
				oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
				break;
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default: 
				oclHandles.error_str += "Unkown reseason";
				break;		
		}	
		throw(oclHandles.error_str);	
	}
	#endif
		
	oclHandles.cl_status = clWaitForEvents(1, &e[0]);

#ifdef ERRMSG
    if (oclHandles.cl_status!= CL_SUCCESS){
    	oclHandles.error_str = "excpetion in _clInvokeKernel2D() -> clWaitForEvents ->";
   		switch(oclHandles.cl_status){
    	case CL_INVALID_VALUE:
			oclHandles.error_str += "CL_INVALID_VALUE";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_EVENT:
			oclHandles.error_str += "CL_INVALID_EVENT";
			break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			oclHandles.error_str += "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			oclHandles.error_str += "Unkown Reason";
			break;
    	}
    	throw(oclHandles.error_str);
    }
#endif
#ifdef PROFILE_
	double t2 = gettime();
	KE += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	release OpenCL memory objects
	@params:
		ob:	the memory object to free or release
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/

void _clFree(cl_mem ob) throw(string){
#ifdef PROFILE_
	double t1 = gettime();	
#endif
	if(ob!=NULL)
		oclHandles.cl_status = clReleaseMemObject(ob);	
#ifdef ERRMSG
	if (oclHandles.cl_status!= CL_SUCCESS){
		oclHandles.error_str = "excpetion in _clFree() ->";
		switch(oclHandles.cl_status){
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;			
			default: 
				oclHandles.error_str += "Unkown reseason";
				break;		
		}
	throw(oclHandles.error_str);
	}
#endif
#ifdef PROFILE_
	double t2 = gettime();
	MF += t2 - t1;
#endif
}

/*------------------------------------------------------------
	@function:	output time profiling information
	@params:	NULL
	@return:	NULL
	@date:		03/04/2011
------------------------------------------------------------*/
void _clStatistics(){
#ifdef PROFILE_
	FILE *fp_pd = fopen("PD_OCL.txt", "a");
	fprintf(fp_pd, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", CC, CR, MA, MF, H2D, D2H, D2D, KE, KC);
	fclose(fp_pd);
#endif	
	return ;
}
#endif //_CL_HELPER_
