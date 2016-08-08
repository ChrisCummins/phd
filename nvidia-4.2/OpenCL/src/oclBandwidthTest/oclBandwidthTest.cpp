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
// 
// *********************************************************************

// standard utilities and systems includes
#include <oclUtils.h>
#include <shrQATest.h>

#include <memory>
#include <iostream>
#include <cassert>

// defines, project
#define MEMCOPY_ITERATIONS  100
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
#define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M

//shmoo mode defines
#define SHMOO_MEMSIZE_MAX     (1 << 26)         //64 M
#define SHMOO_MEMSIZE_START   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_1KB   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_2KB   (1 << 11)         //2 KB
#define SHMOO_INCREMENT_10KB  (10 * (1 << 10))  //10KB
#define SHMOO_INCREMENT_100KB (100 * (1 << 10)) //100 KB
#define SHMOO_INCREMENT_1MB   (1 << 20)         //1 MB
#define SHMOO_INCREMENT_2MB   (1 << 21)         //2 MB
#define SHMOO_INCREMENT_4MB   (1 << 22)         //4 MB
#define SHMOO_LIMIT_20KB      (20 * (1 << 10))  //20 KB
#define SHMOO_LIMIT_50KB      (50 * (1 << 10))  //50 KB
#define SHMOO_LIMIT_100KB     (100 * (1 << 10)) //100 KB
#define SHMOO_LIMIT_1MB       (1 << 20)         //1 MB
#define SHMOO_LIMIT_16MB      (1 << 24)         //16 MB
#define SHMOO_LIMIT_32MB      (1 << 25)         //32 MB

//enums, project
enum testMode { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode { USER_READABLE, CSV };
enum memoryMode { PAGEABLE, PINNED };
enum accessMode { MAPPED, DIRECT };

// CL objects
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_device_id *devices;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(const int argc, const char **argv);
void createQueue(unsigned int device);
void testBandwidth( unsigned int start, unsigned int end, unsigned int increment, 
                    testMode mode, memcpyKind kind, printMode printmode, accessMode accMode, memoryMode memMode, int startDevice, int endDevice);
void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, accessMode accMode, memoryMode memMode, int startDevice, int endDevice);
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                        memcpyKind kind, printMode printmode, accessMode accMode, memoryMode memMode, int startDevice, int endDevice);
void testBandwidthShmoo(memcpyKind kind, printMode printmode,accessMode accMode,  memoryMode memMode, int startDevice, int endDevice);
double testDeviceToHostTransfer(unsigned int memSize, accessMode accMode, memoryMode memMode);
double testHostToDeviceTransfer(unsigned int memSize, accessMode accMode, memoryMode memMode);
double testDeviceToDeviceTransfer(unsigned int memSize);
void printResultsReadable(unsigned int *memSizes, double* bandwidths, unsigned int count, memcpyKind kind, accessMode accMode, memoryMode memMode, int iNumDevs);
void printResultsCSV(unsigned int *memSizes, double* bandwidths, unsigned int count, memcpyKind kind, accessMode accMode, memoryMode memMode, int iNumDevs);
void printHelp(void);

int main(int argc, char** argv) 
{
    shrQAStart(argc, argv);

    // set logfile name and start logs
    shrSetLogFileName ("oclBandwidthTest.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // run the main test
    int iRetVal = runTest(argc, (const char **)argv);

    // finish
    shrQAFinishExit(argc, (const char **)argv, (iRetVal == 0) ? QA_PASSED : QA_FAILED);
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runTest(const int argc, const char **argv)
{
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int startDevice = 0;
    int endDevice = 0;
    int increment = DEFAULT_INCREMENT;
    testMode mode = QUICK_MODE;
    bool htod = false;
    bool dtoh = false;
    bool dtod = false;
    char *modeStr;
    char *device = NULL;
    printMode printmode = USER_READABLE;
    char *memModeStr = NULL;
    memoryMode memMode = PAGEABLE;
    accessMode accMode = DIRECT;

    //process command line args
    if(shrCheckCmdLineFlag( argc, argv, "help"))
    {
        printHelp();
        return 0;
    }

    if(shrCheckCmdLineFlag( argc, argv, "csv"))
    {
        printmode = CSV;
    }

    // Get host memory mode type from command line
    if(shrGetCmdLineArgumentstr(argc, argv, "memory", &memModeStr))
    {
        if(strcmp(memModeStr, "pageable") == 0 )
        {
            memMode = PAGEABLE;
        }
        else if(strcmp(memModeStr, "pinned") == 0)
        {
            memMode = PINNED;
        }
        else
        {
            shrLog("Invalid memory mode - valid modes are pageable or pinned\n");
            shrLog("See --help for more information\n");
            return -1000;
        }
    }
    else
    {
        //default - pageable memory
        memMode = PAGEABLE;
    }
   
    // Access type from command line
    if(shrGetCmdLineArgumentstr(argc, argv, "access", &memModeStr))
    {
        if(strcmp(memModeStr, "direct") == 0)
        {
            accMode = DIRECT;
        }
        else if(strcmp(memModeStr, "mapped") == 0)
        {
            accMode = MAPPED;
        }
        else
        {
            shrLog("Invalid access mode - valid modes are direct or mapped\n");
            shrLog("See --help for more information\n");
            return -2000;
        }
    }
    else
    {
        //default - direct 
        accMode = DIRECT;
    }

    // Get OpenCL platform ID for NVIDIA if available, otherwise default
    cl_platform_id clSelectedPlatformID = NULL; 
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Find out how many devices there are
    cl_uint ciDeviceCount;
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_CPU, 0, NULL, &ciDeviceCount);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        return ciErrNum;
    }
    else if (ciDeviceCount == 0)
    {
        shrLog(" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
        return ciErrNum;
    } 

    // Get command line device options and config accordingly
    if(shrGetCmdLineArgumentstr(argc, argv, "device", &device))
    {
        if(strcmp (device, "all") == 0)
        {
            shrLog("\n!!!Cumulative Bandwidth to be computed from all the devices !!!\n\n");
            startDevice = 0;
            endDevice = (int)(ciDeviceCount-1);
        }
        else
        {
            startDevice = endDevice = atoi(device);
            if(startDevice < 0 || ((size_t)startDevice) >= ciDeviceCount)
            {
                shrLog("\n!!!Invalid GPU number %d given hence default gpu %d will be used !!!\n", startDevice,0);
                startDevice = endDevice = 0;
            }
        }
    }
     
    // Get and log the device info
    shrLog("Running on...\n\n");
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * ciDeviceCount);
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_CPU, ciDeviceCount, devices, &ciDeviceCount);
    for(int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        oclPrintDevName(LOGBOTH, devices[currentDevice]);
        shrLog("\n");
    }
    shrLog("\n");

    // Get command line mode(s) and config accordingly
    if(shrGetCmdLineArgumentstr(argc, argv, "mode", &modeStr))
    {
        //figure out the mode
        if(strcmp(modeStr, "quick") == 0)
        {
            shrLog("Quick Mode\n\n");
            mode = QUICK_MODE;
        }
        else if(strcmp(modeStr, "shmoo") == 0)
        {
            shrLog("Shmoo Mode\n\n");
            mode = SHMOO_MODE;
        }
        else if(strcmp(modeStr, "range") == 0)
        {
            shrLog("Range Mode\n\n");
            mode = RANGE_MODE;
        }
        else
        {
            shrLog("Invalid mode - valid modes are quick, range, or shmoo\n");
            shrLog("See --help for more information\n\n");
            return -3000;
        }
    }
    else
    {
        //default mode - quick
        shrLog("Quick Mode\n\n");
        mode = QUICK_MODE;
    }
    
    if(shrCheckCmdLineFlag(argc, argv, "htod"))
        htod = true;
    if(shrCheckCmdLineFlag(argc, argv, "dtoh"))
        dtoh = true;
    if(shrCheckCmdLineFlag(argc, argv, "dtod"))
        dtod = true;

    if(!htod && !dtoh && !dtod)
    {
        //default:  All
        htod = true;
        dtoh = true;
        dtod = true;
    }

    if(RANGE_MODE == mode)
    {
        if(shrGetCmdLineArgumenti( argc, argv, "start", &start))
        {
            if( start <= 0 )
            {
                shrLog("Illegal argument - start must be greater than zero\n");
                return -4000;
            }   
        }
        else
        {
            shrLog("Must specify a starting size in range mode\n");
            shrLog("See --help for more information\n");
            return -5000;
        }

        if(shrGetCmdLineArgumenti( argc, argv, "end", &end))
        {
            if(end <= 0)
            {
                shrLog("Illegal argument - end must be greater than zero\n");
                return -6000;
            }

            if(start > end)
            {
                shrLog("Illegal argument - start is greater than end\n");
                return -7000;
            }
        }
        else
        {
            shrLog("Must specify an end size in range mode.\n");
            shrLog("See --help for more information\n");
            return -8000;
        }

        if(shrGetCmdLineArgumenti( argc, argv, "increment", &increment))
        {
            if(increment <= 0)
            {
                shrLog("Illegal argument - increment must be greater than zero\n");
                return -9000;
            }
        }
        else
        {
            shrLog("Must specify an increment in user mode\n");
            shrLog("See --help for more information\n");
            return -10000;
        }
    }
   
    // Create the OpenCL context
    cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, NULL);
    if (cxGPUContext == (cl_context)0) 
    {
        shrLog("Failed to create OpenCL context!\n");
        return -11000;    
    }

    // Run tests
    if(htod)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment, 
                      mode, HOST_TO_DEVICE, printmode, accMode, memMode, startDevice, endDevice);
    }                       
    if(dtoh)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, DEVICE_TO_HOST, printmode, accMode, memMode, startDevice, endDevice);
    }                       
    if(dtod)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, DEVICE_TO_DEVICE, printmode, accMode, memMode, startDevice, endDevice);
    }                       

    // Clean up 
    free(memModeStr); 
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(devices)free(devices);
    
    return 0;
}
///////////////////////////////////////////////////////////////////////////////
//  Create command queue for the selected device
///////////////////////////////////////////////////////////////////////////////
void
createQueue(unsigned int device)
{
    // Release if there previous is already one
    if(cqCommandQueue) 
    {
        clReleaseCommandQueue(cqCommandQueue);
    }
  
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, devices[device], CL_QUEUE_PROFILING_ENABLE, NULL);
}
  
///////////////////////////////////////////////////////////////////////////////
//  Run a bandwidth test
///////////////////////////////////////////////////////////////////////////////
void
testBandwidth(unsigned int start, unsigned int end, unsigned int increment, 
              testMode mode, memcpyKind kind, printMode printmode, accessMode accMode, 
              memoryMode memMode, int startDevice, int endDevice)
{
    switch(mode)
    {
    case QUICK_MODE:
        testBandwidthQuick( DEFAULT_SIZE, kind, printmode, accMode, memMode, startDevice, endDevice);
        break;
    case RANGE_MODE:
        testBandwidthRange(start, end, increment, kind, printmode, accMode, memMode, startDevice, endDevice);
        break;
    case SHMOO_MODE: 
        testBandwidthShmoo(kind, printmode, accMode, memMode, startDevice, endDevice);
        break;
    default:  
        break;
    }

}
//////////////////////////////////////////////////////////////////////
//  Run a quick mode bandwidth test
//////////////////////////////////////////////////////////////////////
void
testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, accessMode accMode, 
                   memoryMode memMode, int startDevice, int endDevice)
{
    testBandwidthRange(size, size, DEFAULT_INCREMENT, kind, printmode, accMode, memMode, startDevice, endDevice);
}

///////////////////////////////////////////////////////////////////////
//  Run a range mode bandwidth test
//////////////////////////////////////////////////////////////////////
void
testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                   memcpyKind kind, printMode printmode, accessMode accMode, memoryMode memMode, int startDevice, int endDevice)
{
    //count the number of copies we're going to run
    unsigned int count = 1 + ((end - start) / increment);
    
    unsigned int * memSizes = (unsigned int *)malloc(count * sizeof( unsigned int ));
    double* bandwidths = (double*)malloc(count * sizeof(double));

    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (unsigned int i = 0; i < count; i++)
        bandwidths[i] = 0.0;

    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        // Allocate command queue for the device (dealloc first if already allocated)
        createQueue(currentDevice);

        //run each of the copies
        for(unsigned int i = 0; i < count; i++)
        {
            memSizes[i] = start + i * increment;
            switch(kind)
            {
            case DEVICE_TO_HOST:    bandwidths[i] += testDeviceToHostTransfer(memSizes[i], accMode, memMode);
                break;
            case HOST_TO_DEVICE:    bandwidths[i] += testHostToDeviceTransfer(memSizes[i], accMode, memMode);
                break;
            case DEVICE_TO_DEVICE:  bandwidths[i] += testDeviceToDeviceTransfer(memSizes[i]);
                break;
            }
        }
    } // Complete the bandwidth computation on all the devices

    //print results
    if(printmode == CSV)
    {
        printResultsCSV(memSizes, bandwidths, count, kind, accMode, memMode, (1 + endDevice - startDevice));
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count, kind, accMode, memMode, (1 + endDevice - startDevice));
    }

    //clean up
    free(memSizes);
    free(bandwidths);
}

//////////////////////////////////////////////////////////////////////////////
// Intense shmoo mode - covers a large range of values with varying increments
//////////////////////////////////////////////////////////////////////////////
void testBandwidthShmoo(memcpyKind kind, printMode printmode, accessMode accMode, 
                   memoryMode memMode, int startDevice, int endDevice)
{
    //count the number of copies to make
    unsigned int count = 1 + (SHMOO_LIMIT_20KB  / SHMOO_INCREMENT_1KB)
        + ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB)
        + ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB)
        + ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB)
        + ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB)
        + ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB)
        + ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

    unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
    double* bandwidths = (double*)malloc(count * sizeof(double));

    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (unsigned int i = 0; i < count; i++)
        bandwidths[i] = 0.0;
   
    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        // Allocate command queue for the device (dealloc first if already allocated)
        createQueue(currentDevice);

        //Run the shmoo
        int iteration = 0;
        unsigned int memSize = 0;
        while(memSize <= SHMOO_MEMSIZE_MAX )
        {
            if(memSize < SHMOO_LIMIT_20KB )
            {
                memSize += SHMOO_INCREMENT_1KB;
            }
            else if( memSize < SHMOO_LIMIT_50KB)
            {
                memSize += SHMOO_INCREMENT_2KB;
            }
            else if( memSize < SHMOO_LIMIT_100KB)
            {
                memSize += SHMOO_INCREMENT_10KB;
            }
            else if( memSize < SHMOO_LIMIT_1MB)
            {
                memSize += SHMOO_INCREMENT_100KB;
            }
            else if( memSize < SHMOO_LIMIT_16MB)
            {
                memSize += SHMOO_INCREMENT_1MB;
            }
            else if( memSize < SHMOO_LIMIT_32MB)
            {
                memSize += SHMOO_INCREMENT_2MB;
            }
            else 
            {
                memSize += SHMOO_INCREMENT_4MB;
            }

            memSizes[iteration] = memSize;
            switch(kind)
            {
            case DEVICE_TO_HOST:    bandwidths[iteration] += testDeviceToHostTransfer(memSizes[iteration], accMode, memMode);
                break;
            case HOST_TO_DEVICE:    bandwidths[iteration] += testHostToDeviceTransfer(memSizes[iteration], accMode, memMode);
                break;
            case DEVICE_TO_DEVICE:  bandwidths[iteration] += testDeviceToDeviceTransfer(memSizes[iteration]);
                break;
            }
            iteration++;
            shrLog(".");
        }
    } // Complete the bandwidth computation on all the devices

    //print results
    shrLog("\n");
    if( CSV == printmode)
    {
        printResultsCSV(memSizes, bandwidths, count,  kind, accMode, memMode, (endDevice - startDevice));
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count, kind, accMode, memMode, (endDevice - startDevice));
    }

    //clean up
    free(memSizes);
    free(bandwidths);
}

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
double testDeviceToHostTransfer(unsigned int memSize, accessMode accMode, memoryMode memMode)
{
    double elapsedTimeInSec = 0.0;
    double bandwidthInMBs = 0.0;
    unsigned char *h_data = NULL;
    cl_mem cmPinnedData = NULL;
    cl_mem cmDevData = NULL;
    cl_int ciErrNum = CL_SUCCESS;

    //allocate and init host memory, pinned or conventional
    if(memMode == PINNED)
    {
        // Create a host buffer
        cmPinnedData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        // Get a mapped pointer
        h_data = (unsigned char*)clEnqueueMapBuffer(cqCommandQueue, cmPinnedData, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //initialize 
        for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
        {
            h_data[i] = (unsigned char)(i & 0xff);
        }

        // unmap and make data in the host buffer valid
        ciErrNum = clEnqueueUnmapMemObject(cqCommandQueue, cmPinnedData, (void*)h_data, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    else 
    {
        // standard host alloc
        h_data = (unsigned char *)malloc(memSize);

        //initialize 
        for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
        {
            h_data[i] = (unsigned char)(i & 0xff);
        }
    }

    // allocate device memory 
    cmDevData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // initialize device memory 
    if(memMode == PINNED)
    {
	    // Get a mapped pointer
        h_data = (unsigned char*)clEnqueueMapBuffer(cqCommandQueue, cmPinnedData, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, &ciErrNum);	        

        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, memSize, h_data, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    else
    {
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, memSize, h_data, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Sync queue to host, start timer 0, and copy data from GPU to Host
    ciErrNum = clFinish(cqCommandQueue);
    shrDeltaT(0);
    if(accMode == DIRECT)
    { 
        // DIRECT:  API access to device buffer 
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, memSize, h_data, 0, NULL, NULL);
            oclCheckError(ciErrNum, CL_SUCCESS);
        }
        ciErrNum = clFinish(cqCommandQueue);
        oclCheckError(ciErrNum, CL_SUCCESS);
    } 
    else 
    {
        // MAPPED: mapped pointers to device buffer for conventional pointer access
        void* dm_idata = clEnqueueMapBuffer(cqCommandQueue, cmDevData, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            memcpy(h_data, dm_idata, memSize);
        }
        ciErrNum = clEnqueueUnmapMemObject(cqCommandQueue, cmDevData, dm_idata, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    
    //get the the elapsed time in seconds
    elapsedTimeInSec = shrDeltaT(0);
    
    //calculate bandwidth in MB/s
    bandwidthInMBs = ((double)memSize * (double)MEMCOPY_ITERATIONS) / (elapsedTimeInSec * (double)(1 << 20));

    //clean up memory
    if(cmDevData)clReleaseMemObject(cmDevData);
    if(cmPinnedData) 
    {
	    clEnqueueUnmapMemObject(cqCommandQueue, cmPinnedData, (void*)h_data, 0, NULL, NULL);	
	    clReleaseMemObject(cmPinnedData);	
    }
    h_data = NULL;

    return bandwidthInMBs;
}
///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
double testHostToDeviceTransfer(unsigned int memSize, accessMode accMode, memoryMode memMode)
{
    double elapsedTimeInSec = 0.0;
    double bandwidthInMBs = 0.0;
    unsigned char* h_data = NULL;
    cl_mem cmPinnedData = NULL;
    cl_mem cmDevData = NULL;
    cl_int ciErrNum = CL_SUCCESS;

    // Allocate and init host memory, pinned or conventional
    if(memMode == PINNED)
   { 
        // Create a host buffer
        cmPinnedData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        // Get a mapped pointer
        h_data = (unsigned char*)clEnqueueMapBuffer(cqCommandQueue, cmPinnedData, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        //initialize 
        for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
        {
            h_data[i] = (unsigned char)(i & 0xff);
        }
	
        // unmap and make data in the host buffer valid
        ciErrNum = clEnqueueUnmapMemObject(cqCommandQueue, cmPinnedData, (void*)h_data, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
		h_data = NULL;  // buffer is unmapped
    }
    else 
    {
        // standard host alloc
        h_data = (unsigned char *)malloc(memSize);

        //initialize 
        for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
        {
            h_data[i] = (unsigned char)(i & 0xff);
        }
    }

    // allocate device memory 
    cmDevData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Sync queue to host, start timer 0, and copy data from Host to GPU
    clFinish(cqCommandQueue);
    shrDeltaT(0);
    if(accMode == DIRECT)
    { 
	    if(memMode == PINNED) 
        {
            // Get a mapped pointer
            h_data = (unsigned char*)clEnqueueMapBuffer(cqCommandQueue, cmPinnedData, CL_TRUE, CL_MAP_READ, 0, memSize, 0, NULL, NULL, &ciErrNum);
            oclCheckError(ciErrNum, CL_SUCCESS);
	    }

        // DIRECT:  API access to device buffer 
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
                ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, memSize, h_data, 0, NULL, NULL);
                oclCheckError(ciErrNum, CL_SUCCESS);
        }
        ciErrNum = clFinish(cqCommandQueue);
        oclCheckError(ciErrNum, CL_SUCCESS);
    } 
    else 
    {
        // MAPPED: mapped pointers to device buffer and conventional pointer access
        void* dm_idata = clEnqueueMapBuffer(cqCommandQueue, cmDevData, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);
		if(memMode == PINNED ) 
		{
			h_data = (unsigned char*)clEnqueueMapBuffer(cqCommandQueue, cmPinnedData, CL_TRUE, CL_MAP_READ, 0, memSize, 0, NULL, NULL, &ciErrNum); 
            oclCheckError(ciErrNum, CL_SUCCESS); 
        } 
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            memcpy(dm_idata, h_data, memSize);
        }
        ciErrNum = clEnqueueUnmapMemObject(cqCommandQueue, cmDevData, dm_idata, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    
    //get the the elapsed time in seconds
    elapsedTimeInSec = shrDeltaT(0);
    
    //calculate bandwidth in MB/s
    bandwidthInMBs = ((double)memSize * (double)MEMCOPY_ITERATIONS)/(elapsedTimeInSec * (double)(1 << 20));

    //clean up memory
    if(cmDevData)clReleaseMemObject(cmDevData);
    if(cmPinnedData) 
    {
	    clEnqueueUnmapMemObject(cqCommandQueue, cmPinnedData, (void*)h_data, 0, NULL, NULL);
	    clReleaseMemObject(cmPinnedData);
    }
    h_data = NULL;

    return bandwidthInMBs;
}
///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
double testDeviceToDeviceTransfer(unsigned int memSize)
{
    double elapsedTimeInSec = 0.0;
    double bandwidthInMBs = 0.0;
    unsigned char* h_idata = NULL;
    cl_int ciErrNum = CL_SUCCESS;
    
    //allocate host memory
    h_idata = (unsigned char *)malloc( memSize );
        
    //initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char) (i & 0xff);
    }

    // allocate device input and output memory and initialize the device input memory
    cl_mem d_idata = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, memSize, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cl_mem d_odata = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, memSize, NULL, &ciErrNum);         
    oclCheckError(ciErrNum, CL_SUCCESS);
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, d_idata, CL_TRUE, 0, memSize, h_idata, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Sync queue to host, start timer 0, and copy data from one GPU buffer to another GPU bufffer
    clFinish(cqCommandQueue);
    shrDeltaT(0);
    for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ciErrNum = clEnqueueCopyBuffer(cqCommandQueue, d_idata, d_odata, 0, 0, memSize, 0, NULL, NULL);                
        oclCheckError(ciErrNum, CL_SUCCESS);
    }    

    // Sync with GPU
    clFinish(cqCommandQueue);
    
    //get the the elapsed time in seconds
    elapsedTimeInSec = shrDeltaT(0);
    
    // Calculate bandwidth in MB/s 
    //      This is for kernels that read and write GMEM simultaneously 
    //      Obtained Throughput for unidirectional block copies will be 1/2 of this #
    bandwidthInMBs = 2.0 * ((double)memSize * (double)MEMCOPY_ITERATIONS)/(elapsedTimeInSec * (double)(1 << 20));

    //clean up memory on host and device
    free(h_idata);
    clReleaseMemObject(d_idata);
    clReleaseMemObject(d_odata);

    return bandwidthInMBs;
}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, double* bandwidths, unsigned int count, memcpyKind kind, accessMode accMode, memoryMode memMode, int iNumDevs)
{
    // log config information 
    if (kind == DEVICE_TO_DEVICE)
    {
        shrLog("Device to Device Bandwidth, %i Device(s)\n", iNumDevs);
    }
    else 
    {
        if (kind == DEVICE_TO_HOST)
        {
            shrLog("Device to Host Bandwidth, %i Device(s), ", iNumDevs);
        }
        else if (kind == HOST_TO_DEVICE)
        {
            shrLog("Host to Device Bandwidth, %i Device(s), ", iNumDevs);
        }
        if(memMode == PAGEABLE)
        {
            shrLog("Paged memory");
        }
        else if (memMode == PINNED)
        {
            shrLog("Pinned memory");
        }
        if(accMode == DIRECT)
        {
            shrLog(", direct access\n");
        }
        else if (accMode == MAPPED)
        {
            shrLog(", mapped access\n");
        }
    }

    shrLog("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    unsigned int i; 
    for(i = 0; i < (count - 1); i++)
    {
        shrLog("   %u\t\t\t%s%.1f\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
    }
    shrLog("   %u\t\t\t%s%.1f\n\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
}

///////////////////////////////////////////////////////////////////////////
//print results in a database format
///////////////////////////////////////////////////////////////////////////
void printResultsCSV(unsigned int *memSizes, double* bandwidths, unsigned int count, memcpyKind kind, accessMode accMode, memoryMode memMode, int iNumDevs)
{
    unsigned int i; 
    double dSeconds = 0.0;
    std::string sConfig;
        
    // log config information 
    if (kind == DEVICE_TO_DEVICE)
    {
        sConfig += "D2D";
    }
    else 
    {
        if (kind == DEVICE_TO_HOST)
        {
            sConfig += "D2H";
        }
        else if (kind == HOST_TO_DEVICE)
        {
            sConfig += "H2D";
        }

        if(memMode == PAGEABLE)
        {
            sConfig += "-Paged";
        }
        else if (memMode == PINNED)
        {
            sConfig += "-Pinned";
        }

        if(accMode == DIRECT)
        {
            sConfig += "-Direct";            
        }
        else if (accMode == MAPPED)
        {
            sConfig += "-Mapped";            
        }
    }


    for(i = 0; i < count; i++)
    {
        dSeconds = (double)memSizes[i] / (bandwidths[i] * (double)(1<<20));
        shrLogEx(LOGBOTH | MASTER, 0, "oclBandwidthTest-%s, Bandwidth = %.1f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %i\n", sConfig.c_str(), bandwidths[i], dSeconds, memSizes[i], iNumDevs);
    }
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    shrLog("Usage:  oclBandwidthTest [OPTION]...\n");
    shrLog("Test the bandwidth for device to host, host to device, and device to device transfers\n");
    shrLog("\n");
    shrLog("Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
    shrLog("./oclBandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh\n");

    shrLog("\n");
    shrLog("Options:\n");
    shrLog("--help\tDisplay this help menu\n");
    shrLog("--csv\tPrint results as a CSV\n");
    shrLog("--device=[device_number]\tSpecify the device device to be used\n");
    shrLog("  all - compute cumulative bandwidth on all the devices\n");
    shrLog("  0,1,2,...,n - Specify a GPU device to be used to run this test\n");
    shrLog("--access=[ACCESSMODE]\tSpecify which memory access mode to use\n");
    shrLog("  direct   - direct device memory\n");
    shrLog("  mapped   - mapped device memory\n");
    shrLog("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
    shrLog("  pageable  - pageable system memory\n");
    shrLog("  pinned    - pinned system memory\n");
    shrLog("--mode=[MODE]\tSpecify the mode to use\n");
    shrLog("  quick - performs a quick measurement\n");
    shrLog("  range - measures a user-specified range of values\n");
    shrLog("  shmoo - performs an intense shmoo of a large range of values\n");

    shrLog("--htod\tMeasure host to device transfers\n");   
    shrLog("--dtoh\tMeasure device to host transfers\n");
    shrLog("--dtod\tMeasure device to device transfers\n");
    
    shrLog("Range mode options\n");
    shrLog("--start=[SIZE]\tStarting transfer size in bytes\n");
    shrLog("--end=[SIZE]\tEnding transfer size in bytes\n");
    shrLog("--increment=[SIZE]\tIncrement size in bytes\n");
}
