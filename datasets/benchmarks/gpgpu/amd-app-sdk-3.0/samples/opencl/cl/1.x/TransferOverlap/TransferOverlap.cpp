#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "TransferOverlap.hpp"
#include <string>

struct _memFlags
{
    cl_mem_flags f;
    const char *s;
}
memFlags[] =
{
    { CL_MEM_READ_ONLY,              "CL_MEM_READ_ONLY" },
    { CL_MEM_WRITE_ONLY,             "CL_MEM_WRITE_ONLY" },
    { CL_MEM_READ_WRITE,             "CL_MEM_READ_WRITE" },
    { CL_MEM_ALLOC_HOST_PTR,         "CL_MEM_ALLOC_HOST_PTR" },
    { CL_MEM_USE_PERSISTENT_MEM_AMD, "CL_MEM_USE_PERSISTENT_MEM_AMD"}
};

int nFlags = sizeof(memFlags) / sizeof(memFlags[0]);

int
TransferOverlap::setupTransferOverlap()
{
    // Increase the input valuse if -e option is not used
    if(sampleArgs->verify)
    {
        nLoops = 1;
        nSkip = 0;
        nKLoops = 10;
        nBytes = 1024 * 1024;
    }

    timeLog = new TestLog(nLoops * 50);
    CHECK_ALLOCATION(timeLog, "Failed to allocate host memory. (timeLog)");

    // Educated guess of optimal work size
    int minBytes = MAX_WAVEFRONT_SIZE * sizeof(cl_uint) * 4;

    nBytes = (nBytes / minBytes) * minBytes;
    nBytes = nBytes < minBytes ? minBytes : nBytes;
    nItems = nBytes / (4 * sizeof(cl_uint));

    int maxThreads = nBytes / (4 * sizeof( cl_uint ));
    nThreads = deviceInfo.maxComputeUnits * numWavefronts * MAX_WAVEFRONT_SIZE;

    if(nThreads > maxThreads)
    {
        nThreads = maxThreads;
    }
    else
    {
        while(nItems % nThreads != 0)
        {
            nThreads += MAX_WAVEFRONT_SIZE;
        }
    }

    nBytesResult = (nThreads / MAX_WAVEFRONT_SIZE) * sizeof(cl_uint) * 2;

    return SDK_SUCCESS;
}

int
TransferOverlap::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("TransferOverlap_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int
TransferOverlap::setupCL(void)
{
    cl_int status = 0;
    size_t deviceListSize;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_GPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_GPU;
        }
    }

    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = CECL_CREATE_CONTEXT_FROM_TYPE(cps,
                                      dType,
                                      NULL,
                                      NULL,
                                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // First, get the size of device list data
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              0,
                              NULL,
                              &deviceListSize);
    CHECK_OPENCL_ERROR(status, "clGetContextInfo failed.");

    int deviceCount = (int)(deviceListSize / sizeof(cl_device_id));
    retValue = validateDeviceId(sampleArgs->deviceId, deviceCount);
    CHECK_ERROR(retValue, SDK_SUCCESS, "validateDeviceId() failed");

    // Now allocate memory for device list based on the size we got earlier
    devices = (cl_device_id*)malloc(deviceListSize);
    CHECK_ALLOCATION(devices, "Failed to allocate memory (devices).");

    // Now, get the device list data
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              devices,
                              NULL);
    CHECK_OPENCL_ERROR(status, "clGetContextInfo failed.");

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Check device extensions
    if(!strstr(deviceInfo.extensions, "cl_khr_local_int32_base_atomics"))
    {
        OPENCL_EXPECTED_ERROR("Device does not support cl_khr_local_int32_base_atomics extension!");
    }

    // Get OpenCL device version
    std::string deviceVersionStr = std::string(deviceInfo.deviceVersion);
    size_t vStart = deviceVersionStr.find(" ", 0);
    size_t vEnd = deviceVersionStr.find(" ", vStart + 1);
    std::string vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);

    // Check of OPENCL_C_VERSION if device version is 1.1 or later
#ifdef CL_VERSION_1_1
    if(deviceInfo.openclCVersion)
    {
        // Exit if OpenCL C device version is 1.0
        deviceVersionStr = std::string(deviceInfo.openclCVersion);
        vStart = deviceVersionStr.find(" ", 0);
        vStart = deviceVersionStr.find(" ", vStart + 1);
        vEnd = deviceVersionStr.find(" ", vStart + 1);
        vStrVal = deviceVersionStr.substr(vStart + 1, vEnd - vStart - 1);
        if(vStrVal.compare("1.0") <= 0)
        {
            OPENCL_EXPECTED_ERROR("Unsupported device! Required CL_DEVICE_OPENCL_C_VERSION as 1.1");
        }
    }
    else
    {
        OPENCL_EXPECTED_ERROR("Unsupported device! Required CL_DEVICE_OPENCL_C_VERSION as 1.1");
    }
#else
    OPENCL_EXPECTED_ERROR("Unsupported device! Required CL_DEVICE_OPENCL_C_VERSION as 1.1");
#endif
    //Setup application data
    if(setupTransferOverlap() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    queue = CECL_CREATE_COMMAND_QUEUE(context, devices[sampleArgs->deviceId], 0,
                                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

    inputBuffer1 = CECL_BUFFER(context, inFlags, nBytes, NULL, &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER() failed.(inputBuffer1).");

    inputBuffer2 = CECL_BUFFER(context, inFlags, nBytes, NULL, &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER() failed.(inputBuffer2).");

    resultBuffer1 = CECL_BUFFER(context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   nBytesResult,
                                   NULL,
                                   &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER() failed.(resultBuffer1).");

    resultBuffer2 = CECL_BUFFER(context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   nBytesResult,
                                   NULL,
                                   &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER() failed.(resultBuffer2).");

    /* create a CL program using the kernel source */
    buildProgramData buildData;
    buildData.kernelName = std::string("TransferOverlap_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    /* ConstantBuffer bandwidth from single access */
    readKernel = CECL_KERNEL(program, "readKernel", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(readKernel).");

    size_t kernelWorkGroupSize = 0;

    /* Check whether specified local group size is possible on current kernel */
    status = CECL_GET_KERNEL_WORK_GROUP_INFO(readKernel,
                                      devices[sampleArgs->deviceId],
                                      CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(size_t),
                                      &kernelWorkGroupSize,
                                      0);
    CHECK_OPENCL_ERROR(status, "CECL_GET_KERNEL_WORK_GROUP_INFO() failed.");

    // If local groupSize exceeds the maximum supported on kernel  fall back
    if(MAX_WAVEFRONT_SIZE > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << MAX_WAVEFRONT_SIZE << std::endl;
            std::cout << "Max Group Size supported on the kernel(readKernel) : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
        }
        return SDK_FAILURE;
    }

    writeKernel = CECL_KERNEL(program, "writeKernel", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL() failed.(writeKernel)");

    // Check whether specified local group size is possible on current kernel
    status = CECL_GET_KERNEL_WORK_GROUP_INFO(
                 writeKernel,
                 devices[sampleArgs->deviceId],
                 CL_KERNEL_WORK_GROUP_SIZE,
                 sizeof(size_t),
                 &kernelWorkGroupSize,
                 NULL);
    CHECK_OPENCL_ERROR(status, "CECL_GET_KERNEL_WORK_GROUP_INFO() failed.");

    // If local group size exceeds the maximum supported on kernel fall back
    if(MAX_WAVEFRONT_SIZE > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << MAX_WAVEFRONT_SIZE << std::endl;
            std::cout << "Max Group Size supported on the kernel(writeKernel) : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
        }
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
TransferOverlap::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* numLoops = new Option;

    numLoops->_sVersion = "i";
    numLoops->_lVersion = "iterations";
    numLoops->_description = "Number of timing loops";
    numLoops->_type = CA_ARG_INT;
    numLoops->_value = &nLoops;

    sampleArgs->AddOption(numLoops);
    delete numLoops;

    Option* skipLoops = new Option;

    skipLoops->_sVersion = "s";
    skipLoops->_lVersion = "skip";
    skipLoops->_description = "skip first n iterations for average";
    skipLoops->_type = CA_ARG_INT;
    skipLoops->_value = &nSkip;

    sampleArgs->AddOption(skipLoops);
    delete skipLoops;

    Option* kernelLoops = new Option;

    kernelLoops->_sVersion = "k";
    kernelLoops->_lVersion = "kernelLoops";
    kernelLoops->_description = "Number of loops in kernel";
    kernelLoops->_type = CA_ARG_INT;
    kernelLoops->_value = &nKLoops;

    sampleArgs->AddOption(kernelLoops);
    delete kernelLoops;

    Option* size = new Option;

    size->_sVersion = "x";
    size->_lVersion = "size";
    size->_description = "Size in bytes";
    size->_type = CA_ARG_INT;
    size->_value = &nBytes;

    sampleArgs->AddOption(size);
    delete size;

    Option* wavefronts = new Option;

    wavefronts->_sVersion = "w";
    wavefronts->_lVersion = "wavefronts";
    wavefronts->_description = "Number of wavefronts per compute unit";
    wavefronts->_type = CA_ARG_INT;
    wavefronts->_value = &numWavefronts;

    sampleArgs->AddOption(wavefronts);
    delete wavefronts;

    Option* inputFlags = new Option;

    inputFlags->_sVersion = "I";
    inputFlags->_lVersion = "inMemFlag";
    inputFlags->_description = "Memory flags for input buffer "
                               "\n\t\t 0 for CL_MEM_READ_ONLY"
                               "\n\t\t 1 CL_MEM_WRITE_ONLY"
                               "\n\t\t 2 CL_MEM_READ_WRITE"
                               "\n\t\t 3 CL_MEM_ALLOC_HOST_PTR"
                               "\n\t\t 4 CL_MEM_USE_PERSISTENT_MEM_AMD\n";
    inputFlags->_type = CA_ARG_INT;
    inputFlags->_value = &inFlagsValue;

    sampleArgs->AddOption(inputFlags);
    delete inputFlags;

    Option* noOverlapStr = new Option;

    noOverlapStr->_sVersion = "n";
    noOverlapStr->_lVersion = "noOverlap";
    noOverlapStr->_description = "Do not overlap memset() with kernel";
    noOverlapStr->_type = CA_NO_ARGUMENT;
    noOverlapStr->_value = &noOverlap;

    sampleArgs->AddOption(noOverlapStr);
    delete noOverlapStr;


    Option* pringLogStr = new Option;

    pringLogStr->_sVersion = "l";
    pringLogStr->_lVersion = "log";
    pringLogStr->_description = "Prints complete timing log";
    pringLogStr->_type = CA_NO_ARGUMENT;
    pringLogStr->_value = &printLog;

    sampleArgs->AddOption(pringLogStr);
    delete pringLogStr;

    return SDK_SUCCESS;
}

int
TransferOverlap::parseExtraCommandLineOptions(int argc, char**argv)
{
    if(sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    //Handle extra options which are not default in SDKUtils
    while(--argc)
    {
        if(strcmp(argv[argc], "-I") == 0 ||  strcmp(argv[argc], "-inMemFlag") == 0)
        {
            int f = atoi( argv[ argc+1 ] );
            if( f < nFlags )
            {
                inFlags |= memFlags[ f ].f;
            }
        }
    }

    cl_mem_flags f = CL_MEM_ALLOC_HOST_PTR |
                     CL_MEM_USE_PERSISTENT_MEM_AMD |
                     CL_MEM_READ_ONLY |
                     CL_MEM_WRITE_ONLY |
                     CL_MEM_READ_WRITE;

    if( (inFlags & f) == 0 )
    {
        inFlags |= CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    f  = CL_MEM_READ_ONLY |
         CL_MEM_WRITE_ONLY |
         CL_MEM_READ_WRITE;

    if( (inFlags & f) == 0 )
    {
        inFlags |= CL_MEM_READ_ONLY;
    }

    nSkip = nLoops > nSkip ? nSkip : 0;

    return SDK_SUCCESS;
}

int
TransferOverlap::setup()
{
    int status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    return SDK_SUCCESS;
}

int
TransferOverlap::verifyResultBuffer(cl_mem resultBuffer, bool firstLoop)
{
    cl_int   status;
    cl_event event;
    void *ptrResult = NULL;

    t.Reset();
    t.Start();

    ptrResult = (void*)CECL_MAP_BUFFER(
                    queue,
                    resultBuffer,
                    CL_TRUE,
                    CL_MAP_READ,
                    0,
                    nBytesResult,
                    0,
                    NULL,
                    NULL,
                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER() failed.(resultBuffer)");

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s %8.2lf GB/s\n", "CECL_MAP_BUFFER(MAP_WRITE)",
        t.GetElapsedTime(),
        nBytesResult,
        1);

    t.Reset();
    t.Start();

    cl_uint sum = 0;
    for( int i = 0; i < nThreads / MAX_WAVEFRONT_SIZE; i++ )
    {
        sum += ((cl_uint*)ptrResult)[i];
    }

    bool results;
    if( (sum != nBytes / sizeof(cl_uint)) && !firstLoop )
    {
        results = false;
    }
    else
    {
        results = true;
    }

    t.Stop();
    timeLog->Timer("%32s  %lf s\n", "CPU reduction", t.GetElapsedTime(), nBytes, 1);

    if(results)
    {
        timeLog->Msg( "%32s\n", "verification ok" );
    }
    else
    {
        correctness = false;
        timeLog->Error( "%32s\n", "verification FAILED" );
        std::cout << "Failed\n";
        return SDK_FAILURE;
    }

    t.Reset();
    t.Start();

    status = clEnqueueUnmapMemObject(queue, resultBuffer, (void *) ptrResult, 0,
                                     NULL, &event);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject() failed.(resultBuffer)");

    status = clWaitForEvents(1, &event);
    CHECK_OPENCL_ERROR(status, "clWaitForEvents()");

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s %8.2lf GB/s\n", "clEnqueueUnmapMemObject()",
        t.GetElapsedTime(),
        nBytesResult,
        1);

    // Release event
    status = clReleaseEvent(event);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent(event) failed.");

    return SDK_SUCCESS;
}

int
TransferOverlap::launchKernel(cl_mem inputBuffer, cl_mem resultBuffer,
                              unsigned char v)
{
    cl_uint vKernel = 0;
    cl_int status = 0;

    for(int i = 0; i < sizeof(cl_uint); i++)
    {
        vKernel |= v << (i * 8);
    }

    nItemsPerThread = nItems / nThreads;

    globalWorkSize = nThreads;
    localWorkSize = MAX_WAVEFRONT_SIZE;

    status = CECL_SET_KERNEL_ARG(readKernel, 0, sizeof(void *), (void *)&inputBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG() failed.(inputBuffer)");

    status = CECL_SET_KERNEL_ARG(readKernel, 1, sizeof(void *), (void *) &resultBuffer);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG() failed.(resultBuffer)");

    status = CECL_SET_KERNEL_ARG(readKernel, 2, sizeof(cl_uint),
                            (void *)&nItemsPerThread);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG() failed.(nItemsPerThread)");

    status = CECL_SET_KERNEL_ARG(readKernel, 3, sizeof(cl_uint), (void *) &vKernel);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG() failed.(vKernel)");

    status = CECL_SET_KERNEL_ARG(readKernel, 4, sizeof(cl_uint), (void *) &nKLoops);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG() failed.(nKLoops)");

    cl_event  event;
    cl_event *evPtr;

    if( noOverlap )
    {
        evPtr = &event;
    }
    else
    {
        evPtr = NULL;
    }

    t.Reset();
    t.Start();

    status = CECL_ND_RANGE_KERNEL(
                 queue,
                 readKernel,
                 1,
                 NULL,
                 &globalWorkSize,
                 &localWorkSize,
                 0,
                 NULL,
                 evPtr);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL() failed.");

    if(noOverlap)
    {
        status = clWaitForEvents(1, evPtr);
        CHECK_OPENCL_ERROR(status, "clWaitForEvents() failed.");
    }
    else
    {
        status = clFlush(queue);
        CHECK_OPENCL_ERROR(status, "clFlush() failed.");
    }

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s\n", "CECL_ND_RANGE_KERNEL()",
        t.GetElapsedTime(),
        nBytes,
        nKLoops);

    // Release the event
    if(noOverlap)
    {
        status = clReleaseEvent(*evPtr);
        CHECK_OPENCL_ERROR(status, "clReleaseEvent(*evPtr) failed.");
    }

    return SDK_SUCCESS;
}

void*
TransferOverlap::launchMapBuffer(cl_mem buffer, cl_event *mapEvent)
{
    cl_int status;
    void *ptr = NULL;

    t.Reset();
    t.Start();

    ptr = (void *)CECL_MAP_BUFFER(
              queue,
              buffer,
              CL_FALSE,
              CL_MAP_WRITE,
              0,
              nBytes,
              0,
              NULL,
              mapEvent,
              &status);
    if(ptr == NULL)
    {
        std::cout << "CECL_MAP_BUFFER(buffer) failed.";
        return NULL;
    }

    status = clFlush(queue);
    if(status != CL_SUCCESS)
    {
        std::cout << "clFlush() failed.";
        return NULL;
    }

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s %8.2lf GB/s\n", "CECL_MAP_BUFFER(MAP_WRITE)",
        t.GetElapsedTime(),
        nBytes,
        1);

    return ptr;
}

int
TransferOverlap::fillBuffer(cl_mem buffer, cl_event *mapEvent, void *ptr,
                            unsigned char v)
{
    cl_int status;
    cl_event event;

    t.Reset();
    t.Start();

    status = clWaitForEvents(1, mapEvent);
    CHECK_OPENCL_ERROR(status, "clWaitForEvents() failed.");

    memset(ptr, v, nBytes);

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s %8.2lf GB/s\n", "clWaitForEvents() + memset()",
        t.GetElapsedTime(),
        nBytes,
        1);

    // Release event
    status = clReleaseEvent(*mapEvent);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent(*mapEvent) failed.");

    t.Reset();
    t.Start();

    status = clEnqueueUnmapMemObject(
                 queue,
                 buffer,
                 (void *) ptr,
                 0,
                 NULL,
                 &event);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject(buffer) failed.");

    status = clWaitForEvents(1, &event);
    CHECK_OPENCL_ERROR(status, "clWaitForEvents() failed.");

    t.Stop();
    timeLog->Timer(
        "%32s  %lf s %8.2lf GB/s\n", "clEnqueueUnmapMemObject()",
        t.GetElapsedTime(),
        nBytes,
        1);

    // Release event
    status = clReleaseEvent(event);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent(event) failed.");

    return SDK_SUCCESS;
}

int
TransferOverlap::runOverlapTest()
{
    int nl = nLoops;
    int status = SDK_SUCCESS;

    void *inPtr1 = NULL;
    void *inPtr2 = NULL;

    cl_event lastBuf1MapEvent;
    cl_event lastBuf2MapEvent;

    bool firstLoop = true;

    // Start with inputBuffer1 mapped
    inPtr1 = (void *)CECL_MAP_BUFFER(
                 queue,
                 inputBuffer1,
                 CL_FALSE,
                 CL_MAP_WRITE,
                 0,
                 nBytes,
                 0,
                 NULL,
                 &lastBuf1MapEvent,
                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER(inputBuffer1) failed.");

    status = clFlush(queue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    CPerfCounter lt;

    lt.Reset();
    lt.Start();

    while(nl--)
    {
        timeLog->loopMarker();
        unsigned char v = (unsigned char)(nl & 0xff);

        /* 1. Host acquires and fills inputBuffer1. Unless this
         is the first loop, this happens concurrently with the
         preceding kernel execution.*/
        timeLog->Msg( "\n%s\n\n", "Acquire and fill: inputBuffer1" );
        status = fillBuffer(inputBuffer1, &lastBuf1MapEvent, inPtr1, v);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        /* This is a CPU/GPU synchronization point, as all commands in the
         in-order queue before the preceding cl*Unmap() are now finished.
         We can accurately sample the per-loop timer here.*/
        lt.Stop();
        timeLog->Timer(
            "\n%s %f s\n", "Loop time",
            lt.GetElapsedTime(),
            nBytes,
            1);
        lt.Reset();
        lt.Start();

        /* 2. Launch map of inputBuffer2. The map needs to precede
         the next kernel launch in the in-order queue, otherwise waiting
         for the map to finish would also wait for the kernel to
         finish.*/
        timeLog->Msg("\n%s\n\n", "Launch map: inputBuffer2");
        inPtr2 = launchMapBuffer(inputBuffer2, &lastBuf2MapEvent);
        if(inPtr2 == NULL)
        {
            return SDK_FAILURE;
        }

        // Verify result of kernel for inputBuffer2
        timeLog->Msg("\n%s\n\n", "Verify: resultBuffer2");
        status = verifyResultBuffer(resultBuffer2, firstLoop);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // 3. Asynchronous launch of kernel for inputBuffer1
        timeLog->Msg("\n%s\n\n", "Launch GPU kernel: inputBuffer1");
        status = launchKernel(inputBuffer1, resultBuffer1, v);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        /* 4. Host acquires and fills inputBuffer2. This happens
         concurrently with the preceding kernel execution.*/
        timeLog->Msg("\n%s\n\n", "Acquire and fill: inputBuffer2");
        status = fillBuffer(inputBuffer2, &lastBuf2MapEvent, inPtr2, v);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        /* 5. Launch map of inputBuffer1. The map needs to precede
         the next kernel launch in the in-order queue, otherwise waiting
         for the map to finish would also wait for the kernel to
         finish.*/
        timeLog->Msg("\n%s\n\n", "Launch map: inputBuffer1");
        inPtr1 = launchMapBuffer(inputBuffer1, &lastBuf1MapEvent);
        if(inPtr1 == NULL)
        {
            return SDK_FAILURE;
        }

        // Verify result of kernel for inputBuffer1
        timeLog->Msg("\n%s\n\n", "Verify: resultBuffer1");
        status = verifyResultBuffer(resultBuffer1, firstLoop);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // 6. Asynchronous launch of kernel for inputBuffer2
        timeLog->Msg("\n%s\n\n", "Launch GPU kernel: inputBuffer2");

        status = launchKernel(inputBuffer2, resultBuffer2, v);
        if(status != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        timeLog->Msg("%s\n", "");
        firstLoop = false;
    }

    status = clFinish(queue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");

    status = clReleaseEvent(lastBuf1MapEvent);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent() failed.");

    return SDK_SUCCESS;
}

int
TransferOverlap::run()
{
    CPerfCounter gt;
    gt.Reset();
    gt.Start();

    int status = runOverlapTest();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    gt.Stop();
    testTime = gt.GetElapsedTime();

    if(sampleArgs->verify)
    {
        if(correctness)
        {
            std::cout << "Passed!\n" << std::endl;
        }
        else
        {
            std::cout << "Failed!\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
TransferOverlap::printStats()
{
    if(!sampleArgs->quiet)
    {
#ifdef _WIN32
        std::cout << "Build:               _WINxx" ;
#ifdef _DEBUG
        std::cout << " DEBUG";
#else
        std::cout <<  " release";
#endif
        std::cout << "\n" ;
#else
#ifdef NDEBUG
        std::cout << "Build:               release\n";
#else
        std::cout << "Build:               DEBUG\n";
#endif
#endif

        std::cout << "GPU work items:        " << nThreads << std::endl;
        std::cout << "Buffer size:           " << nBytes << std::endl;
        std::cout << "Timing loops:          " << nLoops << std::endl;
        std::cout << "Kernel loops:          " << nKLoops << std::endl;
        std::cout << "Wavefronts/SIMD:       " << numWavefronts << std::endl;
        std::cout << "memset/kernel overlap: " << (noOverlap ? "no" : "yes") <<
                  std::endl;
        std::cout << "inputBuffer:           ";

        for( int i = 0; i < sizeof(memFlags) / sizeof(memFlags[0]); i++)
            if(inFlags & memFlags[i].f)
            {
                std::cout << memFlags[i].s;
            }

        std::cout << std::endl;

        if(printLog)
        {
            std:: cout << "\nLOOP ITERATIONS\n"
                       << "---------------\n\n";
            timeLog->printLog();
        }

        std::cout <<
                  "\nAVERAGES (over loops" << nSkip << " - " << nLoops - 1 <<
                  ", use -l to show complete log)\n" <<
                  "--------\n\n";

        timeLog->printSummary(nSkip);
        std::cout << "\nComplete test time:" << testTime << " s\n\n";
    }
}

int
TransferOverlap::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    if(inputBuffer1)
    {
        status = clReleaseMemObject(inputBuffer1);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject() failed.");
    }

    if(inputBuffer2)
    {
        status = clReleaseMemObject(inputBuffer2);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject() failed.");
    }

    if(resultBuffer1)
    {
        status = clReleaseMemObject(resultBuffer1);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject() failed.");
    }

    if(resultBuffer2)
    {
        status = clReleaseMemObject(resultBuffer2);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject() failed.");
    }

    status = clReleaseKernel(readKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel() failed.");

    status = clReleaseKernel(writeKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel() failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram() failed.");

    status = clReleaseCommandQueue(queue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue() failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext() failed.");

    delete timeLog;
    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    TransferOverlap clTransferOverlap;

    if(clTransferOverlap.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clTransferOverlap.parseExtraCommandLineOptions(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clTransferOverlap.sampleArgs->isDumpBinaryEnabled())
    {
        return clTransferOverlap.genBinaryImage();
    }
    else
    {
        int state = clTransferOverlap.setup();
        if(state != SDK_SUCCESS)
        {
            return state;
        }

        if(clTransferOverlap.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clTransferOverlap.printStats();
        if(clTransferOverlap.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}
