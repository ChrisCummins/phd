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

#include "AtomicCounters.hpp"


int
AtomicCounters::setupAtomicCounters()
{
    // Make sure length is multiples of GROUP_SIZE
    length = (length / GROUP_SIZE);
    length = length ? length * GROUP_SIZE : GROUP_SIZE;
    // Allocate the memory for input array
    input = (cl_uint*)malloc(length * sizeof(cl_uint));
    CHECK_ALLOCATION(input, "Allocation failed(input)");
    // Set the input data
    value = 2;
    for(cl_uint i = 0; i < length; ++i)
    {
        input[i] = (cl_uint)(rand() % 5);
    }
    if(! sampleArgs->quiet)
    {
        printArray<cl_uint>("Input Arry", input, 256, 1);
    }
    return SDK_SUCCESS;
}

int
AtomicCounters::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("AtomicCounters_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if( sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string( sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int
AtomicCounters::setupCL(void)
{
    cl_int status = 0;
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
            std::cout << "GPU not found. Falling back to CPU" << std::endl;
            dType = CL_DEVICE_TYPE_GPU;
        }
    }
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed.");
    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed.");
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = CECL_CREATE_CONTEXT_FROM_TYPE(cps,
                                      dType,
                                      NULL,
                                      NULL,
                                      &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");
    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed ");
    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed" );
    // Check device extensions
    if(!strstr(deviceInfo.extensions, "cl_ext_atomic_counters_32"))
    {
        OPENCL_EXPECTED_ERROR("Device does not support cl_ext_atomic_counters_32 extension!");
    }
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
    if(setupAtomicCounters() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    commandQueue = CECL_CREATE_COMMAND_QUEUE(context, devices[sampleArgs->deviceId],
                                        props, &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed(commandQueue)");
    // Set Persistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }
    // Create buffer for input array
    inBuf = CECL_BUFFER(context, inMemFlags, length * sizeof(cl_uint), NULL,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(inBuf)");
    // Set up data for input array
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 inBuf,
                 CL_FALSE,
                 0,
                 length * sizeof(cl_uint),
                 input,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(inBuf) failed..");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush(commandQueue) failed.");
    counterOutBuf = CECL_BUFFER(
                        context,
                        CL_MEM_READ_WRITE,
                        sizeof(cl_uint),
                        NULL,
                        &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(counterOutBuf).");
    globalOutBuf = CECL_BUFFER(
                       context,
                       CL_MEM_READ_WRITE,
                       sizeof(cl_uint),
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed.(globalOutBuf).");
    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("AtomicCounters_Kernels.cl");
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
    // ConstantBuffer bandwidth from single access
    counterKernel = CECL_KERNEL(program, "atomicCounters", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(counterKernel).");
    globalKernel = CECL_KERNEL(program, "globalAtomics", &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL(globalKernel) failed.");
    status = kernelInfoC.setKernelWorkGroupInfo(counterKernel,
             devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "kernelInfo.setKernelWorkGroupInfo failed");
    status = kernelInfoG.setKernelWorkGroupInfo(globalKernel,
             devices[sampleArgs->deviceId]);
    CHECK_OPENCL_ERROR(status, "kernelInfo.setKernelWorkGroupInfo failed");
    if(counterWorkGroupSize > kernelInfoC.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << counterWorkGroupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel(readKernel) : "
                      << kernelInfoC.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfoC.kernelWorkGroupSize << std::endl;
        }
        counterWorkGroupSize = kernelInfoC.kernelWorkGroupSize;
    }
    if(globalWorkGroupSize > kernelInfoG.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << globalWorkGroupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel(writeKernel) : "
                      << kernelInfoG.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfoG.kernelWorkGroupSize << std::endl;
        }
        globalWorkGroupSize = kernelInfoG.kernelWorkGroupSize;
    }
    // Wait for event and release event
    status = waitForEventAndRelease(&writeEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(writeEvt) failed.");
    return SDK_SUCCESS;
}

int
AtomicCounters::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL Resources Initialization failed");
    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Allocation failed(array_length)");
    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the Input array";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);
    delete array_length;

    Option* numLoops = new Option;
    CHECK_ALLOCATION(numLoops, "Allocation failed(numLoops)");
    numLoops->_sVersion = "i";
    numLoops->_lVersion = "iterations";
    numLoops->_description = "Number of timing loops";
    numLoops->_type = CA_ARG_INT;
    numLoops->_value = &iterations;
    sampleArgs->AddOption(numLoops);
    delete numLoops;
    return SDK_SUCCESS;
}

int
AtomicCounters::setup()
{
    return setupCL();
}

void
AtomicCounters::cpuRefImplementation()
{
    for(cl_uint i = 0; i < length; ++i)
        if(value == input[i])
        {
            refOut++;
        }
}

int AtomicCounters::verifyResults()
{
    if(sampleArgs->verify)
    {
        // Calculate the reference output
        cpuRefImplementation();
        // Compare the results and see if they match
        if(refOut == counterOut && refOut == globalOut)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}

int
AtomicCounters::runAtomicCounterKernel()
{
    cl_int status = CL_SUCCESS;
    // Set Global and Local work items
    size_t globalWorkItems = length;
    size_t localWorkItems = counterWorkGroupSize;
    // Initialize the counter value
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 counterOutBuf,
                 CL_FALSE,
                 0,
                 sizeof(cl_uint),
                 &initValue,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(counterOutBuf) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush(commandQueue)failed.");
    // Wait for event and release event
    status = waitForEventAndRelease(&writeEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(writeEvt) failed.");
    // Set kernel arguments
    status = CECL_SET_KERNEL_ARG(counterKernel, 0, sizeof(cl_mem), &inBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(inBuf) failed.");
    status = CECL_SET_KERNEL_ARG(counterKernel, 1, sizeof(cl_uint), &value);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(value) failed.");
    status = CECL_SET_KERNEL_ARG(counterKernel, 2, sizeof(cl_mem), &counterOutBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(counterOutBuf) failed.");
    // Run Kernel
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 counterKernel,
                 1,
                 NULL,
                 &globalWorkItems,
                 &localWorkItems,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL(counterKernel) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush(commandQueue) failed.");
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
                     ndrEvt,
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     sizeof(cl_int),
                     &eventStatus,
                     NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventInfo(ndrEvt) failed.");
    }
    cl_ulong startTime;
    cl_ulong endTime;
    // Get profiling information
    status = clGetEventProfilingInfo(
                 ndrEvt,
                 CL_PROFILING_COMMAND_START,
                 sizeof(cl_ulong),
                 &startTime,
                 NULL);
    CHECK_OPENCL_ERROR(status,
                       "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START) failed.");
    status = clGetEventProfilingInfo(
                 ndrEvt,
                 CL_PROFILING_COMMAND_END,
                 sizeof(cl_ulong),
                 &endTime,
                 NULL);
    CHECK_OPENCL_ERROR(status,
                       "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END) failed.");
    double sec = 1e-9 * (endTime - startTime);
    kTimeAtomCounter += sec;
    status = clReleaseEvent(ndrEvt);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent(ndrEvt) failed.");
    // Get the occurrences of Value from atomicKernel
    cl_event readEvt;
    status = CECL_READ_BUFFER(
                 commandQueue,
                 counterOutBuf,
                 CL_FALSE,
                 0,
                 sizeof(cl_uint),
                 &counterOut,
                 0,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(counterOutBuf) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");
    // Wait for event and release event
    status = waitForEventAndRelease(&readEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(readEvt) failed.");
    return SDK_SUCCESS;
}


int
AtomicCounters::runGlobalAtomicKernel()
{
    cl_int status = CL_SUCCESS;
    // Set Global and Local work items
    size_t globalWorkItems = length;
    size_t localWorkItems = globalWorkGroupSize;
    // Initialize the counter value
    cl_event writeEvt;
    status = CECL_WRITE_BUFFER(
                 commandQueue,
                 globalOutBuf,
                 CL_FALSE,
                 0,
                 sizeof(cl_uint),
                 &initValue,
                 0,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER(globalOutBuf) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");
    // Wait for event and release event
    status = waitForEventAndRelease(&writeEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(writeEvt) failed.");
    // Set kernel arguments
    status = CECL_SET_KERNEL_ARG(globalKernel, 0, sizeof(cl_mem), &inBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(inBuf) failed.");
    status = CECL_SET_KERNEL_ARG(globalKernel, 1, sizeof(cl_uint), &value);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(value) failed.");
    status = CECL_SET_KERNEL_ARG(globalKernel, 2, sizeof(cl_mem), &globalOutBuf);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG(globalOutBuf) failed.");
    // Run Kernel
    cl_event ndrEvt;
    status = CECL_ND_RANGE_KERNEL(
                 commandQueue,
                 globalKernel,
                 1,
                 NULL,
                 &globalWorkItems,
                 &localWorkItems,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL(globalKernel) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush(commandQueue) failed.");
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
                     ndrEvt,
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     sizeof(cl_int),
                     &eventStatus,
                     NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventInfo(ndrEvt) failed.");
    }
    cl_ulong startTime;
    cl_ulong endTime;
    // Get profiling information
    status = clGetEventProfilingInfo(
                 ndrEvt,
                 CL_PROFILING_COMMAND_START,
                 sizeof(cl_ulong),
                 &startTime,
                 NULL);
    CHECK_OPENCL_ERROR(status,
                       "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START) failed.");
    status = clGetEventProfilingInfo(
                 ndrEvt,
                 CL_PROFILING_COMMAND_END,
                 sizeof(cl_ulong),
                 &endTime,
                 NULL);
    CHECK_OPENCL_ERROR(status,
                       "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END) failed.");
    double sec = 1e-9 * (endTime - startTime);
    kTimeAtomGlobal += sec;
    status = clReleaseEvent(ndrEvt);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent(ndrEvt) failed.");
    // Get the occurrences of Value from atomicKernel
    cl_event readEvt;
    status = CECL_READ_BUFFER(
                 commandQueue,
                 globalOutBuf,
                 CL_FALSE,
                 0,
                 sizeof(cl_uint),
                 &globalOut,
                 0,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(globalOutBuf) failed.");
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush() failed.");
    // Wait for event and release event
    status = waitForEventAndRelease(&readEvt);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease(readEvt) failed.");
    return SDK_SUCCESS;
}

int
AtomicCounters::run()
{
    // Warm up Atomic counter kernel
    for(int i = 0; i < 2 && iterations != 1; i++)
        if(runAtomicCounterKernel())
        {
            return SDK_FAILURE;
        }
    std::cout << "Executing Kernels for " << iterations << " iterations" <<
              std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    kTimeAtomCounter = 0;
    // Run the kernel for a number of iterations
    for(int i = 0; i < iterations; i++)
        if(runAtomicCounterKernel())
        {
            return SDK_FAILURE;
        }
    // Compute total time
    kTimeAtomCounter /= iterations;
    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>("Atomic Counter Output", &counterOut, 1, 1);
    }
    // Warm up Global atomics kernel
    for(int i = 0; i < 2 && iterations != 1; i++)
        if(runGlobalAtomicKernel())
        {
            return SDK_FAILURE;
        }
    kTimeAtomGlobal = 0;
    // Run the kernel for a number of iterations
    for(int i = 0; i < iterations; i++)
        if(runGlobalAtomicKernel())
        {
            return SDK_FAILURE;
        }
    // Compute total time
    kTimeAtomGlobal /= iterations;
    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>("Global atomics Output", &globalOut, 1, 1);
    }
    return SDK_SUCCESS;
}

void
AtomicCounters::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Occurrences", "AtomicsCounter(sec)", "GlobalAtomics(sec)"};
        std::string stats[4];
        stats[0]  = toString(length, std::dec);
        stats[1]  = toString(counterOut, std::dec);
        stats[2]  = toString(kTimeAtomCounter, std::dec);
        stats[3]  = toString(kTimeAtomGlobal, std::dec);
        printStatistics(strArray, stats, 4);
    }
}

int
AtomicCounters::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;
    status = clReleaseMemObject(inBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(inBuf) failed.");
    status = clReleaseMemObject(counterOutBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(counterOutBuf) failed.");
    status = clReleaseMemObject(globalOutBuf);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject(globalOutBuf) failed.");
    status = clReleaseKernel(counterKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel(counterKernel) failed.");
    status = clReleaseKernel(globalKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel(globalKernel) failed.");
    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram(program) failed.");
    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue(commandQueue) failed.");
    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext(context) failed.");
    free(input);
    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    int status = 0;
    AtomicCounters clAtomicCounters;
    if(clAtomicCounters.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    if(clAtomicCounters.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    if(clAtomicCounters.sampleArgs->isDumpBinaryEnabled())
    {
        return clAtomicCounters.genBinaryImage();
    }
    status = clAtomicCounters.setup();
    if(status != SDK_SUCCESS)
    {
        return status;
    }
    if(clAtomicCounters.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    if(clAtomicCounters.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    if(clAtomicCounters.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    clAtomicCounters.printStats();
    return SDK_SUCCESS;
}
