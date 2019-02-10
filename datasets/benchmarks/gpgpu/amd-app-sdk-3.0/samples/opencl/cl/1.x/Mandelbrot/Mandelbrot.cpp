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

#include <iostream>
using namespace std;

#include "Mandelbrot.hpp"

union uchar4
{
    struct __uchar_four
    {
        unsigned char s0;
        unsigned char s1;
        unsigned char s2;
        unsigned char s3;
    } ch;
    cl_uint num;
};

struct int4;
struct float4
{
    float s0;
    float s1;
    float s2;
    float s3;

    float4 operator * (float4 &fl)
    {
        float4 temp;
        temp.s0 = (this->s0) * fl.s0;
        temp.s1 = (this->s1) * fl.s1;
        temp.s2 = (this->s2) * fl.s2;
        temp.s3 = (this->s3) * fl.s3;
        return temp;
    }

    float4 operator * (float scalar)
    {
        float4 temp;
        temp.s0 = (this->s0) * scalar;
        temp.s1 = (this->s1) * scalar;
        temp.s2 = (this->s2) * scalar;
        temp.s3 = (this->s3) * scalar;
        return temp;
    }

    float4 operator + (float4 &fl)
    {
        float4 temp;
        temp.s0 = (this->s0) + fl.s0;
        temp.s1 = (this->s1) + fl.s1;
        temp.s2 = (this->s2) + fl.s2;
        temp.s3 = (this->s3) + fl.s3;
        return temp;
    }

    float4 operator - (float4 fl)
    {
        float4 temp;
        temp.s0 = (this->s0) - fl.s0;
        temp.s1 = (this->s1) - fl.s1;
        temp.s2 = (this->s2) - fl.s2;
        temp.s3 = (this->s3) - fl.s3;
        return temp;
    }

    friend float4 operator * (float scalar, float4 &fl);
    friend float4 convert_float4(int4 i);
};

float4 operator * (float scalar, float4 &fl)
{
    float4 temp;
    temp.s0 = fl.s0 * scalar;
    temp.s1 = fl.s1 * scalar;
    temp.s2 = fl.s2 * scalar;
    temp.s3 = fl.s3 * scalar;
    return temp;
}


struct double4
{
    double s0;
    double s1;
    double s2;
    double s3;

    double4 operator * (double4 &fl)
    {
        double4 temp;
        temp.s0 = (this->s0) * fl.s0;
        temp.s1 = (this->s1) * fl.s1;
        temp.s2 = (this->s2) * fl.s2;
        temp.s3 = (this->s3) * fl.s3;
        return temp;
    }

    double4 operator * (double scalar)
    {
        double4 temp;
        temp.s0 = (this->s0) * scalar;
        temp.s1 = (this->s1) * scalar;
        temp.s2 = (this->s2) * scalar;
        temp.s3 = (this->s3) * scalar;
        return temp;
    }

    double4 operator + (double4 &fl)
    {
        double4 temp;
        temp.s0 = (this->s0) + fl.s0;
        temp.s1 = (this->s1) + fl.s1;
        temp.s2 = (this->s2) + fl.s2;
        temp.s3 = (this->s3) + fl.s3;
        return temp;
    }

    double4 operator - (double4 fl)
    {
        double4 temp;
        temp.s0 = (this->s0) - fl.s0;
        temp.s1 = (this->s1) - fl.s1;
        temp.s2 = (this->s2) - fl.s2;
        temp.s3 = (this->s3) - fl.s3;
        return temp;
    }

    friend double4 operator * (double scalar, double4 &fl);
    friend double4 convert_double4(int4 i);
};

double4 operator * (double scalar, double4 &fl)
{
    double4 temp;
    temp.s0 = fl.s0 * scalar;
    temp.s1 = fl.s1 * scalar;
    temp.s2 = fl.s2 * scalar;
    temp.s3 = fl.s3 * scalar;
    return temp;
}

struct int4
{
    int s0;
    int s1;
    int s2;
    int s3;

    int4 operator * (int4 &fl)
    {
        int4 temp;
        temp.s0 = (this->s0) * fl.s0;
        temp.s1 = (this->s1) * fl.s1;
        temp.s2 = (this->s2) * fl.s2;
        temp.s3 = (this->s3) * fl.s3;
        return temp;
    }

    int4 operator * (int scalar)
    {
        int4 temp;
        temp.s0 = (this->s0) * scalar;
        temp.s1 = (this->s1) * scalar;
        temp.s2 = (this->s2) * scalar;
        temp.s3 = (this->s3) * scalar;
        return temp;
    }

    int4 operator + (int4 &fl)
    {
        int4 temp;
        temp.s0 = (this->s0) + fl.s0;
        temp.s1 = (this->s1) + fl.s1;
        temp.s2 = (this->s2) + fl.s2;
        temp.s3 = (this->s3) + fl.s3;
        return temp;
    }

    int4 operator - (int4 fl)
    {
        int4 temp;
        temp.s0 = (this->s0) - fl.s0;
        temp.s1 = (this->s1) - fl.s1;
        temp.s2 = (this->s2) - fl.s2;
        temp.s3 = (this->s3) - fl.s3;
        return temp;
    }

    int4 operator += (int4 fl)
    {
        s0 += fl.s0;
        s1 += fl.s1;
        s2 += fl.s2;
        s3 += fl.s3;
        return (*this);
    }

    friend float4 convert_float4(int4 i);
    friend double4 convert_double4(int4 i);
};

float4 convert_float4(int4 i)
{
    float4 temp;
    temp.s0 = (float)i.s0;
    temp.s1 = (float)i.s1;
    temp.s2 = (float)i.s2;
    temp.s3 = (float)i.s3;
    return temp;
}

double4 convert_double4(int4 i)
{
    double4 temp;
    temp.s0 = (double)i.s0;
    temp.s1 = (double)i.s1;
    temp.s2 = (double)i.s2;
    temp.s3 = (double)i.s3;
    return temp;
}

inline float native_log2(float in)
{
    return log(in)/log(2.0f);
}

inline float native_cos(float in)
{
    return cos(in);
}

inline double native_log2(double in)
{
    return log(in)/log(2.0f);
}

inline double native_cos(double in)
{
    return cos(in);
}


#ifndef min
int min(int a1, int a2)
{
    return ((a1 < a2) ? a1 : a2);
}
#endif

int
Mandelbrot::setupMandelbrot()
{
    cl_uint sizeBytes;

    // allocate and init memory used by host.
    sizeBytes = width * height * sizeof(cl_uchar4);
    output = (cl_uint *) malloc(sizeBytes);
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

    if(sampleArgs->verify)
    {
        verificationOutput = (cl_uint *)malloc(sizeBytes);
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");
    }

    return SDK_SUCCESS;
}

int
Mandelbrot::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("MandelBrot_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(enableFMA)
    {
        binaryData.flagsStr.append("-D MUL_ADD=fma ");
    }
    else
    {
        binaryData.flagsStr.append("-D MUL_ADD=mad ");
    }

    if(enableDouble)
    {
        binaryData.flagsStr.append("-D ENABLE_DOUBLE ");
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int
Mandelbrot::setupCL(void)
{
    cl_int status = 0;
    size_t deviceListSize;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_GPU;
    }
    else //sampleArgs->deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_GPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    // Use NULL for backward compatibility
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;

    context = CECL_CREATE_CONTEXT_FROM_TYPE(
                  cprops,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR(status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // First, get the size of device list data
    status = clGetContextInfo(
                 context,
                 CL_CONTEXT_DEVICES,
                 0,
                 NULL,
                 &deviceListSize);
    CHECK_OPENCL_ERROR(status, "clGetContextInfo failed.");

    int deviceCount = (int)(deviceListSize / sizeof(cl_device_id));

    status = validateDeviceId(sampleArgs->deviceId, deviceCount);
    CHECK_ERROR(status, SDK_SUCCESS, "validateDeviceId() failed");

    (devices) = (cl_device_id *)malloc(deviceListSize);
    CHECK_ALLOCATION((devices), "Failed to allocate memory (devices).");

    // Now, get the device list data
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              (devices),
                              NULL);
    CHECK_OPENCL_ERROR(status, "clGetGetContextInfo failed.");



    numDevices = (cl_uint)(deviceListSize/sizeof(cl_device_id));
    numDevices = min(MAX_DEVICES, numDevices);

    if(numDevices != 1 && sampleArgs->isLoadBinaryEnabled())
    {
        expectedError("--load option is not supported if devices are more one");
        return SDK_EXPECTED_FAILURE;
    }

    if(numDevices == 3)
    {
        if(!sampleArgs->quiet)
        {
            cout << "Number of devices must be even,"
                 << "\nChanging number of devices from three to two\n";
        }
        numDevices = 2;
    }

    // Set numDevices to 1 if devicdeId option is used
    if(sampleArgs->isDeviceIdEnabled())
    {
        numDevices = 1;
        devices[0] = devices[sampleArgs->deviceId];
        sampleArgs->deviceId = 0;
    }

    std::string flagsStr = std::string("");
    if(enableDouble)
    {
        // Check whether the device supports double-precision
        int khrFP64 = 0;
        int amdFP64 = 0;
        for (cl_uint i = 0; i < numDevices; i++)
        {
            char deviceExtensions[8192];

            // Get device extensions
            status = clGetDeviceInfo(devices[i],
                                     CL_DEVICE_EXTENSIONS,
                                     sizeof(deviceExtensions),
                                     deviceExtensions,
                                     0);
            CHECK_OPENCL_ERROR(status, "clGetDeviceInfo failed.(extensions)");

            // Check if cl_khr_fp64 extension is supported
            if(strstr(deviceExtensions, "cl_khr_fp64"))
            {
                khrFP64++;
            }
            else
            {
                // Check if cl_amd_fp64 extension is supported
                if(!strstr(deviceExtensions, "cl_amd_fp64"))
                {
                    OPENCL_EXPECTED_ERROR("Device does not support cl_amd_fp64 extension!");
                }
                else
                {
                    amdFP64++;
                }
            }
        }

        if(khrFP64 == numDevices)
        {
            flagsStr.append("-D KHR_DP_EXTENSION ");
        }
        else if(amdFP64 == numDevices)
        {
            flagsStr.append("");
        }
        else
        {
            expectedError("All devices must have same extension(either cl_amd_fp64 or cl_khr_fp64)!");
            return SDK_EXPECTED_FAILURE;
        }
    }

    for (cl_uint i = 0; i < numDevices; i++)
    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        if(sampleArgs->timing)
        {
            prop |= CL_QUEUE_PROFILING_ENABLE;
        }

        commandQueue[i] = CECL_CREATE_COMMAND_QUEUE(
                              context,
                              devices[i],
                              prop,
                              &status);
        CHECK_OPENCL_ERROR(status, "CECL_CREATE_COMMAND_QUEUE failed.");

        outputBuffer[i] = CECL_BUFFER(
                              context,
                              CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              (sizeof(cl_uint) * width * height) / numDevices,
                              NULL,
                              &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (outputBuffer)");
    }

    // create a CL program using the kernel source
    SDKFile kernelFile;

    if(enableFMA)
    {
        flagsStr.append("-D MUL_ADD=fma ");
    }
    else
    {
        flagsStr.append("-D MUL_ADD=mad ");
    }

    if(enableDouble)
    {
        flagsStr.append("-D ENABLE_DOUBLE ");
    }

    std::string kernelPath = getPath();
    if(sampleArgs->isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs->loadBinary.c_str());

        if(kernelFile.readBinaryFromFile(kernelPath.c_str()) != SDK_SUCCESS)
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        const char * binary = kernelFile.source().c_str();
        size_t binarySize = kernelFile.source().size();
        program = clCreateProgramWithBinary(context,
                                            1,
                                            &devices[sampleArgs->deviceId],
                                            (const size_t *)&binarySize,
                                            (const unsigned char**)&binary,
                                            NULL,
                                            &status);
    }
    else
    {
        kernelPath.append("Mandelbrot_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        const char * source = kernelFile.source().c_str();
        size_t sourceSize[] = { strlen(source) };
        program = CECL_PROGRAM_WITH_SOURCE(context,
                                            1,
                                            &source,
                                            sourceSize,
                                            &status);
    }
    CHECK_OPENCL_ERROR(status,"CECL_PROGRAM_WITH_SOURCE failed.");

    // Get additional options

    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load flags file: " << flagsPath << std::endl;
            return SDK_FAILURE;
        }
        flagsFile.replaceNewlineWithSpaces();
        const char * flags = flagsFile.source().c_str();
        flagsStr.append(flags);
    }

    if(flagsStr.size() != 0)
    {
        std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
    }

    /* create a cl program executable for all the devices specified */
    status = CECL_PROGRAM(program,
                            numDevices,
                            devices,
                            flagsStr.c_str(),
                            NULL,
                            NULL);

    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
        {
            for (cl_uint i = 0; i < numDevices; i++)
            {
                cl_int logStatus;
                char * buildLog = NULL;
                size_t buildLogSize = 0;
                logStatus = clGetProgramBuildInfo (program,
                                                   devices[sampleArgs->deviceId],
                                                   CL_PROGRAM_BUILD_LOG,
                                                   buildLogSize,
                                                   buildLog,
                                                   &buildLogSize);
                CHECK_OPENCL_ERROR(logStatus,"clGetProgramBuildInfo failed.");

                buildLog = (char*)malloc(buildLogSize);
                CHECK_ALLOCATION(buildLog, "Failed to allocate host memory.");

                memset(buildLog, 0, buildLogSize);

                logStatus = clGetProgramBuildInfo (program,
                                                   devices[sampleArgs->deviceId],
                                                   CL_PROGRAM_BUILD_LOG,
                                                   buildLogSize,
                                                   buildLog,
                                                   NULL);
                if(checkVal(
                            logStatus,
                            CL_SUCCESS,
                            "clGetProgramBuildInfo failed."))
                {
                    free(buildLog);
                    return SDK_FAILURE;
                }

                std::cout << " \n\t\t\tBUILD LOG\n";
                std::cout << " ************************************************\n";
                std::cout << buildLog << std::endl;
                std::cout << " ************************************************\n";
                free(buildLog);
            }
        }

        CHECK_OPENCL_ERROR(status,"CECL_PROGRAM failed.");

    }

    for (cl_uint i = 0; i < numDevices; i++)
    {
        // get a kernel object handle for a kernel with the given name
        if(enableDouble)
        {
            kernel_vector[i] = CECL_KERNEL(program, "mandelbrot_vector_double",
                                              &status);
        }
        else
        {
            kernel_vector[i] = CECL_KERNEL(program, "mandelbrot_vector_float", &status);
        }

        CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");
    }
    return SDK_SUCCESS;
}


int
Mandelbrot::runCLKernels(void)
{
    cl_int   status;
    cl_kernel kernel;
    cl_event events[MAX_DEVICES];
    cl_int eventStatus = CL_QUEUED;

    size_t globalThreads[1];
    size_t localThreads[1];

    benched = 0;
    globalThreads[0] = (width * height) / numDevices;
    localThreads[0]  = 256;

    globalThreads[0] >>= 2;

    for (cl_uint i = 0; i < numDevices; i++)
    {
        kernel = kernel_vector[i];

        // Check group size against kernelWorkGroupSize
        status = CECL_GET_KERNEL_WORK_GROUP_INFO(kernel,
                                          devices[i],
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t),
                                          &kernelWorkGroupSize,
                                          0);
        CHECK_OPENCL_ERROR(status, "CECL_GET_KERNEL_WORK_GROUP_INFO failed.");

        if((cl_uint)(localThreads[0]) > kernelWorkGroupSize)
        {
            localThreads[0] = kernelWorkGroupSize;
        }

        double aspect = (double)width / (double)height;
        xstep = (xsize / (double)width);
        // Adjust for aspect ratio
        double ysize = xsize / aspect;
        ystep = (-(xsize / aspect) / height);
        leftx = (xpos - xsize / 2.0);
        topy = (ypos + ysize / 2.0 -((double)i * ysize) / (double)numDevices);

        if(i == 0)
        {
            topy0 = topy;
        }

        // Set appropriate arguments to the kernel
        status = CECL_SET_KERNEL_ARG(
                     kernel,
                     0,
                     sizeof(cl_mem),
                     (void *)&outputBuffer[i]);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (outputBuffer)");
        if(enableDouble)
        {
            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         1,
                         sizeof(cl_double),
                         (void *)&leftx);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (leftx)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         2,
                         sizeof(cl_double),
                         (void *)&topy);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (topy)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         3,
                         sizeof(cl_double),
                         (void *)&xstep);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (xstep)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         4,
                         sizeof(cl_double),
                         (void *)&ystep);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (ystep)");
        }
        else
        {
            cl_float leftxF = (float)leftx;
            cl_float topyF = (float)topy;
            cl_float xstepF = (float)xstep;
            cl_float ystepF = (float)ystep;
            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         1,
                         sizeof(cl_float),
                         (void *)&leftxF);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (leftxF)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         2,
                         sizeof(cl_float),
                         (void *)&topyF);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (topyF)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         3,
                         sizeof(cl_float),
                         (void *)&xstepF);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (xstepF)");

            status = CECL_SET_KERNEL_ARG(
                         kernel,
                         4,
                         sizeof(cl_float),
                         (void *)&ystepF);
            CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (ystepF)");
        }
        status = CECL_SET_KERNEL_ARG(
                     kernel,
                     5,
                     sizeof(cl_uint),
                     (void *)&maxIterations);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (maxIterations)");

        // width - i.e number of elements in the array
        status = CECL_SET_KERNEL_ARG(
                     kernel,
                     6,
                     sizeof(cl_int),
                     (void *)&width);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (width)");

        // bench - flag to indicate benchmark mode
        status = CECL_SET_KERNEL_ARG(
                     kernel,
                     7,
                     sizeof(cl_int),
                     (void *)&bench);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (bench)");

        /*
         * Enqueue a kernel run call.
         */
        status = CECL_ND_RANGE_KERNEL(
                     commandQueue[i],
                     kernel,
                     1,
                     NULL,
                     globalThreads,
                     localThreads,
                     0,
                     NULL,
                     &events[i]);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");
    }
    // flush the queues to get things started
    for (cl_uint i = 0; i < numDevices; i++)
    {
        status = clFlush(commandQueue[i]);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");
    }

    // wait for the kernel call to finish execution
    for (cl_uint i = 0; i < numDevices; i++)
    {
        status = waitForEventAndRelease(&events[numDevices-i-1]);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "WaitForEventAndRelease(events[numDevices - i - 1]) Failed");
    }

    if (sampleArgs->timing && bench)
    {
        cl_ulong start;
        cl_ulong stop;
        status = clGetEventProfilingInfo(events[0],
                                         CL_PROFILING_COMMAND_SUBMIT,
                                         sizeof(cl_ulong),
                                         &start,
                                         NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.");

        status = clGetEventProfilingInfo(events[0],
                                         CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong),
                                         &stop,
                                         NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.");

        time = (cl_double)(stop - start)*(cl_double)(1e-09);
    }
    for (cl_uint i = 0; i < numDevices; i++)
    {
        // Enqueue readBuffer
        status = CECL_READ_BUFFER(
                     commandQueue[i],
                     outputBuffer[i],
                     CL_FALSE,
                     0,
                     (width * height * sizeof(cl_int)) / numDevices,
                     output + (width * height / numDevices) * i,
                     0,
                     NULL,
                     &events[i]);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.");
    }

    for (cl_uint i = 0; i < numDevices; i++)
    {
        status = clFlush(commandQueue[i]);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");
    }

    // wait for the kernel call to finish execution
    for (cl_uint i = 0; i < numDevices; i++)
    {
        status = waitForEventAndRelease(&events[numDevices - i - 1]);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "WaitForEventAndRelease(events[numDevices - i - 1]) Failed");
    }

    if (sampleArgs->timing && bench)
    {
        cl_ulong totalIterations = 0;
        for (int i = 0; i < (width * height); i++)
        {
            totalIterations += output[i];
        }
        cl_double flops = 7.0*totalIterations;
        printf("%lf MFLOPs\n", flops*(double)(1e-6)/time);
        printf("%lf MFLOPs according to CPU\n", flops*(double)(1e-6)/totalKernelTime);
        bench = 0;
        benched = 1;
    }
    return SDK_SUCCESS;
}

/**
* Mandelbrot fractal generated with CPU reference implementation
*/

void
Mandelbrot::mandelbrotRefFloat(cl_uint * verificationOutput,
                               cl_float posx,
                               cl_float posy,
                               cl_float stepSizeX,
                               cl_float stepSizeY,
                               cl_int maxIterations,
                               cl_int width,
                               cl_int bench
                              )
{
    int tid;

    for(tid = 0; tid < (height * width / 4); tid++)
    {
        int i = tid%(width/4);
        int j = tid/(width/4);

        int4 veci = {4*i, 4*i+1, 4*i+2, 4*i+3};
        int4 vecj = {j, j, j, j};
        float4 x0;
        x0.s0 = (float)(posx + stepSizeX * (float)veci.s0);
        x0.s1 = (float)(posx + stepSizeX * (float)veci.s1);
        x0.s2 = (float)(posx + stepSizeX * (float)veci.s2);
        x0.s3 = (float)(posx + stepSizeX * (float)veci.s3);
        float4 y0;
        y0.s0 = (float)(posy + stepSizeY * (float)vecj.s0);
        y0.s1 = (float)(posy + stepSizeY * (float)vecj.s1);
        y0.s2 = (float)(posy + stepSizeY * (float)vecj.s2);
        y0.s3 = (float)(posy + stepSizeY * (float)vecj.s3);

        float4 x = x0;
        float4 y = y0;

        cl_int iter=0;
        float4 tmp;
        int4 stay;
        int4 ccount = {0, 0, 0, 0};

        stay.s0 = (x.s0*x.s0 + y.s0*y.s0) <= 4.0f;
        stay.s1 = (x.s1*x.s1 + y.s1*y.s1) <= 4.0f;
        stay.s2 = (x.s2*x.s2 + y.s2*y.s2) <= 4.0f;
        stay.s3 = (x.s3*x.s3 + y.s3*y.s3) <= 4.0f;
        float4 savx = x;
        float4 savy = y;

        for(iter=0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) &&
                (iter < maxIterations); iter+= 16)
        {
            x = savx;
            y = savy;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0f * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0f * tmp * y + y0;

            stay.s0 = (x.s0*x.s0 + y.s0*y.s0) <= 4.0f;
            stay.s1 = (x.s1*x.s1 + y.s1*y.s1) <= 4.0f;
            stay.s2 = (x.s2*x.s2 + y.s2*y.s2) <= 4.0f;
            stay.s3 = (x.s3*x.s3 + y.s3*y.s3) <= 4.0f;

            savx.s0 = (stay.s0 ? x.s0 : savx.s0);
            savx.s1 = (stay.s1 ? x.s1 : savx.s1);
            savx.s2 = (stay.s2 ? x.s2 : savx.s2);
            savx.s3 = (stay.s3 ? x.s3 : savx.s3);
            savy.s0 = (stay.s0 ? y.s0 : savy.s0);
            savy.s1 = (stay.s1 ? y.s1 : savy.s1);
            savy.s2 = (stay.s2 ? y.s2 : savy.s2);
            savy.s3 = (stay.s3 ? y.s3 : savy.s3);
            ccount += stay*16;
        }
        // Handle remainder
        if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))
        {
            iter = 16;
            do
            {
                x = savx;
                y = savy;
                stay.s0 = ((x.s0*x.s0 + y.s0*y.s0) <= 4.0f) && (ccount.s0 < maxIterations);
                stay.s1 = ((x.s1*x.s1 + y.s1*y.s1) <= 4.0f) && (ccount.s1 < maxIterations);
                stay.s2 = ((x.s2*x.s2 + y.s2*y.s2) <= 4.0f) && (ccount.s2 < maxIterations);
                stay.s3 = ((x.s3*x.s3 + y.s3*y.s3) <= 4.0f) && (ccount.s3 < maxIterations);
                tmp = x;
                x = x * x + x0 - y * y;
                y = 2.0f * tmp * y + y0;
                ccount += stay;
                iter--;
                savx.s0 = (stay.s0 ? x.s0 : savx.s0);
                savx.s1 = (stay.s1 ? x.s1 : savx.s1);
                savx.s2 = (stay.s2 ? x.s2 : savx.s2);
                savx.s3 = (stay.s3 ? x.s3 : savx.s3);
                savy.s0 = (stay.s0 ? y.s0 : savy.s0);
                savy.s1 = (stay.s1 ? y.s1 : savy.s1);
                savy.s2 = (stay.s2 ? y.s2 : savy.s2);
                savy.s3 = (stay.s3 ? y.s3 : savy.s3);
            }
            while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);
        }
        x = savx;
        y = savy;
        float4 fc = convert_float4(ccount);

        fc.s0 = (float)ccount.s0 + 1 - native_log2(native_log2(x.s0*x.s0 + y.s0*y.s0));
        fc.s1 = (float)ccount.s1 + 1 - native_log2(native_log2(x.s1*x.s1 + y.s1*y.s1));
        fc.s2 = (float)ccount.s2 + 1 - native_log2(native_log2(x.s2*x.s2 + y.s2*y.s2));
        fc.s3 = (float)ccount.s3 + 1 - native_log2(native_log2(x.s3*x.s3 + y.s3*y.s3));

        float c = fc.s0 * 2.0f * 3.1416f / 256.0f;
        uchar4 color[4];
        color[0].ch.s0 = (unsigned char)(((1.0f + native_cos(c))*0.5f)*255);
        color[0].ch.s1 = (unsigned char)(((1.0f + native_cos(2.0f*c +
                                           2.0f*3.1416f/3.0f))*0.5f)*255);
        color[0].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f*3.1416f/3.0f))
                                          *0.5f)*255);
        color[0].ch.s3 = 0xff;
        if (ccount.s0 == maxIterations)
        {
            color[0].ch.s0 = 0;
            color[0].ch.s1 = 0;
            color[0].ch.s2 = 0;
        }
        if (bench)
        {
            color[0].ch.s0 = ccount.s0 & 0xff;
            color[0].ch.s1 = (ccount.s0 & 0xff00)>>8;
            color[0].ch.s2 = (ccount.s0 & 0xff0000)>>16;
            color[0].ch.s3 = (ccount.s0 & 0xff000000)>>24;
        }
        verificationOutput[4*tid] = color[0].num;

        c = fc.s1 * 2.0f * 3.1416f / 256.0f;
        color[1].ch.s0 = (unsigned char)(((1.0f + native_cos(c))*0.5f)*255);
        color[1].ch.s1 = (unsigned char)(((1.0f + native_cos(2.0f*c +
                                           2.0f*3.1416f/3.0f))*0.5f)*255);
        color[1].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f*3.1416f/3.0f))
                                          *0.5f)*255);
        color[1].ch.s3 = 0xff;
        if (ccount.s1 == maxIterations)
        {
            color[1].ch.s0 = 0;
            color[1].ch.s1 = 0;
            color[1].ch.s2 = 0;
        }
        if (bench)
        {
            color[1].ch.s0 = ccount.s1 & 0xff;
            color[1].ch.s1 = (ccount.s1 & 0xff00)>>8;
            color[1].ch.s2 = (ccount.s1 & 0xff0000)>>16;
            color[1].ch.s3 = (ccount.s1 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+1] = color[1].num;

        c = fc.s2 * 2.0f * 3.1416f / 256.0f;
        color[2].ch.s0 = (unsigned char)(((1.0f + native_cos(c))*0.5f)*255);
        color[2].ch.s1 = (unsigned char)(((1.0f + native_cos(2.0f*c +
                                           2.0f*3.1416f/3.0f))*0.5f)*255);
        color[2].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f*3.1416f/3.0f))
                                          *0.5f)*255);
        color[2].ch.s3 = 0xff;
        if (ccount.s2 == maxIterations)
        {
            color[2].ch.s0 = 0;
            color[2].ch.s1 = 0;
            color[2].ch.s2 = 0;
        }
        if (bench)
        {
            color[2].ch.s0 = ccount.s2 & 0xff;
            color[2].ch.s1 = (ccount.s2 & 0xff00)>>8;
            color[2].ch.s2 = (ccount.s2 & 0xff0000)>>16;
            color[2].ch.s3 = (ccount.s2 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+2] = color[2].num;

        c = fc.s3 * 2.0f * 3.1416f / 256.0f;
        color[3].ch.s0 = (unsigned char)(((1.0f + native_cos(c))*0.5f)*255);
        color[3].ch.s1 = (unsigned char)(((1.0f + native_cos(2.0f*c +
                                           2.0f*3.1416f/3.0f))*0.5f)*255);
        color[3].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f*3.1416f/3.0f))
                                          *0.5f)*255);
        color[3].ch.s3 = 0xff;
        if (ccount.s3 == maxIterations)
        {
            color[3].ch.s0 = 0;
            color[3].ch.s1 = 0;
            color[3].ch.s2 = 0;
        }
        if (bench)
        {
            color[3].ch.s0 = ccount.s3 & 0xff;
            color[3].ch.s1 = (ccount.s3 & 0xff00)>>8;
            color[3].ch.s2 = (ccount.s3 & 0xff0000)>>16;
            color[3].ch.s3 = (ccount.s3 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+3] = color[3].num;
    }
}



/**
* Mandelbrot fractal generated with CPU reference implementation with double compution
*/

void
Mandelbrot::mandelbrotRefDouble(
    cl_uint * verificationOutput,
    cl_double posx,
    cl_double posy,
    cl_double stepSizeX,
    cl_double stepSizeY,
    cl_int maxIterations,
    cl_int width,
    cl_int bench)
{
    int tid;

    for(tid = 0; tid < (height * width / 4); tid++)
    {
        int i = tid%(width/4);
        int j = tid/(width/4);

        int4 veci = {4*i, 4*i+1, 4*i+2, 4*i+3};
        int4 vecj = {j, j, j, j};
        double4 x0;
        x0.s0 = (double)(posx + stepSizeX * (double)veci.s0);
        x0.s1 = (double)(posx + stepSizeX * (double)veci.s1);
        x0.s2 = (double)(posx + stepSizeX * (double)veci.s2);
        x0.s3 = (double)(posx + stepSizeX * (double)veci.s3);
        double4 y0;
        y0.s0 = (double)(posy + stepSizeY * (double)vecj.s0);
        y0.s1 = (double)(posy + stepSizeY * (double)vecj.s1);
        y0.s2 = (double)(posy + stepSizeY * (double)vecj.s2);
        y0.s3 = (double)(posy + stepSizeY * (double)vecj.s3);

        double4 x = x0;
        double4 y = y0;

        cl_int iter=0;
        double4 tmp;
        int4 stay;
        int4 ccount = {0, 0, 0, 0};

        stay.s0 = (x.s0*x.s0 + y.s0*y.s0) <= 4.0;
        stay.s1 = (x.s1*x.s1 + y.s1*y.s1) <= 4.0;
        stay.s2 = (x.s2*x.s2 + y.s2*y.s2) <= 4.0;
        stay.s3 = (x.s3*x.s3 + y.s3*y.s3) <= 4.0;
        double4 savx = x;
        double4 savy = y;

        for(iter=0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) &&
                (iter < maxIterations); iter+= 16)
        {
            x = savx;
            y = savy;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            // Two iterations
            tmp = x * x + x0 - y * y;
            y = 2.0 * x * y + y0;
            x = tmp * tmp + x0 - y * y;
            y = 2.0 * tmp * y + y0;

            stay.s0 = (x.s0*x.s0 + y.s0*y.s0) <= 4.0;
            stay.s1 = (x.s1*x.s1 + y.s1*y.s1) <= 4.0;
            stay.s2 = (x.s2*x.s2 + y.s2*y.s2) <= 4.0;
            stay.s3 = (x.s3*x.s3 + y.s3*y.s3) <= 4.0;

            savx.s0 = (stay.s0 ? x.s0 : savx.s0);
            savx.s1 = (stay.s1 ? x.s1 : savx.s1);
            savx.s2 = (stay.s2 ? x.s2 : savx.s2);
            savx.s3 = (stay.s3 ? x.s3 : savx.s3);
            savy.s0 = (stay.s0 ? y.s0 : savy.s0);
            savy.s1 = (stay.s1 ? y.s1 : savy.s1);
            savy.s2 = (stay.s2 ? y.s2 : savy.s2);
            savy.s3 = (stay.s3 ? y.s3 : savy.s3);
            ccount += stay*16;
        }
        // Handle remainder
        if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))
        {
            iter = 16;
            do
            {
                x = savx;
                y = savy;
                stay.s0 = ((x.s0*x.s0 + y.s0*y.s0) <= 4.0) && (ccount.s0 < maxIterations);
                stay.s1 = ((x.s1*x.s1 + y.s1*y.s1) <= 4.0) && (ccount.s1 < maxIterations);
                stay.s2 = ((x.s2*x.s2 + y.s2*y.s2) <= 4.0) && (ccount.s2 < maxIterations);
                stay.s3 = ((x.s3*x.s3 + y.s3*y.s3) <= 4.0) && (ccount.s3 < maxIterations);
                tmp = x;
                x = x * x + x0 - y * y;
                y = 2.0 * tmp * y + y0;
                ccount += stay;
                iter--;
                savx.s0 = (stay.s0 ? x.s0 : savx.s0);
                savx.s1 = (stay.s1 ? x.s1 : savx.s1);
                savx.s2 = (stay.s2 ? x.s2 : savx.s2);
                savx.s3 = (stay.s3 ? x.s3 : savx.s3);
                savy.s0 = (stay.s0 ? y.s0 : savy.s0);
                savy.s1 = (stay.s1 ? y.s1 : savy.s1);
                savy.s2 = (stay.s2 ? y.s2 : savy.s2);
                savy.s3 = (stay.s3 ? y.s3 : savy.s3);
            }
            while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);
        }
        x = savx;
        y = savy;
        double4 fc = convert_double4(ccount);

        fc.s0 = (double)ccount.s0 + 1 - native_log2(native_log2(x.s0*x.s0 + y.s0*y.s0));
        fc.s1 = (double)ccount.s1 + 1 - native_log2(native_log2(x.s1*x.s1 + y.s1*y.s1));
        fc.s2 = (double)ccount.s2 + 1 - native_log2(native_log2(x.s2*x.s2 + y.s2*y.s2));
        fc.s3 = (double)ccount.s3 + 1 - native_log2(native_log2(x.s3*x.s3 + y.s3*y.s3));

        double c = fc.s0 * 2.0 * 3.1416 / 256.0;
        uchar4 color[4];
        color[0].ch.s0 = (unsigned char)(((1.0 + native_cos(c))*0.5)*255);
        color[0].ch.s1 = (unsigned char)(((1.0 + native_cos(2.0*c + 2.0*3.1416/3.0))
                                          *0.5)*255);
        color[0].ch.s2 = (unsigned char)(((1.0 + native_cos(c - 2.0*3.1416/3.0))*0.5)
                                         *255);
        color[0].ch.s3 = 0xff;
        if (ccount.s0 == maxIterations)
        {
            color[0].ch.s0 = 0;
            color[0].ch.s1 = 0;
            color[0].ch.s2 = 0;
        }
        if (bench)
        {
            color[0].ch.s0 = ccount.s0 & 0xff;
            color[0].ch.s1 = (ccount.s0 & 0xff00)>>8;
            color[0].ch.s2 = (ccount.s0 & 0xff0000)>>16;
            color[0].ch.s3 = (ccount.s0 & 0xff000000)>>24;
        }
        verificationOutput[4*tid] = color[0].num;

        c = fc.s1 * 2.0 * 3.1416 / 256.0;
        color[1].ch.s0 = (unsigned char)(((1.0 + native_cos(c))*0.5)*255);
        color[1].ch.s1 = (unsigned char)(((1.0 + native_cos(2.0*c + 2.0*3.1416/3.0))
                                          *0.5)*255);
        color[1].ch.s2 = (unsigned char)(((1.0 + native_cos(c - 2.0*3.1416/3.0))*0.5)
                                         *255);
        color[1].ch.s3 = 0xff;
        if (ccount.s1 == maxIterations)
        {
            color[1].ch.s0 = 0;
            color[1].ch.s1 = 0;
            color[1].ch.s2 = 0;
        }
        if (bench)
        {
            color[1].ch.s0 = ccount.s1 & 0xff;
            color[1].ch.s1 = (ccount.s1 & 0xff00)>>8;
            color[1].ch.s2 = (ccount.s1 & 0xff0000)>>16;
            color[1].ch.s3 = (ccount.s1 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+1] = color[1].num;

        c = fc.s2 * 2.0 * 3.1416 / 256.0;
        color[2].ch.s0 = (unsigned char)(((1.0 + native_cos(c))*0.5)*255);
        color[2].ch.s1 = (unsigned char)(((1.0 + native_cos(2.0*c + 2.0*3.1416/3.0))
                                          *0.5)*255);
        color[2].ch.s2 = (unsigned char)(((1.0 + native_cos(c - 2.0*3.1416/3.0))*0.5)
                                         *255);
        color[2].ch.s3 = 0xff;
        if (ccount.s2 == maxIterations)
        {
            color[2].ch.s0 = 0;
            color[2].ch.s1 = 0;
            color[2].ch.s2 = 0;
        }
        if (bench)
        {
            color[2].ch.s0 = ccount.s2 & 0xff;
            color[2].ch.s1 = (ccount.s2 & 0xff00)>>8;
            color[2].ch.s2 = (ccount.s2 & 0xff0000)>>16;
            color[2].ch.s3 = (ccount.s2 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+2] = color[2].num;

        c = fc.s3 * 2.0 * 3.1416 / 256.0;
        color[3].ch.s0 = (unsigned char)(((1.0 + native_cos(c))*0.5)*255);
        color[3].ch.s1 = (unsigned char)(((1.0 + native_cos(2.0*c + 2.0*3.1416/3.0))
                                          *0.5)*255);
        color[3].ch.s2 = (unsigned char)(((1.0 + native_cos(c - 2.0*3.1416/3.0))*0.5)
                                         *255);
        color[3].ch.s3 = 0xff;
        if (ccount.s3 == maxIterations)
        {
            color[3].ch.s0 = 0;
            color[3].ch.s1 = 0;
            color[3].ch.s2 = 0;
        }
        if (bench)
        {
            color[3].ch.s0 = ccount.s3 & 0xff;
            color[3].ch.s1 = (ccount.s3 & 0xff00)>>8;
            color[3].ch.s2 = (ccount.s3 & 0xff0000)>>16;
            color[3].ch.s3 = (ccount.s3 & 0xff000000)>>24;
        }
        verificationOutput[4*tid+3] = color[3].num;
    }
}


int Mandelbrot::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* image_width = new Option;
    CHECK_ALLOCATION(image_width, "Memory allocation error.\n");

    image_width->_sVersion = "W";
    image_width->_lVersion = "width";
    image_width->_description = "width of the mandelbrot image";
    image_width->_type = CA_ARG_INT;
    image_width->_value = &width;
    sampleArgs->AddOption(image_width);
    delete image_width;

    Option* image_height = new Option;
    CHECK_ALLOCATION(image_height, "Memory allocation error.\n");

    image_height->_sVersion = "H";
    image_height->_lVersion = "height";
    image_height->_description = "height of the mandelbrot image";
    image_height->_type = CA_ARG_INT;
    image_height->_value = &height;
    sampleArgs->AddOption(image_height);
    delete image_height;

    Option* xpos_param = new Option;
    CHECK_ALLOCATION(xpos_param, "Memory allocation error.\n");

    xpos_param->_sVersion = "x";
    xpos_param->_lVersion = "xpos";
    xpos_param->_description = "xpos to generate the mandelbrot fractal";
    xpos_param->_type = CA_ARG_STRING;
    xpos_param->_value = &xpos_str;
    sampleArgs->AddOption(xpos_param);
    delete xpos_param;

    Option* ypos_param = new Option;
    CHECK_ALLOCATION(ypos_param, "Memory allocation error.\n");

    ypos_param->_sVersion = "y";
    ypos_param->_lVersion = "ypos";
    ypos_param->_description = "ypos to generate the mandelbrot fractal";
    ypos_param->_type = CA_ARG_STRING;
    ypos_param->_value = &ypos_str;
    sampleArgs->AddOption(ypos_param);
    delete ypos_param;

    Option* xsize_param = new Option;
    CHECK_ALLOCATION(xsize_param, "Memory allocation error.\n");

    xsize_param->_sVersion = "xs";
    xsize_param->_lVersion = "xsize";
    xsize_param->_description = "Width of window for the mandelbrot fractal";
    xsize_param->_type = CA_ARG_STRING;
    xsize_param->_value = &xsize_str;
    sampleArgs->AddOption(xsize_param);
    delete xsize_param;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;
    sampleArgs->AddOption(num_iterations);
    delete num_iterations;


    Option* num_double = new Option;
    CHECK_ALLOCATION(num_double, "Memory allocation error.\n");

    num_double->_lVersion = "double";
    num_double->_description = "Enable double data type.(Default : float)";
    num_double->_type = CA_NO_ARGUMENT;
    num_double->_value = &enableDouble;
    sampleArgs->AddOption(num_double);
    delete num_double;

    Option* num_FMA = new Option;
    CHECK_ALLOCATION(num_FMA, "Memory allocation error.\n");

    num_FMA->_lVersion = "fma";
    num_FMA->_description =
        "Enable Fused Multiply-Add(FMA).(Default : Multiply-Add)";
    num_FMA->_type = CA_NO_ARGUMENT;
    num_FMA->_value = &enableFMA;
    sampleArgs->AddOption(num_FMA);
    delete num_FMA;


    if (xpos_str != "")
    {
        xpos = atof(xpos_str.c_str());
    }
    if (ypos_str != "")
    {
        ypos = atof(ypos_str.c_str());
    }
    if (xsize_str != "")
    {
        xsize = atof(xsize_str.c_str());
    }
    else
    {
        xsize = 4.0;
    }
    return SDK_SUCCESS;
}

int Mandelbrot::setup()
{
    // Make sure width is a multiple of 4
    width = (width + 3) & ~(4 - 1);

    iterations = 1;

    if(setupMandelbrot()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int returnVal = setupCL();
    if(returnVal != SDK_SUCCESS)
    {
        return returnVal;
    }

    sampleTimer->stopTimer(timer);

    setupTime = (cl_double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int Mandelbrot::run()
{
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    return SDK_SUCCESS;
}

int Mandelbrot::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        if(enableDouble)
            mandelbrotRefDouble(
                verificationOutput,
                leftx,
                topy0,
                xstep,
                ystep,
                maxIterations,
                width,
                bench);
        else
            mandelbrotRefFloat(
                verificationOutput,
                (cl_float)leftx,
                (cl_float)topy0,
                (cl_float)xstep,
                (cl_float)ystep,
                maxIterations,
                width,
                bench);

        int i, j;
        int counter = 0;

        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                uchar4 temp_ver, temp_out;
                temp_ver.num = verificationOutput[j * width + i];
                temp_out.num = output[j * width + i];

                unsigned char threshold = 2;

                if( ((temp_ver.ch.s0 - temp_out.ch.s0) > threshold) ||
                        ((temp_out.ch.s0 - temp_ver.ch.s0) > threshold) ||

                        ((temp_ver.ch.s1 - temp_out.ch.s1) > threshold) ||
                        ((temp_out.ch.s1 - temp_ver.ch.s1) > threshold) ||

                        ((temp_ver.ch.s2 - temp_out.ch.s2) > threshold) ||
                        ((temp_out.ch.s2 - temp_ver.ch.s2) > threshold) ||

                        ((temp_ver.ch.s3 - temp_out.ch.s3) > threshold) ||
                        ((temp_out.ch.s3 - temp_ver.ch.s3) > threshold))
                {
                    counter++;
                }

            }
        }

        int numPixels = height * width;
        double ratio = (double)counter / numPixels;

        // compare the results and see if they match

        if( ratio < 0.002)
        {
            std::cout<<"Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }

    }

    return SDK_SUCCESS;
}

void Mandelbrot::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Width", "Height", "Time(sec)", "KernelTime(sec)"};
        std::string stats[4];

        sampleTimer->totalTime = setupTime + totalKernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(totalKernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int Mandelbrot::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    for (cl_uint i = 0; i < numDevices; i++)
    {
        status = clReleaseKernel(kernel_vector[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

        status = clReleaseMemObject(outputBuffer[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

        status = clReleaseCommandQueue(commandQueue[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");
    }

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)
    FREE(output);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}

cl_uint Mandelbrot::getWidth(void)
{
    return width;
}

cl_uint Mandelbrot::getHeight(void)
{
    return height;
}


cl_uint * Mandelbrot::getPixels(void)
{
    return output;
}

cl_bool Mandelbrot::showWindow(void)
{
    return !sampleArgs->quiet && !sampleArgs->verify;
}
