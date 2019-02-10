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


#include "FluidSimulation2D.hpp"
#include "ColorScale.h"
#include <GL/glut.h>
#include <cmath>
#include <malloc.h>

// Directions
double e[9][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1}};

// Weights
cl_double w[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

// Omega
const double omega = 1.2f;

// Verify flag
bool verifyFlag = 0;

FluidSimulation2D* me;           /**< Pointing to FluidSimulation2D class */
cl_bool display;
GLuint texnum;
ColorScale bluewhite(6);
bool drawBoundary = false;
bool addVelocity = false;
int oldx, oldy;

int drawVelocities = 1;
int drawParticles = 1;

int frames = 0;
int t0 = 0, te;

// Calculates equivalent distribution
double computefEq(cl_double weight, double dir[2], double rho,
                  cl_double2 velocity)
{
    double u2 = velocity.s[0] * velocity.s[0] + velocity.s[1] * velocity.s[1];
    double eu = dir[0] * velocity.s[0] + dir[1] * velocity.s[1];

    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

// Returns the velocity at (x, y) location relative to lattice
cl_double2 FluidSimulation2D::getVelocity(int x, int y)
{
    int pos = x + dims[0] * y;
    return u[pos];
}

// Returns the state of (x, y) position relative to Lattice
cl_bool FluidSimulation2D::isBoundary(int x, int y)
{
    return h_type[x + dims[0] * y];
}

// Returns the state of (x, y) position relative to Lattice
bool FluidSimulation2D::isFluid(int x, int y)
{
    if(h_type[x + dims[0] * y] == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void FluidSimulation2D::setUOutput(int x, int y, double v[2])
{
    double rho, uu[2];

    int pos = x + dims[0] * y;

    // Calculate density from input distribution
    rho = h_of0[pos] + h_of1234[pos * 4 + 0] +  h_of1234[pos * 4 + 1] +
          h_of1234[pos * 4 + 2] + h_of1234[pos * 4 + 3] +
          h_of5678[pos * 4 + 0] + h_of5678[pos * 4 + 1] + h_of5678[pos * 4 + 2] +
          h_of5678[pos * 4 + 3];

    uu[0] = u[pos].s[0];
    uu[1] = u[pos].s[1];

    // Increase the speed by input speed
    uu[0] += v[0];
    uu[1] += v[1];

    cl_double2 newVel;
    newVel.s[0] = uu[0];
    newVel.s[1] = uu[1];

    // Calculate new distribution based on input speed
    h_if0[pos] = computefEq(w[0], e[0], rho, newVel);
    h_if1234[pos * 4 + 0] = computefEq(w[1], e[1], rho, newVel);
    h_if1234[pos * 4 + 1] = computefEq(w[2], e[2], rho, newVel);
    h_if1234[pos * 4 + 2] = computefEq(w[3], e[3], rho, newVel);
    h_if1234[pos * 4 + 3] = computefEq(w[4], e[4], rho, newVel);

    h_if5678[pos * 4 + 0] = computefEq(w[5], e[5], rho, newVel);
    h_if5678[pos * 4 + 1] = computefEq(w[6], e[6], rho, newVel);
    h_if5678[pos * 4 + 2] = computefEq(w[7], e[7], rho, newVel);
    h_if5678[pos * 4 + 3] = computefEq(w[8], e[8], rho, newVel);
}


void FluidSimulation2D::setSite(int x, int y, bool cellType, double u[2])
{
    // Set type
    h_type[x + dims[0] * y] = cellType;

    // Set velocity and calculate new distributions
    setUOutput(x, y, u);

}

void FluidSimulation2D::reset()
{
    // Initial velocity is 0
    cl_double2 u0;
    u0.s[0] = u0.s[1] = 0.0f;

    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            int pos = x + y * dims[0];

            double den = 10.0f;

            // Initialize the velocity buffer
            u[pos] = u0;

            //rho[pos] = 10.0f;
            h_if0[pos] = computefEq(w[0], e[0], den, u0);
            h_if1234[pos * 4 + 0] = computefEq(w[1], e[1], den, u0);
            h_if1234[pos * 4 + 1] = computefEq(w[2], e[2], den, u0);
            h_if1234[pos * 4 + 2] = computefEq(w[3], e[3], den, u0);
            h_if1234[pos * 4 + 3] = computefEq(w[4], e[4], den, u0);

            h_if5678[pos * 4 + 0] = computefEq(w[5], e[5], den, u0);
            h_if5678[pos * 4 + 1] = computefEq(w[6], e[6], den, u0);
            h_if5678[pos * 4 + 2] = computefEq(w[7], e[7], den, u0);
            h_if5678[pos * 4 + 3] = computefEq(w[8], e[8], den, u0);

            // Initialize boundary cells
            if (x == 0 || x == (dims[0] - 1) || y == 0 || y == (dims[1] - 1))
            {
                h_type[pos] = 1;
            }

            // Initialize fluid cells
            else
            {
                h_type[pos] = 0;
            }
        }
    }
}


int
FluidSimulation2D::setupFluidSimulation2D()
{

    size_t temp = dims[0] * dims[1];

    // Allocate memory for host buffers
    h_if0 = (cl_double*)malloc(sizeof(cl_double) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_if0)");

    h_if1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_if1324)");

    h_if5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_if5678)");

    h_of0 = (cl_double*)malloc(sizeof(cl_double) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_of0)");

    h_of1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_of1234)");

    h_of5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_of5678)");

    if(sampleArgs->verify)
    {
        v_of0 = (cl_double*)malloc(sizeof(cl_double) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_of0)");

        v_of1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_of1234)");

        v_of5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_of5678)");

        v_ef0 = (cl_double*)malloc(sizeof(cl_double) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_ef0)");

        v_ef1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_ef1234)");

        v_ef5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
        CHECK_ALLOCATION(h_if0, "Memory allocation failed(v_ef5678)");
    }


    h_type = (cl_bool*)malloc(sizeof(cl_bool) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(h_type)");

    rho = (cl_double*)malloc(sizeof(cl_double) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(rho)");

    u = (cl_double2*)malloc(sizeof(cl_double2) * temp);
    CHECK_ALLOCATION(h_if0, "Memory allocation failed(u)");

    reset();

    return SDK_SUCCESS;
}

int
FluidSimulation2D::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("FluidSimulation2D_Kernels.cl");
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
FluidSimulation2D::setupCL()
{
    cl_int status = CL_SUCCESS;
    cl_device_type dType;

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


    // If we could find our platform, use it. Otherwise use just available platform.

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = CECL_CREATE_CONTEXT_FROM_TYPE(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUE failed.");
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, 0, "SDKDeviceInfo::setDeviceInfo() failed");


    std::string buildOptions = std::string("");
    // Check if cl_khr_fp64 extension is supported
    if(strstr(deviceInfo.extensions, "cl_khr_fp64"))
    {
        buildOptions.append("-D KHR_DP_EXTENSION");
    }
    else
    {
        // Check if cl_amd_fp64 extension is supported
        if(!strstr(deviceInfo.extensions, "cl_amd_fp64"))
        {
            reqdExtSupport = false;
            OPENCL_EXPECTED_ERROR("Device does not support cl_amd_fp64 extension!");
        }
    }


    /*
    * Create and initialize memory objects
    */

    size_t temp = dims[0] * dims[1];
    d_if0 = CECL_BUFFER(context,
                           CL_MEM_READ_WRITE,
                           sizeof(cl_double) * temp,
                           0,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_if0)");

    status = CECL_WRITE_BUFFER(commandQueue,
                                  d_if0,
                                  1,
                                  0,
                                  sizeof(cl_double) * temp,
                                  h_if0,
                                  0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if0)");

    d_if1234 = CECL_BUFFER(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_double4) * temp,
                              0,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_if1234)");

    status = CECL_WRITE_BUFFER(commandQueue,
                                  d_if1234,
                                  1,
                                  0,
                                  sizeof(cl_double4) * temp,
                                  h_if1234,
                                  0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if1234)");

    d_if5678 = CECL_BUFFER(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_double4) * temp,
                              0,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_if5678)");

    status = CECL_WRITE_BUFFER(commandQueue,
                                  d_if5678,
                                  1,
                                  0,
                                  sizeof(cl_double4) * temp,
                                  h_if5678,
                                  0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if5678)");

    d_of0 = CECL_BUFFER(context,
                           CL_MEM_READ_WRITE,
                           sizeof(cl_double) * temp,
                           0,
                           &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_of0)");

    d_of1234 = CECL_BUFFER(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_double4) * temp,
                              0,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_of1234)");

    d_of5678 = CECL_BUFFER(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_double4) * temp,
                              0,
                              &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (d_of5678)");

    status = clEnqueueCopyBuffer(commandQueue,
                                 d_if0,
                                 d_of0,
                                 0, 0, sizeof(cl_double) * temp,
                                 0, 0, 0);
    CHECK_OPENCL_ERROR(status, "clEnqueueCopyBuffer failed. (d_if0->d_of0)");

    status = clEnqueueCopyBuffer(commandQueue,
                                 d_if1234,
                                 d_of1234,
                                 0, 0, sizeof(cl_double4) * temp,
                                 0, 0, 0);
    CHECK_OPENCL_ERROR(status, "clEnqueueCopyBuffer failed. (d_if1234->d_of1234)");

    status = clEnqueueCopyBuffer(commandQueue,
                                 d_if5678,
                                 d_of5678,
                                 0, 0, sizeof(cl_double4) * temp,
                                 0, 0, 0);
    CHECK_OPENCL_ERROR(status, "clEnqueueCopyBuffer failed. (d_if5678->d_of5678)");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    //Constant arrays
    type = CECL_BUFFER(context,
                          inMemFlags,
                          sizeof(cl_bool) * temp,
                          0,
                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (type)");

    weight = CECL_BUFFER(context,
                            CL_MEM_READ_ONLY,
                            sizeof(cl_double) * 9,
                            0,
                            &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (weight)");

    status = CECL_WRITE_BUFFER(commandQueue,
                                  weight,
                                  1, 0, sizeof(cl_double) * 9,
                                  w,
                                  0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (weight)");

    velocity = CECL_BUFFER(context,
                              CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(cl_double2) * temp,
                              0, &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (velocity)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("FluidSimulation2D_Kernels.cl");
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
    CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    kernel = CECL_KERNEL(
                 program,
                 "lbm",
                 &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}


int
FluidSimulation2D::setupCLKernels()
{
    cl_int status;

    // Set appropriate arguments to the kernel

    // initialize direction buffer
    for(int i = 0; i < 8; i++)
    {
        dirX.s[i] = e[i + 1][0];
        dirY.s[i] = e[i + 1][1];
    }

    // Set kernel arguments
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), &d_if0);
    status |= CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_mem), &d_of0);
    status |= CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), &d_if1234);
    status |= CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), &d_of1234);
    status |= CECL_SET_KERNEL_ARG(kernel, 4, sizeof(cl_mem), &d_if5678);
    status |= CECL_SET_KERNEL_ARG(kernel, 5, sizeof(cl_mem), &d_of5678);
    status |= CECL_SET_KERNEL_ARG(kernel, 6, sizeof(cl_mem), &type);
    status |= CECL_SET_KERNEL_ARG(kernel, 7, sizeof(cl_double8), &dirX);
    status |= CECL_SET_KERNEL_ARG(kernel, 8, sizeof(cl_double8), &dirY);
    status |= CECL_SET_KERNEL_ARG(kernel, 9, sizeof(cl_mem), &weight);
    status |= CECL_SET_KERNEL_ARG(kernel, 10, sizeof(cl_double), &omega);

    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARGs failed.");

    status =  kernelInfo.setKernelWorkGroupInfo(kernel,
              devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "setKErnelWorkGroupInfo() failed");

    if(kernelInfo.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local memory on device" << std::endl;
        return SDK_FAILURE;
    }

    if(groupSize > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }
        groupSize = kernelInfo.kernelWorkGroupSize;
    }
    return SDK_SUCCESS;
}

int
FluidSimulation2D::runCLKernels()
{
    cl_int status;
    static int i = 1;
    size_t temp = dims[0] * dims[1];

    //Enqueue write data to device

    // Write the cell type data each frame
    cl_event typeEvt;
    status = CECL_WRITE_BUFFER(commandQueue,
                                  type,
                                  CL_FALSE,
                                  0,
                                  sizeof(cl_bool) * temp,
                                  h_type,
                                  0, 0, &typeEvt);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (h_type)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&typeEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(typeEvt) Failed");

    cl_event inWriteEvt1;
    cl_event inWriteEvt2;
    cl_event inWriteEvt3;
    // If odd frame (starts from odd frame)
    // Then inputs : d_if0, d_if1234, d_if5678
    // Outputs : d_of0, f_of1234, d_of5678
    // Else they are swapped
    if(i % 2)
    {
        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_if0,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double) * temp,
                                      h_if0,
                                      0, 0, &inWriteEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if0)");

        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_if1234,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double4) * temp,
                                      h_if1234,
                                      0, 0, &inWriteEvt2);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if1234)");

        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_if5678,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double4) * temp,
                                      h_if5678,
                                      0, 0, &inWriteEvt3);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_if5678)");
    }
    else
    {
        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_of0,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double) * temp,
                                      h_if0,
                                      0, 0, &inWriteEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_of0)");

        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_of1234,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double4) * temp,
                                      h_if1234,
                                      0, 0, &inWriteEvt2);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_of1234)");

        status = CECL_WRITE_BUFFER(commandQueue,
                                      d_of5678,
                                      CL_FALSE,
                                      0,
                                      sizeof(cl_double4) * temp,
                                      h_if5678,
                                      0, 0, &inWriteEvt3);
        CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. (d_of5678)");
    }

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inWriteEvt1);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inWriteEvt1) Failed");

    status = waitForEventAndRelease(&inWriteEvt2);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inWriteEvt2) Failed");

    status = waitForEventAndRelease(&inWriteEvt3);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(inWriteEvt3) Failed");

    // Set kernel arguments
    status = CECL_SET_KERNEL_ARG(kernel, 0, sizeof(cl_mem), &d_if0);
    status |= CECL_SET_KERNEL_ARG(kernel, 1, sizeof(cl_mem), &d_of0);
    status |= CECL_SET_KERNEL_ARG(kernel, 2, sizeof(cl_mem), &d_if1234);
    status |= CECL_SET_KERNEL_ARG(kernel, 3, sizeof(cl_mem), &d_of1234);
    status |= CECL_SET_KERNEL_ARG(kernel, 4, sizeof(cl_mem), &d_if5678);
    status |= CECL_SET_KERNEL_ARG(kernel, 5, sizeof(cl_mem), &d_of5678);
    status |= CECL_SET_KERNEL_ARG(kernel, 6, sizeof(cl_mem), &type);
    status |= CECL_SET_KERNEL_ARG(kernel, 7, sizeof(cl_double8), &dirX);
    status |= CECL_SET_KERNEL_ARG(kernel, 8, sizeof(cl_double8), &dirY);
    status |= CECL_SET_KERNEL_ARG(kernel, 9, sizeof(cl_mem), &weight);
    status |= CECL_SET_KERNEL_ARG(kernel, 10, sizeof(cl_double), &omega);
    status |= CECL_SET_KERNEL_ARG(kernel, 11, sizeof(cl_mem), &velocity);

    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARGs failed.)");

    size_t localThreads[2] = {groupSize, 1};
    size_t globalThreads[2] = {dims[0], dims[1]};

    status = CECL_ND_RANGE_KERNEL(commandQueue,
                                    kernel,
                                    2,
                                    0,
                                    globalThreads,
                                    localThreads,
                                    0, 0, 0);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.)");

    status = clFinish(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFinish failed.");

    cl_event velocityEvt;
    status = CECL_READ_BUFFER(commandQueue,
                                 velocity,
                                 CL_FALSE,
                                 0,
                                 sizeof(cl_double2) * temp,
                                 u,
                                 0, 0, &velocityEvt);
    CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(velocity)");

    cl_event outReadEvt1;
    cl_event outReadEvt2;
    cl_event outReadEvt3;
    //Read back the data into host buffer
    if(i % 2)
    {
        status = CECL_READ_BUFFER(commandQueue,
                                     d_of0,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double) * temp,
                                     h_of0,
                                     0, 0, &outReadEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_of0)");

        status = CECL_READ_BUFFER(commandQueue,
                                     d_of1234,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double4) * temp,
                                     h_of1234,
                                     0, 0, &outReadEvt2);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_of1234)");

        status = CECL_READ_BUFFER(commandQueue,
                                     d_of5678,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double4) * temp,
                                     h_of5678,
                                     0, 0, &outReadEvt3);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_of5678)");
    }
    else
    {
        status = CECL_READ_BUFFER(commandQueue,
                                     d_if0,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double) * temp,
                                     h_of0,
                                     0, 0, &outReadEvt1);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_if0)");

        status = CECL_READ_BUFFER(commandQueue,
                                     d_if1234,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double4) * temp,
                                     h_of1234,
                                     0, 0, &outReadEvt2);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_if1234)");

        status = CECL_READ_BUFFER(commandQueue,
                                     d_if5678,
                                     CL_FALSE,
                                     0,
                                     sizeof(cl_double4) * temp,
                                     h_of5678,
                                     0, 0, &outReadEvt3);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER failed.(d_if5678)");
    }

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&velocityEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(velocityEvt) Failed");

    status = waitForEventAndRelease(&outReadEvt1);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outReadEvt1) Failed");

    status = waitForEventAndRelease(&outReadEvt2);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outReadEvt2) Failed");

    status = waitForEventAndRelease(&outReadEvt3);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(outReadEvt3) Failed");

    // Copy from host output to the next input
    memcpy(h_if0, h_of0, sizeof(cl_double) * temp);
    memcpy(h_if1234, h_of1234, sizeof(cl_double4) * temp);
    memcpy(h_if5678, h_of5678, sizeof(cl_double4) * temp);

    cl_mem temp0, temp1234, temp5678;

    //swap input and output buffers
    temp0 = d_of0;
    temp1234 = d_of1234;
    temp5678 = d_of5678;

    d_of0 = d_if0;
    d_of1234 = d_if1234;
    d_of5678 = d_if5678;

    d_if0 = temp0;
    d_if1234 = temp1234;
    d_if5678 = temp5678;

    i++;

    return SDK_SUCCESS;
}

void FluidSimulation2D::collide(int x, int y)
{
    int pos = x + y * dims[0];

    // Boundary
    if (h_type[pos] == 1)
    {
        // Swap the frequency distributions
        cl_double4 temp1234, temp5678;

        v_ef0[pos] = h_if0[pos];

        temp1234.s[0] = h_if1234[pos * 4 + 0];
        temp1234.s[1] = h_if1234[pos * 4 + 1];
        temp1234.s[2] = h_if1234[pos * 4 + 2];
        temp1234.s[3] = h_if1234[pos * 4 + 3];

        v_ef1234[pos * 4 + 0] = temp1234.s[2];
        v_ef1234[pos * 4 + 1] = temp1234.s[3];
        v_ef1234[pos * 4 + 2] = temp1234.s[0];
        v_ef1234[pos * 4 + 3] = temp1234.s[1];

        temp5678.s[0] = h_if5678[pos * 4 + 0];
        temp5678.s[1] = h_if5678[pos * 4 + 1];
        temp5678.s[2] = h_if5678[pos * 4 + 2];
        temp5678.s[3] = h_if5678[pos * 4 + 3];

        v_ef5678[pos * 4 + 0] = temp5678.s[2];
        v_ef5678[pos * 4 + 1] = temp5678.s[3];
        v_ef5678[pos * 4 + 2] = temp5678.s[0];
        v_ef5678[pos * 4 + 3] = temp5678.s[1];

        rho[pos] = 0;
        u[pos].s[0] = u[pos].s[1] = 0;
    }
    //Fluid
    else
    {
        cl_double2 vel;

        // Calculate density from input distribution
        cl_double den = h_if0[pos] + h_if1234[pos * 4 + 0] +  h_if1234[pos * 4 + 1] +
                        h_if1234[pos * 4 + 2] + h_if1234[pos * 4 + 3] +
                        h_if5678[pos * 4 + 0] + h_if5678[pos * 4 + 1] + h_if5678[pos * 4 + 2] +
                        h_if5678[pos * 4 + 3];

        // Calculate velocity vector in x-direction
        vel.s[0] = h_if0[pos] * e[0][0] + h_if1234[pos * 4 + 0] * e[1][0] +
                   h_if1234[pos * 4 + 1] * e[2][0]
                   + h_if1234[pos * 4 + 2] * e[3][0] + h_if1234[pos * 4 + 3] * e[4][0] +
                   h_if5678[pos * 4 + 0] * e[5][0]
                   + h_if5678[pos * 4 + 1] * e[6][0] + h_if5678[pos * 4 + 2] * e[7][0] +
                   h_if5678[pos * 4 + 3] * e[8][0];

        // Calculate velocity vector in y-direction
        vel.s[1] = h_if0[pos] * e[0][1] + h_if1234[pos * 4 + 0] * e[1][1] +
                   h_if1234[pos * 4 + 1] * e[2][1]
                   + h_if1234[pos * 4 + 2] * e[3][1] + h_if1234[pos * 4 + 3] * e[4][1] +
                   h_if5678[pos * 4 + 0] * e[5][1]
                   + h_if5678[pos * 4 + 1] * e[6][1] + h_if5678[pos * 4 + 2] * e[7][1] +
                   h_if5678[pos * 4 + 3] * e[8][1];

        vel.s[0] /= den;
        vel.s[0] /= den;

        // Calculate Equivalent distribution
        v_ef0[pos] = computefEq(w[0], e[0], den, vel);
        v_ef1234[pos * 4 + 0] = computefEq(w[1], e[1], den, vel);
        v_ef1234[pos * 4 + 1] = computefEq(w[2], e[2], den, vel);
        v_ef1234[pos * 4 + 2] = computefEq(w[3], e[3], den, vel);
        v_ef1234[pos * 4 + 3] = computefEq(w[4], e[4], den, vel);

        v_ef5678[pos * 4 + 0] = computefEq(w[5], e[5], den, vel);
        v_ef5678[pos * 4 + 1] = computefEq(w[6], e[6], den, vel);
        v_ef5678[pos * 4 + 2] = computefEq(w[7], e[7], den, vel);
        v_ef5678[pos * 4 + 3] = computefEq(w[8], e[8], den, vel);

        v_ef0[pos] = (1 - omega) * h_if0[pos] + omega * v_ef0[pos];
        v_ef1234[pos * 4 + 0] = (1 - omega) * h_if1234[pos * 4 + 0] + omega *
                                v_ef1234[pos * 4 + 0];
        v_ef1234[pos * 4 + 1] = (1 - omega) * h_if1234[pos * 4 + 1] + omega *
                                v_ef1234[pos * 4 + 1];
        v_ef1234[pos * 4 + 2] = (1 - omega) * h_if1234[pos * 4 + 2] + omega *
                                v_ef1234[pos * 4 + 2];
        v_ef1234[pos * 4 + 3] = (1 - omega) * h_if1234[pos * 4 + 3] + omega *
                                v_ef1234[pos * 4 + 3];
        v_ef5678[pos * 4 + 0] = (1 - omega) * h_if5678[pos * 4 + 0] + omega *
                                v_ef5678[pos * 4 + 0];
        v_ef5678[pos * 4 + 1] = (1 - omega) * h_if5678[pos * 4 + 1] + omega *
                                v_ef5678[pos * 4 + 1];
        v_ef5678[pos * 4 + 2] = (1 - omega) * h_if5678[pos * 4 + 2] + omega *
                                v_ef5678[pos * 4 + 2];
        v_ef5678[pos * 4 + 3] = (1 - omega) * h_if5678[pos * 4 + 3] + omega *
                                v_ef5678[pos * 4 + 3];
    }
}

void FluidSimulation2D::streamToNeighbors(int x, int y)
{
    if (x == 0 || x == dims[0]-1 || y == 0 || y == dims[1]-1)
    {
        return;
    }

    for (int k = 0; k < 9; k++)
    {
        int nx = x + (int)e[k][0];
        int ny = y + (int)e[k][1];

        int pos = nx + dims[0] * ny;
        switch(k)
        {
        case 0:
            v_of0[pos] = v_ef0[pos];
            break;
        case 1:
            v_of1234[pos * 4 + 0] = v_ef1234[pos * 4 + 0];
            break;
        case 2:
            v_of1234[pos * 4 + 1] = v_ef1234[pos * 4 + 1];
            break;
        case 3:
            v_of1234[pos * 4 + 2] = v_ef1234[pos * 4 + 2];
            break;
        case 4:
            v_of1234[pos * 4 + 3] = v_ef1234[pos * 4 + 3];
            break;
        case 5:
            v_of5678[pos * 4 + 0] = v_ef5678[pos * 4 + 0];
            break;
        case 6:
            v_of5678[pos * 4 + 1] = v_ef5678[pos * 4 + 1];
            break;
        case 7:
            v_of5678[pos * 4 + 2] = v_ef5678[pos * 4 + 2];
            break;
        case 8:
            v_of5678[pos * 4 + 3] = v_ef5678[pos * 4 + 3];
            break;

        }
    }
}


/*
* lbm simulation on cpu
*/
void
FluidSimulation2D::CPUReference()
{
    // Copy from host output to the next input
    memcpy(v_of0, h_if0, sizeof(cl_double) * dims[0] * dims[1]);
    memcpy(v_of1234, h_if1234, sizeof(cl_double4) * dims[0] * dims[1]);
    memcpy(v_of5678, h_if5678, sizeof(cl_double4) * dims[0] * dims[1]);

    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            collide(x, y);
        }
    }
    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            streamToNeighbors(x, y);
        }
    }

    int flag0 = 0;
    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            int pos = x + y * dims[0];
            if(h_of0[pos] - v_of0[pos] > 1e-5)
            {
                flag0 = 1;
                break;
            }
        }
    }

    int flag1234 = 0;
    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            int pos = x + y * dims[0];
            if(h_of1234[pos] - v_of1234[pos] > 1e-2)
            {
                std::cout << pos << "=" << h_of1234[pos] - v_of1234[pos] << std::endl;
                flag1234 = 1;
                break;
            }
        }
    }

    int flag5678 = 0;
    for (int y = 0; y < dims[1]; y++)
    {
        for (int x = 0; x < dims[0]; x++)
        {
            int pos = x + y * dims[0];
            if(h_of5678[pos] - v_of5678[pos] > 1e-2)
            {
                std::cout << pos << "=" << h_of1234[pos] - v_of1234[pos] << std::endl;
                flag5678 = 1;
                break;
            }
        }
    }

    if(!flag0 && !flag1234 && !flag5678)
    {
        verifyFlag = 1;
    }

}

int
FluidSimulation2D::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option *num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations,
                     "Error. Failed to allocate memory (num_iterations)\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    return SDK_SUCCESS;
}

int
FluidSimulation2D::setup()
{
    if(setupFluidSimulation2D() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);

    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    display = !sampleArgs->quiet && !sampleArgs->verify;

    return SDK_SUCCESS;
}

/**
* @brief Initialize GL
*/
void
GLInit()
{
    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texnum);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    unsigned char bitmap[LBWIDTH * LBHEIGHT * 4]; // rgba unsigned bytes

    double m, r, g, b;

    for(int y = 0; y < LBHEIGHT; y++)
    {
        for(int x = 0; x < LBWIDTH; x++)
        {
            if(me->isBoundary(x , y))
            {
                r = g = b = 0;
            }
            else
            {
                cl_double2 vel = me->getVelocity(x, y);
                m = sqrt(vel.s[0] * vel.s[0] + vel.s[1] * vel.s[1]);
                bluewhite.GetColor(m * 20, r, g, b);
            }

            bitmap[(x + y * LBWIDTH) * 4 + 0] = (unsigned char)(r * 255);
            bitmap[(x + y * LBWIDTH) * 4 + 1] = (unsigned char)(g * 255);
            bitmap[(x + y * LBWIDTH) * 4 + 2] = (unsigned char)(b * 255);
            bitmap[(x + y * LBWIDTH) * 4 + 3] = 255;
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, LBWIDTH, LBHEIGHT, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, bitmap);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glEnable(GL_TEXTURE_2D);

    // setup 2d pixel plotting camera
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, (GLdouble) winwidth, 0.0f, (GLdouble) winheight, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, winwidth, winheight);

    glBegin(GL_QUADS);

    glColor3f(1.0, 1.0, 1.0);

    glTexCoord2f(0.0, 0.0);
    glVertex2i(0, 0);

    glTexCoord2f(1.0, 0.0);
    glVertex2i(winwidth, 0);

    glTexCoord2f(1.0, 1.0);
    glVertex2i(winwidth, winheight);

    glTexCoord2f(0.0, 1.0);
    glVertex2i(0, winheight);

    glEnd();

    glDisable(GL_TEXTURE_2D);

    glFlush();
    glutSwapBuffers();
}

void update()
{
    me->runCLKernels();

    // redraw
    glutPostRedisplay();

    frames++;

    te = glutGet(GLUT_ELAPSED_TIME);

    // every second approximately
    if (te - t0 >= 1000)
    {
        char title[80];
        sprintf(title, "Lattice Boltzmann demo    %.1f fps", (1000.0*frames/(te-t0)));
        glutSetWindowTitle(title);

        frames = 0;
        t0 = te;
    }
}

void mouse(int button, int state, int x, int y)
{
    double u[2] = {0, 0};

    drawBoundary = false;
    addVelocity = false;

    x = (double)x * (double)((double)LBWIDTH / (double)winwidth);
    y = (double)y * (double)((double)LBHEIGHT / (double)winheight);

    if (state == GLUT_DOWN)
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            if (me->isFluid(x, y) && x >= 0 && x < LBWIDTH && y >= 0 && y < LBHEIGHT)
            {
                addVelocity = true;
                oldx = x;
                oldy = y;
            }
        }

        if (button == GLUT_RIGHT_BUTTON)
        {
            drawBoundary = true;

            if (x >= 0 && x < LBWIDTH && y >= 0 && y < LBHEIGHT)
            {
                me->setSite(x, LBHEIGHT - 1-y, 1, u);
            }
        }
    }
}

void motion(int x, int y)
{
    double m, u[2] = {0, 0};

    x = (double)x * (double)((double)LBWIDTH / (double)winwidth);
    y = (double)y * (double)((double)LBHEIGHT / (double)winheight);

    if (drawBoundary && (x >= 0 && x < LBWIDTH && y >= 0 && y < LBHEIGHT))
    {
        me->setSite(x, LBHEIGHT - 1 - y, 1, u);
    }

    if (addVelocity && (x >= 0 && x < LBWIDTH && y >= 0 && y < LBHEIGHT))
    {
        if (me->isFluid(x, y))
        {
            u[0] = (x - oldx);
            u[1] = (oldy - y);

            m = sqrt(u[0] * u[0] + u[1] * u[1]);
            u[0] /= (1 + 2 * m);
            u[1] /= (1 + 2 * m);

            me->setSite(x, LBHEIGHT - 1 - y, 0, u);
        }
    }
}

void reshape(int w, int h)
{
    winwidth = w;
    winheight = h;
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
    {
        exit(0);
    }

    if (key == 'r')
    {
        me->reset();
    }
    else if (key == 'v')
    {
        drawVelocities *= -1;
    }
    else if (key == 'p')
    {
        drawParticles *= -1;
    }
}


int
FluidSimulation2D::run()
{
    if(!reqdExtSupport)
    {
        return SDK_SUCCESS;
    }

    // Arguments are set and execution call is enqueued on command buffer
    if(setupCLKernels() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(display == 0)
    {
        // Warm up
        for(int i = 0; i < 2 && iterations != 1; ++i)
        {
            runCLKernels();
        }

        int timer = sampleTimer->createTimer();
        sampleTimer->resetTimer(timer);
        sampleTimer->startTimer(timer);

        for(int i = 0; i < iterations; ++i)
        {
            runCLKernels();
        }

        sampleTimer->stopTimer(timer);
        // Compute kernel time
        kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;
    }

    return SDK_SUCCESS;
}

int
FluidSimulation2D::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
        * it overwrites the input array with the output
        */
        CPUReference();

        // compare the results and see if they match
        if(verifyFlag)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
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
FluidSimulation2D::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };

        std::string stats[4];
        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(dims[0], std::dec);
        stats[1] = toString(dims[1], std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int
FluidSimulation2D::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(d_if0);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(d_if0)");

    status = clReleaseMemObject(d_of0);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(d_of0)");

    status = clReleaseMemObject(d_if1234);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(d_if1234)");

    status = clReleaseMemObject(d_of1234);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(d_of1234)");

    status = clReleaseMemObject(d_if5678);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed. (d_if5678)");

    status = clReleaseMemObject(d_of5678);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(d_of5678)");

    status = clReleaseMemObject(type);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(type)");

    status = clReleaseMemObject(weight);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(weight)");

    status = clReleaseMemObject(velocity);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(velocity)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed. (context)");

    return SDK_SUCCESS;
}

FluidSimulation2D::~FluidSimulation2D()
{
    /* release program resources */
    FREE(h_if0);
    FREE(h_if1234);
    FREE(h_if1234);
    FREE(h_of0);
    FREE(h_of1234);
    FREE(h_of1234);
    FREE(v_ef0);
    FREE(v_ef1234);
    FREE(v_ef5678);
    FREE(v_of0);
    FREE(v_of1234);
    FREE(v_of1234);
    FREE(h_type);
    FREE(h_weight);
    FREE(devices);
}


int
main(int argc, char * argv[])
{
    FluidSimulation2D clFluidSim;
    me = &clFluidSim;

    // create color scale
    bluewhite.AddPoint(0.0, 0, 0, 0);
    bluewhite.AddPoint(0.2, 0, 0, 1);
    bluewhite.AddPoint(0.4, 0, 1, 1);
    bluewhite.AddPoint(0.8, 0, 1, 0);
    bluewhite.AddPoint(1.6, 1, 1, 0);
    bluewhite.AddPoint(3.2, 1, 0, 0);

    // Initialize
    if(clFluidSim.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clFluidSim.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clFluidSim.sampleArgs->isDumpBinaryEnabled())
    {
        return clFluidSim.genBinaryImage();
    }
    else
    {
        // Setup
        int status = clFluidSim.setup();
        if(status != SDK_SUCCESS)
        {
            if(status == SDK_EXPECTED_FAILURE)
            {
                return SDK_SUCCESS;
            }
            else
            {
                return SDK_FAILURE;
            }
        }

        // Run
        if(clFluidSim.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // VerifyResults
        if(clFluidSim.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clFluidSim.printStats();

        if(display)
        {
            // Run in  graphical window if requested
            glutInit(&argc, argv);
            glutInitWindowPosition(100,10);
            glutInitWindowSize(400,400);
            glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
            glutCreateWindow("LBM simulation");
            GLInit();
            glutDisplayFunc(render);
            glutIdleFunc(update);
            glutMouseFunc(mouse);
            glutMotionFunc(motion);
            glutReshapeFunc(reshape);
            glutKeyboardFunc(keyboard);

            std::cout << "Use Left-Mouse button to move the fluid\n";
            std::cout << "Use Right-Mouse button to draw boundary\n";
            std::cout << "Press r to reset the simulation\n";

            glutMainLoop();
        }

        // Cleanup
        if(clFluidSim.cleanup()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}
