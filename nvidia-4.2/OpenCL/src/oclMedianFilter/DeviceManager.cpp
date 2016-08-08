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
 
 #include "DeviceManager.h"

DeviceManager::DeviceManager(cl_platform_id cpPlatform, cl_uint* uiNumAllDevs, void (*pCleanup)(int))
{
    // Get the number of GPU devices available to the platform
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, uiNumAllDevs);
    uiDevCount = *uiNumAllDevs;

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiDevCount, cdDevices, NULL);

    // Allocations for perfs, loads and useful devices
    fLoadProportions = new float[uiDevCount];    
    uiUsefulDevs = new cl_uint[uiDevCount];
    fDevPerfs = new float[uiDevCount]; 
    uiUsefulDevCt = 0;
}

DeviceManager::~DeviceManager(void)
{
    delete [] cdDevices;
    delete [] fLoadProportions;
    delete [] uiUsefulDevs;
    delete [] fDevPerfs;
}

// Helper to determine balanced load proportions for multiGPU config using perf estimation
//*****************************************************************************
int DeviceManager::GetDevLoadProportions(bool bNV)
{
    shrLog("  \nDetermining Device Load Proportions based upon Peformance Estimate...\n");
    int iBestDevice = 0;                     // var to keep track of device with best estimated perf
    float fBestPerf = -1.0e10;               // var to keep track of best estimated perf
    float fTotalPerf = 0.0f;                 // accumulator for total perf 
    const float fOverhead = 15000.0f;        // runtime cost of using an additional device   

    // Estimate dev perf and total perf for all devs available to the platform
    for (cl_uint i = 0; i < uiDevCount; i++)
    {
        cl_uint uiComputeUnits, uiCores, uiClockFreq;
        cl_int ciErrNum;

        // CL_DEVICE_MAX_COMPUTE_UNITS 
        ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiComputeUnits), &uiComputeUnits, NULL);
        if (CL_SUCCESS != ciErrNum)
        {
            return ciErrNum;
        }

        // # of CUDA Cores if NV Platform
        uiCores = uiComputeUnits;
        if (bNV)
        {
            int iDevCapMajor = oclGetDevCap(cdDevices[i])/10;
            int iDevCapMinor = oclGetDevCap(cdDevices[i]) % 10;
            uiCores *= ConvertSMVer2Cores(iDevCapMajor, iDevCapMinor); 
        }

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uiClockFreq), &uiClockFreq, NULL);
        if (CL_SUCCESS != ciErrNum)
        {
            return ciErrNum;
        }

        // Get individual device perf and accumulate
        // Note: To achieve better load proportions for each GPU an overhead penalty is subtracted from the computed device perf 
        // If negative perf results, this means the dev will be a drag... don't use it, unless it's the only one
        fDevPerfs[i] = (float)(uiCores * uiClockFreq) - fOverhead;
        shrLog("    Device %d perf:\t(%u cores) * (%u clock freq) - %.0f\t= %.0f", i, uiCores, uiClockFreq, fOverhead, fDevPerfs[i]);
        if (fDevPerfs[i] > 0.0f) 
        {
            shrLog("\t(Perf > Overhead)\n");
            fTotalPerf += fDevPerfs[i];
            uiUsefulDevs[uiUsefulDevCt++] = i;
        }
        else
        {
            shrLog("\t(Perf < Overhead)\n");
        }

        // trap the best perf and perf dev
        if (fDevPerfs[i] > fBestPerf) 
        {
            fBestPerf = fDevPerfs[i];
            iBestDevice = i;
        }
    }

    // Log best device found
    shrLog("\n    Best Perf Device (or tied for best) = Device %d\n", iBestDevice);

    // Handle the case when there are no fast (useful) devices 
    if (uiUsefulDevCt == 0)
    {
        fLoadProportions[0] = 1.0f;
        uiUsefulDevs[0] = iBestDevice;
    }
    else 
    {
        // Compute/assign load proportions
        for (cl_uint i = 0; i < uiUsefulDevCt; i++)
        {
            fLoadProportions[i] = fDevPerfs[uiUsefulDevs[i]]/fTotalPerf;
        }
    }
    return CL_SUCCESS;
}
