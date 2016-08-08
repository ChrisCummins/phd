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
 
 #pragma once

#include<oclUtils.h>

// Class to get/hold OpenCL device information and calculate device load balancing information based upon estimated perf
class DeviceManager
{
public:
    DeviceManager(cl_platform_id cpPlatform, cl_uint* uiNumAllDevs, void (*pCleanup)(int));
    ~DeviceManager(void);

    int GetDevLoadProportions(bool bNV);

    cl_uint* uiUsefulDevs;      // Indexed list of devices worth using
    cl_uint uiUsefulDevCt;      // Number of devices to be used
    float* fLoadProportions;    // Proportions to divide up work among GPU's used
    cl_device_id* cdDevices;    // OpenCL device list

private:
    cl_uint uiDevCount;         // total # of devices available to the platform
    float* fDevPerfs;           // individual device perfs
};
