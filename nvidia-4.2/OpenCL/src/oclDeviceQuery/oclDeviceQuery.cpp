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

/* This sample queries and logs the properties of the CUDA devices present in the system. */

// Standard utilities and common systems includes
#include <oclUtils.h>
#include <shrQATest.h>

#include <memory>
#include <iostream>
#include <cassert>

#ifdef UNIX
#include <sstream>
#include <fstream>
#endif

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    shrQAStart(argc, argv);

    // start logs
    shrSetLogFileName ("oclDeviceQuery.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    bool bPassed = true;
    std::string sProfileString = "oclDeviceQuery, Platform Name = ";

    // Get OpenCL platform ID for NVIDIA if avaiable, otherwise default
    shrLog("OpenCL SW Info:\n\n");
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL; 
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    } 
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", Platform Version = ";

    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_VERSION: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    } 
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", SDK Revision = ";

    // Log OpenCL SDK Revision # 
    shrLog(" OpenCL SDK Revision: \t%s\n\n\n", OCL_SDKREVISION);
    sProfileString += OCL_SDKREVISION;
    sProfileString += ", NumDevs = ";

    // Get and log OpenCL device info 
    cl_uint ciDeviceCount;
    cl_device_id *devices;
    shrLog("OpenCL Device Info:\n\n");
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_CPU, 0, NULL, &ciDeviceCount);

    // check for 0 devices found or errors... 
    if (ciDeviceCount == 0)
    {
        shrLog(" No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
        bPassed = false;
        sProfileString += "0";
    } 
    else if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    else
    {
        // Get and log the OpenCL device ID's
        shrLog(" %u devices found supporting OpenCL:\n\n", ciDeviceCount);
        char cTemp[2];
        #ifdef WIN32
            sprintf_s(cTemp, 2*sizeof(char), "%u", ciDeviceCount);
        #else
            sprintf(cTemp, "%u", ciDeviceCount);
        #endif
        sProfileString += cTemp;
        if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL)
        {
           shrLog(" Failed to allocate memory for devices !!!\n\n");
           bPassed = false;
        }
        ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_CPU, ciDeviceCount, devices, &ciDeviceCount);
        if (ciErrNum == CL_SUCCESS)
        {
            //Create a context for the devices
            cl_context cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog("Error %i in clCreateContext call !!!\n\n", ciErrNum);
                bPassed = false;
            }
            else 
            {
                // show info for each device in the context
                for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
                {  
                    shrLog(" ---------------------------------\n");
                    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
                    shrLog(" Device %s\n", cBuffer);
                    shrLog(" ---------------------------------\n");
                    oclPrintDevInfo(LOGBOTH, devices[i]);
                    sProfileString += ", Device = ";
                    sProfileString += cBuffer;
                }

                // Determine and show image format support 
                cl_uint uiNumSupportedFormats = 0;

                // 2D
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, 
                                           CL_MEM_OBJECT_IMAGE2D,   
                                           NULL, NULL, &uiNumSupportedFormats);
                cl_image_format* ImageFormats = new cl_image_format[uiNumSupportedFormats];
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, 
                                           CL_MEM_OBJECT_IMAGE2D,   
                                           uiNumSupportedFormats, ImageFormats, NULL);
                shrLog("  ---------------------------------\n");
                shrLog("  2D Image Formats Supported (%u)\n", uiNumSupportedFormats); 
                shrLog("  ---------------------------------\n");
                shrLog("  %-6s%-16s%-22s\n\n", "#", "Channel Order", "Channel Type"); 
                for(unsigned int i = 0; i < uiNumSupportedFormats; i++) 
                {  
                    shrLog("  %-6u%-16s%-22s\n", (i + 1),
                            oclImageFormatString(ImageFormats[i].image_channel_order), 
                            oclImageFormatString(ImageFormats[i].image_channel_data_type));
                }
                shrLog("\n"); 
                delete [] ImageFormats;

                // 3D
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, 
                                           CL_MEM_OBJECT_IMAGE3D,   
                                           NULL, NULL, &uiNumSupportedFormats);
                ImageFormats = new cl_image_format[uiNumSupportedFormats];
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, 
                                           CL_MEM_OBJECT_IMAGE3D,   
                                           uiNumSupportedFormats, ImageFormats, NULL);
                shrLog("  ---------------------------------\n");
                shrLog("  3D Image Formats Supported (%u)\n", uiNumSupportedFormats); 
                shrLog("  ---------------------------------\n");
                shrLog("  %-6s%-16s%-22s\n\n", "#", "Channel Order", "Channel Type"); 
                for(unsigned int i = 0; i < uiNumSupportedFormats; i++) 
                {  
                    shrLog("  %-6u%-16s%-22s\n", (i + 1),
                            oclImageFormatString(ImageFormats[i].image_channel_order), 
                            oclImageFormatString(ImageFormats[i].image_channel_data_type));
                }
                shrLog("\n"); 
                delete [] ImageFormats;
            }
        }
        else
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            bPassed = false;
        }
    }

    // masterlog info
    sProfileString += "\n";
    shrLogEx(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // Log system info(for convenience:  not specific to OpenCL) 
    shrLog("\nSystem Info: \n\n");
    #ifdef _WIN32
        SYSTEM_INFO stProcInfo;         // processor info struct
        OSVERSIONINFO stOSVerInfo;      // Win OS info struct
        SYSTEMTIME stLocalDateTime;     // local date / time struct 

        // processor
        SecureZeroMemory(&stProcInfo, sizeof(SYSTEM_INFO));
        GetSystemInfo(&stProcInfo);

        // OS
        SecureZeroMemory(&stOSVerInfo, sizeof(OSVERSIONINFO));
        stOSVerInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
        GetVersionEx(&stOSVerInfo);

        // date and time
        GetLocalTime(&stLocalDateTime); 

        // write time and date to logs
        shrLog(" Local Time/Date = %i:%i:%i, %i/%i/%i\n", 
            stLocalDateTime.wHour, stLocalDateTime.wMinute, stLocalDateTime.wSecond, 
            stLocalDateTime.wMonth, stLocalDateTime.wDay, stLocalDateTime.wYear); 

        // write proc and OS info to logs
        shrLog(" CPU Arch: %i\n CPU Level: %i\n # of CPU processors: %u\n Windows Build: %u\n Windows Ver: %u.%u %s\n\n\n", 
            stProcInfo.wProcessorArchitecture, stProcInfo.wProcessorLevel, stProcInfo.dwNumberOfProcessors, 
            stOSVerInfo.dwBuildNumber, stOSVerInfo.dwMajorVersion, stOSVerInfo.dwMinorVersion, 
            (stOSVerInfo.dwMajorVersion >= 6)? "(Windows Vista / Windows 7)" : "");
    #endif

    #ifdef MAC
	#else
    #ifdef UNIX
        char timestr[255];
        time_t now = time(NULL);
        struct tm  *ts;

        ts = localtime(&now);
        
        strftime(timestr, 255, " %H:%M:%S, %m/%d/%Y",ts);
        
        // write time and date to logs
        shrLog(" Local Time/Date = %s\n",  timestr); 

        // write proc and OS info to logs
        
        // parse /proc/cpuinfo
        std::ifstream cpuinfo( "/proc/cpuinfo" ); // open the file in /proc        
        std::string tmp;

        int cpu_num = 0;
        std::string cpu_name = "none";        

        do {
            cpuinfo >> tmp;
            
            if( tmp == "processor" )
                cpu_num++;
            
            if( tmp == "name" )
            {
                cpuinfo >> tmp; // skip :

                std::stringstream tmp_stream("");
                do {
                    cpuinfo >> tmp;
                    if (tmp != std::string("stepping")) {
                        tmp_stream << tmp.c_str() << " ";
                    }
                }
                while (tmp != std::string("stepping"));
                
                cpu_name = tmp_stream.str();
            }

        }
        while ( (! cpuinfo.eof()) );

        // Linux version
        std::ifstream version( "/proc/version" );
        char versionstr[255];

        version.getline(versionstr, 255);

        shrLog(" CPU Name: %s\n # of CPU processors: %u\n %s\n\n\n", 
               cpu_name.c_str(),cpu_num,versionstr);
    #endif
    #endif

    // finish
    shrQAFinishExit(argc, (const char **)argv, (bPassed ? QA_PASSED : QA_FAILED) ); 
}
