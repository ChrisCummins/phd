/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef BOLTSAMPLE_H_
#define BOLTSAMPLE_H_


/******************************************************************************
* Included header files                                                       *
******************************************************************************/
#include <bolt/BoltVersion.h>
#include <bolt/cl/bolt.h>

#include <CL/opencl.h>

#include "SDKUtil.hpp"

#define CHECK_BOLT_ERROR(actual, msg) CHECK_ERROR(actual, SDK_SUCCESS, msg)


/******************************************************************************
* namespace appsdk                                                           *
******************************************************************************/
namespace appsdk
{

/******************************************************************************
* BoltVersionStr                                                              *
* @brief struct definition that contains Bolt library version information     *
******************************************************************************/
struct BoltVersionStr
{
    int major;      /**< Bolt major release number */
    int minor;      /**< Bolt minor release number */
    int patch;      /**< Bolt build number */

    /**
    ***************************************************************************
    * @brief Constructor of BoltVersionStr. These version numbers come
    *        directly from the Bolt header files, and represent the version
    *        of header that the app is compiled against
    **************************************************************************/
    BoltVersionStr()
    {
        major = BoltVersionMajor;
        minor = BoltVersionMinor;
        patch = BoltVersionPatch;
    }
};


/******************************************************************************
* BoltSample                                                                  *
* Class implements various resources required by the test & initializes the   *
* resources used by tests                                                     *
******************************************************************************/
class BoltCommandArgs :  public SDKCmdArgsParser
{
    protected:
        BoltVersionStr boltVerStr;     /**< Bolt version string */

    public:
        bolt::cl::control
        *boltControlObj;      /**< Reference to default bolt control object. Used to change bolt beahviour */
        double totalTime;                       /**< Total Time taken by the Sample */
        std::string
        runMode;                    /**< Cmd Line Option- Bolt explicit device selection */
        int iterations;
        int samples;

    protected:
        /**
        ***************************************************************************
        * @fn initialize
        * @brief Initialize the resources used by tests
        * @return SDK_SUCCESS on success, SDK_FAILURE otherwise
        **************************************************************************/
        int initialize()
        {
            int defaultOptions = 7;
            Option *optionList = new Option[defaultOptions];
            CHECK_ALLOCATION(optionList, "Error. Failed to allocate memory (optionList)\n");
            optionList[0]._sVersion = "";
            optionList[0]._lVersion = "device";
            optionList[0]._description = "Explicit device selection for Bolt";
            std::string optionStr = "[auto|";
            optionStr.append(  "OpenCL" );
            optionStr.append("|SerialCpu");
            optionStr.append( ((enable_tbb)? "|MultiCoreCpu" : "") );
            optionStr.append("]");
            optionList[0]._usage = optionStr;
            optionList[0]._type = CA_ARG_STRING;
            optionList[0]._value = &runMode;
            optionList[1]._sVersion = "q";
            optionList[1]._lVersion = "quiet";
            optionList[1]._description = "Quiet mode. Suppress most text output.";
            optionList[1]._usage = "";
            optionList[1]._type = CA_NO_ARGUMENT;
            optionList[1]._value = &quiet;
            optionList[2]._sVersion = "e";
            optionList[2]._lVersion = "verify";
            optionList[2]._description = "Verify results against reference implementation.";
            optionList[2]._usage = "";
            optionList[2]._type = CA_NO_ARGUMENT;
            optionList[2]._value = &verify;
            optionList[3]._sVersion = "t";
            optionList[3]._lVersion = "timing";
            optionList[3]._description = "Print timing related statistics.";
            optionList[3]._usage = "";
            optionList[3]._type = CA_NO_ARGUMENT;
            optionList[3]._value = &timing;
            optionList[4]._sVersion = "v";
            optionList[4]._lVersion = "version";
            optionList[4]._description = "Bolt lib & runtime version string.";
            optionList[4]._usage = "";
            optionList[4]._type = CA_NO_ARGUMENT;
            optionList[4]._value = &version;
            optionList[5]._sVersion = "x";
            optionList[5]._lVersion = "samples";
            optionList[5]._description = "Number of sample input values.";
            optionList[5]._usage = "[value]";
            optionList[5]._type = CA_ARG_INT;
            optionList[5]._value = &samples;
            optionList[6]._sVersion = "i";
            optionList[6]._lVersion = "iterations";
            optionList[6]._description = "Number of iterations.";
            optionList[6]._usage = "[value]";
            optionList[6]._type = CA_ARG_INT;
            optionList[6]._value = &iterations;
            _numArgs = defaultOptions;
            _options = optionList;
            return SDK_SUCCESS;
        }

    public:
        bool enable_tbb;                        /**< Option passed by sample - if ENABLE_TBB is set */

        /**
        ***************************************************************************
        * @brief Constructor of BoltSample to initialize member variables
        * @param strSampleName Name of the Sample
        * @param numSamples Number of sample input values
        **************************************************************************/
        BoltCommandArgs(unsigned numSamples, bool enableTBB=false)
        {
            boltControlObj = NULL;
            samples = numSamples;
            iterations = 1;
            enable_tbb = enableTBB;
            initialize();
        }

        /**
        ***************************************************************************
        * @brief Destructor of BoltSample
        **************************************************************************/
        ~BoltCommandArgs()
        {
        }

        /**
        ***************************************************************************
        * @fn parseCommandLine
        * @brief parses the command line options given by user
        * @param argc Number of elements in cmd line input
        * @param argv Array of char* storing the CmdLine Options
        * @return SDK_SUCCESS on success, SDK_FAILURE otherwise
        **************************************************************************/
        int parseCommandLine(int argc, char **argv)
        {
            if(!parse(argv,argc))
            {
                usage();
                if(isArgSet("h",true) == true)
                {
                    exit(SDK_SUCCESS);
                }
                return SDK_FAILURE;
            }
            if(isArgSet("h", true) == true)
            {
                usage();
                exit(SDK_SUCCESS);
            }
            if(isArgSet("v", true) || isArgSet("version", false))
            {
                std::cout << "APP SDK version : " << std::endl
                          << sampleVerStr.c_str() << std::endl;
                std::cout << "Bolt version : " << std::endl
                          << getBoltVerStr().c_str() << std::endl;
                exit(SDK_SUCCESS);
            }
            if(samples <= 0)
            {
                std::cout << "Number input samples should be more than Zero"
                          << std::endl << "Exiting..." << std::endl;
                return SDK_FAILURE;
            }
            if(iterations <= 0)
            {
                std::cout << "Number iterations should be more than Zero"
                          << std::endl << "Exiting..." << std::endl;
                return SDK_FAILURE;
            }
            if( !(runMode.empty()) )                    // If device was specified
            {
                if( (strComparei(runMode, "opencl") == false) &&
                        (strComparei(runMode, "serialcpu") == false) &&
                        (strComparei(runMode, "auto") == false) &&
                        (strComparei(runMode, "multicorecpu") == false) )
                {
                    std::cout << "Specified device is incorrect"
                              << std::endl << "Exiting..." << std::endl;
                    return SDK_FAILURE;
                }
            }
            return SDK_SUCCESS;
        }


        /**
        ******************************************************************************
        * @fn displayRunmodeInfo
        * @brief Displays selected device or run-mode from default bolt-control object
        * @return void
        ******************************************************************************/
        void displayRunmodeInfo()
        {
            if(boltControlObj == NULL)
            {
                std::cout << "boltControlObj is not initialized!";
                return;
            }
            switch(boltControlObj->getForceRunMode())
            {
            case (bolt::cl::control::SerialCpu):
                std::cout << "Running in serial cpu mode" << std::endl;
                break;
            case (bolt::cl::control::MultiCoreCpu):
                std::cout << "Running in multi-core cpu mode(TBB)" << std::endl;
                break;
            case (bolt::cl::control::OpenCL):
            case (bolt::cl::control::Automatic):
                std::cout << "Selected device: "
                          << boltControlObj->getDevice().getInfo<CL_DEVICE_NAME>() << std::endl;
                break;
            default :
                std::cout << "Invalid runmode" << std::endl;
            }
        }


        /**
        ***************************************************************************
        * @fn getBoltVerStr
        * @brief Returns Bolt Version string
        * @return std::string containing Bolt lib & runtime version
        **************************************************************************/
        std::string getBoltVerStr()
        {
            char str[1024];
            unsigned libMajor = 0, libMinor = 0, libPatch = 0;
            bolt::cl::getVersion( libMajor, libMinor, libPatch );
#if defined (_WIN32) && !defined(__MINGW32__)
            sprintf_s(str, 256,
                      "Application compiled with Bolt: v%d.%d.%d\nBolt library compiled with Bolt: v%d.%d.%d",
                      boltVerStr.major,
                      boltVerStr.minor,
                      boltVerStr.patch,
                      libMajor,
                      libMinor,
                      libPatch);
#else
            sprintf(str,
                    "Application compiled with Bolt: v%d.%d.%d\nBolt library compiled with Bolt: v%d.%d.%d",
                    boltVerStr.major,
                    boltVerStr.minor,
                    boltVerStr.patch,
                    libMajor,
                    libMinor,
                    libPatch);
#endif
            return std::string(str);
        }

        /**
         * waitForEventAndRelease
         * waits for a event to complete and release the event afterwards
         * @param event cl_event object
         * @return 0 if success else nonzero
         */
        static int waitForEventAndRelease(cl_event *event)
        {
            cl_int status = CL_SUCCESS;
            cl_int eventStatus = CL_QUEUED;
            while(eventStatus != CL_COMPLETE)
            {
                status = clGetEventInfo(
                             *event,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int),
                             &eventStatus,
                             NULL);
                CHECK_BOLT_ERROR(status, "clGetEventEventInfo Failed with Error Code:");
            }
            status = clReleaseEvent(*event);
            CHECK_BOLT_ERROR(status, "clReleaseEvent Failed with Error Code:");
            return SDK_SUCCESS;
        }


};
}
#endif