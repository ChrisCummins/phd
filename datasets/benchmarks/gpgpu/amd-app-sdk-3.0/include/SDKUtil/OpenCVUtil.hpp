/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef OPENCV_UTIL_H_
#define OPENCV_UTIL_H_

/******************************************************************************
* Included header files                                                       *
******************************************************************************/

#include <opencv2/core/version.hpp>
#include <opencv2/core/core.hpp>


#include "SDKUtil.hpp"

namespace appsdk
{
/******************************************************************************
* OpenCVCommandArgs                                                        *
* Class implements various resources required by the test & initializes the   *
* resources used by tests                                                     *
******************************************************************************/
class OpenCVCommandArgs : public SDKCmdArgsParser
{
  bool version;            /**< Cmd Line Option- if version */

public:

  int iterations;  /*!<Number of iterations for kernel execution.*/

  /**
  ***************************************************************************
  * @fn initialize
  * @brief Initialize the resources used by tests
  * @return SDK_SUCCESS on success, SDK_FAILURE otherwise
  **************************************************************************/
  int initialize();

  /**
  ***************************************************************************
  * @brief Constructor of OpenCVCommandArgs to initialize member
           variables
  **************************************************************************/
  OpenCVCommandArgs();
  
  /**
  ***************************************************************************
  * @brief Destructor of OpenCVCommandArgs
  **************************************************************************/
  ~OpenCVCommandArgs();

  /**
  ***************************************************************************
  * @fn parseCommandLine
  * @brief parses the command line options given by user
  * @param argc Number of elements in cmd line input
  * @param argv Array of char* storing the CmdLine Options
  * @return SDK_SUCCESS on success, SDK_FAILURE otherwise
  **************************************************************************/
  int parseCommandLine(int argc, char **argv);
  
};

/******************************************************************************
* Implementation of OpenCVCommandArgs::initialize()                        *
******************************************************************************/
int OpenCVCommandArgs::initialize()
{
  int defaultOptions = 4;

  Option *optionList = new Option[defaultOptions];
  CHECK_ALLOCATION(optionList,
                   "Error. Failed to allocate memory (optionList)\n");

  optionList[0]._sVersion = "q";
  optionList[0]._lVersion = "quiet";
  optionList[0]._description = "Quiet mode. Suppress most text output.";
  optionList[0]._type = CA_NO_ARGUMENT;
  optionList[0]._value = &quiet;

  optionList[1]._sVersion = "e";
  optionList[1]._lVersion = "verify";
  optionList[1]._description =
                 "Verify results against reference implementation.";
  optionList[1]._type = CA_NO_ARGUMENT;
  optionList[1]._value = &verify;

  optionList[2]._sVersion = "t";
  optionList[2]._lVersion = "timing";
  optionList[2]._description = "Print timing related statistics.";
  optionList[2]._type = CA_NO_ARGUMENT;
  optionList[2]._value = &timing;

  optionList[3]._sVersion = "v";
  optionList[3]._lVersion = "version";
  optionList[3]._description = "OpenCV lib & runtime version string.";
  optionList[3]._type = CA_NO_ARGUMENT;
  optionList[3]._value = &version;

  _numArgs = defaultOptions;
  _options = optionList;
              
  return SDK_SUCCESS;
}

/******************************************************************************
* Implementation of OpenCVCommandArgs::parseCommandLine()                  *
******************************************************************************/
int OpenCVCommandArgs::parseCommandLine(int argc, char**argv)
{
    if(!parse(argv,argc))
    {
      usage();
      if(isArgSet("h",true) == true)
        exit(SDK_SUCCESS);
      return SDK_FAILURE;
    }
    if(isArgSet("h",true) == true)
    {
      usage();
      exit(SDK_SUCCESS);
    }
    if(isArgSet("v", true)
       || isArgSet("version", false))
    {
      std::cout << "APP SDK version : " << sampleVerStr.c_str() << std::endl;
      exit(0);
    }
  

  return SDK_SUCCESS;
}

/******************************************************************************
* Implementation of OpenCVCommandArgs::OpenCVCommandArgs()              *
******************************************************************************/
OpenCVCommandArgs::OpenCVCommandArgs() : SDKCmdArgsParser()
{
  iterations = 1;
  //initialize();
}

/******************************************************************************
* Implementation of OpenCVCommandArgs::~OpenCVCommandArgs()             *
******************************************************************************/
OpenCVCommandArgs::~OpenCVCommandArgs()
{
}
}
#endif
