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


#include "BoxFilterGLSAT.hpp"
#include "BoxFilterGLSeparable.hpp"
#include <cmath>

int
main(int argc, char* argv[])
{
    /**
     * By default SAT version runs
     * If verification option is given then both versions are verified (SAT and Separable)
     * If sep is mentioned with verification option then only Separable version runs
     * If sat is mentioned with verification option then only SAT version runs
     */

    bool flag_sat = false;
    bool flag_sep = false;
    bool flag_verify = false;
    bool separable_verification = false;
    bool sat_verification = false;

    // Process command line options
    for(int i = 0; i < argc; i++)
    {
        if(!strcmp(argv[i], "-sep"))
        {
            flag_sep = true;
        }
        if(!strcmp(argv[i], "-sat"))
        {
            flag_sat = true;
        }
        if(!strcmp(argv[i], "-e"))
        {
            flag_verify = true;
        }
    }

    // If -sep is specified with -e then only Separable version's verification runs
    if(flag_sep == true && flag_verify == true)
    {
        separable_verification = true;
    }

    // If -sat is specified with -e then only SAT version's verification runs
    if(flag_sat == true && flag_verify == true)
    {
        sat_verification = true;
    }

    // If nothing is specified then by default SAT version runs
    if(flag_sep == false && flag_sat == false)
    {
        flag_sat = true;
    }
    /**
    * Don't use GLUT for windows
    * Reason: GLUT always creates a context on the primary device in MultiGPU environment.
    * For Linux, we can create context on any device.
    */

    if(flag_sat && !separable_verification)
    {
        std::cout << "Running SAT version.. " << std::endl;

        BoxFilterGLSAT clBoxFilterGLSAT;
        BoxFilterGLSAT::boxFilterGLSAT = &clBoxFilterGLSAT;

        if(clBoxFilterGLSAT.initialize() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSAT.sampleArgs->parseCommandLine(argc, argv))
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSAT.sampleArgs->isDumpBinaryEnabled())
        {
            return clBoxFilterGLSAT.genBinaryImage();
        }

        cl_int retValue = clBoxFilterGLSAT.setup();
        if(retValue != SDK_SUCCESS)
        {
            return (retValue == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;
        }

        if(clBoxFilterGLSAT.run() !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSAT.verifyResults() !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSAT.cleanup()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clBoxFilterGLSAT.printStats();
    }
    if(flag_sep && !sat_verification)
    {
        std::cout << "Running Separable version.. " << std::endl;

        BoxFilterGLSeparable clBoxFilterGLSeparable;
        BoxFilterGLSeparable::boxFilterGLSeparable = &clBoxFilterGLSeparable;

        if(clBoxFilterGLSeparable.initialize() !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSeparable.sampleArgs->parseCommandLine(argc,
                argv) !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSeparable.sampleArgs->isDumpBinaryEnabled())
        {
            return clBoxFilterGLSeparable.genBinaryImage();
        }

        cl_int retValue = clBoxFilterGLSeparable.setup();
        if(retValue != SDK_SUCCESS)
        {
            return (retValue == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;
        }

        if(clBoxFilterGLSeparable.run() !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSeparable.verifyResults()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBoxFilterGLSeparable.cleanup() !=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clBoxFilterGLSeparable.printStats();
    }
    return SDK_SUCCESS;
}



