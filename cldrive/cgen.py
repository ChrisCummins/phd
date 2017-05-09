# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
from enum import Enum
from functools import partial

import numpy as np


from cldrive import *


def escape_c_string(s):
    def escape_line(l):
        x = l
        return '"{}\\n"'.format(x)

    return '\n'.join(escape_line(l) for l in s.split('\n') if len(l.strip()))


def emit_c(env: OpenCLEnvironment, src: str, inputs: np.array,
           gsize: NDRange, lsize: NDRange, timeout: int=-1,
           optimizations: bool=True, profiling: bool=False,
           debug: bool=False) -> np.array:
    """
    Generate C code to drive kernel.

    Parameters
    ----------
    env : OpenCLEnvironment
        The OpenCL environment to run the kernel in.
    src : str
        The OpenCL kernel source.
    inputs : np.array
        The input data to the kernel.
    optimizations : bool, optional
        Whether to enable or disbale OpenCL compiler optimizations.
    profiling : bool, optional
        If true, print OpenCLevent times for data transfers and kernel
        executions to stderr.
    timeout : int, optional
        Cancel execution if it has not completed after this many seconds.
        A value <= 0 means never time out.
    debug : bool, optional
        If true, silence the OpenCL compiler.

    Returns
    -------
    str
        Code which can be compiled using a C compiler to drive the kernel.

    Raises
    ------
    ValueError
        If input types are incorrect.
    TypeError
        If an input is of an incorrect type.
    LogicError
        If the input types do not match OpenCL kernel types.
    PorcelainError
        If the OpenCL subprocess exits with non-zero return  code.
    RuntimeError
        If OpenCL program fails to build or run.

    Examples
    --------
    TODO
    """
    c = """\
/* generated using cldrive <https://github.com/ChrisCummins/cldrive> */
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char *kernel_src = \\
"""
    c += escape_c_string(src)

    c += """;
const char *kernel_name = "{}";
""".format(kernel_name(src))

    c += """
int main() {
        int err;

        cl_uint num_platforms;
        cl_platform_id platform_id;
        err = clGetPlatformIDs(
                /* cl_uint num_entries */ 1,
                /* cl_platform_id *platforms */ &platform_id,
                /* cl_uint *num_platforms */ &num_platforms);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clGetPlatformIDs CL_INVALID_VALUE\\n");
                        return 1;
        }

        cl_device_id device_id;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_PLATFORM:
                        fprintf(stderr, "clGetDeviceIDs CL_INVALID_PLATFORM\\n");
                        return 1;
                case CL_INVALID_DEVICE_TYPE:
                        fprintf(stderr, "clGetDeviceIDs CL_INVALID_DEVICE_TYPE\\n");
                        return 1;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clGetDeviceIDs CL_INVALID_VALUE\\n");
                        return 1;
                case CL_DEVICE_NOT_FOUND:
                        fprintf(stderr, "clGetDeviceIDs CL_DEVICE_NOT_FOUND\\n");
                        return 1;
                default:
                        fprintf(stderr, "clGetDeviceIDs %d\\n", err);
                        return 1;
        }

        cl_context ctx = clCreateContext(
                /* cl_context_properties *properties */ NULL,
                /* cl_uint num_devices */ 1,
                /* const cl_device_id *devices */ &device_id,
                /* void *pfn_notify */ NULL,
                /* void *user_data */ NULL,
                /* cl_int *errcode_ret */ &err);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_PLATFORM:
                        fprintf(stderr, "clCreateContext CL_INVALID_PLATFORM\\n");
                        return 1;
                case CL_INVALID_DEVICE:
                        fprintf(stderr, "clCreateContext CL_INVALID_DEVICE\\n");
                        return 1;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clCreateContext CL_INVALID_VALUE\\n");
                        return 1;
                case CL_DEVICE_NOT_AVAILABLE:
                        fprintf(stderr, "clCreateContext CL_DEVICE_NOT_AVAILABLE\\n");
                        return 1;
                case CL_OUT_OF_HOST_MEMORY:
                        fprintf(stderr, "clCreateContext CL_OUT_OF_HOST_MEMORY\\n");
                        return 1;
                default:
                        fprintf(stderr, "clCreateContext %d\\n", err);
                        return 1;
        }

        cl_command_queue queue = clCreateCommandQueue(
                /* cl_context context */ ctx,
                /* cl_device_id device */ device_id,
                /* cl_command_queue_properties properties */ 0,
                /* cl_int *errcode_ret */ &err);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_CONTEXT:
                        fprintf(stderr, "clCreateCommandQueue CL_INVALID_CONTEXT\\n");
                        return 1;
                case CL_INVALID_DEVICE:
                        fprintf(stderr, "clCreateCommandQueue CL_INVALID_DEVICE\\n");
                        return 1;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clCreateCommandQueue CL_INVALID_VALUE\\n");
                        return 1;
                case CL_INVALID_QUEUE_PROPERTIES:
                        fprintf(stderr, "clCreateCommandQueue CL_INVALID_QUEUE_PROPERTIES\\n");
                        return 1;
                case CL_OUT_OF_HOST_MEMORY:
                        fprintf(stderr, "clCreateCommandQueue CL_OUT_OF_HOST_MEMORY\\n");
                        return 1;
                default:
                        fprintf(stderr, "clCreateCommandQueue %d\\n", err);
                        return 1;
        }

        cl_program program = clCreateProgramWithSource(ctx, 1, (const char **) &kernel_src, NULL, &err);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_CONTEXT:
                        fprintf(stderr, "clCreateProgramWithSource CL_INVALID_CONTEXT\\n");
                        return 1;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clCreateProgramWithSource CL_INVALID_VALUE\\n");
                        return 1;
                case CL_OUT_OF_HOST_MEMORY:
                        fprintf(stderr, "clCreateProgramWithSource CL_OUT_OF_HOST_MEMORY\\n");
                        return 1;
                default:
                        fprintf(stderr, "clCreateProgramWithSource %d\\n", err);
                        return 1;
        }

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
                size_t len;
                char buffer[2048];

                fprintf(stderr, "clBuildProgram build log:\\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                fprintf(stderr, "%s\\n", buffer);
                exit(1);

                switch (err) {
                        case CL_INVALID_PROGRAM:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_PROGRAM\\n");
                                return 1;
                        case CL_INVALID_VALUE:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_VALUE\\n");
                                return 1;
                        case CL_INVALID_DEVICE:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_DEVICE\\n");
                                return 1;
                        case CL_INVALID_BINARY:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_BINARY\\n");
                                return 1;
                        case CL_INVALID_BUILD_OPTIONS:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_BUILD_OPTIONS\\n");
                                return 1;
                        case CL_INVALID_OPERATION:
                                fprintf(stderr, "clGetProgramBuildInfo CL_INVALID_OPERATION\\n");
                                return 1;
                        case CL_COMPILER_NOT_AVAILABLE:
                                fprintf(stderr, "clGetProgramBuildInfo CL_COMPILER_NOT_AVAILABLE\\n");
                                return 1;
                        case CL_BUILD_PROGRAM_FAILURE:
                                fprintf(stderr, "clGetProgramBuildInfo CL_BUILD_PROGRAM_FAILURE\\n");
                                return 1;
                        case CL_OUT_OF_HOST_MEMORY:
                                fprintf(stderr, "clGetProgramBuildInfo CL_OUT_OF_HOST_MEMORY\\n");
                                return 1;
                        default:
                                fprintf(stderr, "clGetProgramBuildInfo %d\\n", err);
                                return 1;
                }
        }

        cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
        switch (err) {
                case CL_SUCCESS:
                        break;
                case CL_INVALID_PROGRAM:
                        fprintf(stderr, "clCreateContext CL_INVALID_PROGRAM\\n");
                        return 1;
                case CL_INVALID_PROGRAM_EXECUTABLE:
                        fprintf(stderr, "clCreateContext CL_INVALID_PROGRAM_EXECUTABLE\\n");
                        return 1;
                case CL_INVALID_VALUE:
                        fprintf(stderr, "clCreateContext CL_INVALID_VALUE\\n");
                        return 1;
                case CL_INVALID_KERNEL_NAME:
                        fprintf(stderr, "clCreateContext CL_INVALID_KERNEL_NAME\\n");
                        return 1;
                case CL_INVALID_KERNEL_DEFINITION:
                        fprintf(stderr, "clCreateContext CL_INVALID_KERNEL_DEFINITION\\n");
                        return 1;
                case CL_OUT_OF_HOST_MEMORY:
                        fprintf(stderr, "clCreateContext CL_OUT_OF_HOST_MEMORY\\n");
                        return 1;
                default:
                        fprintf(stderr, "clCreateContext %d\\n", err);
                        return 1;
        }

        return 0;
}
"""
    return c
