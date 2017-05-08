#!/usr/bin/env python3
"""
The Reproducer-er (TM)

Reproduce suspicious results.

Usage: ./the-reproducerer

NOTE: requires running ./analyze.py first


What it does
------------
  Figure out what OpenCL devices we have on this system
  Fetch all suspicious entries from the DB for the available OpenCL devices
  For each suspicious entry:
    Attempt to reproduce it.
    If successful, generate C code for a standalone binary, and a bug report.
    Else, record a failure to reproduce.
    Additionally, warn if any experimental setup has changed
        (e.g. different software versions).
"""
import cldrive
import os
import pyopencl as cl
import sqlalchemy as sql
import sys

from argparse import ArgumentParser
from subprocess import Popen, PIPE
from labm8 import fs
from pathlib import Path
from progressbar import ProgressBar

import db
from db import *
from lib import *


def escape_c_string(s):
    return '\n'.join('"{}\\n"'.format(x) for x in s.split('\n') if len(x))


def make_c_source(kernel_src):
    src = """\
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


const char *src = \\
"""
    src += escape_c_string(kernel_src)

    src += """;
const char *kernel_name = "{}";
""".format(cldrive.kernel_name(kernel_src))

    src += """
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

        cl_program program = clCreateProgramWithSource(ctx, 1, (const char **) &src, NULL, &err);
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
    return src


def reproduce_clgen_build_failures(result):
    import analyze
    import clgen_run_cldrive
    print(result)

    ### Reproduce using Python
    flags = result.params.to_flags()
    cli = cldrive_cli(result.testbed.platform, result.testbed.device, *flags)

    # runtime, status, stdout, stderr = clgen_run_cldrive.drive(
    #     cli, result.program.src)

    # new_result = CLgenResult(
    #     program=result.program, params=result.params, testbed=result.testbed,
    #     cli=" ".join(cli), status=status, runtime=runtime,
    #     stdout=stdout, stderr=stderr)

    # analyze.analyze_cldrive_result(new_result, CLgenResult, session)

    # if new_result.classification != result.classification:
    #     print("could not reproduce result")
    #     sys.exit(1)

    # print(">>>> Reproduced using cldrive")

    ### Reproduce using C standalone binary
    src = make_c_source(result.program.src)

    with open(result.program.id + '.c', 'w') as outfile:
        print(src, file=outfile)

    # TODO: portable -I and -l flags
    cli = ['gcc', '-xc', '-', '-lOpenCL']
    process = Popen(cli, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    universal_newlines=True)
    stdout, stderr = process.communicate(src)
    print('stdout:', stdout.rstrip())
    print('stderr:', stderr.rstrip())
    print('status:', process.returncode)
    if process.returncode:
        print("Failed to compile binary for", result.program.id)
        sys.exit(1)

    cli = ['./a.out']
    process = Popen(cli, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = process.communicate(src)
    print('stdout:', stdout.rstrip())
    print('stderr:', stderr.rstrip())
    print('status:', process.returncode)

    if process.returncode:
        print(">>> Reproduced using standalone binary")
        sys.exit(0)
    else:
        print(">>> Failed to reproduce using standalone binary")


def get_available_cldrive_envs():
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    args = parser.parse_args()

    db.init(args.hostname)
    session = db.make_session()

    for env in cldrive.all_envs():
        testbed = db.get_testbed(session, env.platform, env.device)

        clgen_build_failures = session.query(CLgenResult)\
            .filter(CLgenResult.testbed == testbed)\
            .filter(CLgenResult.classification == 'Build failure')
        for result in clgen_build_failures:
            reproduce_clgen_build_failures(result)

    print("done.")
