#!/usr/bin/env python3
import re
from argparse import ArgumentParser
from collections import namedtuple
from subprocess import Popen, PIPE
from time import time
from typing import Dict, List, Tuple, NewType

import cldrive
import progressbar
from labm8 import fs

import db
from db import *
from lib import *

status_t = NewType('status_t', int)
return_t = namedtuple('return_t', ['runtime', 'status', 'stdout', 'stderr'])


def verify_params(platform: str, device: str, optimizations: bool,
                  global_size: tuple, local_size: tuple,
                  stderr: str) -> None:
    """ verify that expected params match actual as reported by cldrive """
    optimizations = "on" if optimizations else "off"

    actual_platform = None
    actual_device = None
    actual_optimizations = None
    actual_global_size = None
    actual_local_size = None
    for line in stderr.split('\n'):
        if line.startswith("[cldrive] Platform: "):
            actual_platform_name = re.sub(r"^\[cldrive\] Platform: ", "", line).rstrip()
        elif line.startswith("[cldrive] Device: "):
            actual_device_name = re.sub(r"^\[cldrive\] Device: ", "", line).rstrip()
        elif line.startswith("[cldrive] OpenCL optimizations: "):
            actual_optimizations = re.sub(r"^\[cldrive\] OpenCL optimizations: ", "", line).rstrip()

        # global size
        match = re.match('^\[cldrive\] 3-D global size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # local size
        match = re.match('^\[cldrive\] 3-D local size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # check if we've collected everything:
        if (actual_platform and actual_device and actual_optimizations and
            actual_global_size and actual_local_size):
            assert(actual_platform == platform)
            assert(actual_device == device)
            assert(actual_optimizations == optimizations)
            assert(actual_global_size == global_size)
            assert(actual_local_size == local_size)
            return


def drive(command: List[str], src: str) -> Tuple[float, int, str, str]:
    """ invoke cldrive on source """
    start_time = time()

    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src.encode('utf-8'))
    stderr = stderr.decode('utf-8')

    runtime = time() - start_time

    return return_t(
        runtime=runtime, status=status_t(process.returncode),
        stdout=stdout, stderr=stderr)


def get_num_progs_to_run(session: db.session_t,
                         testbed: Testbed, params: cldriveParams):
    subquery = session.query(CLgenResult.program_id).filter(
        CLgenResult.testbed_id == testbed.id, CLgenResult.params_id == params.id)
    num_ran = session.query(CLgenProgram.id).filter(CLgenProgram.id.in_(subquery)).count()
    subquery = session.query(CLgenResult.program_id).filter(
        CLgenResult.testbed_id == testbed.id)
    total = session.query(CLgenProgram.id).count()
    return num_ran, total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-H", "--hostname", type=str, default="cc1",
        help="MySQL database hostname")
    parser.add_argument(
        "platform", metavar="<platform name>", help="OpenCL platform name")
    parser.add_argument(
        "device", metavar="<device name>", help="OpenCL device name")
    parser.add_argument(
        "--no-opts", action="store_true",
        help="Disable OpenCL optimizations (on by default)")
    parser.add_argument(
        "-s", "--size", metavar="<size>", type=int, default=64,
        help="size of input arrays to generate (default: 64)")
    parser.add_argument(
        "-i", "--generator", metavar="<{rand,arange,zeros,ones}>", default="arange",
        help="input generator to use, one of: {rand,arange,zeros,ones} (default: arange)")
    parser.add_argument(
        "--scalar-val", metavar="<float>", type=float, default=None,
        help="values to assign to scalar inputs (default: <size> argumnent)")
    parser.add_argument(
        "-g", "--gsize", type=str, default="128,16,1",
        help="Comma separated global sizes (default: 128,16,1)")
    parser.add_argument(
        "-l", "--lsize", type=str, default="32,1,1",
        help="Comma separated global sizes (default: 32,1,1)")
    args = parser.parse_args()

    gsize = cldrive.NDRange.from_str(args.gsize)
    lsize = cldrive.NDRange.from_str(args.lsize)
    env = cldrive.make_env(platform=args.platform, device=args.device)

    db.init(args.hostname)  # initialize db engine

    with Session(commit=False) as session:
        testbed = get_testbed(session, args.platform, args.device)

        params = db.get_or_create(
            session, cldriveParams, size=args.size, generator=args.generator,
            scalar_val=args.scalar_val, gsize_x=gsize.x, gsize_y=gsize.y,
            gsize_z=gsize.z, lsize_x=lsize.x, lsize_y=lsize.y, lsize_z=lsize.z,
            optimizations=not args.no_opts)

        harness = clgen_mkharness.mkharness(session, env, program, params)

        flags = params.to_flags()
        cli = cldrive_cli(args.platform, args.device, *flags)

        print(testbed)
        print(" ".join(cli))

        # progress bar
        num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
        bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

        # main execution loop:
        while True:
            # get the next program to run
            subquery = session.query(CLgenResult.program_id).filter(
                CLgenResult.testbed == testbed, CLgenResult.params == params)
            program = session.query(CLgenProgram).filter(
                ~CLgenProgram.id.in_(subquery)).order_by(CLgenProgram.id).first()

            # we have no program to run
            if not program:
                break

            runtime, status, stdout, stderr = drive(cli, program.src)

            # assert that executed params match expected
            verify_params(platform=args.platform, device=args.device,
                          optimizations=params.optimizations,
                          global_size=params.gsize, local_size=params.lsize,
                          stderr=stderr)

            # create new result
            result = CLgenResult(
                program=program, params=params, testbed=testbed,
                cli=" ".join(cli), status=status, runtime=runtime,
                stdout=stdout, stderr=stderr)

            # record result
            session.add(result)
            session.commit()

            # update progress bar
            num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

    print("done.")
