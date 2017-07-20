#!/usr/bin/env python3
import os
import platform
import progressbar
import pyopencl as cl
import re
from collections import deque, namedtuple
from argparse import ArgumentParser
from labm8 import fs
from tempfile import NamedTemporaryFile
from time import time, strftime
from typing import Dict, List, Tuple

import clsmith
import cldrive

import analyze
import db
from db import *
from lib import *


def get_platform_name(platform_id):
    platform = cl.get_platforms()[platform_id]
    return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
    platform = cl.get_platforms()[platform_id]
    device = platform.get_devices()[device_id]
    return device.get_info(cl.device_info.NAME)


def get_driver_version(platform_id, device_id):
    platform = cl.get_platforms()[platform_id]
    device = platform.get_devices()[device_id]
    return device.get_info(cl.device_info.DRIVER_VERSION)


def cl_launcher(src: str, platform_id: int, device_id: int,
                *args) -> Tuple[float, int, str, str]:
    """ Invoke cl launcher on source """
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.cl') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return clsmith.cl_launcher(tmp.name, platform_id, device_id, *args,
                                   timeout=os.environ.get("TIMEOUT", 60))


def verify_params(platform: str, device: str, optimizations: bool,
                  global_size: tuple, local_size: tuple,
                  stderr: str) -> None:
    """ verify that expected params match actual as reported by CLsmith """
    optimizations = "on" if optimizations else "off"

    actual_platform = None
    actual_device = None
    actual_optimizations = None
    actual_global_size = None
    actual_local_size = None
    for line in stderr.split('\n'):
        if line.startswith("Platform: "):
            actual_platform_name = re.sub(r"^Platform: ", "", line).rstrip()
        elif line.startswith("Device: "):
            actual_device_name = re.sub(r"^Device: ", "", line).rstrip()
        elif line.startswith("OpenCL optimizations: "):
            actual_optimizations = re.sub(r"^OpenCL optimizations: ", "", line).rstrip()

        # global size
        match = re.match('^3-D global size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D global size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D global size \d+ = \[(\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), 0, 0)

        # local size
        match = re.match('^3-D local size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D local size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D local size \d+ = \[(\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), 0, 0)

        # check if we've collected everything:
        if (actual_platform and actual_device and actual_optimizations and
            actual_global_size and actual_local_size):
            assert(actual_platform == platform)
            assert(actual_device == device)
            assert(actual_optimizations == optimizations)
            assert(actual_global_size == global_size)
            assert(actual_local_size == local_size)
            return


def parse_ndrange(ndrange: str) -> Tuple[int, int, int]:
    components = ndrange.split(',')
    assert(len(components) == 3)
    return (int(components[0]), int(components[1]), int(components[2]))


def get_num_progs_to_run(session: db.session_t,
                         testbed: Testbed, params: cl_launcherParams):
    subquery = session.query(CLSmithResult.program_id).filter(
        CLSmithResult.testbed_id == testbed.id, CLSmithResult.params_id == params.id)
    num_ran = session.query(CLSmithProgram.id).filter(CLSmithProgram.id.in_(subquery)).count()
    subquery = session.query(CLSmithResult.program_id).filter(
        CLSmithResult.testbed_id == testbed.id)
    total = session.query(CLSmithProgram.id).count()
    return num_ran, total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("platform_id", metavar="<platform-id>", type=int,
                        help="OpenCL platform ID")
    parser.add_argument("device_id", metavar="<device-id>", type=int,
                        help="OpenCL device ID")
    parser.add_argument("--no-opts", action="store_true",
                        help="Disable OpenCL optimizations (on by default)")
    parser.add_argument("-g", "--gsize", type=str, default="128,16,1",
                        help="Comma separated global sizes (default: 128,16,1)")
    parser.add_argument("-l", "--lsize", type=str, default="32,1,1",
                        help="Comma separated global sizes (default: 32,1,1)")
    args = parser.parse_args()

    # Parse command line options
    platform_id = args.platform_id
    device_id = args.device_id

    optimizations = not args.no_opts
    gsize = parse_ndrange(args.gsize)
    lsize = parse_ndrange(args.lsize)

    # get testbed information
    platform_name = get_platform_name(platform_id)
    device_name = get_device_name(platform_id, device_id)
    driver_version = get_driver_version(platform_id, device_id)

    db.init(args.hostname)  # initialize db engine

    with Session() as session:
        testbed = get_testbed(session, platform_name, device_name)

        params = db.get_or_create(session, cl_launcherParams,
            optimizations = optimizations,
            gsize_x = gsize[0], gsize_y = gsize[1], gsize_z = gsize[2],
            lsize_x = lsize[0], lsize_y = lsize[1], lsize_z = lsize[2])
        flags = params.to_flags()

        print(testbed)
        print(params)

        # progress bar
        num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
        bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

        # programs to run, and results to push to database
        inbox = deque()
        outbox = deque()

        def next_batch():
            """
            Fill the inbox with jobs to run, empty the outbox of jobs we have
            run.
            """
            BATCH_SIZE = 100
            devname = testbed.device.strip()
            print(f"\nnext CLSmith batch for {devname} at", strftime("%H:%M:%S"))
            # update the counters
            num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

            # fill inbox
            done = session.query(CLSmithResult.program_id).filter(
                CLSmithResult.testbed == testbed, CLSmithResult.params == params)
            notdone = session.query(CLSmithProgram).filter(
                ~CLSmithProgram.id.in_(done)).order_by(CLSmithProgram.id).limit(BATCH_SIZE)
            for program in notdone:
                inbox.append(program)

            # empty outbox
            while len(outbox):
                session.add(outbox.popleft())
            session.commit()

        try:
            while True:
                # get the next batch of programs to run
                if not len(inbox):
                    next_batch()
                # we have no programs to run
                if not len(inbox):
                    break

                # get next program to run
                program = inbox.popleft()

                # drive the program
                runtime, status, stdout, stderr = cl_launcher(
                    program.src, platform_id, device_id, *flags)

                # assert that executed params match expected
                verify_params(platform=platform_name, device=device_name,
                              optimizations=params.optimizations,
                              global_size=params.gsize, local_size=params.lsize,
                              stderr=stderr)

                # create new result
                result = CLSmithResult(
                    program=program, params=params, testbed=testbed,
                    flags=" ".join(flags), status=status, runtime=runtime,
                    stdout=stdout, stderr=stderr)
                result.outcome = analyze.get_cl_launcher_outcome(result)
                outbox.append(result)

                # update progress bar
                num_ran += 1
                bar.update(min(num_ran, num_to_run))
        finally:
            # flush any remaining results
            next_batch()

    print("done.")
