#!/usr/bin/env python3
import progressbar
import re

from argparse import ArgumentParser
from labm8 import fs
from tempfile import NamedTemporaryFile

import clinfo
import clsmith
import db

from db import Program, Testbed, Params, Result, Session


def cl_launcher(src: str, platform_id: int, device_id: int, *args):
    """ Invoke cl launcher on source """
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.cl') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return clsmith.cl_launcher(tmp.name, platform_id, device_id, *args)


def verify_params(platform: str, device: str, optimizations: bool,
                  global_size: tuple, local_size: tuple,
                  stderr: str):
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


def parse_ndrange(ndrange):
    components = ndrange.split(',')
    assert(len(components) == 3)
    return (int(components[0]), int(components[1]), int(components[2]))


def get_num_progs_to_run(session, testbed, params):
    subquery = session.query(Result.program_id).filter(
        Result.testbed_id == testbed.id, Result.params_id == params.id)
    num_ran = session.query(Program.id).filter(Program.id.in_(subquery)).count()
    subquery = session.query(Result.program_id).filter(
        Result.testbed_id == testbed.id)
    total = session.query(Program.id).count()
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
    platform_name = clinfo.get_platform_name(platform_id)
    device_name = clinfo.get_device_name(platform_id, device_id)
    driver_version = clinfo.get_driver_version(platform_id, device_id)

    db.init(args.hostname)  # initialize db engine

    with Session() as session:
        testbed = db.get_or_create(session, Testbed,
            platform=platform_name, device=device_name, driver=driver_version)

        params = db.get_or_create(session, Params,
            optimizations = optimizations,
            gsize_x = gsize[0], gsize_y = gsize[1], gsize_z = gsize[2],
            lsize_x = lsize[0], lsize_y = lsize[1], lsize_z = lsize[2])
        flags = params.to_flags()

        print(testbed)
        print(params)

        # progress bar
        num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
        bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

        # main execution loop:
        while True:
            # get the next program to run
            subquery = session.query(Result.program_id).filter(
                Result.testbed_id == testbed.id, Result.params_id == params.id)
            program = session.query(Program).filter(
                ~Program.id.in_(subquery)).order_by(Program.id).first()

            # we have no program to run
            if not program:
                break

            runtime, status, stdout, stderr = cl_launcher(
                program.src, platform_id, device_id, *flags)

            # assert that executed params match expected
            verify_params(platform=platform_name, device=device_name,
                          optimizations=params.optimizations,
                          global_size=params.gsize, local_size=params.lsize,
                          stderr=stderr)

            # create new result
            result = Result(
                program=program, params=params, testbed=testbed,
                flags=" ".join(flags), status=status, runtime=runtime,
                stdout=stdout, stderr=stderr)

            # record result
            session.add(result)
            session.commit()

            # update progress bar
            num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

    print("\ndone.")
