#!/usr/bin/env python

import pyopencl as cl
import os
from time import sleep

from argparse import ArgumentParser
from itertools import product
from subprocess import Popen

from db import *

def get_platform_name(platform_id):
    platform = cl.get_platforms()[platform_id]
    return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
    platform = cl.get_platforms()[platform_id]
    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
    device = ctx.get_info(cl.context_info.DEVICES)[device_id]
    return device.get_info(cl.device_info.NAME)

def main():
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("platform_id", metavar="<platform-id>", type=int,
                        help="OpenCL platform ID")
    parser.add_argument("device_id", metavar="<device-id>", type=int,
                        help="OpenCL device ID")
    args = parser.parse_args()

    # get testbed information
    platform_id, device_id = args.platform_id, args.device_id
    platform_name = get_platform_name(platform_id)
    device_name = get_device_name(platform_id, device_id)

    init(args.hostname)
    with Session(commit=False) as session:
        testbed = str(get_testbed(session, platform_name, device_name))
    print(testbed)

    cl_launcher_scripts = [
        'clsmith-run-cl_launcher.py',
    ]
    cldrive_scripts = [
        'clgen-run-cldrive.py',
        'clsmith-run-cldrive.py',
        'github-run-cldrive.py',
    ]

    cldrive_script_args = [
        ['-g', '1,1,1',    '-l', '1,1,1',  '-s', '256',  '-i', 'arange'],
        ['-g', '1,1,1',    '-l', '1,1,1',  '-s', '256',  '-i', 'arange', '--no-opts'],
        ['-g', '128,16,1', '-l', '32,1,1', '-s', '4096', '-i', 'arange'],
        ['-g', '128,16,1', '-l', '32,1,1', '-s', '4096', '-i', 'arange', '--no-opts'],
    ]
    cl_launcher_script_args = [
        ['-g', '1,1,1',    '-l', '1,1,1'],
        ['-g', '1,1,1',    '-l', '1,1,1',  '--no-opts'],
        ['-g', '128,16,1', '-l', '32,1,1'],
        ['-g', '128,16,1', '-l', '32,1,1', '--no-opts'],
    ]

    cl_launcher_jobs = list(product(cl_launcher_scripts, cl_launcher_script_args))
    cldrive_jobs = list(product(cldrive_scripts, cldrive_script_args))

    print(len(cl_launcher_jobs), "cl_launcher jobs,",
          len(cldrive_jobs), "cldrive jobs")

    try:
        while True:
            for script, args in cl_launcher_jobs:
                cmd = [script, str(platform_id), str(device_id)] + args
                print('\033[1m', *cmd, '\033[0m')
                p = Popen(['python'] + cmd)
                p.communicate()
                if p.returncode:
                    print("\033[1m\033[91m>>>> EXIT STATUS", p.returncode, '\033[0m')

            for script, args in cldrive_jobs:
                cmd = [script, platform_name, device_name] + args
                print('\033[1m', *cmd, '\033[0m')
                p = Popen(['python'] + cmd)
                p.communicate()
                if p.returncode:
                    print("\033[1m\033[91m>>>> EXIT STATUS", p.returncode, '\033[0m')
    except KeyboardInterrupt:
        print("\ninterrupting, abort.")


if __name__ == "__main__":
    main()
