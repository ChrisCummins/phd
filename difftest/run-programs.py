#!/usr/bin/env python

import pyopencl as cl
import os
from time import sleep

from argparse import ArgumentParser
from itertools import product
from subprocess import Popen

import db
from db import *

def get_platform_name(platform_id):
    platform = cl.get_platforms()[platform_id]
    return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
    platform = cl.get_platforms()[platform_id]
    device = platform.get_devices()[device_id]
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

    db_hostname = args.hostname
    db_url = db.init(db_hostname)
    print("connected to", db_url)
    with Session(commit=False) as session:
        testbed = str(get_testbed(session, platform_name, device_name))
    print(testbed)

    co_scripts = [
        # 'clgen_run_co.py'
    ]
    cl_launcher_scripts = [
        'clsmith-run-cl_launcher.py',
        # 'clgen_run_cl_launcher.py',
    ]
    cldrive_scripts = [
        'clgen_run_cldrive.py',
        # 'clsmith-run-cldrive.py',
        # 'github-run-cldrive.py',
    ]

    co_script_args = [
        [],
        ['--no-opts'],
        ['--with-kernel'],
        ['--with-kernel', '--no-opts'],
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

    timeout = '3h'

    co_jobs = [
        ['timeout', '-s9', timeout, 'python', script, "--hostname", db_hostname, platform_name, device_name] + args
         for script, args in product(co_scripts, co_script_args)
    ]
    cl_launcher_jobs = [
        ['timeout', '-s9', timeout, 'python', script, "--hostname", db_hostname, str(platform_id), str(device_id)] + args
        for script, args in product(cl_launcher_scripts, cl_launcher_script_args)
    ]
    cldrive_jobs = [
        ['timeout', '-s9', timeout, 'python', script, "--hostname", db_hostname, platform_name, device_name] + args
        for script, args in product(cldrive_scripts, cldrive_script_args)
    ]
    jobs = co_jobs + cl_launcher_jobs + cldrive_jobs
    i = 0

    try:
        while len(jobs):
            i += 1
            job = jobs.pop(0)
            nremaining = len(jobs)

            # run job
            print(f'\033[1mjob {i} ({nremaining} remaining)\033[0m', *job)
            p = Popen(job)
            p.communicate()

            # repeat job if it fails
            if p.returncode == -9:
                print('\033[1m\033[91m>>>> TIMEOUT', p.returncode, '\033[0m')
                jobs.append(job)
            elif p.returncode:
                print('\033[1m\033[91m>>>> EXIT STATUS', p.returncode, '\033[0m')
                jobs.append(job)

    except KeyboardInterrupt:
        print("\ninterrupting, abort.")


if __name__ == "__main__":
    main()
