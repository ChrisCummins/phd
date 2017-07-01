#!/usr/bin/env python

import pyopencl as cl
import random
import os
import sys
from time import sleep

from argparse import ArgumentParser
from itertools import product
from subprocess import Popen

import db
from db import *

from clgen_run_cl_launcher import *


def reproduce(result_id):
    TABLE = cl_launcherCLgenResult

    with Session(commit=False) as s:
        result = s.query(TABLE).filter(TABLE.id == result_id).first()
        if not result:
            raise KeyError(f"no result with ID {result_id}")

        flags = result.params.to_flags()
        program = result.program

        try:
            platform_id = result.testbed.platform_id()
            device_id = result.testbed.device_id()
        except KeyError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

        runtime, status, stdout, stderr = cl_launcher(
                program.src, platform_id, device_id, *flags)

        reproduced = True
        if stderr != result.stderr:
            reproduced = False
            print("stderr differs")
        if stdout != result.stdout:
            reproduced = False
            print("stdout differs")

        return reproduced


def main():
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-r", "--result", dest="result_id", type=int, default=None,
                        help="results ID")
    args = parser.parse_args()

    # get testbed information
    db_hostname = args.hostname
    db_url = db.init(db_hostname)

    if not reproduce(args.result_id):
        sys.exit(1)


if __name__ == "__main__":
    main()
