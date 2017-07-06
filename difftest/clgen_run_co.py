#!/usr/bin/env python3
import re
import sqlalchemy as sql
import sys

from argparse import ArgumentParser
from collections import namedtuple
from subprocess import Popen, PIPE
from time import time
from typing import Dict, List, Tuple, NewType

import cldrive
import progressbar
from labm8 import fs

import co
import db
from db import *
from lib import *


def get_num_progs_to_run(session: db.session_t,
                         testbed: Testbed, params: coParams):
    subquery = session.query(coCLgenResult.program_id).filter(
        coCLgenResult.testbed_id == testbed.id, coCLgenResult.params_id == params.id)
    num_ran = session.query(CLgenProgram.id).filter(CLgenProgram.id.in_(subquery)).count()
    subquery = session.query(coCLgenResult.program_id).filter(
        coCLgenResult.testbed_id == testbed.id)
    total = session.query(CLgenProgram.id).count()
    return num_ran, total


def get_next_program(session: db.session_t,
                     testbed: Testbed, params: coParams):
    subquery = session.query(coCLgenResult.program_id).filter(
        coCLgenResult.testbed == testbed, coCLgenResult.params == params)
    program = session.query(CLgenProgram).filter(
        ~CLgenProgram.id.in_(subquery)).order_by(CLgenProgram.id).first()
    return program


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
        "--with-kernel", action="store_true",
        help="Build kernel after program")
    args = parser.parse_args()

    db.init(args.hostname)  # initialize db engine

    with Session(commit=False) as session:
        testbed = get_testbed(session, args.platform, args.device)

        params = db.get_or_create(
            session, coParams, optimizations=not args.no_opts,
            build_kernel=args.with_kernel)
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
            program = get_next_program(session, testbed, params)

            # we have no program to run
            if not program:
                break

            runtime, status, stdout, stderr = co.drive(cli, program.src)

            # assert that executed params match expected
            co.verify_params(platform=args.platform, device=args.device,
                             optimizations=params.optimizations_on_off,
                             stderr=stderr)

            # create new result
            result = coCLgenResult(
                program=program, params=params, testbed=testbed,
                status=status, runtime=runtime, stdout=stdout, stderr=stderr)

            # record result
            session.add(result)
            session.commit()

            # update progress bar
            num_ran, num_to_run = get_num_progs_to_run(session, testbed, params)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

    print("done.")
