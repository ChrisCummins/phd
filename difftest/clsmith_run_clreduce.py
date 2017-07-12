#!/usr/bin/env python3
import re
import fileinput
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from subprocess import Popen, PIPE
from time import time
from typing import Dict, List, Tuple, NewType
from tempfile import TemporaryDirectory
from progressbar import ProgressBar

import clgen_mkharness
import cldrive
import progressbar
import subprocess
from labm8 import fs

import analyze
import db
from db import *
from lib import *

# paths to clreduce library
CLREDUCE_DIR = fs.abspath('..', 'lib', 'clreduce')
CREDUCE = fs.abspath(CLREDUCE_DIR, 'build_creduce', 'creduce', 'creduce')
OCLGRIND = fs.abspath(CLREDUCE_DIR, 'build_oclgrind', 'oclgrind')
CLSMITH_DIR = fs.abspath('..', 'lib', 'CLSmith', 'build')
CLSMITH_RUNTIME_DIR = fs.abspath('..', 'lib', 'CLSmith', 'runtime')
CL_LAUNCHER = fs.abspath(CLSMITH_DIR, 'cl_launcher')
INTERESTING_TEST = fs.abspath(CLREDUCE_DIR, 'interestingness_tests', 'wrong_code_bug.py')

status_t = NewType('status_t', int)
return_t = namedtuple('return_t', ['runtime', 'status', 'log', 'src'])


def get_num_results_to_reproduce(session: db.session_t, testbed: Testbed):
    num_ran = session.query(CLSmithReduction.id).join(CLSmithResult)\
        .filter(CLSmithResult.testbed_id == testbed.id).count()
    total = session.query(CLSmithResult.id)\
        .filter(CLSmithResult.classification == 'w',
                CLSmithResult.testbed_id == testbed.id).count()
    return num_ran, total


def remove_preprocessor_comments(test_case_name):
    """ written by the CLreduce folks """
    for line in fileinput.input(test_case_name, inplace=True):
        if re.match(r'^# \d+ "[^"]*"', line):
            continue
        print(line, end="")


def run_reduction(s, result: CLSmithResult) -> return_t:
    """
    Note as a side effect this method modified environment variables.
    """
    start_time = time()
    print("reduction")

    oracle_testbed_id = s.query(Testbed.id).filter(Testbed.platform == "Oclgrind").filter()
    assert oracle_testbed_id
    oracle_run = s.query(CLSmithResult)\
        .filter(CLSmithResult.testbed_id == oracle_testbed_id,
                CLSmithResult.program_id == result.program_id,
                CLSmithResult.params_id == result.params_id).first()
    if oracle_run:
        assert oracle_run.stdout != result.stdout
    else:
        print("warning: no oracle run on file", file=sys.stderr)

    with TemporaryDirectory(prefix='clreduce-') as tmpdir:
        path = fs.path(tmpdir, "kernel.cl")

        # move headers into place
        for header in [x for x in fs.ls(CLSMITH_DIR, abspaths=True) if x.endswith('.h')]:
            fs.cp(header, tmpdir)

        # put kernel
        kernel = fs.path(tmpdir, "kernel.cl")
        with open(kernel, 'w') as outfile:
            print(result.program.src, file=outfile)

        cmd = ["clang", "-I", CLSMITH_DIR, "-I", CLSMITH_RUNTIME_DIR,
               "-E", "-CC", "-o", kernel, path]
        pp = subprocess.run(cmd, timeout=60, check=True)
        remove_preprocessor_comments(kernel)

        # Get OpenCL indexes
        env = cldrive.make_env(platform=result.testbed.platform,
                               device=result.testbed.device)
        platform_id, device_id = env.ids()

        # setup env
        os.chdir(tmpdir)
        optimized = 'optimised' if result.params.optimizations else 'unoptimised'
        os.environ['CREDUCE_TEST_OPTIMISATION_LEVEL'] = optimized
        os.environ['CREDUCE_TEST_CASE'] = path
        os.environ['OCLGRIND'] = OCLGRIND
        os.environ['CREDUCE_TEST_CL_LAUNCHER'] = CL_LAUNCHER
        os.environ['CREDUCE_TEST_PLATFORM'] = str(platform_id)
        os.environ['CREDUCE_TEST_DEVICE'] = str(device_id)

        cmd = ['perl', '--', CREDUCE, '--n', '4', '--timing', INTERESTING_TEST, path]

        # Run the actual reduction
        out = []
        process = subprocess.Popen(cmd, universal_newlines=True, bufsize=1,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                out.append(line)
        process.wait()

        status = process.returncode

        with open(kernel) as infile:
            src = infile.read()

    runtime = time() - start_time
    return CLSmithReduction(result=result, runtime=runtime, status=status,
                            src=src, log='\n'.join(out))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-H", "--hostname", type=str, default="cc1",
        help="MySQL database hostname")
    parser.add_argument(
        "platform", metavar="<platform name>", help="OpenCL platform name")
    parser.add_argument(
        "device", metavar="<device name>", help="OpenCL device name")
    args = parser.parse_args()

    db.init(args.hostname)  # initialize db engine

    assert fs.isexe(CREDUCE)
    assert fs.isexe(INTERESTING_TEST)

    with Session(commit=False) as s:
        testbed = get_testbed(s, args.platform, args.device)

        # progress bar
        num_ran, total = get_num_results_to_reproduce(s, testbed)
        bar = progressbar.ProgressBar(init_value=num_ran, max_value=total)

        # main execution loop:
        while True:
            # get the next result to reduce
            done = s.query(CLSmithReduction.id)
            result = s.query(CLSmithResult).filter(
                CLSmithResult.testbed_id == testbed.id,
                CLSmithResult.classification == "w",
                ~CLSmithResult.id.in_(done)).order_by(CLSmithResult.id).first()

            if not result:
                break

            reduction = run_reduction(s, result)

            # record result
            s.add(reduction)
            s.commit()

            # update progress bar
            num_ran, total = get_num_results_to_reproduce(s, testbed)
            bar.max_value = total
            bar.update(min(num_ran, total))

    print("done.")
