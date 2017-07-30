#!/usr/bin/env python3
import progressbar
import re
import subprocess
from argparse import ArgumentParser
from collections import deque, namedtuple
from labm8 import crypto, fs
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from time import time, strftime
from typing import Dict, List, Tuple, NewType

import analyze
import cldrive
import clgen_mkharness
import util
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


def drive_testcase(s: db.session_t, testcase: CLgenTestCase,
                   env: cldrive.OpenCLEnvironment, platform_id: int,
                   device_id: int, timeout: int=60) -> return_t:
    """ run CLgen program test harness """
    harness = clgen_mkharness.mkharness(s, env, testcase)

    with NamedTemporaryFile(prefix='cldrive-harness-', delete=False) as tmpfile:
        path = tmpfile.name
    try:
        clgen_mkharness.compile_harness(
            harness.src, path, platform_id=platform_id, device_id=device_id)

        cmd = ['timeout', '-s9', str(timeout), tmpfile.name]

        start_time = time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        try:
            stdout = stdout.decode('utf-8')
        except UnicodeError as e:
            stdout = '<-- UTF-ERROR -->'

        try:
            stderr = stderr.decode('utf-8')
        except UnicodeError as e:
            stderr = '<-- UTF-ERROR -->'
        runtime = time() - start_time

        return return_t(
            runtime=runtime, status=status_t(proc.returncode),
            stdout=stdout, stderr=stderr)
    finally:
        fs.rm(path)


def get_num_to_run(session: db.session_t, testbed: Testbed, optimizations: int=None):
    num_ran = session.query(sql.sql.func.count(CLgenResult.id))\
                .filter(CLgenResult.testbed_id == testbed.id)
    total = session.query(sql.sql.func.count(CLgenTestCase.id))

    if optimizations is not None:
        num_ran = num_ran.join(CLgenTestCase).join(cldriveParams)\
            .filter(cldriveParams.optimizations == optimizations)
        total = total.join(cldriveParams)\
            .filter(cldriveParams.optimizations == optimizations)

    return num_ran.scalar(), total.scalar()


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
        "--opt", action="store_true", help="Only test with optimizations on")
    parser.add_argument(
        "--no-opt", action="store_true", help="Only test with optimizations disabled")
    args = parser.parse_args()

    env = cldrive.make_env(platform=args.platform, device=args.device)
    platform_id, device_id = env.ids()

    optimizations = None
    if args.opt:
        optimizations = 1
    if args.no_opt:
        optimizations = 0

    db.init(args.hostname)  # initialize db engine

    with Session(commit=False) as session:
        testbed = get_testbed(session, args.platform, args.device)
        devname = util.device_str(testbed.device)
        print(devname)

        # progress bar
        num_ran, num_to_run = get_num_to_run(session, testbed, optimizations)
        bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

        # testcases to run, and results to push to database
        inbox = deque()
        outbox = deque()

        def next_batch():
            """
            Fill the inbox with jobs to run, empty the outbox of jobs we have
            run.
            """
            BATCH_SIZE = 100
            print(f"\nnext CLgen batch for {devname} at", strftime("%H:%M:%S"))
            # update the counters
            num_ran, num_to_run = get_num_to_run(session, testbed, optimizations)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

            # fill inbox
            done = session.query(CLgenResult.testcase_id).filter(
                CLgenResult.testbed == testbed)
            if optimizations is not None:
                done = done.join(CLgenTestCase).join(cldriveParams)\
                        .filter(cldriveParams.optimizations == optimizations)

            todo = session.query(CLgenTestcase)\
                        .filter(~CLgenTestcase.id.in_(done))\
                        .order_by(CLgenTestcase.program_id,
                                  CLgenTestcase.params_id)\
                        .limit(BATCH_SIZE)
            if optimizations is not None:
                todo = todo.join(cldriveParams)\
                            .filter(cldriveParams.optimizations == optimizations)

            for testcase in todo:
                inbox.append(testcase)

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
                testcase = inbox.popleft()

                # drive the testcase
                try:
                    runtime, status, stdout, stderr = drive_testcase(
                        session, testcase, params, env, platform_id, device_id)

                    # assert that executed params match expected
                    if stderr != '<-- UTF-ERROR -->':
                        verify_params(platform=args.platform, device=args.device,
                                      optimizations=params.optimizations,
                                      global_size=params.gsize, local_size=params.lsize,
                                      stderr=stderr)

                    # create new result
                    stdout_ = util.escape_stdout(stdout)
                    stdout = get_or_create(
                        s, CLgenStdout,
                        hash=crypto.sha1_str(stdout_), stdout=stdout_)

                    stderr_ = util.escape_stderr(stderr)
                    stderr = get_or_create(
                        s, CLgenStderr,
                        hash=crypto.sha1_str(stderr_), stderr=stderr_)

                    result = CLgenResult(
                        testbed_id=testbed.id,
                        testcase_id=testcase.id,
                        status=status,
                        runtime=runtime,
                        stdout_id=stdout.id,
                        stderr_id=stderr.id)
                    result.outcome = OUTCOMES_TO_INT[analyze.get_cldrive_outcome(result)]
                    outbox.append(result)

                    # update progress bar
                    num_ran += 1
                    bar.update(min(num_ran, num_to_run))
                except clgen_mkharness.HarnessCompilationError:
                    print("program:", program.id)
        finally:
            # flush any remaining results
            next_batch()

    print("done.")
