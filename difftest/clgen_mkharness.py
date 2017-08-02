#!/usr/bin/env python3
"""
Create test harnesses for CLgen programs using cldrive.
"""
import cldrive
import clgen
import os
import sys
import sqlalchemy as sql
import subprocess

from itertools import product
from argparse import ArgumentParser
from labm8 import fs
from pathlib import Path
from progressbar import ProgressBar
from time import time
from typing import List
from tempfile import NamedTemporaryFile

import db
import clgen_mkharness
from db import *


class HarnessCompilationError(ValueError):
    pass


def mkharness(s, env: cldrive.OpenCLEnvironment, testcase: CLgenTestCase):
    """ generate a self-contained C program for the given test case """
    if testcase.harness:
        return testcase.harness[0]

    data_generator = cldrive.Generator.from_str(params.generator)
    gsize = cldrive.NDRange(params.gsize_x, params.gsize_y, params.gsize_z)
    lsize = cldrive.NDRange(params.lsize_x, params.lsize_y, params.lsize_z)
    compile_only = False

    try:
        # generate the full test harness
        start_time = time()
        inputs = cldrive.make_data(
            src=program.src, size=params.size,
            data_generator=data_generator, scalar_val=params.scalar_val)

        src = cldrive.emit_c(
            env, src=program.src, inputs=inputs, gsize=gsize, lsize=lsize,
            optimizations=params.optimizations)
    except Exception:
        try:
            # could not generate the full test harness, try and create a kernel
            start_time = time()
            compile_only = True
            src = cldrive.emit_c(
                env, src=program.src, inputs=[], gsize=gsize, lsize=lsize,
                optimizations=params.optimizations, compile_only=True,
                create_kernel=True)
        except Exception:
            # could not create a kernel, so create a compile-only stub
            start_time = time()
            src = cldrive.emit_c(env, src=program.src, inputs=[], gsize=gsize, lsize=lsize,
                optimizations=params.optimizations, compile_only=True,
                create_kernel=False)

    generation_time = time() - start_time

    try:
        with NamedTemporaryFile(prefix='cldrive-harness-') as tmpfile:
            start_time = time()
            compile_harness(src, tmpfile.name)
            compile_time = time() - start_time

        harness = CLgenHarness(
            id=testcase.id,
            cldrive_version=cldrive.__version__,
            src=src,
            compile_only=compile_only,
            generation_time=generation_time,
            compile_time=compile_time)

        s.add(harness)
        s.commit()

        return harness
    except ValueError:
        print("\nharness compilation failed!", file=sys.stderr)
        print("program:", program.id, file=sys.stderr)
        print("params:", params.id, file=sys.stderr)
        print(src, file=sys.stderr)


def compile_harness(src: str, path: str='a.out', platform_id=None,
                    device_id=None, cc: str='gcc',
                    flags: List[str]=["-std=c99", "-Wno-deprecated-declarations", "-lOpenCL"],
                    timeout: int=60) -> None:
    """ compile harness binary from source """
    cmd = ['timeout', '-s9', str(timeout), cc, '-xc', '-', '-o', str(path)] + flags
    if platform_id is not None:
        cmd.append(f'-DPLATFORM_ID={platform_id}')
    if device_id is not None:
        cmd.append(f'-DDEVICE_ID={device_id}')

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(src.encode('utf-8'))
    if not proc.returncode == 0:
        raise HarnessCompilationError(f'harness compilation failed with returncode {proc.returncode}')
    return path


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-p", "--program-id", type=int,
                        help="Program ID to generate test harnesses for")
    args = parser.parse_args()

    db.init(args.hostname)

    env = cldrive.make_env()

    with Session() as s:
        if args.program_id:
            q = s.query(CLgenTestCase)\
                    .filter(CLgenTestCase.program_id == args.program_id)
        else:
            done = s.query(CLgenHarness.id)
            q = s.query(CLgenTestCase).filter(~CLgenTestCase.id.in_(done))

        for testcase in ProgressBar(max_value=q.count())(q):
            mkharness(s, env, testcase)

    print("done.")
