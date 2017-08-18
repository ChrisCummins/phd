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

from collections import namedtuple
from itertools import product
from argparse import ArgumentParser
from labm8 import fs
from pathlib import Path
from progressbar import ProgressBar
from time import time
from typing import List, Tuple
from tempfile import NamedTemporaryFile

import db
from db import *


class HarnessCompilationError(ValueError):
    pass


harness_t = namedtuple('harness_t', ['generation_time', 'compile_only', 'src'])


def mkharness_src(testcase: Testcase) -> harness_t:
    """ generate a self-contained C program for the given test case """
    program = testcase.program
    threads = testcase.threads

    gsize = cldrive.NDRange(threads.gsize_x, threads.gsize_y, threads.gsize_z)
    lsize = cldrive.NDRange(threads.lsize_x, threads.lsize_y, threads.lsize_z)
    size = max(gsize.product * 2, 256)
    compile_only = False

    try:
        # generate a compile-and-execute test harness
        start_time = time()
        src = cldrive.emit_c(
            src=program.src, size=size, start_at=1,# TODO: testcase.input_seed
            gsize=gsize, lsize=lsize,
            scalar_val=size)
    except Exception:
        # create a compile-only stub if not possible
        start_time = time()
        src = cldrive.emit_c(
            src=program.src, size=0, start_at=1,# TODO: testcase.input_seed
            gsize=gsize, lsize=lsize, compile_only=True)

    generation_time = time() - start_time

    return harness_t(generation_time, compile_only, src)


def mkharness(testcase: Testcase) -> harness_t:
    """ generate a self-contained C program for the given test case and add it to the database """
    generation_time, compile_only, src = mkharness_src(testcase)

    with NamedTemporaryFile(prefix='cldrive-harness-') as tmpfile:
        start_time = time()
        compile_harness(src, tmpfile.name)
        compile_time = time() - start_time

    return harness_t(generation_time, compile_only, src)


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
            q = s.query(Testcase)\
                    .filter(Testcase.program_id == args.program_id)
        else:
            #done = s.query(CLgenHarness.id)
            q = s.query(Testcase).filter(Testcase.harness == 1) #.filter(~Testcase.id.in_(done))

        for testcase in ProgressBar(max_value=q.count())(q):
            mkharness(testcase)

    print("done.")
