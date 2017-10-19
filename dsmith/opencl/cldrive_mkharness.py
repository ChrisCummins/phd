#!/usr/bin/env python3
"""
Create test harnesses for cldrive programs.
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

import dsmith
from dsmith import db
from dsmith.db import *


class HarnessCompilationError(ValueError):
    pass


harness_t = namedtuple('harness_t', ['generation_time', 'compile_only', 'src'])
default_cflags = ["-std=c99", "-Wno-deprecated-declarations", "-lOpenCL"]


def mkharness(testcase: Testcase) -> harness_t:
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


def compile_harness(src: str, path: str='a.out', platform_id=None,
                    device_id=None, cc: str='gcc',
                    flags: List[str]=default_cflags,
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
        raise HarnessCompilationError(
            f'harness compilation failed with returncode {proc.returncode}')
    return path
