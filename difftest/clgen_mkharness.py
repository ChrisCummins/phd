#!/usr/bin/env python3
"""
Create cldrive test harnesses.
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

import db
from db import *

def mkharness(s, env: cldrive.OpenCLEnvironment, program: db.CLgenProgram,
              params: db.cldriveParams):
    """ generate a self-contained C program for the given test case """
    # return cached harness if one exists
    harness = s.query(CLgenHarness).filter(
        CLgenHarness.program_id == program.id,
        CLgenHarness.params_id == params.id).first()
    if harness:
        return harness

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
    except cldrive.OpenCLValueError as e:
        try:
            # could not generate the full test harness, try and create a kernel
            start_time = time()
            compile_only = True
            src = cldrive.emit_c(
                env, src=program.src, inputs=[], gsize=gsize, lsize=lsize,
                optimizations=params.optimizations, compile_only=True,
                create_kernel=True)
        except cldrive.OpenCLValueError as e:
            # could not create a kernel, so create a compile-only stub
            start_time = time()
            src = cldrive.emit_c(env, src=program.src, inputs=[], gsize=gsize, lsize=lsize,
                optimizations=params.optimizations, compile_only=True,
                create_kernel=False)

    generation_time = time() - start_time

    start_time = time()
    proc = subprocess.Popen(['gcc', '-xc', '-', '-lOpenCL'], stdin=subprocess.PIPE)
    proc.communicate(src.encode('utf-8'))
    if not proc.returncode == 0:
        print(src)
        print(proc.stderr.decode('utf-8'))
        raise ValueError('harness compilation failed')
    compile_time = time() - start_time

    harness = CLgenHarness(
        program_id=program.id,
        params_id=params.id,
        cldrive_version=cldrive.__version__,
        src=src,
        compile_only=compile_only,
        generation_time=generation_time,
        compile_time=compile_time)

    s.add(harness)
    s.commit()
    return harness


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    args = parser.parse_args()

    db.init(args.hostname)

    env = cldrive.make_env()

    with Session() as s:
        programs = s.query(CLgenProgram).all()
        params = s.query(cldriveParams).all()

        todo = list(product(programs, params))

        for program, params in ProgressBar()(todo):
            mkharness(s, env, program, params)

    print("done.")
