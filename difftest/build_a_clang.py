#!/usr/bin/env python3
import re
import fileinput
import os
import sys
import pyopencl as cl
import cldrive
import progressbar
import subprocess
from argparse import ArgumentParser
from collections import deque, namedtuple
from subprocess import Popen, PIPE
from labm8 import fs, crypto
from time import time, strftime
from typing import Dict, List, Tuple, NewType, Union
from tempfile import NamedTemporaryFile, TemporaryDirectory
from progressbar import ProgressBar

import clgen_mkharness
import analyze
import db
import util
from db import *
from lib import *


def get_num_programs_to_build(session: db.session_t, tables: Tableset, clang: str):
    num_ran = session.query(sql.sql.func.count(tables.clangs.id))\
        .filter(tables.clangs.clang == clang)\
        .scalar()
    total = session.query(sql.sql.func.count(tables.programs.id))\
        .scalar()
    return num_ran, total


def build_with_clang(program: Union[CLgenProgram, CLSmithProgram],
                     clang: str, clang_include: str) -> Tuple[int, float, str]:
    with NamedTemporaryFile(prefix='buildaclang-', delete=False) as tmpfile:
        src_path = tmpfile.name
    try:
        with open(src_path, "w") as outfile:
            print(program.src, file=outfile)

        cmd = ['timeout', '-s9', '60s', clang, '-cc1', '-xcl',
               '-I', clang_include, '-finclude-default-header', src_path]

        start_time = time()
        process = subprocess.Popen(cmd, universal_newlines=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()

        return process.returncode, time() - start_time, stderr.strip()

    finally:
        fs.rm(src_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-H", "--hostname", type=str, default="cc1",
        help="MySQL database hostname")
    parser.add_argument("clang", type=str, help="clang version")
    parser.add_argument("--clsmith", action="store_true",
                        help="Only reduce CLSmith results")
    parser.add_argument("--clgen", action="store_true",
                        help="Only reduce CLgen results")
    args = parser.parse_args()

    db.init(args.hostname)  # initialize db engine

    clang = fs.abspath(f"../lib/llvm/build/{args.clang}/bin/clang")
    clang_include = fs.abspath(f"../lib/llvm/build/{args.clang}/lib/clang/{args.clang}/include")

    if not fs.isfile(clang):
        print(f"fatal: clang '{clang}' does not exist")
        sys.exit(1)
    if not fs.isdir(clang_include):
        print(f"fatal: include dir '{clang_include}' does not exist")
        sys.exit(1)

    if args.clgen and args.clsmith:
        tablesets = [CLSMITH_TABLES, CLGEN_TABLES]
    elif args.clsmith:
        tablesets = [CLSMITH_TABLES]
    elif args.clgen:
        tablesets = [CLGEN_TABLES]
    else:
        tablesets = [CLSMITH_TABLES, CLGEN_TABLES]

    with Session(commit=True) as s:

        def next_batch():
            """
            Fill the inbox with jobs to run.
            """
            BATCH_SIZE = 100
            print(f"\nnext {tables.name} batch for clang {args.clang} at", strftime("%H:%M:%S"))
            # update the counters
            num_ran, num_to_run = get_num_programs_to_build(s, tables, args.clang)
            bar.max_value = num_to_run
            bar.update(min(num_ran, num_to_run))

            # fill inbox
            done = s.query(tables.clangs.program_id)\
                .filter(tables.clangs.clang == args.clang)
            todo = s.query(tables.programs)\
                .filter(~tables.programs.id.in_(done))\
                .order_by(tables.programs.date_added)\
                .limit(BATCH_SIZE)

            for program in todo:
                inbox.append(program)

        for tables in tablesets:
            # progress bar
            num_ran, num_to_run = get_num_programs_to_build(s, tables, clang)
            bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

            # testcases to run
            inbox = deque()

            while True:
                # get the next batch of programs to run
                if not len(inbox):
                    next_batch()
                # we have no programs to run
                if not len(inbox):
                    break

                # get next program to run
                program = inbox.popleft()

                status, runtime, stderr = build_with_clang(program, clang, clang_include)

                # create new result
                stderr_ = util.escape_stderr(stderr)
                stderr = get_or_create(
                    s, tables.clang_stderrs,
                    hash=crypto.sha1_str(stderr_), stderr=stderr_)
                s.flush()

                result = tables.clangs(
                    program_id=program.id,
                    clang=args.clang,
                    status=status,
                    runtime=runtime,
                    stderr_id=stderr.id)

                s.add(result)
                s.commit()

                # update progress bar
                num_ran += 1
                bar.update(min(num_ran, num_to_run))
    print("done.")
