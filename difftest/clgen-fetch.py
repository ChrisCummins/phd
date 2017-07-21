#!/usr/bin/env python3
import cldrive
import os
from argparse import ArgumentParser
from pathlib import Path

import sqlalchemy as sql
from labm8 import fs
from progressbar import ProgressBar

import clgen_mkharness
import db
from db import CLgenProgram, Session


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("directory", help="directory containing kernels")
    parser.add_argument("--model", help="model ID")
    parser.add_argument("--sampler", help="sampler ID")
    parser.add_argument("--version", help="clgen version")
    parser.add_argument("--status", type=int, help="preprocessed status")
    parser.add_argument("--cl_launchable", action="store_true",
                        help="kernels have signature '__kernel void entry(...)'")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="max programs to generate, no max if < 0")
    parser.add_argument("--no-harness", action="store_true",
                        help="don't generate cldrive harnesses")
    args = parser.parse_args()

    db.init(args.hostname)

    # get a list of files to import
    paths = [x for x in Path(args.directory).iterdir() if x.is_file()]

    if args.num > 1:  # limit number of imports if user requested
        paths = paths[:args.num]

    with Session() as session:
        # Environment and params for mkharness()
        env = cldrive.make_env()
        params = s.query(cldriveParams).all()

        for path in ProgressBar()(paths):
            kid = os.path.splitext(path.name)[0]
            assert(len(kid) == 40)

            src = fs.read_file(path)

            exists = session.query(CLgenProgram).filter(
                    sql.or_(CLgenProgram.id == kid, CLgenProgram.src == src)
                ).count()

            if not exists:
                program = CLgenProgram(
                    id=kid, clgen_version=args.version, model=args.model,
                    sampler=args.sampler, src=src,
                    status=args.status, cl_launchable=args.cl_launchable)
                session.add(program)
                session.commit()

                # Make test harnesses, if required
                if not args.no_harness:
                    for param in params:
                        clgen_mkharness.mkharness(session, env, program, param)

    print("done.")
