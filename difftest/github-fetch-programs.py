#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from pathlib import Path

import sqlalchemy as sql
from labm8 import fs
from progressbar import ProgressBar

import db
from db import GitHubProgram, Session


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("directory", help="directory containing kernels")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="max programs to generate, no max if < 0")
    args = parser.parse_args()

    db.init(args.hostname)

    # get a list of files to import
    paths = [x for x in Path(args.directory).iterdir() if x.is_file()]

    if args.num > 1:  # limit number of imports if user requested
        paths = paths[:args.num]

    with Session(commit=True) as session:
        for path in ProgressBar()(paths):
            kid = os.path.splitext(path.name)[0]  # strip file extension

            exists = session.query(GitHubProgram).filter(GitHubProgram.id == kid).count()

            if not exists:
                try:
                    p = GitHubProgram(
                        id=kid, src=fs.read_file(path))
                    session.add(p)
                    session.commit()
                except UnicodeError:
                    # at least one of the programs contains non-ASCII char
                    session.rollback()


    print("done.")
