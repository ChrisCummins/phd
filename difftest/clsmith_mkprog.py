#!/usr/bin/env python3
import progressbar
import sqlalchemy as sql

from argparse import ArgumentParser
from labm8 import crypto
from labm8 import fs
from tempfile import NamedTemporaryFile

import clsmith
import db
from db import *


def get_num_progs(session: session_t) -> int:
    return session.query(sql.sql.func.count(CLSmithProgram)).scalar()


def make_program(session: session_t, *flags) -> None:
    """
    Arguments:
        *flags: Additional flags to CLSmith.
    """
    with NamedTemporaryFile(prefix='clsmith-', suffix='.cl') as tmp:
        runtime, status, stdout, stderr = clsmith.clsmith('-o', tmp.name, *flags)

        # A non-zero exit status of clsmith implies that no program was
        # generated.
        if status:
            make_program(*flags)

        src = fs.read_file(tmp.name)
        hash_ = crypto.sha1_str(src)
        dupe = session.query(CLSmithProgram.id)\
            .filter(CLSmithProgram.hash == hash_).first()

        if not dupe:
            program = CLSmithProgram(
                hash=hash_,
                flags=" ".join(flags),
                runtime=runtime,
                src=src,
                linecount=len(src.split('\n')))
            session.add(program)
            session.commit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="max programs to generate, no max if < 0")
    args = parser.parse_args()

    target_num_progs = args.num

    db.init(args.hostname)  # initialize db engine

    with Session() as s:
        numprogs = get_num_progs(s)
        if target_num_progs > 0:
            bar = progressbar.ProgressBar(
                initial_value=numprogs, max_value=target_num_progs)
        else:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

        while target_num_progs < 0 or numprogs < target_num_progs:
            make_program(s)
            numprogs = get_num_progs(s)
            bar.update(numprogs)

    print("done.")
