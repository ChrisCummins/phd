#!/usr/bin/env python3
import progressbar
import sqlalchemy as sql

from argparse import ArgumentParser
from labm8 import crypto
from labm8 import fs
from tempfile import NamedTemporaryFile

import clsmith
import db

from db import CLSmithProgram, Session


def get_num_progs() -> int:
    with db.Session() as session:
        return session.query(db.CLSmithProgram).count()


def make_program(*flags) -> None:
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

        file_id = crypto.sha1_file(tmp.name)
        src = fs.read_file(tmp.name)

        # insert program into the table. If it already exists, ignore it.
        try:
            with Session() as session:
                program = CLSmithProgram(
                    id=file_id, flags=" ".join(flags), runtime=runtime,
                    stdout=stdout, stderr=stderr, src=src)
                session.add(program)
        except sql.exc.IntegrityError:
            # duplicate program already exists
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="max programs to generate, no max if < 0")
    args = parser.parse_args()

    target_num_progs = args.num

    db.init(args.hostname)  # initialize db engine

    numprogs = get_num_progs()
    if target_num_progs > 0:
        bar = progressbar.ProgressBar(initial_value=numprogs,
                                      max_value=target_num_progs)
    else:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    while target_num_progs < 0 or numprogs < target_num_progs:
        make_program()# '--small'
        numprogs = get_num_progs()
        bar.update(numprogs)

    print("done.")
