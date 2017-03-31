#!/usr/bin/env python3
import progressbar
import sqlalchemy as sql

from argparse import ArgumentParser
from labm8 import crypto
from labm8 import fs
from tempfile import NamedTemporaryFile

import clsmith
import db


def get_num_progs() -> int:
    with db.Session() as session:
        return session.query(db.Program).count()


def make_program(*flags) -> None:
    """
    Arguments:
        *flags: Additional flags to CLSmith.
    """
    with NamedTemporaryFile(prefix='clsmith-', suffix='.c') as tmp:

        runtime, stdout, stderr = clsmith.clsmith('-o', tmp.name, *flags)
        file_id = crypto.sha1_file(tmp.name)
        src = fs.read_file(tmp.name)

        # insert program into the table. If it already exists, ignore it.
        try:
            with db.Session() as session:
                program = db.Program(
                    id=file_id, flags=" ".join(flags), runtime=runtime,
                    stdout=stdout, stderr=stderr, src=src)
                session.add(program)
        except sql.exc.IntegrityError:
            pass



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dbpath", metavar="<database>",
                        help="path to database")
    parser.add_argument("-n", "--num", type=int, default=1000)
    args = parser.parse_args()

    dbpath = fs.path(args.dbpath)
    target_num_progs = args.num

    db.init(dbpath)  # initialize db engine

    numprogs = get_num_progs()
    bar = progressbar.ProgressBar(initial_value=numprogs,
                                  max_value=target_num_progs)

    while numprogs < target_num_progs:
        make_program('--small')
        numprogs = get_num_progs()
        bar.update(numprogs)

    print("done.")
