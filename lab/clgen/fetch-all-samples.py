#!/usr/bin/env python
#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
from labm8 import fs

import clgen
from clgen import cache
from clgen import cli
from clgen import dbutil
from clgen import log

__description__ = """
Merge all sampler files into a database.
"""

def merge_all_samplers(outpath):
    if not fs.isfile(outpath):
        dbutil.create_db(outpath)

    db = dbutil.connect(outpath)
    for samplerdir in fs.ls(fs.path(cache.ROOT, "sampler"), abspaths=True):
        inpath = fs.path(samplerdir, "kernels.db")
        if not fs.isfile(inpath):
            continue

        log.info("importing from", fs.basename(samplerdir))
        c = db.cursor()
        c.execute("ATTACH {} AS rhs".format(outpath))
        c.execute("INSERT OR IGNORE INTO PreprocessedFiles "
                  "SELECT * FROM rhs.PreprocessedFiles")
        db.commit()


def main():
    parser = cli.ArgumentParser(description=__description__)
    parser.add_argument("dataset", metavar="<dataset>",
                        help="path to output dataset")
    args = parser.parse_args()

    cli.main(merge_all_samplers, fs.path(args.dataset))


if __name__ == "__main__":
    main()
