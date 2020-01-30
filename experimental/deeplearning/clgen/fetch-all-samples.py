#!/usr/bin/env python
#
# Copyright 2016-2020 Chris Cummins <chrisc.101@gmail.com>.
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
from clgen import cli
from clgen import dbutil
from clgen import explore
from clgen import log

from labm8.py import fs

__description__ = """
Merge kernel datasets.
"""


def get_all_sampler_datasets():
  datasets = []
  sampledirs = []
  for versioncache in fs.ls(fs.path("~/.cache/clgen"), abspaths=True):
    samplerdir = fs.path(versioncache, "sampler")
    if fs.isdir(samplerdir):
      sampledirs += fs.ls(samplerdir, abspaths=True)

  for samplerdir in sampledirs:
    inpath = fs.path(samplerdir, "kernels.db")
    if fs.isfile(inpath):
      datasets.append(inpath)
  return datasets


def merge(outpath, inpaths=[]):
  if not fs.isfile(outpath):
    dbutil.create_db(outpath)
    log.info("created", outpath)

  db = dbutil.connect(outpath)

  if not inpaths:
    inpaths = get_all_sampler_datasets()

  for inpath in inpaths:
    log.info("merging from", inpath)
    c = db.cursor()
    c.execute("ATTACH '{}' AS rhs".format(inpath))
    c.execute(
      "INSERT OR IGNORE INTO ContentFiles " "SELECT * FROM rhs.ContentFiles"
    )
    c.execute(
      "INSERT OR IGNORE INTO PreprocessedFiles "
      "SELECT * FROM rhs.PreprocessedFiles"
    )
    c.execute("DETACH rhs")
    db.commit()

  explore.explore(outpath)


def main():
  parser = cli.ArgumentParser(description=__description__)
  parser.add_argument("dataset", help="path to output dataset")
  parser.add_argument("inputs", nargs="*", help="path to input datasets")
  args = parser.parse_args()

  cli.main(merge, args.dataset, args.inputs)


if __name__ == "__main__":
  main()
