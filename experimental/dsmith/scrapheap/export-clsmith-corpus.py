#!/usr/bin/env python
from argparse import ArgumentParser

import dsmith
import sqlalchemy as sql
from dsmith import db
from dsmith.db import *
from progressbar import ProgressBar

from labm8.py import fs

__description__ = """ Export a CLSmith program codes for assembling CLgen
training corpuses. This requires inlining CLSmith headers.
"""


def main():
  parser = ArgumentParser(description=__description__)
  parser.add_argument("classification")
  parser.add_argument("outdir")
  args = parser.parse_args()

  db.init("cc1")
  session = db.make_session()

  program_ids = [
    x[0] for x in session.query(sql.distinct(CLSmithResult.program_id)) \
      .filter(CLSmithResult.classification == args.classification).all()]

  header = fs.Read(dsmith.data_path("include", "clsmith.h"))

  fs.mkdir(args.outdir)

  for program_id in ProgressBar()(program_ids):
    outpath = fs.path(args.outdir, program_id + ".cl")

    if not fs.exists(outpath):
      program = session.query(CLSmithProgram) \
        .filter(CLSmithProgram.id == program_id).one()

      pre, post = program.src.split('#include "CLSmith.h"')

      inlined = pre + header + post

      with open(outpath, "w") as outfile:
        print(inlined, file=outfile)


if __name__ == "__main__":
  main()
