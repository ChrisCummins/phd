#!/usr/bin/env python3
"""
Run GPUverify on CLgen programs.
"""
from argparse import ArgumentParser

import clgen
from dsmith import db
from dsmith.db import *
from progressbar import ProgressBar

if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  parser.add_argument(
    "-r",
    "--recheck",
    action="store_true",
    help="Re-run on previously verified programs",
  )
  args = parser.parse_args()

  db.init(args.hostname)
  session = db.make_session()

  q = session.query(CLgenProgram)
  if not args.recheck:
    q = q.filter(CLgenProgram.gpuverified == None)

  for program in ProgressBar()(q.all()):
    try:
      clgen.gpuverify(program.src, ["--local_size=64", "--num_groups=128"])
      program.gpuverified = 1
    except clgen.GPUVerifyException:
      program.gpuverified = 0

    session.commit()

  print("done.")
