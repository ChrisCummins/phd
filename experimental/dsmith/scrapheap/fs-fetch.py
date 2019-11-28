#!/usr/bin/env python3
"""
Fetch OpenCL kernels from file system
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

from dsmith import db
from dsmith.db import *
from progressbar import ProgressBar

from labm8.py import fs

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-H",
                      "--hostname",
                      type=str,
                      default="cc1",
                      help="MySQL database hostname")
  parser.add_argument("classname", help="db.py table class: {GitHubProgram}")
  parser.add_argument("directory", help="directory containing kernels")
  parser.add_argument(
      "-n",
      "--num",
      type=int,
      default=-1,
      help="max programs to import, no max if < 0 (default: -1)")
  args = parser.parse_args()

  # get a list of files to import
  paths = [x for x in Path(args.directory).iterdir() if x.is_file()]
  shuffle(paths)

  if args.num > 1:  # limit number of imports if user requested
    paths = paths[:args.num]

  # here be dragons.
  Class = eval(args.classname)

  for path in ProgressBar()(paths):
    db.init(args.hostname)
    kid = os.path.splitext(path.name)[0]  # strip file extension

    try:
      with Session(commit=True) as session:
        exists = session.query(Class).filter(Class.id == kid).count()
        if not exists:
          p = Class(id=kid, src=fs.Read(path))
          session.add(p)
    except UnicodeError:
      # at least one of the programs contains non-ASCII char
      session.rollback()

  print("done.")
