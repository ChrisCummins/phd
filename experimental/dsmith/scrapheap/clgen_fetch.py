#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import cldrive
from dsmith import db
from dsmith.db import *
from progressbar import ProgressBar

from labm8.py import crypto
from labm8.py import fs

# Benchmarked CLgen inference rate (characters per second):
CLGEN_INFERENCE_CPS = 465


def import_clgen_sample(session: session_t,
                        path: Path,
                        cl_launchable: bool = False,
                        harnesses: List[cldriveParams] = [],
                        delete: bool = False) -> None:
  src = fs.Read(path)
  hash_ = crypto.sha1_str(src)

  dupe = s.query(CLgenProgram).filter(CLgenProgram.hash == hash_).first()

  if dupe:
    print(f"warning: ignoring duplicate file {path}")
  elif not len(src):
    print(f"warning: ignoring empty file {path}")
  else:
    program = CLgenProgram(hash=hash_,
                           runtime=len(src) / CLGEN_INFERENCE_CPS,
                           src=src,
                           linecount=len(src.split('\n')),
                           cl_launchable=cl_launchable)
    s.add(program)
    s.commit()

    # Make test harnesses, if required
    if harnesses:
      env = cldrive.make_env()
      for params in harnesses:
        testcase = get_or_create(s,
                                 CLgenTestCase,
                                 program_id=program.id,
                                 params_id=params.id)
        s.flush()
        clgen_mkharness.mkharness(s, env, testcase)

    if delete:
      fs.rm(path)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-H",
                      "--hostname",
                      type=str,
                      default="cc1",
                      help="MySQL database hostname")
  parser.add_argument("directory", help="directory containing kernels")
  parser.add_argument("--cl_launchable",
                      action="store_true",
                      help="kernels have signature '__kernel void entry(...)'")
  parser.add_argument("-n",
                      "--num",
                      type=int,
                      default=-1,
                      help="max programs to import, no max if < 0")
  parser.add_argument("--no-harness",
                      action="store_true",
                      help="don't generate cldrive harnesses")
  parser.add_argument("--delete",
                      action="store_true",
                      help="delete file after import")
  args = parser.parse_args()

  db.init(args.hostname)

  # get a list of files to import
  paths = [path for path in Path(args.directory).iterdir() if path.is_file()]

  if args.num > 1:  # limit number of imports if user requested
    paths = paths[:args.num]

  with Session() as s:
    params = [] if args.no_harness else s.query(cldriveParams).all()

    for path in ProgressBar()(paths):
      import_clgen_sample(s,
                          path,
                          harnesses=params,
                          cl_launchable=args.cl_launchable,
                          delete=args.delete)

  print("done.")
