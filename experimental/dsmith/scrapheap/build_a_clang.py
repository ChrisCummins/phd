#!/usr/bin/env python3
import subprocess
import sys
from argparse import ArgumentParser
from collections import deque
from tempfile import NamedTemporaryFile
from time import strftime
from time import time
from typing import Tuple
from typing import Union

from dsmith import db
from dsmith.db import *
from dsmith.lib import *
from progressbar import ProgressBar

from labm8.py import crypto
from labm8.py import fs


def get_num_programs_to_build(session: db.session_t, tables: Tableset,
                              clang: str):
  num_ran = session.query(sql.sql.func.count(tables.clangs.id)) \
    .filter(tables.clangs.clang == clang) \
    .scalar()
  total = session.query(sql.sql.func.count(tables.programs.id)) \
    .scalar()
  return num_ran, total


def create_stderr(s: session_t, tables: Tableset,
                  stderr: str) -> CLgenClangStderr:
  assertion_ = util.get_assertion(s, tables.clang_assertions, stderr)
  unreachable_ = util.get_unreachable(s, tables.clang_unreachables, stderr)
  terminate_ = util.get_terminate(s, tables.clang_terminates, stderr)

  errs = sum(1 if x else 0 for x in [assertion_, unreachable_, terminate_])

  if errs > 1:
    raise LookupError(f"Multiple errors types found in: {stderr}\n\n" +
                      f"Assertion: {assertion_}\n" +
                      f"Unreachable: {unreachable_}\n" +
                      f"Terminate: {terminate_}")

  stderr = tables.clang_stderrs(hash=hash_,
                                stderr=stderr,
                                assertion=assertion_,
                                unreachable=unreachable_,
                                terminate=terminate_)
  s.add(stderr)
  s.flush()
  return stderr


def build_with_clang(program: Union[CLgenProgram, CLSmithProgram],
                     clang: str) -> Tuple[int, float, str]:
  with NamedTemporaryFile(prefix='buildaclang-', delete=False) as tmpfile:
    src_path = tmpfile.name
  try:
    with open(src_path, "w") as outfile:
      print(program.src, file=outfile)

    cmd = ['timeout', '-s9', '60s', clang, '-cc1', '-xcl', src_path]

    start_time = time()
    process = subprocess.Popen(cmd,
                               universal_newlines=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    _, stderr = process.communicate()

    return process.returncode, time() - start_time, stderr.strip()

  finally:
    fs.rm(src_path)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-H",
                      "--hostname",
                      type=str,
                      default="cc1",
                      help="MySQL database hostname")
  parser.add_argument("clang", type=str, help="clang version")
  parser.add_argument("--clsmith",
                      action="store_true",
                      help="Only reduce CLSmith results")
  parser.add_argument("--clgen",
                      action="store_true",
                      help="Only reduce CLgen results")
  parser.add_argument("--recheck",
                      action="store_true",
                      help="Re-check existing errors")
  args = parser.parse_args()

  db.init(args.hostname)  # initialize db engine

  clang = fs.abspath(f"../lib/llvm/build/{args.clang}/bin/clang")

  if not args.recheck and not fs.isfile(clang):
    print(f"fatal: clang '{clang}' does not exist")
    sys.exit(1)

  if args.clgen and args.clsmith:
    tablesets = [CLSMITH_TABLES, CLGEN_TABLES]
  elif args.clsmith:
    tablesets = [CLSMITH_TABLES]
  elif args.clgen:
    tablesets = [CLGEN_TABLES]
  else:
    tablesets = [CLSMITH_TABLES, CLGEN_TABLES]

  with Session(commit=True) as s:

    def next_batch():
      """
      Fill the inbox with jobs to run.
      """
      BATCH_SIZE = 1000
      print(f"\nnext {tables.name} batch for clang {args.clang} at",
            strftime("%H:%M:%S"))
      # update the counters
      num_ran, num_to_run = get_num_programs_to_build(s, tables, args.clang)
      bar.max_value = num_to_run
      bar.update(min(num_ran, num_to_run))

      # fill inbox
      done = s.query(tables.clangs.program_id) \
        .filter(tables.clangs.clang == args.clang)
      todo = s.query(tables.programs) \
        .filter(~tables.programs.id.in_(done)) \
        .order_by(tables.programs.date_added) \
        .limit(BATCH_SIZE)

      for program in todo:
        inbox.append(program)

    for tables in tablesets:
      if args.recheck:
        q = s.query(tables.clang_stderrs)
        for stderr in ProgressBar(max_value=q.count())(q):
          assertion_ = util.get_assertion(s, tables.clang_assertions,
                                          stderr.stderr)
          unreachable_ = util.get_unreachable(s, tables.clang_unreachables,
                                              stderr.stderr)
          terminate_ = util.get_terminate(s, tables.clang_terminates,
                                          stderr.stderr)

          errs = sum(
              1 if x else 0 for x in [assertion_, unreachable_, terminate_])
          if errs > 1:
            raise LookupError(f"Multiple errors types found in: {stderr}\n\n" +
                              f"Assertion: {assertion_}\n" +
                              f"Unreachable: {unreachable_}\n" +
                              f"Terminate: {terminate_}")

          if assertion_ != stderr.assertion:
            print("updating assertion")
            stderr.assertion = assertion_
          if unreachable_ != stderr.unreachable:
            print("updating unreachable")
            stderr.unreachable = unreachable_
          if terminate_ != stderr.terminate:
            print("updating terminate")
            stderr.terminate = terminate_
      else:
        # progress bar
        num_ran, num_to_run = get_num_programs_to_build(s, tables, clang)
        bar = ProgressBar(init_value=num_ran, max_value=num_to_run)

        # testcases to run
        inbox = deque()

        while True:
          # get the next batch of programs to run
          if not len(inbox):
            next_batch()
          # we have no programs to run
          if not len(inbox):
            break

          # get next program to run
          program = inbox.popleft()

          status, runtime, stderr_ = build_with_clang(program, clang)

          # create new result
          hash_ = crypto.sha1_str(stderr_)
          q = s.query(tables.clang_stderrs.id) \
            .filter(tables.clang_stderrs.hash == hash_) \
            .first()

          if q:
            stderr_id = q[0]
          else:
            stderr_id = create_stderr(s, tables, stderr_).id

          result = tables.clangs(program_id=program.id,
                                 clang=args.clang,
                                 status=status,
                                 runtime=runtime,
                                 stderr_id=stderr_id)

          s.add(result)
          s.commit()

          # update progress bar
          num_ran += 1
          bar.update(min(num_ran, num_to_run))
  print("done.")
