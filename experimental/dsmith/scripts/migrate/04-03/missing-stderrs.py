#!/usr/bin/env python3
"""
Create test harnesses for CLgen programs using cldrive.
"""
import json
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import db
import sqlalchemy as sql
from db import *
from progressbar import ProgressBar

from labm8.py import crypto
from labm8.py import fs

if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  parser.add_argument(
    "--commit", action="store_true", help="Commit changes (default is dry-run)"
  )
  args = parser.parse_args()

  db.init(args.hostname)

  to_del = deque()

  with Session(commit=False) as s:

    def flush():
      if args.commit:
        s.commit()
        while len(to_del):
          fs.rm(to_del.popleft())

    print("Importing CLgen programs ...")
    paths = [p for p in Path("export/clgen/program").iterdir()]
    for i, path in enumerate(ProgressBar()(paths)):
      with open(path) as infile:
        data = json.loads(infile.read())

      new_id = (
        s.query(CLgenProgram.id)
        .filter(CLgenProgram.hash == crypto.sha1_str(data["src"]))
        .scalar()
      )

      idx = CLgenProgramTranslation(old_id=data["id"], new_id=new_id)
      s.add(idx)

      to_del.append(path)
      if i and not i % 1000:
        flush()
    flush()

    PROGRAMS = dict(
      (old_id, new_id)
      for old_id, new_id in s.query(
        CLgenProgramTranslation.old_id, CLgenProgramTranslation.new_id
      ).all()
    )

    print("Import CLgen results ...")
    paths = [p for p in Path("export/clgen/result").iterdir()]

    STDERR_MISSING_ID = 645948

    num_todo = (
      s.query(sql.sql.func.count(CLgenResult.id))
      .filter(CLgenResult.stderr_id == STDERR_MISSING_ID)
      .scalar()
    )
    num_done = 0

    bar = ProgressBar(max_value=num_todo)

    for path in paths:
      with open(path) as infile:
        data = json.loads(infile.read())

      program_id = PROGRAMS[data["program"]]

      testcase = (
        s.query(CLgenTestCase)
        .filter(
          CLgenTestCase.program_id == program_id,
          CLgenTestCase.params_id == data["params"],
        )
        .scalar()
      )

      testbed_id = data["testbed"]

      result = (
        s.query(CLgenResult)
        .filter(
          CLgenResult.testbed_id == testbed_id,
          CLgenResult.testcase_id == testcase.id,
          CLgenResult.stderr_id == STDERR_MISSING_ID,
        )
        .first()
      )

      to_del.append(path)

      if result:
        stderr_ = util.escape_stderr(data["stderr"])
        stderr = get_or_create(
          s, CLgenStderr, hash=crypto.sha1_str(stderr_), stderr=stderr_
        )
        s.flush()
        assert isinstance(stderr.id, int)
        result.stderr_id = stderr.id

        num_done += 1
        bar.update(num_done)

        if not num_done % 1000:
          flush()

        if num_done == num_todo:
          break
    flush()

  print("done.")
