#!/usr/bin/env python3
"""
Create test harnesses for CLgen programs using cldrive.
"""
from argparse import ArgumentParser
from collections import deque

import db
from db import *

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

    # print("Importing CLSmith programs ...")
    # paths = [x for x in Path("export/clsmith/program").iterdir()]
    # for i, path in enumerate(ProgressBar()(paths)):
    #     with open(path) as infile:
    #         data = json.loads(infile.read())

    #     program = CLSmithProgram(
    #         hash=crypto.sha1_str(data["src"]),
    #         date=datetime.datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%S"),
    #         flags=data["flags"],
    #         runtime=data["runtime"],
    #         src=data["src"],
    #         linecount=len(data["src"].split("\n")))
    #     s.add(program)
    #     s.flush()

    #     idx = CLSmithProgramTranslation(
    #         old_id=data["id"],
    #         new_id=program.id)
    #     s.add(idx)

    #     to_del.append(path)
    #     if i and not i % 1000:
    #         flush()
    # flush()

    # print("Import CLSmith results ...")
    # paths = [p for p in Path("export/clsmith/result").iterdir()]
    # for i, path in enumerate(ProgressBar()(paths)):
    #     with open(path) as infile:
    #         data = json.loads(infile.read())

    #     program_id = s.query(CLSmithProgramTranslation.new_id)\
    #         .filter(CLSmithProgramTranslation.old_id == data["program"]).scalar()

    #     testcase = get_or_create(
    #         s, CLSmithTestCase,
    #         program_id=program_id,
    #         params_id=data["params"])

    #     stdout_ = util.escape_stdout(data["stdout"])
    #     stdout = get_or_create(
    #         s, CLSmithStdout,
    #         hash=crypto.sha1_str(stdout_), stdout=stdout_)

    #     stderr_ = util.escape_stderr(data["stderr"])
    #     stderr = get_or_create(
    #         s, CLSmithStderr,
    #         hash=crypto.sha1_str(stderr_), stderr=stderr_)
    #     s.flush()

    #     testbed_id = data["testbed"]

    #     dupe = s.query(CLSmithResult.id)\
    #         .filter(CLSmithResult.testbed_id==testbed_id,
    #                 CLSmithResult.testcase_id==testcase.id).first()

    #     if dupe:
    #         print(f"\nwarning: ignoring duplicate CLSmith result {path}")
    #     else:
    #         result = get_or_create(
    #             s, CLSmithResult,
    #             testbed_id=testbed_id,
    #             testcase_id=testcase.id,
    #             date=datetime.datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%S"),
    #             status=data["status"],
    #             runtime=data["runtime"],
    #             stdout_id=stdout.id,
    #             stderr_id=stderr.id,
    #             outcome=data["outcome"])
    #         s.add(result)
    #         to_del.append(path)

    #     if i and not i % 1000:
    #         flush()
    # flush()

    # print("Importing CLgen programs ...")
    # paths = [p for p in Path("export/clgen/program").iterdir()]
    # for i, path in enumerate(ProgressBar()(paths)):
    #     with open(path) as infile:
    #         data = json.loads(infile.read())

    #     program = CLgenProgram(
    #         hash=crypto.sha1_str(data["src"]),
    #         date_added=datetime.datetime.strptime(data["date_added"], "%Y-%m-%dT%H:%M:%S"),
    #         runtime=len(data["src"]) / 465,
    #         src=data["src"],
    #         linecount=len(data["src"].split("\n")),
    #         cl_launchable=data["cl_launchable"],
    #         gpuverified=data["gpuverified"],
    #         throws_warnings=data["throws_warnings"])
    #     s.add(program)
    #     s.flush()

    #     idx = CLgenProgramTranslation(
    #         old_id=data["id"],
    #         new_id=program.id)
    #     s.add(idx)

    #     to_del.append(path)
    #     if i and not i % 1000:
    #         flush()
    # flush()

    # print("Import CLgen harnesses ...")
    # paths = [p for p in Path("export/clgen/harness").iterdir()]
    # for i, path in enumerate(ProgressBar()(paths)):
    #     with open(path) as infile:
    #         data = json.loads(infile.read())

    #     program_id = PROGRAMS[data["program"]]

    #     testcase = get_or_create(
    #         s, CLgenTestCase,
    #         program_id=program_id,
    #         params_id=data["params"])
    #     s.flush()

    #     dupe = s.query(CLgenHarness.id).filter(CLgenHarness.id == testcase.id).first()

    #     if dupe:
    #         print(f"\nwarning: ignoring duplicate CLgen harness {path}")
    #     else:
    #         harness = CLgenHarness(
    #             id=testcase.id,
    #             date=datetime.datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%S"),
    #             cldrive_version=data["cldrive"],
    #             src=data["src"],
    #             compile_only=data["compile_only"],
    #             generation_time=data["generation_time"],
    #             compile_time=data["compile_time"],
    #             gpuverified=data["gpuverified"],
    #             oclverified=data["oclverified"])
    #         s.add(harness)
    #         to_del.append(path)

    #     if i and not i % 1000:
    #         flush()
    # flush()

    # print("Import CLgen results ...")
    # # classifications = np.loadtxt(Path(fs.path("~/class.numbers")), delimiter='\n')
    # paths = [p for p in Path("export/clgen/result").iterdir()]
    # for i, path in enumerate(ProgressBar()(paths)):
    #     with open(path) as infile:
    #         data = json.loads(infile.read())

    #     program_id = PROGRAMS[data["program"]]

    #     testcase = get_or_create(
    #         s, CLgenTestCase,
    #         program_id=program_id,
    #         params_id=data["params"])
    #     s.flush()

    #     testbed_id = data["testbed"]
    #     # classification = classifications[data["id"]]

    #     dupe = s.query(CLgenResult.id)\
    #         .filter(CLgenResult.testbed_id==testbed_id,
    #                 CLgenResult.testcase_id==testcase.id).first()

    #     if dupe:
    #         print(f"\nwarning: ignoring duplicate CLgen result {path}")
    #     else:
    #         stdout_ = util.escape_stdout(data["stdout"])
    #         stdout = get_or_create(
    #             s, CLgenStdout,
    #             hash=crypto.sha1_str(stdout_), stdout=stdout_)

    #         stderr_ = util.escape_stderr(data["stderr"])
    #         stderr = get_or_create(
    #             s, CLgenStderr,
    #             hash=crypto.sha1_str(stderr_), stderr=stderr_)
    #         s.flush()

    #         result = CLgenResult(
    #             testbed_id=testbed_id,
    #             testcase_id=testcase.id,
    #             date=datetime.datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%S"),
    #             status=data["status"],
    #             runtime=data["runtime"],
    #             stdout_id=stdout.id,
    #             stderr_id=stderr.id,
    #             outcome=OUTCOMES_TO_INT[data["outcome"]])
    #         s.add(result)

    #     to_del.append(path)
    #     if i and not i % 1000:
    #         flush()
    # flush()

  print("done.")
