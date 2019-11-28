#!/usr/bin/env python3
"""
Create test harnesses for CLgen programs using cldrive.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import analyze
import db
from db import *
from progressbar import ProgressBar

from labm8.py import fs

OUTCOMES = {
  "bf": 1,
  "bc": 2,
  "bto": 3,
  "c": 4,
  "to": 5,
  "pass": 6,
  None: None,
}

CLASSIFICATIONS = {
  "w": 1,
  "bf": 2,
  "c": 3,
  "to": 4,
  "pass": 5,
  None: None,
}

if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  args = parser.parse_args()

  db.init(args.hostname)

  with Session(commit=False) as s:
    # Export results
    #
    print("Exporting CLgen results ...")
    fs.mkdir("export/clgen/result")

    # Pick up where we left off
    done = set(
      [int(fs.basename(path)) for path in Path("export/clgen/result").iterdir()]
    )
    print(len(done), "done")
    ids = set([x[0] for x in s.query(CLgenResult.id).all()])
    print(len(ids), "in total")
    todo = ids - done
    print(len(todo), "todo")

    for result_id in ProgressBar()(todo):
      result = s.query(CLgenResult).filter(CLgenResult.id == result_id).scalar()

      with open(f"export/clgen/result/{result.id}", "w") as outfile:
        print(
          json.dumps(
            {
              "id": result.id,
              "testbed": result.testbed_id,
              "program": result.program_id,
              "params": result.params_id,
              "date": result.date.isoformat(),
              "status": result.status,
              "runtime": result.runtime,
              "stdout": result.stdout,
              "stderr": result.stderr,
              "outcome": analyze.get_cldrive_outcome(result),
            }
          ),
          file=outfile,
        )

    # Export harnesses
    #
    print("Exporting CLgen harnesses ...")
    fs.mkdir("export/clgen/harness")

    # Pick up where we left off
    done = set(
      [
        int(fs.basename(path))
        for path in Path("export/clgen/harness").iterdir()
      ]
    )
    print(len(done), "done")
    ids = set([x[0] for x in s.query(CLgenHarness.id)])
    print(len(ids), "in total")
    todo = ids - done
    print(len(todo), "todo")

    for harness_id in ProgressBar()(todo):
      harness == s.query(CLgenHarness).filter(
        CLgenHarness.id == harness_id
      ).scalar()

      with open(f"export/harness/{harness.id}", "w") as outfile:
        print(
          json.dumps(
            {
              "id": harness.id,
              "program": harness.program.id,
              "params": harness.params_id,
              "date": harness.date.isoformat(),
              "cldrive": harness.cldrive_version,
              "src": harness.src,
              "compile_only": harness.compile_only,
              "generation_time": harness.generation_time,
              "compile_time": harness.compile_time,
              "gpuverified": harness.gpuverified,
              "oclverified": harness.oclverified,
            }
          ),
          file=outfile,
        )

    # Export programs
    #
    print("Exporting CLgen programs ...")
    fs.mkdir("export/clgen/program")

    # Pick up where we left off
    done = set(
      [fs.basename(path) for path in Path("export/clgen/program").iterdir()]
    )
    print(len(done), "done")
    ids = set([x[0] for x in s.query(CLgenProgram.id)])
    print(len(ids), "in total")
    todo = ids - done
    print(len(todo), "todo")

    for program_id in ProgressBar()(todo):
      program = (
        s.query(CLgenProgram).filter(CLgenProgram.id == program_id).scalar()
      )

      with open(f"export/program/{program.id}", "w") as outfile:
        print(
          json.dumps(
            {
              "id": program.id,
              "date_added": program.date_added.isoformat(),
              "clgen": program.clgen_version,
              "src": program.src,
              "cl_launchable": program.cl_launchable,
              "gpuverified": program.gpuverified,
              "throws_warnings": program.throws_warnings,
            }
          ),
          file=outfile,
        )

    # Export programs
    #
    print("Exporting CLSmith programs ...")
    fs.mkdir("export/clsmith/program")

    # Pick up where we left off
    done = set(
      [fs.basename(path) for path in Path("export/clsmith/program").iterdir()]
    )
    print(len(done), "done")
    ids = set([x[0] for x in s.query(CLSmithProgram.id)])
    print(len(ids), "in total")
    todo = ids - done
    print(len(todo), "todo")

    for program_id in ProgressBar()(todo):
      program = (
        s.query(CLSmithProgram).filter(CLSmithProgram.id == program_id).scalar()
      )

      with open(f"export/clsmith/program/{program.id}", "w") as outfile:
        print(
          json.dumps(
            {
              "id": program.id,
              "date": program.date.isoformat(),
              "flags": program.flags,
              "runtime": program.runtime,
              "src": program.src,
            }
          ),
          file=outfile,
        )

    # Export results
    #
    print("Exporting CLSmith results ...")
    fs.mkdir("export/clsmith/result")

    # Pick up where we left off
    done = set(
      [
        int(fs.basename(path))
        for path in Path("export/clsmith/result").iterdir()
      ]
    )
    print(len(done), "done")
    ids = set([x[0] for x in s.query(CLSmithResult.id)])
    print(len(ids), "in total")
    todo = ids - done
    print(len(todo), "todo")

    for result_id in ProgressBar()(todo):
      result = (
        s.query(CLSmithResult).filter(CLSmithResult.id == result_id).scalar()
      )

      with open(f"export/clsmith/result/{result.id}", "w") as outfile:
        print(
          json.dumps(
            {
              "id": result.id,
              "testbed": result.testbed_id,
              "program": result.program_id,
              "params": result.params_id,
              "date": result.date.isoformat(),
              "status": result.status,
              "runtime": result.runtime,
              "stdout": result.stdout,
              "stderr": result.stderr,
              "outcome": OUTCOMES[result.outcome],
              "classification": CLASSIFICATIONS[result.classification],
            }
          ),
          file=outfile,
        )

  print("done.")
