#!/usr/bin/env python3
"""
The Reproducer-er (TM)

Reproduce suspicious results.

Usage: ./the-reproducerer

NOTE: requires running ./analyze.py first


What it does
------------
  Figure out what OpenCL devices we have on this system
  Fetch all suspicious entries from the DB for the available OpenCL devices
  For each suspicious entry:
    Attempt to reproduce it.
    If successful, generate C code for a standalone binary, and a bug report.
    Else, record a failure to reproduce.
    Additionally, warn if any experimental setup has changed
        (e.g. different software versions).
"""
import faulthandler

faulthandler.enable()

import sys

from argparse import ArgumentParser
from collections import Counter
from labm8.py import fs

from dsmith import db
from dsmith.db import *
from dsmith.lib import *


def reproduce_clgen_build_failures(result):
  import analyze
  import clgen_run_cldrive

  print(result)

  ### Reproduce using Python
  flags = result.params.to_flags()
  cli = cldrive_cli(result.testbed.platform, result.testbed.device, *flags)

  runtime, status, stdout, stderr = clgen_run_cldrive.drive(
    cli, result.program.src
  )

  new_result = CLgenResult(
    program=result.program,
    params=result.params,
    testbed=result.testbed,
    cli=" ".join(cli),
    status=status,
    runtime=runtime,
    stdout=stdout,
    stderr=stderr,
  )

  analyze.analyze_cldrive_result(new_result, CLgenResult, session)

  if new_result.classification != result.classification:
    print("could not reproduce result")
    sys.exit(1)

  print(">>>> Reproduced using cldrive")

  ### Reproduce using C standalone binary
  with open(result.program.id + ".cl", "w") as outfile:
    print(result.program.src, file=outfile)

  # TODO:
  # env = cldrive.OpenCLEnvironment(platform=result.testbed.platform,
  #                                 device=result.testbed.device)
  # src = cldrive.emit_c(env, result.program.src)
  # with open(result.program.id + '.c', 'w') as outfile:
  #     print(src, file=outfile)

  # # TODO: portable -I and -l flags
  # cli = ['gcc', '-xc', '-', '-lOpenCL']
  # process = Popen(cli, stdin=PIPE, stdout=PIPE, stderr=PIPE,
  #                 universal_newlines=True)
  # stdout, stderr = process.communicate(src)
  # print('stdout:', stdout.rstrip())
  # print('stderr:', stderr.rstrip())
  # print('status:', process.returncode)
  # if process.returncode:
  #     print("Failed to compile binary for", result.program.id)
  #     sys.exit(1)

  # cli = ['./a.out']
  # process = Popen(cli, stdout=PIPE, stderr=PIPE, universal_newlines=True)
  # stdout, stderr = process.communicate(src)
  # print('stdout:', stdout.rstrip())
  # print('stderr:', stderr.rstrip())
  # print('status:', process.returncode)

  # if process.returncode:
  #     print(">>> Reproduced using standalone binary")
  #     sys.exit(0)
  # else:
  #     print(">>> Failed to reproduce using standalone binary")


def generate_report_base(result):
  return f"""\
OpenCL Platform:   {result.testbed.platform}
OpenCL Device:     {result.testbed.device}
Driver version:    {result.testbed.driver}
OpenCL version:    {result.testbed.opencl}

Operating System:  {result.testbed.host}
"""


def generate_wrong_code_report(result):
  def summarize_stdout(stdout):
    components = [x for x in stdout.split(",") if x != ""]
    ncomponents = len(components)
    if len(set(components)) == 1:
      return f"'{components[0]},' x {ncomponents}"
    else:
      return stdout

  results = (
    session.query(CLSmithResult)
    .filter(
      CLSmithResult.program == result.program,
      CLSmithResult.params == result.params,
      CLSmithResult.status == 0,
    )
    .all()
  )

  if len(results) > 2:
    # Use voting to pick oracle.
    outputs = [r.stdout for r in results]
    majority_output, majority_count = Counter(outputs).most_common(1)[0]
    if majority_count == 1:  # no majority
      result.classification = "Wrong code"
    elif result.stdout != majority_output:
      result.classification = "Wrong code"
  elif len(results) == 2:
    if results[0].stdout != results[1].stdout:
      majority_count = 1
      result.classification = "Wrong code"
      majority_output = "[UNKNOWN]"
    else:
      majority_count = 2

  majority_devices = [r.testbed for r in results if r.stdout == majority_output]

  kernel_nlines = len(result.program.src.split("\n"))

  majority_str = "\n    - ".join(t.device for t in majority_devices)

  program_output = summarize_stdout(result.stdout)
  expected_output = summarize_stdout(majority_output)

  return (
    generate_report_base(result)
    + f"""
Global size:       {result.params.gsize}
Workgroup size:    {result.params.lsize}
Optimizations:     {result.params.optimizations_on_off}

OpenCL kernel:     {result.program.id} ({kernel_nlines} lines)
Program output:    {program_output}
Expected output:   {expected_output}

Majority devices:  {majority_count}
    - {majority_str}\
"""
  )


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  args = parser.parse_args()

  db.init(args.hostname)
  session = db.make_session()

  clsmith_wrong_code_programs = session.query(CLSmithResult).filter(
    CLSmithResult.classification == "w"
  )
  fs.mkdir("../data/difftest/unreduced/clsmith/wrong_code")
  fs.mkdir("../data/difftest/unreduced/clsmith/wrong_code/reports")
  for result in clsmith_wrong_code_programs:
    vendor = vendor_str(result.testbed.platform)

    with open(
      f"../data/difftest/unreduced/clsmith/wrong_code/{vendor}-{result.program.id}.cl",
      "w",
    ) as outfile:
      print(result.program.src, file=outfile)

    with open(
      f"../data/difftest/unreduced/clsmith/wrong_code/reports/{vendor}-{result.id}.txt",
      "w",
    ) as outfile:
      print(outfile.name)
      print(generate_wrong_code_report(result), file=outfile)

  # for env in cldrive.all_envs():
  #     testbed = db.get_testbed(session, env.platform, env.device)

  #     clsmith_wrong_code_bugs = session.query(CLSmithResult)\
  #         .filter(CLSmithResult.testbed == testbed)\
  #         .filter(CLSmithResult.classification == "Wrong code")
  #     fs.mkdir("../data/difftest/unreduced/clsmith/wrong_code")
  #     for result in clsmith_wrong_code_bugs:
  #         print(result.program.id)
  #         with open(f"../data/difftest/unreduced/clsmith/wrong_code/{result.program.id}.cl", "w") as outfile:
  #             print(result.program.src, file=outfile)

  # clgen_build_failures = session.query(CLgenResult)\
  #     .filter(CLgenResult.testbed == testbed)\
  #     .filter(CLgenResult.classification == 'Build failure')
  # for result in clgen_build_failures:
  #     reproduce_clgen_build_failures(result)

  print("done.")
