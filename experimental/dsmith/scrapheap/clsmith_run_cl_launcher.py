#!/usr/bin/env python3
import os
import re
from argparse import ArgumentParser
from collections import deque
from tempfile import NamedTemporaryFile
from time import strftime
from typing import Tuple

import progressbar
from dsmith import clsmith
from dsmith import db
from dsmith.db import *
from dsmith.lib import *

from labm8.py import crypto
from third_party.py.pyopencl import pyopencl as cl


def get_platform_name(platform_id):
  platform = cl.get_platforms()[platform_id]
  return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
  platform = cl.get_platforms()[platform_id]
  device = platform.get_devices()[device_id]
  return device.get_info(cl.device_info.NAME)


def get_driver_version(platform_id, device_id):
  platform = cl.get_platforms()[platform_id]
  device = platform.get_devices()[device_id]
  return device.get_info(cl.device_info.DRIVER_VERSION)


def cl_launcher(
  src: str, platform_id: int, device_id: int, *args
) -> Tuple[float, int, str, str]:
  """ Invoke cl launcher on source """
  with NamedTemporaryFile(prefix="cl_launcher-", suffix=".cl") as tmp:
    tmp.write(src.encode("utf-8"))
    tmp.flush()

    return clsmith.cl_launcher(
      tmp.name,
      platform_id,
      device_id,
      *args,
      timeout=os.environ.get("TIMEOUT", 60),
    )


def verify_params(
  platform: str,
  device: str,
  optimizations: bool,
  global_size: tuple,
  local_size: tuple,
  stderr: str,
) -> None:
  """ verify that expected params match actual as reported by CLsmith """
  optimizations = "on" if optimizations else "off"

  actual_platform = None
  actual_device = None
  actual_optimizations = None
  actual_global_size = None
  actual_local_size = None
  for line in stderr.split("\n"):
    if line.startswith("Platform: "):
      actual_platform_name = re.sub(r"^Platform: ", "", line).rstrip()
    elif line.startswith("Device: "):
      actual_device_name = re.sub(r"^Device: ", "", line).rstrip()
    elif line.startswith("OpenCL optimizations: "):
      actual_optimizations = re.sub(
        r"^OpenCL optimizations: ", "", line
      ).rstrip()

    # global size
    match = re.match("^3-D global size \d+ = \[(\d+), (\d+), (\d+)\]", line)
    if match:
      actual_global_size = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
      )
    match = re.match("^2-D global size \d+ = \[(\d+), (\d+)\]", line)
    if match:
      actual_global_size = (int(match.group(1)), int(match.group(2)), 0)
    match = re.match("^1-D global size \d+ = \[(\d+)\]", line)
    if match:
      actual_global_size = (int(match.group(1)), 0, 0)

    # local size
    match = re.match("^3-D local size \d+ = \[(\d+), (\d+), (\d+)\]", line)
    if match:
      actual_local_size = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
      )
    match = re.match("^2-D local size \d+ = \[(\d+), (\d+)\]", line)
    if match:
      actual_local_size = (int(match.group(1)), int(match.group(2)), 0)
    match = re.match("^1-D local size \d+ = \[(\d+)\]", line)
    if match:
      actual_local_size = (int(match.group(1)), 0, 0)

    # check if we've collected everything:
    if (
      actual_platform
      and actual_device
      and actual_optimizations
      and actual_global_size
      and actual_local_size
    ):
      assert actual_platform == platform
      assert actual_device == device
      assert actual_optimizations == optimizations
      assert actual_global_size == global_size
      assert actual_local_size == local_size
      return


def parse_ndrange(ndrange: str) -> Tuple[int, int, int]:
  components = ndrange.split(",")
  assert len(components) == 3
  return (int(components[0]), int(components[1]), int(components[2]))


def get_num_to_run(
  session: db.session_t, testbed: Testbed, optimizations: int = None
):
  num_ran = session.query(sql.sql.func.count(CLSmithResult.id)).filter(
    CLSmithResult.testbed_id == testbed.id
  )
  total = session.query(sql.sql.func.count(CLSmithTestCase.id))

  if optimizations is not None:
    num_ran = (
      num_ran.join(CLSmithTestCase)
      .join(cl_launcherParams)
      .filter(cl_launcherParams.optimizations == optimizations)
    )
    total = total.join(cl_launcherParams).filter(
      cl_launcherParams.optimizations == optimizations
    )

  return num_ran.scalar(), total.scalar()


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  parser.add_argument(
    "platform_id", metavar="<platform-id>", type=int, help="OpenCL platform ID"
  )
  parser.add_argument(
    "device_id", metavar="<device-id>", type=int, help="OpenCL device ID"
  )
  parser.add_argument(
    "--opt", action="store_true", help="Only test with optimizations on"
  )
  parser.add_argument(
    "--no-opt",
    action="store_true",
    help="Only test with optimizations disabled",
  )
  args = parser.parse_args()

  # Parse command line options
  platform_id = args.platform_id
  device_id = args.device_id

  # get testbed information
  platform_name = get_platform_name(platform_id)
  device_name = get_device_name(platform_id, device_id)
  driver_version = get_driver_version(platform_id, device_id)

  optimizations = None
  if args.opt and args.no_opt:
    pass  # both flags
  elif args.opt:
    optimizations = 1
  elif args.no_opt:
    optimizations = 0

  db.init(args.hostname)  # initialize db engine

  with Session() as session:
    testbed = get_testbed(session, platform_name, device_name)
    devname = util.device_str(testbed.device)

    # progress bar
    num_ran, num_to_run = get_num_to_run(session, testbed, optimizations)
    bar = progressbar.ProgressBar(init_value=num_ran, max_value=num_to_run)

    # programs to run, and results to push to database
    inbox = deque()

    def next_batch():
      """
      Fill the inbox with jobs to run.
      """
      BATCH_SIZE = 100
      print(f"\nnext CLSmith batch for {devname} at", strftime("%H:%M:%S"))
      # update the counters
      num_ran, num_to_run = get_num_to_run(session, testbed, optimizations)
      bar.max_value = num_to_run
      bar.update(min(num_ran, num_to_run))

      # fill inbox
      done = session.query(CLSmithResult.testcase_id).filter(
        CLSmithResult.testbed == testbed
      )
      if optimizations is not None:
        done = (
          done.join(CLSmithTestCase)
          .join(cl_launcherParams)
          .filter(cl_launcherParams.optimizations == optimizations)
        )

      todo = (
        session.query(CLSmithTestCase)
        .filter(~CLSmithTestCase.id.in_(done))
        .order_by(CLSmithTestCase.program_id, CLSmithTestCase.params_id)
      )
      if optimizations is not None:
        todo = todo.join(cl_launcherParams).filter(
          cl_launcherParams.optimizations == optimizations
        )

      todo = todo.limit(BATCH_SIZE)
      for testcase in todo:
        inbox.append(testcase)

    try:
      while True:
        # get the next batch of programs to run
        if not len(inbox):
          next_batch()
        # we have no programs to run
        if not len(inbox):
          break

        # get next program to run
        testcase = inbox.popleft()

        program = testcase.program
        params = testcase.params
        flags = params.to_flags()

        # drive the program
        runtime, status, stdout, stderr = cl_launcher(
          program.src, platform_id, device_id, *flags
        )

        # assert that executed params match expected
        verify_params(
          platform=platform_name,
          device=device_name,
          optimizations=params.optimizations,
          global_size=params.gsize,
          local_size=params.lsize,
          stderr=stderr,
        )

        # create new result
        stdout_ = util.escape_stdout(stdout)
        stdout = get_or_create(
          session, CLSmithStdout, hash=crypto.sha1_str(stdout_), stdout=stdout_
        )

        stderr_ = util.escape_stderr(stderr)
        stderr = get_or_create(
          session, CLSmithStderr, hash=crypto.sha1_str(stderr_), stderr=stderr_
        )
        session.flush()

        result = CLSmithResult(
          testbed_id=testbed.id,
          testcase_id=testcase.id,
          status=status,
          runtime=runtime,
          stdout_id=stdout.id,
          stderr_id=stderr.id,
          outcome=analyze.get_cl_launcher_outcome(status, runtime, stderr_),
        )

        session.add(result)
        session.commit()

        # update progress bar
        num_ran += 1
        bar.update(min(num_ran, num_to_run))
    finally:
      # flush any remaining results
      next_batch()

  print("done.")
