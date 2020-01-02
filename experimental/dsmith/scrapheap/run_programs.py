#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import product
from subprocess import Popen

from dsmith import db
from dsmith.db import *

from third_party.py.pyopencl import pyopencl as cl


def get_platform_name(platform_id):
  platform = cl.get_platforms()[platform_id]
  return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
  platform = cl.get_platforms()[platform_id]
  device = platform.get_devices()[device_id]
  return device.get_info(cl.device_info.NAME)


def main():
  parser = ArgumentParser(description="Collect difftest results for a device")
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
  parser.add_argument(
    "--clsmith", action="store_true", help="Only run CLSmith test cases"
  )
  parser.add_argument(
    "--clgen", action="store_true", help="Only run CLgen test cases"
  )
  parser.add_argument(
    "-t",
    "--timeout",
    type=str,
    default="3h",
    help="timeout(1) duration for batches (default: 3h)",
  )
  args = parser.parse_args()

  # get testbed information
  platform_id, device_id = args.platform_id, args.device_id
  platform_name = get_platform_name(platform_id)
  device_name = get_device_name(platform_id, device_id)

  db_hostname = args.hostname
  db_url = db.init(db_hostname)
  print("connected to", db_url)
  with Session(commit=False) as session:
    testbed = str(get_testbed(session, platform_name, device_name))
  print(testbed)

  cl_launcher_scripts = [
    "clsmith_run_cl_launcher.py",
  ]
  cldrive_scripts = [
    "clgen_run_cldrive.py",
  ]

  if args.opt:
    script_args = [["--opt"]]
  elif args.no_opt:
    script_args = [["--no-opt"]]
  else:
    script_args = [["--opt", "--no-opt"]]

  timeout = str(args.timeout)
  cl_launcher_jobs = [
    [
      "timeout",
      "-s9",
      timeout,
      "python",
      script,
      "--hostname",
      db_hostname,
      str(platform_id),
      str(device_id),
    ]
    + args
    for script, args in product(cl_launcher_scripts, script_args)
  ]
  cldrive_jobs = [
    [
      "timeout",
      "-s9",
      timeout,
      "python",
      script,
      "--hostname",
      db_hostname,
      platform_name,
      device_name,
    ]
    + args
    for script, args in product(cldrive_scripts, script_args)
  ]

  # Determine which jobs to run
  jobs = []
  if not args.clgen:
    jobs += cl_launcher_jobs
  if not args.clsmith:
    jobs += cldrive_jobs

  i = 0

  try:
    while len(jobs):
      i += 1
      job = jobs.pop(0)
      nremaining = len(jobs)

      # run job
      print(f"\033[1mjob {i} ({nremaining} remaining)\033[0m", *job)
      p = Popen(job)
      p.communicate()

      # repeat job if it fails
      if p.returncode == -9:
        print("\033[1m\033[91m>>>> TIMEOUT", p.returncode, "\033[0m")
        jobs.append(job)
      elif p.returncode:
        print("\033[1m\033[91m>>>> EXIT STATUS", p.returncode, "\033[0m")
        jobs.append(job)

      print("done")

  except KeyboardInterrupt:
    print("\ninterrupting, abort.")


if __name__ == "__main__":
  main()
