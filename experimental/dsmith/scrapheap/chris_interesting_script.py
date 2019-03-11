#!/usr/bin/env python

import argparse
import logging
import subprocess
import sys
import threading

from dsmith import clsmith

LAUNCHER = clsmith.cl_launcher_path
LAUNCHER_OPTS = ["-l", "1,1,1", "-g", "1,1,1"]
OCLGRIND = "oclgrind"
OCLGRIND_OPTS = [
    "--max-errors", "16", "--build-options", "-O0", "-Wall", "--uninitialized"
]

reference_platforms = ["AMD", "Intel"]
device = 0

timeout = 60.0  # seconds

logfile = None


class RunInThread:
  stdout = None
  stderr = None
  timed_out = False

  def __init__(self, command, timeout=30):
    # Start the thread
    thread = threading.Thread(target=self.__run, args=(command,))
    thread.start()

    # Wait for it to finish or to timeout
    thread.join(timeout)
    if thread.is_alive():
      self.process.terminate()
      thread.join(5)
      if thread.is_alive():
        self.process.kill()
        thread.join()
      self.timed_out = True

  def __run(self, command):
    # Start the process
    self.process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    # Get its output (blocking call)
    self.stdout, self.stderr = self.process.communicate()


def verify_with_oclgrind(clprogram):
  # Execute Oclgrind in a separate thread
  run = RunInThread(
      [OCLGRIND] + OCLGRIND_OPTS +
      [LAUNCHER, "-f", clprogram, "-p", "0", "-d", "0"] + LAUNCHER_OPTS,
      timeout * 30)

  # Check to see if Oclgrind actually completed successfully
  if run.timed_out:
    app.Debug("Oclgrind: Timed out")
    return False

  stdout, stderr = run.stdout, run.stderr
  app.Debug("Oclgrind:\nOut: %r\nErr: %r\n", stdout, stderr)
  # Check if the compilation process was successful
  compiled = "ompilation terminated successfully" in stderr
  if not compiled:
    app.Debug("Oclgrind: Compilation failed")
    return False

  # Check for issues in the Oclgrind output
  if "ninitialized value" in stderr:
    app.Debug("Oclgrind: Uninitialized value detected")
    return False
  if "initialized address" in stderr:
    app.Debug("Oclgrind: Uninitialized address detected")
    return False

  # We didn't find any issues so everything must be okay
  return True


def get_reference_run(clprogram):
  result = {}
  for platform in range(len(reference_platforms)):
    platform_name = reference_platforms[platform]
    result[platform_name] = None

    run = RunInThread(
        [LAUNCHER, "-f", clprogram, "-p",
         str(platform + 1), "-d",
         str(device)] + LAUNCHER_OPTS, timeout)

    stdout, stderr = run.stdout, run.stderr
    app.Debug("Reference[%s]:\nOut: %r\nErr: %r\n", platform_name, stdout,
              stderr)

    compiled = "ompilation terminated successfully" in stderr
    if compiled and not run.timed_out:
      result[platform_name] = stdout.split("\n")
    else:
      if run.timed_out:
        app.Debug("Reference[%s]: Timed out", platform_name)
        result[platform_name] = platform_name + " timed out!"
      else:
        app.Debug("Reference[%s]: Compilation failed", platform_name)
        result[platform_name] = platform_name + " did not compile!"
      return None

    # Accumulate all the results and check if they all match
    # This is done in the loop so that we can exit immediatelly after the
    # first mismatch
    _rand_key = list(result.keys())[0]
    random_result = result[_rand_key]
    if len(random_result) < 2 or not random_result[1]:
      app.Debug("Reference: No result")
      return None
    if len(random_result) > 1:
      if "error" in random_result[1]:
        app.Debug("\"error\" found in output")
        return False
      if "Error" in random_result[1]:
        app.Debug("\"Error\" found in output")
        return False
      if random_result[1] == "0,":
        app.Debug("Zero output is not interesting")
        return False
    all_match = all(result[x] == random_result for x in result.keys())
    if not all_match:
      app.Debug("Reference: Mismatched results (\n%r\n)", result)
      return None

  # They are all the same anyway
  random_result = result[_rand_key]
  return random_result


def get_ocl_run(clprogram):
  result = None
  platform_name = "OCL"
  run = RunInThread(
      [LAUNCHER, "-f", clprogram, "-p",
       str(0), "-d", str(device)] + LAUNCHER_OPTS, timeout)

  out, err = run.stdout, run.stderr
  app.Debug("OCL:\nOut: %r\nErr: %r\n", out, err)

  compiled = "ompilation terminated successfully" in out

  if compiled and not run.timed_out:
    result = out.split("\n")
  else:
    if run.timed_out:
      app.Debug("OCL: Timed out")
      result = platform_name + " timed out!"
    else:
      app.Debug("OCL: Compilation failed")
      result = platform_name + " did not compile!"
    return None

  return result


def run(clprogram, no_oclgrind, vectors=True):
  reference = get_reference_run(clprogram)
  ocl = None
  if reference:
    ocl = get_ocl_run(clprogram)

  if ocl and reference and ocl != reference:
    if no_oclgrind:
      return True
    return verify_with_oclgrind(clprogram)

  return False


def main(argv):
  global logfile

  argparser = argparse.ArgumentParser(
      description="Check if the OpenCL kernel is interesting.")
  argparser.add_argument('--logfile', action='store')
  argparser.add_argument('--loglevel', action='store')
  argparser.add_argument('--vectors', action='store', default=True)
  argparser.add_argument('--no-oclgrind', action='store_true')
  argparser.add_argument(
      'clprogram',
      metavar="KERNEL",
      nargs='?',
      default="CLProgram.cl",
      help="The kernel file to run (CLProgram.cl by default)")
  args = argparser.parse_args(argv[1:])

  if args.loglevel or args.logfile:
    log_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(level=log_level, filename=args.logfile)

  clprogram = args.clprogram
  no_oclgrind = args.no_oclgrind
  logfile = args.logfile

  app.Debug("%r\n", argv)

  if run(clprogram, no_oclgrind, args.vectors):
    sys.exit(0)
  else:
    sys.exit(1)


if __name__ == "__main__":
  main(sys.argv)
