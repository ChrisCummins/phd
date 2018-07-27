"""A python wrapper around CLSmith, a random generator of OpenCL C programs.

CLSmith is developed by Chris Lidbury <christopher.lidbury10imperial.ac.uk>.
See https://github.com/ChrisLidbury/CLSmith.

This file can be executed as a binary in order to invoke CLSmith. Note you must
use '--' to prevent this script from attempting to parse the args, and a second
'--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/clsmith [-- -- <args>]
"""
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
from absl import app
from absl import flags

from lib.labm8 import bazelutil


FLAGS = flags.FLAGS

CLSMITH = bazelutil.DataPath('CLSmith/CLSmith')


class CLSmithError(EnvironmentError):
  """Error thrown in case CLSmith fails."""

  def __init__(self, msg: str, returncode: int):
    self.msg = msg
    self.returncode = returncode

  def __repr__(self):
    return str(self.msg)


@contextlib.contextmanager
def TemporaryWorkingDir(prefix: str = None) -> pathlib.Path:
  old_directory = os.getcwd()
  with tempfile.TemporaryDirectory(prefix=prefix) as d:
    os.chdir(d)
    yield pathlib.Path(d)
  os.chdir(old_directory)


def RunClsmith(*opts) -> str:
  """Generate and return a CLSmith program.

  Args:
    opts: A list of command line options to pass to CLSmith binary.

  Returns:
    The generated source code as a string.
  """
  with TemporaryWorkingDir(prefix='clsmith_') as d:
    proc = subprocess.Popen([CLSMITH] + list(opts), stderr=subprocess.PIPE,
                            universal_newlines=True)
    _, stderr = proc.communicate()
    if proc.returncode:
      raise CLSmithError(msg=stderr, returncode=proc.returncode)
    with open(d / 'CLProg.c') as f:
      src = f.read()
  return src


def main(argv):
  """Main entry point."""
  try:
    print(RunClsmith(*argv[1:]))
  except CLSmithError as e:
    print(e, file=sys.stderr)
    sys.exit(e.returncode)


if __name__ == '__main__':
  app.run(main)
