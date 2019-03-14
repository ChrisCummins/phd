# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A python wrapper around CLSmith, a random generator of OpenCL C programs.

CLSmith is developed by Chris Lidbury <christopher.lidbury10imperial.ac.uk>.
See https://github.com/ChrisLidbury/CLSmith.

This file can be executed as a binary in order to invoke CLSmith. Note you must
use '--' to prevent this script from attempting to parse the args, and a second
'--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/clsmith [-- -- <args>]
"""
import subprocess
import sys

from labm8 import app
from labm8 import bazelutil
from labm8 import fs

FLAGS = app.FLAGS

CLSMITH = bazelutil.DataPath('CLSmith/CLSmith')


class CLSmithError(EnvironmentError):
  """Error thrown in case CLSmith fails."""

  def __init__(self, msg: str, returncode: int):
    self.msg = msg
    self.returncode = returncode

  def __repr__(self):
    return str(self.msg)


def Exec(*opts) -> str:
  """Generate and return a CLSmith program.

  Args:
    opts: A list of command line options to pass to CLSmith binary.

  Returns:
    The generated source code as a string.
  """
  with fs.TemporaryWorkingDir(prefix='clsmith_') as d:
    proc = subprocess.Popen(
        [CLSMITH] + list(opts), stderr=subprocess.PIPE, universal_newlines=True)
    _, stderr = proc.communicate()
    if proc.returncode:
      raise CLSmithError(msg=stderr, returncode=proc.returncode)
    with open(d / 'CLProg.c') as f:
      src = f.read()
  return src


def main(argv):
  """Main entry point."""
  try:
    print(Exec(*argv[1:]))
  except CLSmithError as e:
    print(e, file=sys.stderr)
    sys.exit(e.returncode)


if __name__ == '__main__':
  app.RunWithArgs(main)
