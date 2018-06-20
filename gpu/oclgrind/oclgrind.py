"""A python wrapper around oclgrind, an OpenCL device simulator and debugger.

Oclgrind is developed by James Price <j.price@bristol.ac.uk>.
See https://github.com/jrprice/Oclgrind/wiki for the oclgrind documentation.

This file can be executed as a binary in order to invoke oclgrind. Note you must
use '--' to prevent this script from attempting to parse the args, and a second
'--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //gpu/oclgrind -- -- --version
"""
import subprocess
import sys
import typing

from absl import app
from absl import flags
from absl import logging

from gpu.cldrive import env
from lib.labm8 import bazelutil
from lib.labm8 import system


FLAGS = flags.FLAGS

# The path to the oclgrind binary.
_OCLGRIND_PKG = 'oclgrind_linux' if system.is_linux() else 'oclgrind_mac'
OCLGRIND_PATH = bazelutil.DataPath(f'{_OCLGRIND_PKG}/bin/oclgrind')


def Exec(argv: typing.List[str]) -> subprocess.Popen:
  """Execute a command using oclgrind.

  This creates a Popen process, executes it, and sets the stdout and stderr
  attributes to the process output.

  Args:
    argv: A list of arguments to pass to the oclgrind executable.

  Returns:
    A Popen instance, with string stdout and stderr attributes set.
  """
  cmd = [str(OCLGRIND_PATH)] + argv
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
  stdout, stderr = process.communicate()
  process.stdout, process.stderr = stdout, stderr
  return process


class OpenCLEnvironment(env.OpenCLEnvironment):
  """A mock cldrive OpenCLEnvironment for oclgrind."""

  def __init__(self):
    super(OpenCLEnvironment, self).__init__(None, None)

  @property
  def platform_name(self):
    """The name of the platform."""
    return 'Oclgrind'

  @property
  def device_name(self):
    """The name of the device."""
    return 'Oclgrind Simulator'

  @property
  def driver_version(self):
    """The driver version."""
    # This must be updated when @oclgrind package is updated.
    return 'Oclgrind 18.3'

  @property
  def opencl_version(self):
    """The supported OpenCL version."""
    # This must be updated when @oclgrind package is updated.
    return '1.2'

  @property
  def device_type(self):
    """The name of the device type."""
    return 'Emulator'


def main(argv):
  """Main entry point."""
  proc = Exec(argv[1:])
  print(proc.stdout)
  print(proc.stderr, file=sys.stderr)
  sys.exit(proc.returncode)


if __name__ == '__main__':
  app.run(main)
