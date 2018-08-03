"""A python wrapper around opt, the LLVM optimizer.

opt is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke opt. Note you must
use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:opt [-- <script_args> [-- <opt_args>]]
"""
import subprocess
import sys
import typing
from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import system

from compilers.llvm import llvm


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'opt_timeout_seconds', 60,
    'The maximum number of seconds to allow process to run.')

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'

# Path to opt binary.
OPT = bazelutil.DataPath(f'{_LLVM_REPO}/bin/opt')


class OptException(llvm.LlvmError):
  """An error from opt."""
  pass


def Exec(args: typing.List[str], timeout_seconds: int = 60) -> subprocess.Popen:
  """Run LLVM's optimizer.

  Args:
    args: A list of arguments to pass to binary.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(OPT)] + args
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=True)
  stdout, stderr = process.communicate()
  process.stdout = stdout
  process.stderr = stderr
  return process


def main(argv):
  """Main entry point."""
  try:
    print(Exec(argv[1:], timeout_seconds=FLAGS.opt_timeout_seconds))
  except (llvm.LlvmTimeout, OptException) as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
