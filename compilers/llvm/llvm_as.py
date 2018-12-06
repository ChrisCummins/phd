"""A python wrapper around llvm-as, the LLVM assembler.

llvm-as is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke llvm-as. Note you
must use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:llvm_as [-- <script_args> [-- <llvm_as_args>]]
"""
import subprocess
import sys
import typing

from absl import app
from absl import flags
from absl import logging

from compilers.llvm import llvm
from labm8 import bazelutil
from labm8 import system


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'llvm_as_timeout_seconds', 60,
    'The maximum number of seconds to allow process to run.')

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'

# Path to llvm-as binary.
LLVM_AS = bazelutil.DataPath(f'{_LLVM_REPO}/bin/llvm-as')


class LlvmAsError(llvm.LlvmError):
  """An error from llvm-as."""
  pass


def Exec(args: typing.List[str],
         stdin: typing.Optional[str] = None,
         timeout_seconds: int = 60) -> subprocess.Popen:
  """Run llvm-as.

  Args:
    args: A list of arguments to pass to binary.
    stdin: Optional input string to pass to binary.
    timeout_seconds: The number of seconds to allow llvm-as to run for.

  Returns:
    A Popen instance with stdout and stderr set to strings.

  Raises:
    LlvmTimeout: If llvm-as does not complete before timeout_seconds.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(LLVM_AS)] + args
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      stdin=subprocess.PIPE if stdin else None, universal_newlines=True)
  if stdin:
    stdout, stderr = process.communicate(stdin)
  else:
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f'llvm-as timed out after {timeout_seconds}s')
  process.stdout = stdout
  process.stderr = stderr
  return process


def main(argv: typing.List[str]):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.llvm_as_timeout_seconds)
    if proc.stdout:
      print(proc.stdout)
    if proc.stderr:
      print(proc.stderr, file=sys.stderr)
    sys.exit(proc.returncode)
  except llvm.LlvmTimeout as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
