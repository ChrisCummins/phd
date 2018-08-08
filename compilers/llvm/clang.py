"""A python wrapper around clang, the LLVM compiler.

clang is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke clang. Note you
must use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:clang [-- <script_args> [-- <clang_args>]]
"""
import pathlib
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
    'clang_timeout_seconds', 60,
    'The maximum number of seconds to allow process to run.')

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'

# Path to clang binary.
CLANG = bazelutil.DataPath(f'{_LLVM_REPO}/bin/clang')


class ClangException(llvm.LlvmError):
  """An error from clang."""
  pass


def Exec(args: typing.List[str], timeout_seconds: int = 60) -> subprocess.Popen:
  """Run clang.

  Args:
    args: A list of arguments to pass to binary.
    timeout_seconds: The number of seconds to allow clang to run for.

  Returns:
    A Popen instance with stdout and stderr set to strings.

  Raises:
    LlvmTimeout: If clang does not complete before timeout_seconds.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(CLANG)] + args
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=True)
  stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f'clang timed out after {timeout_seconds}s')
  process.stdout = stdout
  process.stderr = stderr
  return process


def Compile(input_path: pathlib.Path,
            binary_path: pathlib.Path,
            copts: typing.Optional[typing.List[str]] = None,
            timeout_seconds: int = 60) -> pathlib.Path:
  """Compile bytecode file to a binary.

  Args:
    input_path: The path of the input bytecode file.
    binary_path: The path of the binary to generate.
    copts: A list of additional flags to pass to the compiler.
    timeout_seconds: The number of seconds to allow clang to run for.

  Returns:
    The binary_path.

  Raises:
    ValueError: If the compilation fails.
  """
  copts = copts or []
  proc = Exec([str(input_path), '-o', str(binary_path)] + copts,
              timeout_seconds=timeout_seconds)
  if proc.returncode == 9:
    raise llvm.LlvmTimeout(f'opt timed out after {timeout_seconds} seconds')
  elif proc.returncode:
    raise ClangException(
        f'clang exited with returncode {proc.returncode}: {proc.stderr}')
  if not binary_path.is_file():
    raise ClangException(f"Binary file not generated: '{binary_path}'")
  return binary_path


def main(argv):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.clang_timeout_seconds)
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
