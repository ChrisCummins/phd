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
"""A python wrapper around clang-format, a tool to format code.

clang-format is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke clang-format. Note you
must use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:clang_format \
     [-- <script_args> [-- <clang_format_args>]]
"""
import fileinput
import subprocess
import sys
import typing

from compilers.llvm import llvm
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import system

FLAGS = app.FLAGS

app.DEFINE_string(
  "clang_format_file_suffix", ".c", "The file name suffix to assume for files."
)
app.DEFINE_integer(
  "clang_format_timeout_seconds",
  60,
  "The maximum number of seconds to allow process to run.",
)

_LLVM_REPO = "llvm_linux" if system.is_linux() else "llvm_mac"

# Path to clang-format binary.
CLANG_FORMAT = bazelutil.DataPath(f"{_LLVM_REPO}/bin/clang-format")


class ClangFormatException(llvm.LlvmError):
  """An error from clang-format."""

  pass


def Exec(
  text: str, suffix: str, args: typing.List[str], timeout_seconds: int = 60
) -> str:
  """Run clang-format on a source.

  Args:
    text: The source code to run through clang-format.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    args: A list of additional arguments to pass to binary.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    LlvmTimeout: If clang-format does not complete before timeout_seconds.
  """
  cmd = [
    "timeout",
    "-s9",
    str(timeout_seconds),
    str(CLANG_FORMAT),
    "-assume-filename",
    f"input{suffix}",
  ] + args
  app.Log(3, "$ %s", " ".join(cmd))
  process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(text)
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f"clang-format timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ClangFormatException(stderr)
  return stdout


def main(argv):
  """Main entry point."""
  try:
    print(
      Exec(
        fileinput.input(),
        FLAGS.clang_format_file_suffix,
        argv[1:],
        timeout_seconds=FLAGS.clang_format_timeout_seconds,
      )
    )
  except (llvm.LlvmTimeout, ClangFormatException) as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  app.RunWithArgs(main)
