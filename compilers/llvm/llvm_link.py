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
"""A python wrapper around llvm-link, the LLVM bitcode linker.

llvm-link is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke llvm-link. Note you
must use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:llvm_link [-- <script_args> [-- <llvm_link_args>]]
"""
import pathlib
import subprocess
import sys
import typing

from compilers.llvm import llvm
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import system

FLAGS = app.FLAGS

app.DEFINE_integer(
  "llvm_link_timeout_seconds",
  60,
  "The maximum number of seconds to allow process to run.",
)

_LLVM_REPO = "llvm_linux" if system.is_linux() else "llvm_mac"

# Path to llvm-link binary.
LLVM_LINK = bazelutil.DataPath(f"{_LLVM_REPO}/bin/llvm-link")


def Exec(args: typing.List[str], timeout_seconds: int = 60) -> subprocess.Popen:
  """Run LLVM's bitcode linker.

  Args:
    args: A list of arguments to pass to binary.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  cmd = ["timeout", "-s9", str(timeout_seconds), str(LLVM_LINK)] + args
  app.Log(3, "$ %s", " ".join(cmd))
  process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
  )
  stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f"llvm-link timed out after {timeout_seconds}s")
  process.stdout = stdout
  process.stderr = stderr
  return process


def LinkBitcodeFilesToBytecode(
  input_paths: typing.List[pathlib.Path],
  output_path: pathlib.Path,
  linkopts: typing.Optional[typing.List[str]] = None,
  timeout_seconds: int = 60,
) -> pathlib.Path:
  """Link multiple bitcode files to a single bytecode file.

  Args:
    input_paths: A list of input bitcode files.
    output_path: The bytecode file to generate.
    linkopts: A list of additional flags to pass to llvm-link.
    timeout_seconds: The number of seconds to allow llvm-link to run for.

  Returns:
    The output_path.
  """
  if output_path.is_file():
    output_path.unlink()
  linkopts = linkopts or []
  proc = Exec(
    [str(x) for x in input_paths] + ["-o", str(output_path), "-S"] + linkopts,
    timeout_seconds=timeout_seconds,
  )
  if proc.returncode:
    raise ValueError(f"Failed to link bytecode: {proc.stderr}")
  if not output_path.is_file():
    raise ValueError(f"Bytecode file {output_path} not linked.")
  return output_path


def main(argv):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.llvm_link_timeout_seconds)
    if proc.stdout:
      print(proc.stdout)
    if proc.stderr:
      print(proc.stderr, file=sys.stderr)
    sys.exit(proc.returncode)
  except llvm.LlvmTimeout as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  app.RunWithArgs(main)
