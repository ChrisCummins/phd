# Copyright 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""A python wrapper around opt, the LLVM optimizer.

opt is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke opt. Note you must
use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:opt [-- <script_args> [-- <opt_args>]]
"""
import functools
import json
import pathlib
import subprocess
import sys
import typing

from compilers.llvm import llvm
from labm8.py import app
from labm8.py import bazelutil

FLAGS = app.FLAGS

app.DEFINE_integer(
  "opt_timeout_seconds",
  60,
  "The maximum number of seconds to allow process to run.",
)

# Path to opt binary.
OPT = bazelutil.DataPath("phd/third_party/llvm/opt")

# The list of LLVM opt transformation passes.
# See: https://llvm.org/docs/Passes.html#transform-passes
TRANSFORM_PASSES = json.loads(
  bazelutil.DataString("phd/compilers/llvm/opt_transform_passes.json")
)

# Valid optimization levels. Same as for clang, but without -Ofast.
OPTIMIZATION_LEVELS = {"-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"}


class OptException(llvm.LlvmError):
  """An error from opt."""

  pass


def ValidateOptimizationLevel(opt: str) -> str:
  """Check that the requested optimization level is valid.

  Args:
    opt: The optimization level.

  Returns:
    The input argument.

  Raises:
    ValueError: If optimization level is not valid.
  """
  if opt in OPTIMIZATION_LEVELS:
    return opt
  raise ValueError(
    f"Invalid opt optimization level '{opt}'. "
    f"Valid levels are: {OPTIMIZATION_LEVELS}"
  )


def Exec(
  args: typing.List[str],
  stdin: typing.Optional[typing.Union[str, bytes]] = None,
  timeout_seconds: int = 60,
  universal_newlines: bool = True,
  log: bool = True,
  opt: typing.Optional[pathlib.Path] = None,
) -> subprocess.Popen:
  """Run LLVM's optimizer.

  Args:
    args: A list of arguments to pass to the opt binary.
    stdin: Optional input to pass to binary. If universal_newlines is set, this
      should be a string. If not, it should be bytes.
    timeout_seconds: The number of seconds to allow opt to run for.
    universal_newlines: Argument passed to Popen() of opt process.
    log: If true, print executed command to DEBUG log.
    opt: An optional `opt` binary path to use.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  opt = opt or OPT
  cmd = ["timeout", "-s9", str(timeout_seconds), str(opt)] + args
  app.Log(3, "$ %s", " ".join(cmd))
  process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdin=subprocess.PIPE if stdin else None,
    universal_newlines=universal_newlines,
  )
  if stdin:
    stdout, stderr = process.communicate(stdin)
  else:
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f"clang timed out after {timeout_seconds}s")
  process.stdout = stdout
  process.stderr = stderr
  return process


def RunOptPassOnBytecode(
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  opts: typing.List[str],
  timeout_seconds: int = 60,
) -> pathlib.Path:
  """Run opt pass(es) on a bytecode file.

  Args:
    input_path: The input bytecode file.
    output_path: The file to generate.
    opts: Additional flags to pass to opt.
    timeout_seconds: The number of seconds to allow opt to run for.

  Returns:
    The output_path.

  Raises:
    OptException: In case of error.
    LlvmTimeout: If the process times out.
  """
  # We don't care about the output of opt, but we will try and decode it if
  # opt fails.
  proc = Exec(
    [str(input_path), "-o", str(output_path), "-S"] + opts,
    timeout_seconds=timeout_seconds,
    universal_newlines=False,
  )
  if proc.returncode == 9:
    raise llvm.LlvmTimeout(f"opt timed out after {timeout_seconds} seconds")
  elif proc.returncode:
    try:
      stderr = proc.stderr.decode("utf-8")
      raise OptException(
        f"clang exited with returncode {proc.returncode}: {stderr}"
      )
    except UnicodeDecodeError:
      raise OptException(f"clang exited with returncode {proc.returncode}")
  if not output_path.is_file():
    raise OptException(f"Bytecode file {output_path} not generated")
  return output_path


def GetAllOptimizationsAvailable() -> typing.List[str]:
  """Return the full list of optimizations available.

  Returns:
    A list of strings, where each string is an LLVM opt flag to enable an
    optimization.

  Raises:
    OptException: If unable to interpret opt output.
  """
  # We must disable logging here - this function is invoked to set
  # OPTIMIZATION_PASSES variable below, before flags are parsed.
  proc = Exec(["-help-list-hidden"], log=False)
  lines = proc.stdout.split("\n")
  # Find the start of the list of optimizations.
  for i in range(len(lines)):
    if lines[i] == "  Optimizations available:":
      break
  else:
    raise OptException
  # Find the end of the list of optimizations.
  for j in range(i + 1, len(lines)):
    if not lines[j].startswith("    -"):
      break
  else:
    raise OptException

  # Extract the list of optimizations.
  optimizations = [line[len("    ") :].split()[0] for line in lines[i + 1 : j]]
  if len(optimizations) < 2:
    raise OptException

  return optimizations


@functools.lru_cache(maxsize=1)
def GetAllOptPasses() -> typing.Set[str]:
  """Return all opt passes."""
  return TRANSFORM_PASSES.union(OPTIMIZATION_LEVELS).union(
    set(GetAllOptimizationsAvailable())
  )


def main(argv):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.opt_timeout_seconds)
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
