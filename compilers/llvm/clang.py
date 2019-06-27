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
"""A python wrapper around clang, the LLVM compiler.

clang is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke clang. Note you
must use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:clang [-- <script_args> [-- <clang_args>]]
"""
import collections
import pathlib
import re
import subprocess
import sys
import typing

from compilers.llvm import llvm
from labm8 import app
from labm8 import bazelutil
from labm8 import system

FLAGS = app.FLAGS

app.DEFINE_integer('clang_timeout_seconds', 60,
                   'The maximum number of seconds to allow process to run.')

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'

# Path to clang binary.
CLANG = bazelutil.DataPath(f'{_LLVM_REPO}/bin/clang')

# Valid optimization levels.
OPTIMIZATION_LEVELS = {"-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz"}

# A structured representation of the output of clang's bisect debugging, e.g.
#     $ clang foo.c -mllvm -opt-bisect-limit=-1.
# The output is of the form:
#     BISECT: running pass (<number>) <name> on <target_type> (<target>)
#
# See ClangBisectMessageToInvocation() for the conversion.
OptPassRunInvocation = collections.namedtuple('OptPassRunInvocation',
                                              ['name', 'target', 'target_type'])


class ClangException(llvm.LlvmError):
  """An error from clang."""
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
  raise ValueError(f"Invalid clang optimization level '{opt}'. "
                   f"Valid levels are: {OPTIMIZATION_LEVELS}")


def Exec(args: typing.List[str],
         stdin: typing.Optional[str] = None,
         timeout_seconds: int = 60,
         log: bool = True,
         stdout=subprocess.PIPE,
         stderr=subprocess.PIPE) -> subprocess.Popen:
  """Run clang.

  Args:
    args: A list of arguments to pass to binary.
    stdin: Optional input string to pass to binary.
    timeout_seconds: The number of seconds to allow clang to run for.
    log: If true, print executed command to DEBUG log.

  Returns:
    A Popen instance with stdout and stderr set to strings.

  Raises:
    LlvmTimeout: If clang does not complete before timeout_seconds.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(CLANG)] + args
  if log:
    app.Log(3, '$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd,
      stdout=stdout,
      stderr=stderr,
      stdin=subprocess.PIPE if stdin else None,
      universal_newlines=True)
  if stdin:
    stdout, stderr = process.communicate(stdin)
  else:
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f'clang timed out after {timeout_seconds}s')
  process.stdout = stdout
  process.stderr = stderr
  return process


def Compile(srcs: typing.List[pathlib.Path],
            out: pathlib.Path,
            copts: typing.Optional[typing.List[str]] = None,
            timeout_seconds: int = 60) -> pathlib.Path:
  """Compile input sources.

  This has some minor behavioural differences from calling into clang directly.
  The first is that necessary parent directories for the output path are created
  if necessary, rather than raising an error.

  Args:
    srcs: The path of the input bytecode file.
    out: The path of the binary to generate.
    copts: A list of additional flags to pass to the compiler.
    timeout_seconds: The number of seconds to allow clang to run for.

  Returns:
    The output path.

  Raises:
    FileNotFoundError: If any of the srcs do not exist.
    LlvmTimeout: If the compiler times out.
    ClangException: If the compilation fails.
  """
  copts = copts or []
  # Validate the input srcs.
  for src in srcs:
    if not src.is_file():
      raise FileNotFoundError(f"File not found: '{src}'")
  # Ensure the output directory exists.
  out.parent.mkdir(parents=True, exist_ok=True)

  proc = Exec(
      [str(x) for x in srcs] + ['-o', str(out)] + copts,
      timeout_seconds=timeout_seconds)
  if proc.returncode == 9:
    raise llvm.LlvmTimeout(f'clang timed out after {timeout_seconds} seconds')
  elif proc.returncode:
    raise ClangException(
        f'clang exited with returncode {proc.returncode}: {proc.stderr}')
  if not out.is_file():
    raise ClangException(f"Binary file not generated: '{out}'")
  return out


def Preprocess(src: str,
               copts: typing.Optional[typing.List[str]] = None,
               timeout_seconds: int = 60):
  """Run input code through the compiler frontend to inline macros.

  Args:
    src: The source code to preprocess.
    copts: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The preprocessed code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  copts = copts or []
  cmd = ['-E', '-c', '-', '-o', '-'] + copts
  app.Log(3, '$ %s', ' '.join(cmd))
  process = Exec(cmd, timeout_seconds=timeout_seconds, stdin=src)
  if process.returncode:
    raise ClangException(returncode=process.returncode, stderr=process.stderr)
  return process.stdout


# A regex which matches the a line of clang's bisect debugging output, and
# contains three named match groups: name, target, and target_type.
_CLANG_BISECT_MESSAGE_RE = re.compile(
    r'BISECT: running pass \([\d]+\) (?P<name>.+) on (?P<target_type>\w+)( '
    r'\((?P<target>[^\)]+)\))?')


def ClangBisectMessageToInvocation(line: str) -> OptPassRunInvocation:
  """Parse a line of clang's bisect debugging output into a structured form.

  Args:
    line: The line of output as a string.

  Returns:
    An OptPassRunInvocation tuple.

  Raises:
    ClangException: If the line cannot be parsed.
  """
  m = _CLANG_BISECT_MESSAGE_RE.match(line)
  if not m:
    raise ClangException(msg=f'Cannot interpret line: {line}')
  return OptPassRunInvocation(
      name=m.group('name'),
      target=m.group('target') or '',
      target_type=m.group('target_type'))


def GetOptPasses(cflags: typing.Optional[typing.List[str]] = None,
                 language: typing.Optional[str] = 'c',
                 stubfile: typing.Optional[str] = 'int main() {}'
                ) -> typing.List[OptPassRunInvocation]:
  """Get the list of passes run by opt.

  Args:
    cflags: The cflags passed to clang. Defaults to -O0.
    language: The language to get the pass list for.
    stubfile: The dummy stub file to get the pass list from.

  Returns:
    A list of passes.
  """
  cflags = cflags or ['-O0']
  process = Exec(
      cflags + ['-mllvm', '-opt-bisect-limit=-1', f'-x{language}', '-'],
      stdin=stubfile)
  if process.returncode:
    raise ClangException(f'clang exited with returncode {process.returncode} '
                         f'and stderr: {process.stderr}')
  lines = process.stderr.rstrip().split('\n')
  return [ClangBisectMessageToInvocation(line) for line in lines]
  return lines


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
  app.RunWithArgs(main)
