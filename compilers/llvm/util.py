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
"""High level utilities functions using LLVM."""
import subprocess
import typing

from compilers.llvm import llvm
from compilers.llvm import llvm_as
from compilers.llvm import opt
from labm8 import app

FLAGS = app.FLAGS


def GetOptArgs(cflags: typing.Optional[typing.List[str]] = None
              ) -> typing.List[typing.List[str]]:
  """Get the arguments passed to opt.

  Args:
    cflags: The cflags passed to clang. Defaults to -O0.

  Returns:
    A list of invocation arguments.
  """
  cflags = cflags or ['-O0']
  p1 = subprocess.Popen([llvm_as.LLVM_AS],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE)
  p2 = subprocess.Popen(
      [opt.OPT, '-disable-output', '-debug-pass=Arguments'] + cflags,
      stdin=p1.stdout,
      stderr=subprocess.PIPE,
      universal_newlines=True)
  _, stderr = p2.communicate()
  if p2.returncode:
    raise llvm.LlvmError(stderr)
  args = []
  for line in stderr.rstrip().split('\n'):
    if not line.startswith('Pass Arguments:'):
      raise llvm.LlvmError(f'Cannot interpret line: {line}')
    line = line[len('Pass Arguments:'):]
    args.append(line.split())
    for arg in args[-1]:
      if not arg[0] == '-':
        raise llvm.LlvmError(f'Cannot interpret clang argument: {arg}')
  return args
