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
"""End-to-end compilation pipeline."""
import pathlib
import subprocess

from compilers.llvm import clang
from compilers.llvm import opt
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_LlvmPipeline(tempdir: pathlib.Path):
  """End-to-end test."""
  with open(tempdir / 'foo.c', 'w') as f:
    f.write("""
#include <stdio.h>

int main() {
  int x = 0;
  if (x != 0)
    x = 5; // dead code
  printf("Hello, world!\\n");
  return x;
}
""")

  # Generate bytecode.
  p = clang.Exec([
      str(tempdir / 'foo.c'), '-o',
      str(tempdir / 'foo.ll'), '-S', '-xc++', '-emit-llvm', '-c', '-O0'
  ])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo.ll').is_file()

  # Run an optimization pass.
  p = opt.Exec(
      [str(tempdir / 'foo.ll'), '-o',
       str(tempdir / 'foo2.ll'), '-S', '-dce'])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo2.ll').is_file()

  # Compile bytecode to LLVM IR.
  p = clang.Exec([str(tempdir / 'foo2.ll'), '-o', str(tempdir / 'foo')])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo').is_file()

  out = subprocess.check_output([str(tempdir / 'foo')], universal_newlines=True)
  assert out == 'Hello, world!\n'


if __name__ == '__main__':
  test.Main()
