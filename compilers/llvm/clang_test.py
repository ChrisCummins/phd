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
"""Unit tests for //compilers/llvm/clang.py."""
import pathlib
import pytest

from compilers.llvm import clang
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

_BASIC_CPP_PROGRAM = """
int main() {
  return 0;
}
"""


def _StripPreprocessorLines(out: str):
  return '\n'.join(line for line in out.split('\n') if not line.startswith('#'))


# Exec() tests.


def test_Exec_compile_bytecode(tempdir: pathlib.Path):
  """Test bytecode generation."""
  with open(tempdir / 'foo.cc', 'w') as f:
    f.write(_BASIC_CPP_PROGRAM)
  p = clang.Exec([
      str(tempdir / 'foo.cc'), '-xc++', '-S', '-emit-llvm', '-c', '-o',
      str(tempdir / 'foo.ll')
  ])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo.ll').is_file()


def test_Exec_compile_bytecode_stdin(tempdir: pathlib.Path):
  """Test bytecode generation."""
  p = clang.Exec(
      ['-xc++', '-S', '-emit-llvm', '-c', '-o',
       str(tempdir / 'foo.ll'), '-'],
      stdin=_BASIC_CPP_PROGRAM)
  print(p.stderr)
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo.ll').is_file()


@pytest.mark.parametrize("opt",
                         ("-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(opt: str):
  """Test that valid optimization levels are returned."""
  assert clang.ValidateOptimizationLevel(opt) == opt


@pytest.mark.parametrize(
    "opt",
    (
        "O0",  # missing leading '-'
        "-O4",  # not a real value
        "foo"))  # not a real value
def test_ValidateOptimizationLevel_invalid(opt: str):
  """Test that invalid optimization levels raise an error."""
  with pytest.raises(ValueError) as e_ctx:
    clang.ValidateOptimizationLevel(opt)
  assert opt in str(e_ctx.value)


# Preprocess() tests.


def test_Preprocess_empty_input():
  """Test that Preprocess accepts an empty input."""
  assert _StripPreprocessorLines(clang.Preprocess('')) == '\n'


def test_Preprocess_small_cxx_program():
  """Test pre-processing a small C++ program."""
  assert clang.Preprocess("""
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""",
                          copts=['-xc++']).endswith("""

template<typename T>
T foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""")


def test_Preprocess_missing_include():
  """Test that Preprocessor error is raised on missing #include."""
  with pytest.raises(clang.ClangException) as e_info:
    clang.Preprocess('#include "my-missing-file.h"')
  assert "'my-missing-file.h' file not found" in str(e_info.value.stderr)


def test_GetOptPasses_O0():
  """Black box opt passes test for -O0."""
  args = clang.GetOptPasses(['-O0'])
  assert args


if __name__ == '__main__':
  test.Main()
