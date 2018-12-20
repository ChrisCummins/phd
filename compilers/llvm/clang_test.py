"""Unit tests for //compilers/llvm/clang.py."""
import pathlib
import sys
import tempfile
import typing

import pytest
from absl import app
from absl import flags

from compilers.llvm import clang


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)


def _StripPreprocessorLines(out: str):
  return '\n'.join(line for line in out.split('\n') if not line.startswith('#'))


# Exec() tests.

def test_Exec_compile_bytecode(tempdir: pathlib.Path):
  """Test bytecode generation."""
  with open(tempdir / 'foo.cc', 'w') as f:
    f.write("""
#include <iostream>

int main() {
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
""")
  p = clang.Exec([str(tempdir / 'foo.cc'), '-xc++', '-S', '-emit-llvm', '-c',
                  '-o', str(tempdir / 'foo.ll')])
  assert not p.returncode
  assert not p.stderr
  assert not p.stdout
  assert (tempdir / 'foo.ll').is_file()


def test_Exec_compile_bytecode_stdin(tempdir: pathlib.Path):
  """Test bytecode generation."""
  p = clang.Exec(['-xc++', '-S', '-emit-llvm', '-c',
                  '-o', str(tempdir / 'foo.ll'), '-'], stdin="""
#include <iostream>

int main() {
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
""")
  print(p.stderr)
  assert not p.returncode
  assert not p.stderr
  assert not p.stdout
  assert (tempdir / 'foo.ll').is_file()


@pytest.mark.parametrize(
    "opt", ("-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(opt: str):
  """Test that valid optimization levels are returned."""
  assert clang.ValidateOptimizationLevel(opt) == opt


@pytest.mark.parametrize(
    "opt", ("O0",  # missing leading '-'
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
""", copts=['-xc++']).endswith("""

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


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
