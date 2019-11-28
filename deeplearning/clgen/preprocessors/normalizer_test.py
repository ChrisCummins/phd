# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/clgen/preprocessors/normalizer.py."""
import subprocess

import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import normalizer
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class MockProcess:
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, *args):
    return "", ""


def test_NormalizeIdentifiers_process_command(mocker):
  """Test the clang_rewriter comand which is run."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(0)
  normalizer.NormalizeIdentifiers("", ".c", ["-foo"])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ["timeout", "-s9", "60"]


def test_NormalizeIdentifiers_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang_rewriter returns with SIGKILL."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(9)
  with test.Raises(errors.ClangTimeout):
    normalizer.NormalizeIdentifiers("", ".c", [])
  # ClangTimeout inherits from ClangException.
  with test.Raises(errors.ClangException):
    normalizer.NormalizeIdentifiers("", ".c", [])


def test_NormalizeIdentifiers_RewriterException(mocker):
  """Test that ClangException is raised if clang_rewriter returns 204."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(204)
  with test.Raises(errors.RewriterException):
    normalizer.NormalizeIdentifiers("", ".c", [])
  # RewriterException inherits from ClangException.
  with test.Raises(errors.ClangException):
    normalizer.NormalizeIdentifiers("", ".c", [])


def test_NormalizeIdentifiers_empty_c_file():
  """Test that RewriterException is raised on an empty file."""
  with test.Raises(errors.RewriterException):
    normalizer.NormalizeIdentifiers("", ".c", [])


def test_NormalizeIdentifiers_small_c_program():
  """Test the output of a small program."""
  assert (
    normalizer.NormalizeIdentifiers(
      """
int main(int argc, char** argv) {}
""",
      ".c",
      [],
    )
    == """
int A(int a, char** b) {}
"""
  )


def test_NormalizeIdentifiers_variable_names_function_scope():
  """Test that variable name sequence reset for each function."""
  assert (
    normalizer.NormalizeIdentifiers(
      """
int foo(int bar, int car) { int blah = bar; }
int foobar(int hello, int bar) { int test = bar; }
""",
      ".c",
      [],
    )
    == """
int A(int a, int b) { int c = a; }
int B(int a, int b) { int c = b; }
"""
  )


def test_NormalizeIdentifiers_small_cl_program():
  """Test the output of a small OpenCL program."""
  assert (
    normalizer.NormalizeIdentifiers(
      """
kernel void foo(global int* bar) {}
""",
      ".cl",
      [],
    )
    == """
kernel void A(global int* a) {}
"""
  )


def test_NormalizeIdentifiers_c_syntax_error():
  """Test that a syntax error does not prevent normalizer output."""
  assert normalizer.NormalizeIdentifiers("int main@@()! ##", ".c", [])


def test_NormalizeIdentifiers_printf_not_rewritten():
  """Test that a call to printf is not rewritten."""
  assert (
    normalizer.NormalizeIdentifiers(
      """
#include <stdio.h>

int main(int argc, char** argv) {
  printf("Hello, world!\\n");
  return 0;
}
""",
      ".c",
      [],
    )
    == """
#include <stdio.h>

int A(int a, char** b) {
  printf("Hello, world!\\n");
  return 0;
}
"""
  )


def test_NormalizeIdentifiers_undefined_not_rewritten():
  """Test that undefined functions and variables are not rewritten."""
  assert (
    normalizer.NormalizeIdentifiers(
      """
void main(int argc, char** argv) {
  undefined_function(undefined_variable);
}
""",
      ".c",
      [],
    )
    == """
void A(int a, char** b) {
  undefined_function(undefined_variable);
}
"""
  )


# Benchmarks.


def test_benchmark_NormalizeIdentifiers_c_hello_world(benchmark):
  """Benchmark NormalizeIdentifiers for a "hello world" C program."""
  benchmark(
    normalizer.NormalizeIdentifiers,
    """
#include <stdio.h>

int main(int argc, char** argv) {
  printf("Hello, world!\\n");
  return 0;
}
""",
    ".c",
    [],
  )


if __name__ == "__main__":
  test.Main()
