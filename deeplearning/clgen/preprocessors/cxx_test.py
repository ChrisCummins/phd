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
"""Unit tests for :cxx_test.py."""

import pytest
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import cxx
from labm8 import test


FLAGS = flags.FLAGS


# ClangPreprocess() tests.

def test_ClangPreprocess_empty_input():
  """Test that ClangPreprocess accepts an empty input."""
  assert cxx.ClangPreprocess('') == '\n'


def test_ClangPreprocess_small_cxx_program():
  """Test pre-processing a small C++ program."""
  assert cxx.ClangPreprocess("""
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""") == """

template<typename T>
T foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
"""


def test_ClangPreprocess_missing_include():
  """Test that ClangPreprocessor error is raised on missing #include."""
  with pytest.raises(errors.ClangException) as e_info:
    cxx.ClangPreprocess('#include "my-missing-file.h"')
  assert "'my-missing-file.h' file not found" in str(e_info.value)


# Compile() tests.

def test_Compile_empty_input():
  """Test that Compile accepts an empty input."""
  assert cxx.Compile('') == ''


def test_Compile_small_cxx_program():
  """Test Compile on a small C++ program."""
  assert cxx.Compile("""
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""") == """
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
"""


def test_Compile_user_define():
  """Test that Compile accepts a program with a custom #define."""
  assert cxx.Compile("""
#define FLOAT_T float
int A(FLOAT_T* a) {}
""") == """
#define FLOAT_T float
int A(FLOAT_T* a) {}
"""


def test_Compile_syntax_error():
  """Test that Compile rejects a program with invalid syntax."""
  with pytest.raises(errors.ClangException) as e_info:
    cxx.Compile("int mainA2@@1!!!#")
  assert 'error: ' in str(e_info.value)


def test_Compile_undefined_variable():
  """Test that Compile rejects a program with an undefined variable."""
  with pytest.raises(errors.ClangException) as e_info:
    cxx.Compile("""
int main(int argc, char** argv) {
  undefined_variable;
}
""")
  assert 'use of undeclared identifier' in str(e_info.value)


def test_Compile_undefined_function():
  """Test that Compile rejects a program with an undefined function."""
  with pytest.raises(errors.ClangException) as e_info:
    cxx.Compile("""
int main(int argc, char** argv) {
  undefined_function(argc);
}
""")
  assert 'use of undeclared identifier' in str(e_info.value)


def test_Compile_cxx_header():
  """Test that Compile accepts a program using a C++ header."""
  src = """
#include <cassert>
int main(int argc, char** argv) { return 0; }
"""
  assert src == cxx.Compile(src)


def test_Compile_c99_header():
  """Test that Compile accepts a program using a C99 header."""
  src = """
#include <assert.h>
int main(int argc, char** argv) { return 0; }
"""
  assert src == cxx.Compile(src)


# ClangFormat() tests.

def test_ClangFormat_simple_c_program():
  """Test that a simple C program is unchanged."""
  assert cxx.ClangFormat("""
int main(int argc, char** argv) { return 0; }
""") == """
int main(int argc, char** argv) {
  return 0;
}
"""


def test_ClangFormat_pointer_alignment():
  """Test that pointers are positioned left."""
  assert cxx.ClangFormat("""
int * A(int* a, int * b, int *c);
""") == """
int* A(int* a, int* b, int* c);
"""


def test_ClangFormat_undefined_data_type():
  """Test that an undefined data type does not cause an error."""
  assert cxx.ClangFormat("""
int main(MY_TYPE argc, char** argv) { return 0; }
""") == """
int main(MY_TYPE argc, char** argv) {
  return 0;
}
"""


def test_ClangFormat_undefined_variable():
  """Test that an undefined variable does not cause an error."""
  assert cxx.ClangFormat("""
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
""") == """
int main(int argc, char** argv) {
  return UNDEFINED_VARIABLE;
}
"""


def test_ClangFormat_undefined_function():
  """Test that an undefined function does not cause an error."""
  assert cxx.ClangFormat("""
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
""") == """
int main(int argc, char** argv) {
  return UNDEFINED_FUNCTION(0);
}
"""


def test_ClangFormat_invalid_preprocessor_directive():
  """Test that an invalid preprocessor directive does not raise an error."""
  assert cxx.ClangFormat("""
#this_is_not_a_valid_directive
int main(int argc, char** argv) { return 0; }
""") == """
#this_is_not_a_valid_directive
int main(int argc, char** argv) {
  return 0;
}
"""


# NormalizeIdentifiers() tests.

def test_NormalizeIdentifiers_small_cxx_program():
  """Test that rewriter performs as expected for a small C++ program."""
  assert """
#include <iostream>

int A(int a, char** b) {
  int c = 2 * a;
  std::cout << c << ' args' << std::endl;
}
""" == cxx.NormalizeIdentifiers("""
#include <iostream>

int main(int argc, char** argv) {
  int foo = 2 * argc;
  std::cout << foo << ' args' << std::endl;
}
""")


# StripComments() tests.

def test_StripComments_empty_input():
  """Test StripComments on an empty input."""
  assert cxx.StripComments('') == ''


def test_StripComments_only_comment():
  """Test StripComments on an input containing only comments."""
  assert cxx.StripComments('// Just a comment') == ' '
  assert cxx.StripComments('/* Just a comment */') == ' '


def test_StripComments_small_program():
  """Test Strip Comments on a small program."""
  assert cxx.StripComments("""
/* comment header */

int main(int argc, char** argv) { // main function.
  return /* foo */ 0;
}
""") == """
 

int main(int argc, char** argv) {  
  return   0;
}
"""


# Benchmarks.

HELLO_WORLD_CXX = """
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
"""


def test_benchmark_ClangPreprocess_hello_world(benchmark):
  """Benchmark ClangPreprocess on a "hello world" C++ program."""
  benchmark(cxx.ClangPreprocess, HELLO_WORLD_CXX)


def test_benchmark_Compile_hello_world(benchmark):
  """Benchmark Compile on a "hello world" C++ program."""
  benchmark(cxx.Compile, HELLO_WORLD_CXX)


def test_benchmark_StripComments_hello_world(benchmark):
  """Benchmark StripComments on a "hello world" C++ program."""
  benchmark(cxx.StripComments, HELLO_WORLD_CXX)


if __name__ == '__main__':
  test.Main()
