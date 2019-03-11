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
"""Unit tests for //deeplearning/clgen/preprocessors/clang.py."""
import subprocess

import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import clang
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


class MockProcess():
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, *args):
    return '', ''


# StripPreprocessorLines() tests.


def test_StripPreprocessorLines_empty_input():
  assert clang.StripPreprocessorLines('') == ''


def test_StripPreprocessorLines_no_stdin():
  """Test that without a stdin marker is stripped."""
  assert clang.StripPreprocessorLines('The\ncat\nsat\non\nthe\nmat') == ''


def test_StripPreprocessorLines_everything_stripped_before_stdin():
  """Test that everything before the stdin marker is stripped."""
  assert clang.StripPreprocessorLines("""
This will
all be stripped.
# 1 "<stdin>" 2
This will not be stripped.""") == "This will not be stripped."


def test_StripPreprocessorLines_nothing_before_stdin():
  """Test that if nothing exists before stdin marker, nothing is lost."""
  assert clang.StripPreprocessorLines('# 1 "<stdin>" 2\nfoo') == 'foo'


def test_StripPreprocessorLines_hash_stripped():
  """Test that lines which begin with # symbol are stripped."""
  assert clang.StripPreprocessorLines("""# 1 "<stdin>" 2
# This will be stripped.
Foo
This will # not be stripped.
# But this will.""") == """Foo
This will # not be stripped."""


# Preprocess() tests.


def test_Preprocess_process_command(mocker):
  """Test the process comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang.Preprocess('', ['-foo'])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']
  assert cmd[4:] == ['-E', '-c', '-', '-o', '-', '-foo']


def test_Preprocess_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(errors.ClangTimeout):
    clang.Preprocess('', [])
  # ClangTimeout inherits from ClangException.
  with pytest.raises(errors.ClangException):
    clang.Preprocess('', [])


def test_Preprocess_ClangException(mocker):
  """Test that ClangException is raised if clang returns non-zero returncode."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(errors.ClangException):
    clang.Preprocess('', [])


def test_Preprocess_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang.Preprocess('', []) == '\n'


def test_Preprocess_simple_c_program():
  """Test that a simple C program is unchanged."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return 0; }
""", []) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_cflag():
  """Test pre-processing with a custom define in the command line."""
  assert clang.Preprocess(
      """
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_define():
  """Test pre-processing with a #define in the source code."""
  assert clang.Preprocess(
      """
#define MY_TYPE int
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """

int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_data_type():
  """Test that an undefined data type does not cause an error."""
  assert clang.Preprocess(
      """
int main(MY_TYPE argc, char** argv) { return 0; }
""", []) == """
int main(MY_TYPE argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_variable():
  """Test that an undefined variable does not cause an error."""
  assert clang.Preprocess(
      """
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
"""


def test_Preprocess_undefined_function():
  """Test that an undefined function does not cause an error."""
  assert clang.Preprocess(
      """
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
"""


def test_Preprocess_invalid_preprocessor_directive():
  """Test that an invalid preprocessor directive raises an error."""
  with pytest.raises(errors.ClangException) as e_info:
    clang.Preprocess(
        """
#this_is_not_a_valid_directive
int main(int argc, char** argv) { return 0; }
""", [])
  assert "invalid preprocessing directive" in str(e_info.value)


def test_Preprocess_no_strip_processor_lines():
  """Test that stdin marker is preserved if no strip_preprocessor_lines."""
  assert '# 1 "<stdin>" 2' in clang.Preprocess(
      """
int main(int argc, char** argv) { return 0; }
""", [],
      strip_preprocessor_lines=False)


def test_Preprocess_include_stdio_strip():
  """Test that an included file is stripped."""
  out = clang.Preprocess(
      """
#include <stdio.h>
int main(int argc, char** argv) { return NULL; }
""", [])
  # NULL expands to either "((void *)0)" or "((void*)0)". Accept either.
  assert out == """\
int main(int argc, char** argv) { return ((void *)0); }
""" or out == """\
int main(int argc, char** argv) { return ((void*)0); }
"""


# ClangFormat() tests.


def test_ClangFormat_process_command(mocker):
  """Test the clang-format comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang.ClangFormat('', '.cpp')
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']


def test_ClangFormat_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang-format returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(errors.ClangTimeout):
    clang.ClangFormat('', '.cpp')
  # ClangTimeout inherits from ClangException.
  with pytest.raises(errors.ClangException):
    clang.ClangFormat('', '.cpp')


def test_ClangFormat_ClangException(mocker):
  """Test that ClangException is raised if clang-format returns non-zero."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(errors.ClangFormatException):
    clang.ClangFormat('', '.cpp')


def test_ClangFormat_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang.ClangFormat('', '.cpp') == ''


# CompileLlvmBytecode() tests.


def test_CompileLlvmBytecode_command(mocker):
  """Test the clang comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang.CompileLlvmBytecode('', '.c', ['-foo'])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']
  assert cmd[5:] == ['-S', '-emit-llvm', '-o', '-', '-foo']


def test_CompileLlvmBytecode_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(errors.ClangTimeout):
    clang.CompileLlvmBytecode('', '.c', [])
  # ClangTimeout inherits from ClangException.
  with pytest.raises(errors.ClangException):
    clang.CompileLlvmBytecode('', '.c', [])


def test_CompileLlvmBytecode_ClangException(mocker):
  """Test that ClangException is raised if clang returns non-zero."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(errors.ClangException):
    clang.CompileLlvmBytecode('', '.c', [])


def test_CompileLlvmBytecode_empty_c_file():
  """Test that bytecode is produced for an empty C program."""
  bc = clang.CompileLlvmBytecode('', '.c', [])
  assert 'source_filename =' in bc
  assert 'target datalayout =' in bc
  assert 'target triple =' in bc


def test_CompileLlvmBytecode_small_c_program():
  """Test that bytecode is produced for a small C program."""
  bc = clang.CompileLlvmBytecode('int main() {}', '.c', [])
  # The full bytecode output is too fragile to compare against, e.g. it contains
  # the file path, target layout, exact LLVM build version, etc. Instead we just
  # check for some expected lines.
  assert 'source_filename =' in bc
  assert "define i32 @main() #0 {" in bc
  assert "ret i32 0" in bc


def test_CompileLlvmBytecode_small_cl_program():
  """Test that bytecode is produced for a small OpenCL program."""
  bc = clang.CompileLlvmBytecode('kernel void A() {}', '.cl', [])
  assert 'source_filename =' in bc
  assert 'target datalayout =' in bc
  assert 'target triple =' in bc


def test_CompileLlvmBytecode_c_syntax_error():
  """Test that a ClangException is raised for a C program with syntax error."""
  with pytest.raises(errors.ClangException):
    clang.CompileLlvmBytecode('int main@@()! ##', '.c', [])


# Benchmarks.

HELLO_WORLD_C = """
#include <stdio.h>

int main(int argc, char** argv) {
  printf("Hello, world!\\n");
  return 0;
}
"""


def test_benchmark_Preprocess_c_hello_world(benchmark):
  """Benchmark Preprocess for a "hello world" C program."""
  benchmark(clang.Preprocess, HELLO_WORLD_C, [])


def test_benchmark_ClangFormat_c_hello_world(benchmark):
  """Benchmark ClangFormat for a "hello world" C program."""
  benchmark(clang.ClangFormat, HELLO_WORLD_C, '.cpp')


def test_benchmark_CompileLlvmBytecode_c_hello_world(benchmark):
  """Benchmark CompileLlvmBytecode for a "hello world" C program."""
  benchmark(clang.CompileLlvmBytecode, HELLO_WORLD_C, '.c', [])


if __name__ == '__main__':
  test.Main()
