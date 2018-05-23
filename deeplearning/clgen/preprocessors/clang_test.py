"""Unit tests for //deeplearning/clgen/preprocessors/clang.py."""
import subprocess
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import clang


FLAGS = flags.FLAGS


class MockProcess():
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, input):
    return '', ''


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


def test_Preprocess_simple_c_program():
  """Test that a simple C program is unchanged."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return 0; }
""", []) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_cflag():
  """Test pre-processing with a custom define in the command line."""
  assert clang.Preprocess("""
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_define():
  """Test pre-processing with a #define in the source code."""
  assert clang.Preprocess("""
#define MY_TYPE int
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """

int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_data_type():
  """Test that an undefined data type does not cause an error."""
  assert clang.Preprocess("""
int main(MY_TYPE argc, char** argv) { return 0; }
""", []) == """
int main(MY_TYPE argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_variable():
  """Test that an undefined variable does not cause an error."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
"""


def test_Preprocess_undefined_function():
  """Test that an undefined function does not cause an error."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
"""


def test_Preprocess_invalid_preprocessor_directive():
  """Test that an invalid preprocessor directive raises an error."""
  with pytest.raises(errors.ClangException) as e_info:
    clang.Preprocess("""
#this_is_not_a_valid_directive
int main(int argc, char** argv) { return 0; }
""", [])
  assert "invalid preprocessing directive" in str(e_info.value)


def test_Preprocess_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang.Preprocess('', []) == '\n'


def test_Preprocess_no_strip_processor_lines():
  """Test that stdin marker is preserved if no strip_preprocessor_lines."""
  assert '# 1 "<stdin>" 2' in clang.Preprocess("""
int main(int argc, char** argv) { return 0; }
""", [], strip_preprocessor_lines=False)


@pytest.mark.skip()
def test_Preprocess_include_stdio_strip():
  """Test that an included file is stripped."""
  assert clang.Preprocess("""
#include <stdio.h>
int main(int argc, char** argv) { return NULL; }
""", []) == """
int main(int argc, char** argv) { return 0; }
"""


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
