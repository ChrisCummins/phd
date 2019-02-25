"""Unit tests for //compilers/llvm/clang_format.py."""
import subprocess

import pytest
from absl import flags

from compilers.llvm import clang_format
from compilers.llvm import llvm
from labm8 import test

FLAGS = flags.FLAGS


class MockProcess():
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, *args):
    del args
    return '', ''


# Exec() tests.


def test_Exec_process_command(mocker):
  """Test the clang-format command which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang_format.Exec('', '.cpp', [])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']


def test_Exec_timeout(mocker):
  """Test that error is raised if clang-format returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(llvm.LlvmTimeout):
    clang_format.Exec('', '.cpp', [])
  # ClangTimeout inherits from ClangException.
  with pytest.raises(llvm.LlvmError):
    clang_format.Exec('', '.cpp', [])


def test_Exec_error(mocker):
  """Test that error is raised if clang-format returns non-zero."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(clang_format.ClangFormatException):
    clang_format.Exec('', '.cpp', [])


def test_Exec_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang_format.Exec('', '.cpp', []) == ''


if __name__ == '__main__':
  test.Main()
