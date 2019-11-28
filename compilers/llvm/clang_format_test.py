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
"""Unit tests for //compilers/llvm/clang_format.py."""
import subprocess

import pytest

from compilers.llvm import clang_format
from compilers.llvm import llvm
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class MockProcess:
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, *args):
    del args
    return "", ""


# Exec() tests.


def test_Exec_process_command(mocker):
  """Test the clang-format command which is run."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(0)
  clang_format.Exec("", ".cpp", [])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ["timeout", "-s9", "60"]


def test_Exec_timeout(mocker):
  """Test that error is raised if clang-format returns with SIGKILL."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(llvm.LlvmTimeout):
    clang_format.Exec("", ".cpp", [])
  # ClangTimeout inherits from ClangException.
  with pytest.raises(llvm.LlvmError):
    clang_format.Exec("", ".cpp", [])


def test_Exec_error(mocker):
  """Test that error is raised if clang-format returns non-zero."""
  mock_Popen = mocker.patch("subprocess.Popen")
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(clang_format.ClangFormatException):
    clang_format.Exec("", ".cpp", [])


def test_Exec_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang_format.Exec("", ".cpp", []) == ""


if __name__ == "__main__":
  test.Main()
