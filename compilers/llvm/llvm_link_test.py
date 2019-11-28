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
"""Unit tests for //compilers/llvm/llvm_link.py."""
import pathlib

import pytest

from compilers.llvm import llvm_link
from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


def test_LinkBitcodeFilesToBytecode_empty_file(tempdir: pathlib.Path):
  """llvm-link with an empty file."""
  input_path = tempdir / "empty.ll"
  output_path = tempdir / "linked.ll"
  fs.Write(input_path, "".encode("utf-8"))
  llvm_link.LinkBitcodeFilesToBytecode(
    [input_path], output_path, timeout_seconds=5
  )
  assert output_path.is_file()


def test_LinkBitcodeFilesToBytecode_syntax_error(tempdir: pathlib.Path):
  """llvm-link fails when a file contains invalid syntax."""
  input_path = tempdir / "empty.ll"
  output_path = tempdir / "linked.ll"
  fs.Write(input_path, "syntax error!".encode("utf-8"))
  with test.Raises(ValueError) as e_ctx:
    llvm_link.LinkBitcodeFilesToBytecode(
      [input_path], output_path, timeout_seconds=5
    )
  assert str(e_ctx.value).startswith("Failed to link bytecode: ")


if __name__ == "__main__":
  test.Main()
