# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //tools/format/formatters:shell."""
from labm8.py import test
from tools.format.formatters import shell

FLAGS = test.FLAGS


def test_small_shell_program():
  """Test pre-processing a small shell program."""
  text = shell.FormatShell.Format(
    """
foo() {
    echo hello; ls
}"""
  )
  print(text)
  assert (
    text
    == """\
foo() {
  echo hello
  ls
}
"""
  )


def test_small_bats_program():
  """Test pre-processing a small bats program."""
  text = shell.FormatShell.Format(
    """
@test "run version" {
    "$BIN"  --version
}""",
    assumed_filename="test.bats",
  )
  print(text)
  assert (
    text
    == """\
@test "run version" {
  "$BIN" --version
}
"""
  )


if __name__ == "__main__":
  test.Main()
