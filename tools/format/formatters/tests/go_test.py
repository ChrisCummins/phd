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
"""Unit tests for //tools/format/formatters:go."""
from labm8.py import test
from tools.format.formatters import go

FLAGS = test.FLAGS


def test_small_go_program():
  """Test pre-processing a small C++ program."""
  text = go.FormatGo.Format(
    """
package main
import "fmt"
func main() {
  fmt.Println("hello world")
}"""
  )
  assert (
    text
    == """\
package main

import "fmt"

func main() {
\tfmt.Println("hello world")
}
"""
  )


def test_empty_file():
  with test.Raises(go.FormatGo.FormatError):
    go.FormatGo.Format("")


if __name__ == "__main__":
  test.Main()
