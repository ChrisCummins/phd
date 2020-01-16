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
"""Unit tests for //tools/format/formatters:text."""
from labm8.py import test
from tools.format.formatters import text

FLAGS = test.FLAGS


def test_empty_file():
  assert text.FormatText.Format("") == ""


def test_strip_trailing_whitespace():
  assert text.FormatText.Format("Hello   \n") == "Hello\n"


def test_add_newline():
  assert text.FormatText.Format("Hello") == "Hello\n"


@test.XFail(reason="I have not yet implemented EOF clean-up")
def test_multiple_newlines_at_end_of_file():
  assert text.FormatText.Format("Hello\n\n") == "Hello\n"


if __name__ == "__main__":
  test.Main()
