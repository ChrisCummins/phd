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
"""Unit tests for ///common_test.py."""

import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import common
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

# MinimumLineCount3() tests.


def test_MinimumLineCount3_empty_input():
  """Test that MinimumLineCount3 rejects an empty input."""
  with pytest.raises(errors.NoCodeException):
    common.MinimumLineCount3('')


def test_MinimumLineCount3_whitespace_does_not_count():
  """Test that MinimumLineCount3 rejects ignores whitespace."""
  with pytest.raises(errors.NoCodeException):
    common.MinimumLineCount3('\n\n  \n\n  \n\n   \n\n')


def test_MinimumLineCount3_simple_program():
  """Test that MinimumLineCount3 accepts a program with 3 lines."""
  assert common.MinimumLineCount3("""
int main(int argc, char** argv) {
  return 0;
}  
""") == """
int main(int argc, char** argv) {
  return 0;
}  
"""


# StripDuplicateEmptyLines() tests.


def test_StripDuplicateEmptyLines_empty_input():
  """Test StripDuplicateEmptyLines accepts an empty input."""
  assert common.StripDuplicateEmptyLines('') == ''


# Benchmarks.

HELLO_WORLD_C = """
#include <stdio.h>

int main(int argc, char** argv) {
  printf("Hello, world!\\n");
  return 0;
}
"""


def test_benchmark_MinimumLineCount3_c_hello_world(benchmark):
  """Benchmark MinimumLineCount3 on a "hello world" C program."""
  benchmark(common.MinimumLineCount3, HELLO_WORLD_C)


def test_benchmark_StripDuplicateEmptyLines_c_hello_world(benchmark):
  """Benchmark StripDuplicateEmptyLines on a "hello world" C program."""
  benchmark(common.StripDuplicateEmptyLines, HELLO_WORLD_C)


if __name__ == '__main__':
  test.Main()
