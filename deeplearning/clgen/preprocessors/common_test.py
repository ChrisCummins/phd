"""Unit tests for ///common_test.py."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import common


FLAGS = flags.FLAGS


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


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
