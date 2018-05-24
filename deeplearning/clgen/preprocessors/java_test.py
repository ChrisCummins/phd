"""Unit tests for //deeplearning/clgen/preprocessors/java.py."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen.preprocessors import java


FLAGS = flags.FLAGS


# ClangFormat() tests.

def test_ClangFormat_java_hello_world():
  assert java.ClangFormat("""
public class HelloWorld {
    public static void main(String [] args) {
        System.out.println("Hello, World"    );
    } }
""") == """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""


def test_ClangFormat_java_long_line():
  assert java.ClangFormat("""
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {
}
""") == """
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {}
"""


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
