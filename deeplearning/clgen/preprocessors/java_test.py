"""Unit tests for //deeplearning/clgen/preprocessors/java.py."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import java


FLAGS = flags.FLAGS


# ClangFormat() tests.

def test_ClangFormat_hello_world():
  """Test formatting of a "hello world" Java program."""
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


def test_ClangFormat_long_line():
  """Test that extremely long lines are not wrapped."""
  assert java.ClangFormat("""
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {
}
""") == """
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {}
"""


# Compile() tests.

def test_Compile_empty_input():
  """That an empty file is rejected."""
  with pytest.raises(errors.BadCodeException):
    java.Compile('')


def test_Compile_hello_world():
  """Test compilation of a "hello world" Java program."""
  assert java.Compile("""
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
""") == """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""


def test_Compile_class_name_whitespace():
  """Test that compile can infer the class name despite whitespace."""
  assert java.Compile("""
public
class     HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
""") == """
public
class     HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
