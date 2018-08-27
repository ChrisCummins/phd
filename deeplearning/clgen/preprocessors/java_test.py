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


# WrapMethodInClass() tests.

def test_Compile_WrapMethodInClass_hello_world():
  """Test output of wrapping a method in a class."""
  assert java.Compile(java.WrapMethodInClass("""\
private static void Hello() {
  System.out.println("Hello, world!");
}""")) == """\
public class A {
  private static void Hello() {
  System.out.println("Hello, world!");
}
}
"""


def test_Compile_WrapMethodInClass_syntax_error():
  """Test that error is raised if method contains a syntax error."""
  with pytest.raises(errors.BadCodeException):
    java.Compile(java.WrapMethodInClass("!@///"))


def test_Compile_WrapMethodInClass_undefined_symbol():
  """Test that error is raised if method has undefined symbols."""
  with pytest.raises(errors.BadCodeException):
    java.Compile(java.WrapMethodInClass("""
private static void Hello() {
  UndefinedMethod(5);
}
"""))


# InsertShimImports() tests.


def test_Compile_InsertShimImports_WrapMethodInClass_array_list():
  """Test output of wrapping a method in a class."""
  assert """\
private static void Hello() {
  ArrayList<Object> a = new ArrayList<>();
  System.out.println("Hello, world!");
}
""" in java.Compile(java.InsertShimImports(java.WrapMethodInClass("""\
private static void Hello() {
  ArrayList<Object> a = new ArrayList<>();
  System.out.println("Hello, world!");
}
""")))


# JavaRewrite() tests.

def test_JavaRewrite_hello_world():
  """Java rewriter returns unmodified input for bad code."""
  assert java.JavaRewrite("hello, world") == "hello, world\n"


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
