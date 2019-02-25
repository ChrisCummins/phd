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
"""Unit tests for //deeplearning/clgen/preprocessors/java.py."""

import pytest
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import java
from labm8 import test

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
  assert java.Compile(
      java.WrapMethodInClass("""\
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
    java.Compile(
        java.WrapMethodInClass("""
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
""" in java.Compile(
      java.InsertShimImports(
          java.WrapMethodInClass("""\
private static void Hello() {
  ArrayList<Object> a = new ArrayList<>();
  System.out.println("Hello, world!");
}
""")))


# JavaRewrite() tests.


def test_JavaRewrite_hello_world():
  """Java rewriter returns unmodified input for bad code."""
  assert java.JavaRewrite("hello, world") == "hello, world\n"


def test_JavaRewrite_delete_comment():
  """Comment(s) are deleted."""
  assert java.JavaRewrite('/* This is a comment */') == '\n'
  assert java.JavaRewrite('//Line comment') == '\n'
  assert java.JavaRewrite("""
/**
 * Docstring format comment.
 */
// And a line comment.
/* And a C syntax style comment. */
""") == '\n'


def test_JavaRewrite_whitespace():
  """Multiple blank lines and whitespace is stripped."""
  assert java.JavaRewrite('\n\n  \n\t\n') == '\n'


def test_JavaRewrite_rewrite_class_name():
  """Test that class is renamed."""
  assert java.JavaRewrite("""
public class MyJavaClass {
}
""") == """\
public class A {
}
"""


def test_JavaRewrite_rewrite_anonymous_class_names():
  """Test that anonymous classes are renamed."""
  assert java.JavaRewrite("""
public class MyJavaClass {
  private class AnonymousClassA {
  }
  
  private class AnotherPrivateClass {
  }
}
""") == """\
public class A {
\tprivate class B {
\t}

\tprivate class C {
\t}
}
"""


def test_JavaRewrite_rewrite_static_method_argument_names():
  """Test that arguments are renamed."""
  assert java.JavaRewrite("""
public class A {
  public static int myMethod(final int foo, int bar) {
    System.out.println("Hello world! " + bar); 
    return foo + bar;
  }
}
""") == """\
public class A {
\tpublic static int fn_A(final int a, int b) {
\t\tSystem.out.println("Hello world! " + b);
\t\treturn a + b;
\t}
}
"""
  """
private static boolean slowEquals(byte[] a,byte[] b){
  int diff=a.length ^ b.length;
  for (int i=0; i < a.length && i < b.length; i++)   diff|=a[i] ^ b[i];
  return diff == 0;
}
"""


if __name__ == '__main__':
  test.Main()
