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

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import java
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

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


# UnwrapMethodInClass() tests.


def test_UnwrapMethodInClass_hello_world():
  """Test that method is extracted from a class."""
  assert java.UnwrapMethodInClass("""
public class HelloWorld {
  private static void Hello() {
    System.out.println("Hello, world!");
  }
}""") == """\
private static void Hello(){
  System.out.println("Hello, world!");
}
"""


def test_UnwrapMethodInClass_no_methods():
  """Test that error is raised if class doesn't contain a method."""
  with pytest.raises(errors.BadCodeException) as e_ctx:
    java.UnwrapMethodInClass("public class HelloWorld {}")
  assert str(e_ctx.value) == "Expected 1 method, found 0"


def test_UnwrapMethodInClass_multiple_methods():
  """Test that error is raised if class contains multiple methods."""
  with pytest.raises(errors.BadCodeException) as e_ctx:
    java.UnwrapMethodInClass("""
public class HelloWorld {
  public static void Hello() {}
  public static void Goodbye() {}
}""")
  assert str(e_ctx.value) == "Expected 1 method, found 2"


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


def test_JavaRewrite_comments_are_unchanged():
  """Comment(s) are not deleted."""
  assert java.JavaRewrite("""
/**
 * Docstring format comment.
 */
// And a line comment.
/* And a C syntax style comment. */
""") == """\
/**
 * Docstring format comment.
 */
// And a line comment.
/* And a C syntax style comment. */
"""


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


def test_JavaRewrite_conflicting_length_name():
  """Test that rewriter gracefully handles 'length' used as variable."""
  assert java.JavaRewrite("""
public class A {
  public static double[] createEntry(final double[] position){
    int length=position.length;
    int sqrt=(int)Math.sqrt(9);

    return 0.0;
  }
}
""") == """\
public class A {
\tpublic static double[] fn_A(final double[] a) {
\t\tint b = a.length;
\t\tint c = (int) Math.sqrt(9);

\t\treturn 0.0;
\t}
}
"""


if __name__ == '__main__':
  test.Main()
