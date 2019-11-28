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
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# ClangFormat() tests.


def test_ClangFormat_hello_world():
  """Test formatting of a "hello world" Java program."""
  assert (
    java.ClangFormat(
      """
public class HelloWorld {
    public static void main(String [] args) {
        System.out.println("Hello, World"    );
    } }
"""
    )
    == """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""
  )


def test_ClangFormat_long_line():
  """Test that extremely long lines are not wrapped."""
  assert (
    java.ClangFormat(
      """
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {
}
"""
    )
    == """
public class VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
extends VeryVeryLongNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBase {}
"""
  )


# Compile() tests.


def test_Compile_empty_input():
  """That an empty file is rejected."""
  with pytest.raises(errors.BadCodeException):
    java.Compile("")


def test_Compile_hello_world():
  """Test compilation of a "hello world" Java program."""
  assert (
    java.Compile(
      """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""
    )
    == """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""
  )


def test_Compile_class_name_whitespace():
  """Test that compile can infer the class name despite whitespace."""
  assert (
    java.Compile(
      """
public
class     HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""
    )
    == """
public
class     HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""
  )


# WrapMethodInClass() tests.


def test_Compile_WrapMethodInClass_hello_world():
  """Test output of wrapping a method in a class."""
  assert (
    java.Compile(
      java.WrapMethodInClass(
        """\
private static void Hello() {
  System.out.println("Hello, world!");
}"""
      )
    )
    == """\
public class A {
private static void Hello() {
  System.out.println("Hello, world!");
}
}
"""
  )


def test_Compile_WrapMethodInClass_syntax_error():
  """Test that error is raised if method contains a syntax error."""
  with pytest.raises(errors.BadCodeException):
    java.Compile(java.WrapMethodInClass("!@///"))


def test_Compile_WrapMethodInClass_undefined_symbol():
  """Test that error is raised if method has undefined symbols."""
  with pytest.raises(errors.BadCodeException):
    java.Compile(
      java.WrapMethodInClass(
        """
private static void Hello() {
  UndefinedMethod(5);
}
"""
      )
    )


# UnwrapMethodInClass() tests.


def test_UnwrapMethodInClass_hello_world():
  """Test that method is extracted from a class."""
  assert (
    java.UnwrapMethodInClass(
      """
public class HelloWorld {
  private static void Hello() {
    System.out.println("Hello, world!");
  }
}"""
    )
    == """\
private static void Hello(){
  System.out.println("Hello, world!");
}
"""
  )


def test_UnwrapMethodInClass_no_methods():
  """Test that error is raised if class doesn't contain a method."""
  with pytest.raises(errors.BadCodeException) as e_ctx:
    java.UnwrapMethodInClass("public class HelloWorld {}")
  assert str(e_ctx.value) == "Expected 1 method, found 0"


def test_UnwrapMethodInClass_multiple_methods():
  """Test that error is raised if class contains multiple methods."""
  with pytest.raises(errors.BadCodeException) as e_ctx:
    java.UnwrapMethodInClass(
      """
public class HelloWorld {
  public static void Hello() {}
  public static void Goodbye() {}
}"""
    )
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
      java.WrapMethodInClass(
        """\
private static void Hello() {
  ArrayList<Object> a = new ArrayList<>();
  System.out.println("Hello, world!");
}
"""
      )
    )
  )


# JavaRewrite() tests.


def test_JavaRewrite_hello_world():
  """Java rewriter returns unmodified input for bad code."""
  assert java.JavaRewrite("hello, world") == "hello, world\n"


def test_JavaRewrite_comments_are_unchanged():
  """Comment(s) are not deleted."""
  assert (
    java.JavaRewrite(
      """
/**
 * Docstring format comment.
 */
// And a line comment.
/* And a C syntax style comment. */
"""
    )
    == """\
/**
 * Docstring format comment.
 */
// And a line comment.
/* And a C syntax style comment. */
"""
  )


def test_JavaRewrite_whitespace():
  """Multiple blank lines and whitespace is stripped."""
  assert java.JavaRewrite("\n\n  \n\t\n") == "\n"


def test_JavaRewrite_rewrite_class_name():
  """Test that class is renamed."""
  assert (
    java.JavaRewrite(
      """
public class MyJavaClass {
}
"""
    )
    == """\
public class A {
}
"""
  )


def test_JavaRewrite_rewrite_anonymous_class_names():
  """Test that anonymous classes are renamed."""
  assert (
    java.JavaRewrite(
      """
public class MyJavaClass {
  private class AnonymousClassA {
  }
  
  private class AnotherPrivateClass {
  }
}
"""
    )
    == """\
public class A {
\tprivate class B {
\t}
\tprivate class C {
\t}
}
"""
  )


def test_JavaRewrite_rewrite_static_method_argument_names():
  """Test that arguments are renamed."""
  assert (
    java.JavaRewrite(
      """
public class A {
  public static int myMethod(final int foo, int bar) {
    System.out.println("Hello world! " + bar); 
    return foo + bar;
  }
}
"""
    )
    == """\
public class A {
\tpublic static int fn_A(final int a, int b) {
\t\tSystem.out.println("Hello world! " + b);
\t\treturn a + b;
\t}
}
"""
  )


def test_JavaRewrite_conflicting_length_name():
  """Test that rewriter gracefully handles 'length' used as variable."""
  assert (
    java.JavaRewrite(
      """
public class A {
  public static double[] createEntry(final double[] position){
    int length=position.length;
    int sqrt=(int)Math.sqrt(9);
    
    return 0.0;
  }
}
"""
    )
    == """\
public class A {
\tpublic static double[] fn_A(final double[] a) {
\t\tint b = a.length;
\t\tint c = (int) Math.sqrt(9);
\t\treturn 0.0;
\t}
}
"""
  )


def test_JavaRewrite_formats_source():
  """Test that source is formatted."""
  assert (
    java.JavaRewrite(
      """
public class A { public      static
                             void Foo(int a) { return
a;}
}
"""
    )
    == """\
public class A {
\tpublic static void fn_A(int a) {
\t\treturn a;
\t}
}
"""
  )


def test_JavaRewrite_assertion_arg_rename_FAILS():
  """This test highlights a failure of the JavaRewriter.

  Instead of correctly renaming 'x' -> 'a' in the assertion, the assertion is
  treated as the declaration of a new variable (of type 'assert'), and named
  'b'.
  """
  assert (
    java.JavaRewrite(
      """
public class A {
\tpublic static void fn_A(int x) {
\t\tassert x;
\t}
}
"""
    )
    == """\
public class A {
\tpublic static void fn_A(int a) {
\t\tassert b;
\t}
}
"""
  )


def test_JavaRewrite_optional_if_else_braces():
  """Test that if/else braces are inserted."""
  assert (
    java.JavaRewrite(
      """
public class A {
\tpublic static int fn_A(int x) {
\t\tif (x)
\t\t\treturn 1;
\t\telse
\t\t\treturn 0;
\t}
}
"""
    )
    == """\
public class A {
\tpublic static int fn_A(int a) {
\t\tif (a) {
\t\t\treturn 1;
\t\t} else {
\t\t\treturn 0;
\t\t}
\t}
}
"""
  )


def test_JavaRewrite_optional_if_else_braces_on_one_line():
  """Test that if/else braces are inserted when declaration is on one line."""
  assert (
    java.JavaRewrite(
      """
public class A {
\tpublic static int fn_A(int x) {
\t\tif (x) return 1; else return 0;
\t}
}
"""
    )
    == """\
public class A {
\tpublic static int fn_A(int a) {
\t\tif (a) {
\t\t\treturn 1;
\t\t} else {
\t\t\treturn 0;
\t\t}
\t}
}
"""
  )


def test_JavaRewrite_github_testcase_1():
  """Regression test found from scraping GitHub.

  This is an interesting file because it flexes the rewriter's ability to insert
  blocks around one-liner `for` and `if` constructs.

  Original source:
      github.com/0001077192/discord-spicybot
      src/main/java/com/nsa/spicybot/commands/SpicyPointsCommand.java
  """
  assert (
    java.JavaRewrite(
      """
public class A {
public static String format(int num){
  String original="" + num;
  String dummy=original.length() % 3 != 0 ? original.substring(0,original.length() % 3) + "," : "";
  for (int i=original.length() % 3; i < original.length(); i+=3)   dummy+=original.substring(i,i + 3) + ",";
  if (dummy.endsWith(","))   dummy=dummy.substring(0,dummy.length() - 1);
  return dummy;
}
}
"""
    )
    == """\
public class A {
\tpublic static String fn_A(int a) {
\t\tString b = "" + a;
\t\tString c = b.length() % 3 != 0 ? b.substring(0,b.length() % 3) + "," : "";
\t\tfor (int d = b.length() % 3; d < b.length(); d += 3) {
\t\t\tc += b.substring(d, d + 3) + ",";
\t\t}
\t\tif (c.endsWith(",")) {
\t\t\tc = c.substring(0, c.length() - 1);
\t\t}
\t\treturn c;
\t}
}
"""
  )


if __name__ == "__main__":
  test.Main()
