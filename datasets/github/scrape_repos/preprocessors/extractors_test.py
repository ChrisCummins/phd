"""Unit tests for //datasets/github/scrape_repos/preprocessors/extractors.py"""

from datasets.github.scrape_repos.preprocessors import extractors
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

# JavaMethods() tests.


def test_JavaMethods_hello_world():
  """Test that no methods in "hello world"."""
  assert extractors.JavaMethods(None, None, "Hello, world!", None) == []


def test_JavaMethods_simple_file():
  """Test output of simple class file."""
  assert extractors.JavaMethods(
      None, None, """
public class A {

  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
  
  private int foo() { /* comment */ return 5; }
}
""", None) == [
          """\
public static void main(String[] args){
  System.out.println("Hello, world!");
}
""", """\
private int foo(){
  return 5;
}
"""
      ]


def test_JavaMethods_syntax_error():
  """Test that syntax errors are silently ignored."""
  assert extractors.JavaMethods(
      None, None, """
public class A {

  public static void main(String[] args) {
    /@! syntax error!
  }
}
""", None) == ["public static void main(String[] args){\n}\n"]


# JavaStaticMethods() tests.


def test_JavaStaticMethods_simple_file():
  """Test output of simple class file."""
  assert extractors.JavaStaticMethods(
      None, None, """
public class A {

  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
  
  private int foo() { /* comment */ return 5; }
}
""", None) == [
          """\
public static void main(String[] args){
  System.out.println("Hello, world!");
}
"""
      ]


if __name__ == '__main__':
  test.Main()
