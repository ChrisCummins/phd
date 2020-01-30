# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //datasets/github/scrape_repos/preprocessors/extractors.py"""
from datasets.github.scrape_repos.preprocessors import extractors
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# JavaMethods() tests.


def test_JavaMethods_hello_world():
  """Test that no methods in "hello world"."""
  assert extractors.JavaMethods(None, None, "Hello, world!", None) == []


def test_JavaMethods_simple_file():
  """Test output of simple class file."""
  assert (
    extractors.JavaMethods(
      None,
      None,
      """
public class A {

  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }

  private int foo() { /* comment */ return 5; }
}
""",
      None,
    )
    == [
      """\
public static void main(String[] args){
  System.out.println("Hello, world!");
}
""",
      """\
private int foo(){
  return 5;
}
""",
    ]
  )


def test_JavaMethods_syntax_error():
  """Test that syntax errors are silently ignored."""
  assert (
    extractors.JavaMethods(
      None,
      None,
      """
public class A {

  public static void main(String[] args) {
    /@! syntax error!
  }
}
""",
      None,
    )
    == ["public static void main(String[] args){\n}\n"]
  )


# BatchedMethodExtractor() tests.


def test_BatchedMethodExtractor_valid_input():
  extractors.BatchedMethodExtractor(
    [
      """
public class A {
  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
}
    """,
      """
public class A {
  public static void main(String[] args) {
    System.out.println("Hi, world!");
  }
}
    """,
    ]
  ) == [
    """\
public static void main(String[] args) {
  System.out.println("Hello, world!");
}
""",
    """\
public static void main(String[] args) {
  System.out.println("Hi, world!");
}
""",
  ]


# JavaStaticMethods() tests.


def test_JavaStaticMethods_simple_file():
  """Test output of simple class file."""
  assert (
    extractors.JavaStaticMethods(
      None,
      None,
      """
public class A {

  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }

  private int foo() { /* comment */ return 5; }
}
""",
      None,
    )
    == [
      """\
public static void main(String[] args){
  System.out.println("Hello, world!");
}
"""
    ]
  )


def test_JavaStaticMethods_docstring_is_stripped():
  """Test that dosctring is not exported."""
  assert (
    extractors.JavaStaticMethods(
      None,
      None,
      """
public class A {

  /**
   * This is a docstring
   * @param args   Arguments
   */
  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
}
""",
      None,
    )
    == [
      """\
public static void main(String[] args){
  System.out.println("Hello, world!");
}
"""
    ]
  )


if __name__ == "__main__":
  test.Main()
