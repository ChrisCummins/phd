"""Unit tests for //labm8:shell."""
from labm8 import shell
from labm8 import test


def test_EscapeList():
  # Empty list
  words = []
  assert shell.ShellEscapeList(words) == ''

  # Empty string
  words = ['']
  assert shell.ShellEscapeList(words) == "''"

  # Single word
  words = ['foo']
  assert shell.ShellEscapeList(words) == "'foo'"

  # Single word with single quote
  words = ["foo'bar"]
  expected = """   'foo'"'"'bar'   """.strip()
  assert shell.ShellEscapeList(words) == expected
  # .. double quote
  words = ['foo"bar']
  expected = """   'foo"bar'   """.strip()
  assert shell.ShellEscapeList(words) == expected

  # Multiple words
  words = ['foo', 'bar']
  assert shell.ShellEscapeList(words) == "'foo' 'bar'"

  # Words with spaces
  words = ['foo', 'bar', "foo'' ''bar"]
  expected = """   'foo' 'bar' 'foo'"'"''"'"' '"'"''"'"'bar'   """.strip()
  assert shell.ShellEscapeList(words) == expected

  # Now I'm just being mean
  words = ['foo', 'bar', """   ""'"'"   """.strip()]
  expected = """   'foo' 'bar' '""'"'"'"'"'"'"'   """.strip()
  assert shell.ShellEscapeList(words) == expected


if __name__ == '__main__':
  test.Main()
