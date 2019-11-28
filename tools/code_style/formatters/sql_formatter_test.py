"""Unit tests for //tools/code_style/formatters:sql_formatter."""
from labm8.py import test
from tools.code_style.formatters import sql_formatter

FLAGS = test.FLAGS


def test_FormatSql_empty_string():
  assert sql_formatter.FormatSql("") == ""


def test_FormatSql_hello_world():
  assert sql_formatter.FormatSql("Hello world") == "Hello world"


def test_FormatSql_sql_query():
  assert sql_formatter.FormatSql("select count(*),foo.bar from foo;") == """\
SELECT count(*),
       foo.bar
FROM foo;"""


if __name__ == '__main__':
  test.Main()
