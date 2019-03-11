"""Unit tests for //labm8:jsonutil."""

import pytest

from labm8 import app
from labm8 import fs
from labm8 import jsonutil
from labm8 import system
from labm8 import test

FLAGS = app.FLAGS


def test_loads():
  a_str = """{
          "a": 1,  // this has comments
          "b": [1, 2, 3]
      } # end comment
      // begin with comment
      """
  a = jsonutil.loads(a_str)
  assert a == {'a': 1, 'b': [1, 2, 3]}


def test_loads_malformed():
  a_str = """bad json {asd,,}"""
  with pytest.raises(ValueError):
    jsonutil.loads(a_str)


def test_read_file():
  a_str = """{
          "a": 1,  // this has comments
          "b": [1, 2, 3]
      } # end comment
      // begin with comment
      """
  system.echo(a_str, "/tmp/labm8.loaf.json")
  a = jsonutil.read_file("/tmp/labm8.loaf.json")
  assert a == {'a': 1, 'b': [1, 2, 3]}


def test_read_file_bad_path():
  with pytest.raises(fs.File404):
    jsonutil.read_file("/not/a/real/path")
  assert not jsonutil.read_file("/not/a/real/path", must_exist=False)


def test_write_file():
  d1 = {"a": "1", "b": "2"}
  jsonutil.write_file("/tmp/labm8.write_file.json", d1)
  d2 = jsonutil.read_file("/tmp/labm8.write_file.json")
  fs.rm("/tmp/labm8.write_file.json")

  jsonutil.write_file("/tmp/labm8.write_file2.json", d1)
  d3 = jsonutil.read_file("/tmp/labm8.write_file2.json")
  fs.rm("/tmp/labm8.write_file2.json")

  assert d1 == d2 == d3


if __name__ == '__main__':
  test.Main()
