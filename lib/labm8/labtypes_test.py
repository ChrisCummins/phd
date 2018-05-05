"""Unit tests for //lib/labm8:fmt."""
import sys

import pytest
from absl import app

from lib.labm8 import labtypes


# is_str()
def test_is_str():
  assert labtypes.is_str("Hello, World!")
  assert labtypes.is_str(str("Hello, World!"))
  assert not labtypes.is_str("Hello, World!".encode("utf-8"))
  assert not labtypes.is_str(bytes("Hello, World!".encode("utf-8")))
  assert not labtypes.is_str(8)
  assert not labtypes.is_str(['a', 'b', 'c'])
  assert not labtypes.is_str({'a': 'b', 'c': 18})


def test_is_str_seq():
  assert not labtypes.is_str(tuple([1]))
  assert not labtypes.is_str((1, 2))
  assert not labtypes.is_str([1])
  assert not labtypes.is_str([1, 2])


def test_is_str_num():
  assert not labtypes.is_str(1)
  assert not labtypes.is_str(1.3)


def test_is_str_dict():
  assert not labtypes.is_str({"foo": 100})
  assert not labtypes.is_str({10: ["a", "b", "c"]})


# is_dict() tests
def test_is_dict():
  assert labtypes.is_dict({"foo": 100})
  assert labtypes.is_dict({10: ["a", "b", "c"]})


def test_is_dict_str():
  assert not labtypes.is_dict("a")
  assert not labtypes.is_dict("abc")
  assert not labtypes.is_dict(["abc", "def"][0])


def test_is_dict_seq():
  assert not labtypes.is_dict(tuple([1]))
  assert not labtypes.is_dict((1, 2))
  assert not labtypes.is_dict([1])
  assert not labtypes.is_dict([1, 2])


def test_is_dict_num():
  assert not labtypes.is_dict(1)
  assert not labtypes.is_dict(1.3)


# is_seq() tests
def test_is_seq():
  assert labtypes.is_seq(tuple([1]))
  assert labtypes.is_seq((1, 2))
  assert labtypes.is_seq([1])
  assert labtypes.is_seq([1, 2])


def test_is_seq_str():
  assert not labtypes.is_seq("a")
  assert not labtypes.is_seq("abc")
  assert not labtypes.is_seq(["abc", "def"][0])


def test_is_seq_num():
  assert not labtypes.is_seq(1)
  assert not labtypes.is_seq(1.3)


def test_is_seq_dict():
  assert not labtypes.is_seq({"foo": 100})
  assert not labtypes.is_seq({10: ["a", "b", "c"]})


# flatten()
def test_flatten():
  assert labtypes.flatten([[1], [2, 3]]) == [1, 2, 3]


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
