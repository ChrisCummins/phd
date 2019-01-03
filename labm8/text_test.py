"""Unit tests for //labm8:text."""

import pytest
from absl import flags

from labm8 import test
from labm8 import text


FLAGS = flags.FLAGS


# get_substring_idxs()
def test_get_substring_idxs():
  assert [0, 2] == text.get_substring_idxs('a', 'aba')
  assert not text.get_substring_idxs('a', 'bb')


# truncate()
def test_truncate():
  assert "foo" == text.truncate("foo", 100)
  assert "1234567890" == text.truncate("1234567890", 10)
  assert "12345..." == text.truncate("1234567890", 8)
  for i in range(10, 20):
    assert i == len(text.truncate("The quick brown fox jumped "
                                  "over the slow lazy dog", i))


def test_truncate_bad_maxchar():
  with pytest.raises(text.TruncateError):
    text.truncate("foo", -1)
    text.truncate("foo", 3)


# distance()
def test_levenshtein():
  assert 0 == text.levenshtein("foo", "foo")
  assert 1 == text.levenshtein("foo", "fooo")
  assert 3 == text.levenshtein("foo", "")
  assert 1 == text.levenshtein("1234", "1 34")
  assert 1 == text.levenshtein("123", "1 3")


# diff()
def test_diff():
  assert 0 == text.diff("foo", "foo")
  assert 0.25 == text.diff("foo", "fooo")
  assert 1 == text.diff("foo", "")
  assert 0.25 == text.diff("1234", "1 34")
  assert (1 / 3) == text.diff("123", "1 3")


if __name__ == '__main__':
  test.Main()
