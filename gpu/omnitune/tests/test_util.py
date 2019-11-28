from unittest import main

from omnitune import util

from labm8.py.tests.testutil import TestCase


class TestUtil(TestCase):

  # parse_str()
  def test_parse_str(self):
    self._test("abc", util.parse_str("abc"))
    self._test("a\nc", util.parse_str("a\nc"))
    self._test("123", util.parse_str("123"))


if __name__ == '__main__':
  main()
