"""Unit tests for //lib/labm8:fmt."""
import sys

import pytest
from absl import app

from lib.labm8 import fmt


def test_table():
  assert (["foo", "1", "bar", "2"] ==
          fmt.table((("foo", 1), ("bar", 2))).split())


def test_table_columns():
  assert ((["type", "value", "foo", "1", "bar", "2"]) ==
          fmt.table((("foo", 1), ("bar", 2)),
                    columns=("type", "value")).split())


def test_table_bad_columns():
  with pytest.raises(fmt.Error):
    fmt.table((("foo", 1), ("bar", 2)),
              columns=("type", "value", "too", "many", "values"))


def test_table_bad_rows():
  with pytest.raises(fmt.Error):
    fmt.table((("foo", 1), ("bar", 2), ("car",)))


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
