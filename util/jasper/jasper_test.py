"""Unit tests for //util/jasper."""

import pathlib
import pytest

from labm8 import app
from labm8 import test
from util.jasper import jasper

FLAGS = app.FLAGS


def _mock_edit_callback(path: pathlib.Path):
  assert path.is_file()


def _mock_edit_callback_with_query(path: pathlib.Path):
  assert path.is_file()
  with open(path, 'w') as f:
    f.write('select COUNT(*) from database.table')


def test_getQueryFromUserOrDie_no_change():
  with pytest.raises(SystemExit):
    jasper.getQueryFromUserOrDie(_mock_edit_callback)


def test_getQueryFromUserOrDie_with_change():
  # Note that getQueryFromUserOrDie() runs sqlformat on the query, hence the
  # change in formatting and capitalisation.
  assert jasper.getQueryFromUserOrDie(_mock_edit_callback_with_query) == """\
SELECT count(*)
FROM database.table\
"""


if __name__ == '__main__':
  test.Main()
