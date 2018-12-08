"""Unit tests for //labm8:sqlutil."""

import pathlib
import sys
import tempfile

import pytest
from absl import app

from labm8 import sqlutil


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  """A pytest fixture for a temporary directory."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


def test_CreateEngine_sqlite_not_found(tempdir: pathlib.Path):
  """Test DatabaseNotFound for non-existent SQLite database."""
  with pytest.raises(sqlutil.DatabaseNotFound) as e_ctx:
    sqlutil.CreateEngine(f'sqlite:///{tempdir.absolute()}/db.db',
                         create_if_not_exist=False)
  assert e_ctx.value.url == f'sqlite:///{tempdir.absolute()}/db.db'
  assert str(e_ctx.value) == (f"Database not found: "
                              f"'sqlite:///{tempdir.absolute()}/db.db'")


def test_CreateEngine_sqlite_invalid_relpath():
  """Test that relative paths are disabled."""
  with pytest.raises(ValueError) as e_ctx:
    sqlutil.CreateEngine(f'sqlite:///relative.db')
  assert str(e_ctx.value) == "Relative path to SQLite database is not allowed"


def test_CreateEngine_sqlite_created(tempdir: pathlib.Path):
  """Test that SQLite database is found."""
  sqlutil.CreateEngine(f'sqlite:///{tempdir.absolute()}/db.db',
                       create_if_not_exist=True)
  assert (tempdir / 'db.db').is_file()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
