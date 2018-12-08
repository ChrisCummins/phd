"""Unit tests for //labm8:sqlutil."""

import pathlib
import sys
import tempfile

import pytest
import sqlalchemy as sql
from absl import app
from sqlalchemy.ext import declarative

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


def test_CreateEngine_sqlite_in_memory_not_new():
  """Test that error is raised if in-memory table created when not new."""
  with pytest.raises(ValueError) as e_ctx:
    sqlutil.CreateEngine(f'sqlite://', create_if_not_exist=False)
  assert str(e_ctx.value) == ("create_if_exist=False not valid for in-memory "
                              "SQLite database")


def test_CreateEngine_sqlite_created(tempdir: pathlib.Path):
  """Test that SQLite database is found."""
  sqlutil.CreateEngine(f'sqlite:///{tempdir}/db.db',
                       create_if_not_exist=True)
  assert (tempdir / 'db.db').is_file()


def test_Session_GetOrAdd():
  """Test that GetOrAdd() does not create duplicates."""
  base = declarative.declarative_base()

  class Table(base, sqlutil.TablenameFromClassNameMixin):
    """A table containing a single 'value' primary key."""
    value = sql.Column(sql.Integer, primary_key=True)

  # Create the database.
  db = sqlutil.Database(f'sqlite://', base)

  # Create an entry in the database.
  with db.Session(commit=True) as s:
    s.GetOrAdd(Table, value=42)

  # Check that the entry has been added.
  with db.Session() as s:
    assert s.query(Table).count() == 1
    assert s.query(Table).one().value == 42

  # Create another entry. Since a row already exists in the database, this
  # doesn't add anything.
  with db.Session(commit=True) as s:
    s.GetOrAdd(Table, value=42)

  # Check that the database is unchanged.
  with db.Session() as s:
    assert s.query(Table).count() == 1
    assert s.query(Table).one().value == 42


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
