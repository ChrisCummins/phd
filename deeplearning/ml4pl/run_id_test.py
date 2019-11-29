"""Unit tests for //deeplearning/ml4pl:run_id."""
import datetime
import time

import sqlalchemy as sql

from deeplearning.ml4pl import run_id
from labm8.py import sqlutil
from labm8.py import test

FLAGS = test.FLAGS


def test_RunId_FromString():
  run_id_ = run_id.RunId.FromString("foo:191101110001:example01")
  assert run_id_.script_name == "foo"
  assert run_id_.timestamp == "191101110001"
  assert run_id_.hostname == "example01"
  assert run_id_.datetime == datetime.datetime(
    year=2019, month=11, day=1, hour=11, minute=0, second=1
  )


def test_RunId_compare_equal():
  a = run_id.RunId.FromString("foo:191101110001:example01")
  b = run_id.RunId.FromString("foo:191101110001:example01")
  assert a == b


def test_RunId_compare_differnt_script_name():
  a = run_id.RunId.FromString("foo:191101110001:example01")
  b = run_id.RunId.FromString("bar:191101110001:example01")
  assert a != b


def test_RunId_compare_different_hostname():
  a = run_id.RunId.FromString("foo:191101110001:example01")
  b = run_id.RunId.FromString("foo:191101110001:example02")
  assert a != b


def test_RunId_compare_different_timestamp():
  a = run_id.RunId.FromString("foo:191101110001:example01")
  b = run_id.RunId.FromString("foo:191101110002:example01")
  assert a != b


def test_SqlStringColumn_default():
  """Test that default run ID is set on table."""
  base = sql.ext.declarative.declarative_base()

  class Table(base):
    """Table for testing."""

    __tablename__ = "table"
    id = sql.Column(sql.Integer, primary_key=True)
    run_id = run_id.RunId.SqlStringColumn()

  db = sqlutil.Database("sqlite://", base)
  with db.Session(commit=True) as s:
    s.add(Table())
    s.commit()
    table = s.query(Table).first()

    assert isinstance(table.run_id, str)
    assert run_id.RUN_ID == table.run_id
    assert table.run_id == run_id.RUN_ID


def test_SqlStringColumn_no_default():
  """Test when no default value is set."""
  base = sql.ext.declarative.declarative_base()

  class Table(base):
    """Table for testing."""

    __tablename__ = "table"
    id = sql.Column(sql.Integer, primary_key=True)
    run_id = run_id.RunId.SqlStringColumn(default=None)

  db = sqlutil.Database("sqlite://", base)
  with db.Session(commit=True) as s:
    s.add(Table(run_id="foo"))
    s.commit()
    table = s.query(Table).first()

    assert table.run_id == "foo"


def test_ComputeRunId_unique_timestamps():
  """Compute multiple run IDs and check that they change."""
  previous_run_id = None
  end_time = time.time() + 10
  while time.time() < end_time:
    run_id_ = run_id.RunId.GenerateGlobalUnique()
    # Check the length of the run ID.
    assert len(run_id_) <= 40
    # Check that run ID has changed.
    assert run_id_ != previous_run_id
    previous_run_id = run_id_


if __name__ == "__main__":
  test.Main()
