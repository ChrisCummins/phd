# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl:run_id."""
import datetime
import multiprocessing
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


def _MakeARunId(*args):
  """Generate a run ID."""
  return run_id.RunId.GenerateGlobalUnique()


def test_GenerateGlobalUnique_multiprocessed():
  """Generate a bunch of run IDs concurrently and check that they are unique."""
  with multiprocessing.Pool() as p:
    run_ids = list(p.map(_MakeARunId, range(50)))

  assert len(run_ids) == len(set(run_ids))


if __name__ == "__main__":
  test.Main()
