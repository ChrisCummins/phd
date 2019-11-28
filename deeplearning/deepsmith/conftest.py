# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""Shared test fixture functions.

Fixtures defined in this file may be used in any test in this directory. You
do not need to import this file, it is discovered automatically by pytest.

See the conftest.py documentation for more details:
https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
"""
import pathlib

import pytest

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import datastore_pb2
from labm8.py import pbutil


def _ReadTestDataStoreFiles() -> datastore_pb2.DataStoreTestSet:
  """Read the config protos for testing.

  The datastore names are derived from the file names.

  Returns:
    A DataStoreTestSet instance.

  Raises:
    AssertionError: In case of error reading datastore configs.
  """
  paths = list(
    pathlib.Path("deeplearning/deepsmith/tests/data/datastores").iterdir()
  )
  assert paths
  names = [p.stem for p in paths]
  protos = [pbutil.FromFile(path, datastore_pb2.DataStore()) for path in paths]
  datastore_set = datastore_pb2.DataStoreTestSet()
  for name, proto in zip(names, protos):
    # There's no graceful error handling here, but it's important that we don't
    # run tests on a datastore unless it's specifically marked as testonly.
    assert proto.testonly
    dst_proto = datastore_set.values[name]
    dst_proto.MergeFrom(proto)
  assert len(datastore_set.values) == len(protos) == len(names) == len(paths)
  return datastore_set


# Read the proto files once.
_DATASTORE_TESTSET = _ReadTestDataStoreFiles()


@test.Fixture(
  ids=_DATASTORE_TESTSET.values.keys(),
  params=_DATASTORE_TESTSET.values.values(),
  scope="function",
)
def ds(request) -> datastore.DataStore:
  """Create an in-memory SQLite datastore for testing.

  The database is will be empty.

  Returns:
    A DataStore instance.
  """
  return datastore.DataStore(request.param)


@test.Fixture(
  ids=_DATASTORE_TESTSET.values.keys(),
  params=_DATASTORE_TESTSET.values.values(),
  scope="function",
)
def session(request) -> db.session_t:
  """Create a session for an in-memory SQLite datastore.

  The database is will be empty.

  Returns:
    A Session instance.
  """
  with ds(request).Session() as session_:
    yield session_
