"""Shared test fixture functions.

Fixtures defined in this file may be used in any test in this directory. You
do not need to import this file, it is discovered automatically by pytest.

See the conftest.py documentation for more details:
https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
"""
import pytest

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import datastore_pb2


@pytest.fixture
def ds() -> datastore.DataStore:
  """Create an in-memory SQLite datastore for testing.

  Returns:
    A DataStore instance.
  """
  return datastore.DataStore(datastore_pb2.DataStore(
      sqlite=datastore_pb2.DataStore.Sqlite(
          url='sqlite://',
      )
  ))


@pytest.fixture
def session() -> db.session_t:
  """Create a session for an in-memory SQLite datastore.

  Returns:
    A Session instance.
  """
  with ds().Session() as session_:
    yield session_
