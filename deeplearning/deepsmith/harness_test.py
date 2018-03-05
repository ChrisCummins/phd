"""Tests for //deeplearning/deepsmith:harness."""
import sys
import tempfile

import pytest
from absl import app

import deeplearning.deepsmith.harness
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2


@pytest.fixture
def session() -> db.session_t:
  with tempfile.TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
    ds = datastore.DataStore(engine="sqlite", db_dir=tmpdir)
    with ds.Session() as session:
      yield session


def test_Harness_ToProto():
  harness = deeplearning.deepsmith.harness.Harness(
      name="name", version="version"
  )
  proto = harness.ToProto()
  assert proto.name == "name"
  assert proto.version == "version"


def test_Harness_GetOrAdd(session):
  proto = deepsmith_pb2.Harness(
      name="name", version="version"
  )
  harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session, proto
  )

  assert harness.name == "name"
  assert harness.version == "version"


def test_Harness_duplicates(session):
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 0
  proto_a1 = deepsmith_pb2.Harness(name="a", version="version")
  proto_a2 = deepsmith_pb2.Harness(name="a", version="version")
  proto_b = deepsmith_pb2.Harness(name="b", version="version")
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_a1)
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 1
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_b)
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 2
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_a2)
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 2


def test_Harness_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Harness(name="name", version="version")
  harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_in)
  proto_out = harness.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField("name")
  assert proto_in != proto_out


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
