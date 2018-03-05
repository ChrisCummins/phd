"""Tests for //deeplearning/deepsmith:generator."""
import sys
import tempfile

import pytest
from absl import app

import deeplearning.deepsmith.generator
from deeplearning.deepsmith import db
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith.protos import deepsmith_pb2


@pytest.fixture
def session() -> db.session_t:
  with tempfile.TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
    ds = datastore.DataStore(engine="sqlite", db_dir=tmpdir)
    with ds.Session() as session:
      yield session


def test_Generator_ToProto():
  generator = deeplearning.deepsmith.generator.Generator(
      name="name", version="version"
  )
  proto = generator.ToProto()
  assert proto.name == "name"
  assert proto.version == "version"


def test_Generator_GetOrAdd(session):
  proto = deepsmith_pb2.Generator(
      name="name", version="version"
  )
  generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
      session, proto
  )

  assert generator.name == "name"
  assert generator.version == "version"


def test_Generator_duplicates(session):
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 0
  proto_a1 = deepsmith_pb2.Generator(
      name="a", version="version"
  )
  proto_a2 = deepsmith_pb2.Generator(
      name="a", version="version"  # proto_a1 == proto_a2
  )
  proto_b = deepsmith_pb2.Generator(
      name="b", version="version"
  )
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
      session, proto_a1
  )
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 1
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
      session, proto_b
  )
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 2
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
      session, proto_a2
  )
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 2


def test_Generator_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Generator(
      name="name", version="version"
  )
  generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
      session, proto_in
  )
  proto_out = generator.ToProto()
  assert proto_in == proto_out


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
