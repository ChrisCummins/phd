"""Tests for //deeplearning/deepsmith:testbed."""
import sys
import tempfile

import pytest
from absl import app

import deeplearning.deepsmith.testbed
import deeplearning.deepsmith.toolchain
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2


@pytest.fixture
def session() -> db.session_t:
  with tempfile.TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
    ds = datastore.DataStore(engine="sqlite", db_dir=tmpdir)
    with ds.Session() as session:
      yield session


def test_Testbed_ToProto():
  testbed = deeplearning.deepsmith.testbed.Testbed(
      toolchain=deeplearning.deepsmith.toolchain.Toolchain(name="cpp"),
      name="clang",
      version="3.9.0",
      optset=[
        deeplearning.deepsmith.testbed.TestbedOpt(
            name=deeplearning.deepsmith.testbed.TestbedOptName(name="arch"),
            value=deeplearning.deepsmith.testbed.TestbedOptValue(name="x86_64"),
        ),
        deeplearning.deepsmith.testbed.TestbedOpt(
            name=deeplearning.deepsmith.testbed.TestbedOptName(name="build"),
            value=deeplearning.deepsmith.testbed.TestbedOptValue(name="debug+assert"),
        ),
      ],
  )
  proto = testbed.ToProto()
  assert proto.toolchain == "cpp"
  assert proto.name == "clang"
  assert proto.version == "3.9.0"
  assert len(proto.opts) == 2
  assert proto.opts["arch"] == "x86_64"
  assert proto.opts["build"] == "debug+assert"


def test_Testbed_GetOrAdd(session):
  proto = deepsmith_pb2.Testbed(
      toolchain="cpp",
      name="clang",
      version="3.9.0",
      opts={
        "arch": "x86_64",
        "build": "debug+assert"
      },
  )
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session, proto
  )
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2

  assert testbed.toolchain.name == "cpp"
  assert testbed.name == "clang"
  assert testbed.version == "3.9.0"
  assert len(testbed.opts) == 2
  assert testbed.opts["arch"] == "x86_64"
  assert testbed.opts["build"] == "debug+assert"


def test_Testbed_GetOrAdd_duplicates(session):
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0
  proto_a1 = deepsmith_pb2.Testbed(
      toolchain="cpp",
      name="clang",
      version="3.9.0",
      opts={
        "arch": "x86_64",
        "build": "debug+assert",
      },
  )
  proto_a2 = deepsmith_pb2.Testbed(
      toolchain="cpp",
      name="clang",
      version="3.9.0",
      opts={
        "arch": "x86_64",
        "build": "debug+assert",
      },
  )
  proto_b = deepsmith_pb2.Testbed(
      toolchain="cpp",
      name="gcc",
      version="5.0",
      opts={
        "arch": "x86_64",
        "build": "opt",
      },
  )
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_a1)
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_a2)
  # proto_a1 == proto_a2, so the counts should remain unchanged.
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_b)
  # proto_b adds a new testbed, new opt (note the duplicate arch), and
  # two new entries in the TestbedOptSet table.
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 3
  print(session.query(deeplearning.deepsmith.testbed.TestbedOptSet).all())
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 4
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 3


def test_Testbed_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Testbed(
      toolchain="opencl",
      name="nvidia",
      version="1.0.0",
      opts={
        "opencl": "1.2",
        "devtype": "GPU",
      },
  )
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_in)
  proto_out = testbed.ToProto()
  print("PROTO_IN", proto_in)
  print("PROTO_OUT", proto_out)
  print("MAP", proto_in.opts)
  print("MAP", proto_out.opts)
  print(session.query(deeplearning.deepsmith.testbed.TestbedOpt).all())
  print(session.query(deeplearning.deepsmith.testbed.TestbedOptSet).all())
  print(session.query(deeplearning.deepsmith.testbed.TestbedOptName).all())
  print(session.query(deeplearning.deepsmith.testbed.TestbedOptValue).all())
  print(session.query(deeplearning.deepsmith.testbed.Testbed).first())
  assert proto_in == proto_out
  proto_out.ClearField("toolchain")
  assert proto_in != proto_out  # Sanity check.


def test_Testbed_GetOrAdd_no_opts(session):
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session, deepsmith_pb2.Testbed(
          toolchain="toolchain",
          name="name",
          version="version",
          opts={},
      )
  )
  assert testbed.optset_id == 0
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0


def test_Testbed_GetOrAdd_only_different_optset(session):
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session, deepsmith_pb2.Testbed(
          toolchain="toolchain",
          name="name",
          version="version",
          opts={
            "a": "A",
            "b": "B",
            "c": "C",
          },
      )
  )
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session, deepsmith_pb2.Testbed(
          toolchain="toolchain",
          name="name",
          version="version",
          opts={
            "d": "D",
          },
      )
  )
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session, deepsmith_pb2.Testbed(
          toolchain="toolchain",
          name="name",
          version="version",
          opts={},
      )
  )
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 3
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 4
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 4
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 4
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 4


def test_Testbed_GetOrAdd_rollback(session):
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
      session,
      deepsmith_pb2.Testbed(
          toolchain="opencl",
          name="nvidia",
          version="1.0.0",
          opts={
            "opencl": "1.2",
            "devtype": "GPU",
          },
      )
  )
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  session.rollback()
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
