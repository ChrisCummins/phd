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
"""Tests for //deeplearning/deepsmith:testbed."""
import hashlib
import random

import deeplearning.deepsmith.testbed
import deeplearning.deepsmith.toolchain
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.deepsmith.tests.fixtures"]


def test_Testbed_ToProto():
  testbed = deeplearning.deepsmith.testbed.Testbed(
    toolchain=deeplearning.deepsmith.toolchain.Toolchain(string="cpp"),
    name="clang",
    optset=[
      deeplearning.deepsmith.testbed.TestbedOpt(
        name=deeplearning.deepsmith.testbed.TestbedOptName(string="arch"),
        value=deeplearning.deepsmith.testbed.TestbedOptValue(string="x86_64"),
      ),
      deeplearning.deepsmith.testbed.TestbedOpt(
        name=deeplearning.deepsmith.testbed.TestbedOptName(string="build"),
        value=deeplearning.deepsmith.testbed.TestbedOptValue(
          string="debug+assert"
        ),
      ),
    ],
  )
  proto = testbed.ToProto()
  assert proto.toolchain == "cpp"
  assert proto.name == "clang"
  assert len(proto.opts) == 2
  assert proto.opts["arch"] == "x86_64"
  assert proto.opts["build"] == "debug+assert"


def test_Testbed_GetOrAdd(session):
  proto = deepsmith_pb2.Testbed(
    toolchain="cpp",
    name="clang",
    opts={"arch": "x86_64", "build": "debug+assert"},
  )
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto)
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  )
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  )

  assert testbed.toolchain.string == "cpp"
  assert testbed.name == "clang"
  assert len(testbed.optset) == 2
  assert len(testbed.opts) == 2
  assert testbed.opts["arch"] == "x86_64"
  assert testbed.opts["build"] == "debug+assert"


def test_Testbed_GetOrAdd_duplicates(session):
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  )
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 0
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0
  )
  proto_a1 = deepsmith_pb2.Testbed(
    toolchain="cpp",
    name="clang",
    opts={"arch": "x86_64", "build": "debug+assert",},
  )
  proto_a2 = deepsmith_pb2.Testbed(
    toolchain="cpp",
    name="clang",
    opts={"arch": "x86_64", "build": "debug+assert",},
  )
  proto_b = deepsmith_pb2.Testbed(
    toolchain="cpp", name="gcc", opts={"arch": "x86_64", "build": "opt",},
  )
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_a1)
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  )
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  )
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_a2)
  # proto_a1 == proto_a2, so the counts should remain unchanged.
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  )
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  )
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_b)
  # proto_b adds a new testbed, new opt (note the duplicate arch), and
  # two new entries in the TestbedOptSet table.
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 2
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 3
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 4
  )
  assert session.query(deeplearning.deepsmith.toolchain.Toolchain).count() == 1
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 3
  )


def test_Testbed_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Testbed(
    toolchain="cpp",
    name="clang",
    opts={"arch": "x86_64", "build": "debug+assert"},
  )
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(session, proto_in)

  # NOTE: We have to flush before constructing a proto so that SQLAlchemy
  # resolves all of the object IDs.
  session.flush()

  proto_out = testbed.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField("toolchain")
  assert proto_in != proto_out  # Sanity check.


def test_Testbed_GetOrAdd_no_opts(session):
  testbed = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session, deepsmith_pb2.Testbed(toolchain="toolchain", name="name", opts={},)
  )
  empty_md5 = hashlib.md5().digest()
  assert testbed.optset_id == empty_md5
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0
  )


def test_Testbed_GetOrAdd_only_different_optset(session):
  testbed_a = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session,
    deepsmith_pb2.Testbed(
      toolchain="toolchain", name="name", opts={"a": "A", "b": "B", "c": "C",},
    ),
  )
  testbed_b = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session,
    deepsmith_pb2.Testbed(
      toolchain="toolchain", name="name", opts={"d": "D",},
    ),
  )
  testbed_c = deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session, deepsmith_pb2.Testbed(toolchain="toolchain", name="name", opts={},)
  )
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 3
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 4
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 4
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 4
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 4
  )
  assert len(testbed_a.optset) == 3
  assert len(testbed_b.optset) == 1
  assert len(testbed_c.optset) == 0


def test_Testbed_GetOrAdd_rollback(session):
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session,
    deepsmith_pb2.Testbed(
      toolchain="opencl",
      name="nvidia",
      opts={"opencl": "1.2", "devtype": "GPU",},
    ),
  )
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 1
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 2
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 2
  )
  session.rollback()
  assert session.query(deeplearning.deepsmith.testbed.Testbed).count() == 0
  assert session.query(deeplearning.deepsmith.testbed.TestbedOpt).count() == 0
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptSet).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptName).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.testbed.TestbedOptValue).count() == 0
  )


def _AddRandomNewTestbed(session):
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session,
    deepsmith_pb2.Testbed(
      toolchain=str(random.random()),
      name=str(random.random()),
      opts={
        str(random.random()): str(random.random()),
        str(random.random()): str(random.random()),
        str(random.random()): str(random.random()),
      },
    ),
  )
  session.flush()


def test_benchmark_Testbed_GetOrAdd_new(session, benchmark):
  benchmark(_AddRandomNewTestbed, session)


def _AddExistingTestbed(session):
  deeplearning.deepsmith.testbed.Testbed.GetOrAdd(
    session,
    deepsmith_pb2.Testbed(
      toolchain="toolchain", name="name", opts={"a": "a", "b": "b", "c": "c",},
    ),
  )
  session.flush()


def test_benchmark_Testbed_GetOrAdd_existing(session, benchmark):
  benchmark(_AddExistingTestbed, session)


if __name__ == "__main__":
  test.Main()
