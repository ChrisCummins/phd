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
"""Tests for //deeplearning/deepsmith:generator."""
import hashlib
import random

import deeplearning.deepsmith.generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_Generator_ToProto():
  generator = deeplearning.deepsmith.generator.Generator(
    name="name",
    optset=[
      deeplearning.deepsmith.generator.GeneratorOpt(
        name=deeplearning.deepsmith.generator.GeneratorOptName(
          string="version"
        ),
        value=deeplearning.deepsmith.generator.GeneratorOptValue(
          string="1.0.0"
        ),
      ),
      deeplearning.deepsmith.generator.GeneratorOpt(
        name=deeplearning.deepsmith.generator.GeneratorOptName(string="build"),
        value=deeplearning.deepsmith.generator.GeneratorOptValue(
          string="debug+assert"
        ),
      ),
    ],
  )
  proto = generator.ToProto()
  assert proto.name == "name"
  assert len(proto.opts) == 2
  assert proto.opts["version"] == "1.0.0"
  assert proto.opts["build"] == "debug+assert"


def test_Generator_GetOrAdd(session):
  proto = deepsmith_pb2.Generator(
    name="name", opts={"version": "1.0.0", "build": "debug+assert",}
  )
  generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, proto
  )

  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 2
  )

  assert generator.name == "name"
  assert len(generator.optset) == 2
  assert len(generator.opts) == 2
  assert generator.opts["version"] == "1.0.0"
  assert generator.opts["build"] == "debug+assert"


def test_Generator_duplicates(session):
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 0
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 0
  )
  proto_a1 = deepsmith_pb2.Generator(
    name="a", opts={"arch": "x86_64", "build": "debug+assert",},
  )
  proto_a2 = deepsmith_pb2.Generator(  # proto_a1 == proto_a2
    name="a", opts={"arch": "x86_64", "build": "debug+assert",},
  )
  proto_b = deepsmith_pb2.Generator(
    name="b", opts={"arch": "x86_64", "build": "opt",},
  )
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.generator.Generator.GetOrAdd(session, proto_a1)
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 1
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 2
  )
  deeplearning.deepsmith.generator.Generator.GetOrAdd(session, proto_a2)
  # proto_a1 == proto_a2, so the counts should remain unchanged.
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 1
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 2
  )
  deeplearning.deepsmith.generator.Generator.GetOrAdd(session, proto_b)
  # proto_b adds a new generator, new opt (note the duplicate arch), and
  # two new entries in the GeneratorOptSet table.
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 2
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 3
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 4
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 3
  )


def test_Generator_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Generator(
    name="a", opts={"arch": "x86_64", "build": "debug+assert"},
  )
  generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, proto_in
  )
  # NOTE: We have to flush before constructing a proto so that SQLAlchemy
  # resolves all of the object IDs.
  session.flush()

  proto_out = generator.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField("name")
  assert proto_in != proto_out  # Sanity check.


def test_Generator_GetOrAdd_no_opts(session):
  generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, deepsmith_pb2.Generator(name="name", opts={},)
  )
  empty_md5 = hashlib.md5().digest()
  assert generator.optset_id == empty_md5
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 1
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 0
  )


def test_Generator_GetOrAdd_only_different_optset(session):
  generator_a = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session,
    deepsmith_pb2.Generator(name="name", opts={"a": "A", "b": "B", "c": "C",},),
  )
  generator_b = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, deepsmith_pb2.Generator(name="name", opts={"d": "D",},)
  )
  generator_c = deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, deepsmith_pb2.Generator(name="name", opts={},)
  )
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 3
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 4
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 4
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 4
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 4
  )
  assert len(generator_a.optset) == 3
  assert len(generator_b.optset) == 1
  assert len(generator_c.optset) == 0


def test_Generator_GetOrAdd_rollback(session):
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session, deepsmith_pb2.Generator(name="name", opts={"a": "1", "b": "2",},)
  )
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 1
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 2
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 2
  )
  session.rollback()
  assert session.query(deeplearning.deepsmith.generator.Generator).count() == 0
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOpt).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptSet).count() == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptName).count()
    == 0
  )
  assert (
    session.query(deeplearning.deepsmith.generator.GeneratorOptValue).count()
    == 0
  )


def _AddRandomNewGenerator(session):
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session,
    deepsmith_pb2.Generator(
      name=str(random.random()),
      opts={
        str(random.random()): str(random.random()),
        str(random.random()): str(random.random()),
        str(random.random()): str(random.random()),
      },
    ),
  )
  session.flush()


def test_benchmark_Generator_GetOrAdd_new(session, benchmark):
  benchmark(_AddRandomNewGenerator, session)


def _AddExistingGenerator(session):
  deeplearning.deepsmith.generator.Generator.GetOrAdd(
    session,
    deepsmith_pb2.Generator(name="name", opts={"a": "a", "b": "b", "c": "c",},),
  )
  session.flush()


def test_benchmark_Generator_GetOrAdd_existing(session, benchmark):
  benchmark(_AddExistingGenerator, session)


if __name__ == "__main__":
  test.Main()
