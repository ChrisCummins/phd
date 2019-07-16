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
"""Tests for //deeplearning/deepsmith:harness."""
import hashlib
import random

import deeplearning.deepsmith.harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_Harness_ToProto():
  harness = deeplearning.deepsmith.harness.Harness(
      name='name',
      optset=[
          deeplearning.deepsmith.harness.HarnessOpt(
              name=deeplearning.deepsmith.harness.HarnessOptName(
                  string='version'),
              value=deeplearning.deepsmith.harness.HarnessOptValue(
                  string='1.0.0'),
          ),
          deeplearning.deepsmith.harness.HarnessOpt(
              name=deeplearning.deepsmith.harness.HarnessOptName(
                  string='build'),
              value=deeplearning.deepsmith.harness.HarnessOptValue(
                  string='debug+assert'),
          ),
      ],
  )
  proto = harness.ToProto()
  assert proto.name == 'name'
  assert len(proto.opts) == 2
  assert proto.opts['version'] == '1.0.0'
  assert proto.opts['build'] == 'debug+assert'


def test_Harness_GetOrAdd(session):
  proto = deepsmith_pb2.Harness(name='name',
                                opts={
                                    'version': '1.0.0',
                                    'build': 'debug+assert',
                                })
  harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto)

  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 2
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 2

  assert harness.name == 'name'
  assert len(harness.optset) == 2
  assert len(harness.opts) == 2
  assert harness.opts['version'] == '1.0.0'
  assert harness.opts['build'] == 'debug+assert'


def test_Harness_duplicates(session):
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 0
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 0
  proto_a1 = deepsmith_pb2.Harness(
      name='a',
      opts={
          'arch': 'x86_64',
          'build': 'debug+assert',
      },
  )
  proto_a2 = deepsmith_pb2.Harness(  # proto_a1 == proto_a2
      name='a',
      opts={
          'arch': 'x86_64',
          'build': 'debug+assert',
      },
  )
  proto_b = deepsmith_pb2.Harness(
      name='b',
      opts={
          'arch': 'x86_64',
          'build': 'opt',
      },
  )
  assert proto_a1 == proto_a2  # Sanity check.
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_a1)
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 1
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 2
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_a2)
  # proto_a1 == proto_a2, so the counts should remain unchanged.
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 1
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 2
  deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_b)
  # proto_b adds a new harness, new opt (note the duplicate arch), and
  # two new entries in the HarnessOptSet table.
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 2
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 3
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 4
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 3


def test_Harness_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Harness(
      name='a',
      opts={
          'arch': 'x86_64',
          'build': 'debug+assert'
      },
  )
  harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(session, proto_in)
  # NOTE: We have to flush before constructing a proto so that SQLAlchemy
  # resolves all of the object IDs.
  session.flush()

  proto_out = harness.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField('name')
  assert proto_in != proto_out  # Sanity check.


def test_Harness_GetOrAdd_no_opts(session):
  harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session, deepsmith_pb2.Harness(
          name='name',
          opts={},
      ))
  empty_md5 = hashlib.md5().digest()
  assert harness.optset_id == empty_md5
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 1
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 0


def test_Harness_GetOrAdd_only_different_optset(session):
  harness_a = deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session,
      deepsmith_pb2.Harness(
          name='name',
          opts={
              'a': 'A',
              'b': 'B',
              'c': 'C',
          },
      ))
  harness_b = deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session, deepsmith_pb2.Harness(
          name='name',
          opts={
              'd': 'D',
          },
      ))
  harness_c = deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session, deepsmith_pb2.Harness(
          name='name',
          opts={},
      ))
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 3
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 4
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 4
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 4
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 4
  assert len(harness_a.optset) == 3
  assert len(harness_b.optset) == 1
  assert len(harness_c.optset) == 0


def test_Harness_GetOrAdd_rollback(session):
  deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session, deepsmith_pb2.Harness(
          name='name',
          opts={
              'a': '1',
              'b': '2',
          },
      ))
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 1
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 2
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 2
  session.rollback()
  assert session.query(deeplearning.deepsmith.harness.Harness).count() == 0
  assert session.query(deeplearning.deepsmith.harness.HarnessOpt).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptSet).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptName).count() == 0
  assert session.query(
      deeplearning.deepsmith.harness.HarnessOptValue).count() == 0


def _AddRandomNewHarness(session):
  deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session,
      deepsmith_pb2.Harness(
          name=str(random.random()),
          opts={
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
          },
      ))
  session.flush()


def test_benchmark_Harness_GetOrAdd_new(session, benchmark):
  benchmark(_AddRandomNewHarness, session)


def _AddExistingHarness(session):
  deeplearning.deepsmith.harness.Harness.GetOrAdd(
      session,
      deepsmith_pb2.Harness(
          name='name',
          opts={
              'a': 'a',
              'b': 'b',
              'c': 'c',
          },
      ))
  session.flush()


def test_benchmark_Harness_GetOrAdd_existing(session, benchmark):
  benchmark(_AddExistingHarness, session)


if __name__ == '__main__':
  test.Main()
