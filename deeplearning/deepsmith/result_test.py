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
"""Tests for //deeplearning/deepsmith:result."""
import datetime

import deeplearning.deepsmith.client
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.profiling_event
import deeplearning.deepsmith.result
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8.py import app
from labm8.py import labdate
from labm8.py import test

FLAGS = app.FLAGS


def test_Result_ToProto():
  now = datetime.datetime.now()

  result = deeplearning.deepsmith.result.Result(
      testcase=deeplearning.deepsmith.testcase.Testcase(
          toolchain=deeplearning.deepsmith.toolchain.Toolchain(string='cpp'),
          generator=deeplearning.deepsmith.generator.Generator(
              name='generator'),
          harness=deeplearning.deepsmith.harness.Harness(name='harness'),
          inputset=[
              deeplearning.deepsmith.testcase.TestcaseInput(
                  name=deeplearning.deepsmith.testcase.TestcaseInputName(
                      string='src'),
                  value=deeplearning.deepsmith.testcase.TestcaseInputValue(
                      string='void main() {}'),
              ),
              deeplearning.deepsmith.testcase.TestcaseInput(
                  name=deeplearning.deepsmith.testcase.TestcaseInputName(
                      string='data'),
                  value=deeplearning.deepsmith.testcase.TestcaseInputValue(
                      string='[1,2]'),
              ),
          ],
          invariant_optset=[
              deeplearning.deepsmith.testcase.TestcaseInvariantOpt(
                  name=deeplearning.deepsmith.testcase.TestcaseInvariantOptName(
                      string='config'),
                  value=deeplearning.deepsmith.testcase.
                  TestcaseInvariantOptValue(string='opt'),
              ),
          ],
          profiling_events=[
              deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
                  client=deeplearning.deepsmith.client.Client(
                      string='localhost'),
                  type=deeplearning.deepsmith.profiling_event.
                  ProfilingEventType(string='generate',),
                  duration_ms=100,
                  event_start=now,
              ),
              deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
                  client=deeplearning.deepsmith.client.Client(
                      string='localhost'),
                  type=deeplearning.deepsmith.profiling_event.
                  ProfilingEventType(string='foo',),
                  duration_ms=100,
                  event_start=now,
              ),
          ]),
      testbed=deeplearning.deepsmith.testbed.Testbed(
          toolchain=deeplearning.deepsmith.toolchain.Toolchain(string='cpp'),
          name='clang',
          optset=[
              deeplearning.deepsmith.testbed.TestbedOpt(
                  name=deeplearning.deepsmith.testbed.TestbedOptName(
                      string='arch'),
                  value=deeplearning.deepsmith.testbed.TestbedOptValue(
                      string='x86_64'),
              ),
              deeplearning.deepsmith.testbed.TestbedOpt(
                  name=deeplearning.deepsmith.testbed.TestbedOptName(
                      string='build'),
                  value=deeplearning.deepsmith.testbed.TestbedOptValue(
                      string='debug+assert'),
              ),
          ],
      ),
      returncode=0,
      outputset=[
          deeplearning.deepsmith.result.ResultOutput(
              name=deeplearning.deepsmith.result.ResultOutputName(
                  string='stdout'),
              value=deeplearning.deepsmith.result.ResultOutputValue(
                  truncated_value='Hello, world!'),
          ),
          deeplearning.deepsmith.result.ResultOutput(
              name=deeplearning.deepsmith.result.ResultOutputName(
                  string='stderr'),
              value=deeplearning.deepsmith.result.ResultOutputValue(
                  truncated_value=''),
          ),
      ],
      profiling_events=[
          deeplearning.deepsmith.profiling_event.ResultProfilingEvent(
              client=deeplearning.deepsmith.client.Client(string='localhost'),
              type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                  string='exec',),
              duration_ms=500,
              event_start=now,
          ),
          deeplearning.deepsmith.profiling_event.ResultProfilingEvent(
              client=deeplearning.deepsmith.client.Client(string='localhost'),
              type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                  string='overhead',),
              duration_ms=100,
              event_start=now,
          ),
      ],
      outcome_num=6,
  )
  proto = result.ToProto()
  assert proto.testcase.toolchain == 'cpp'
  assert proto.testcase.generator.name == 'generator'
  assert proto.testcase.harness.name == 'harness'
  assert len(proto.testcase.inputs) == 2
  assert proto.testcase.inputs['src'] == 'void main() {}'
  assert proto.testcase.inputs['data'] == '[1,2]'
  assert len(proto.testcase.invariant_opts) == 1
  assert proto.testcase.invariant_opts['config'] == 'opt'
  assert len(proto.testcase.profiling_events) == 2
  assert proto.testcase.profiling_events[0].client == 'localhost'
  assert proto.testcase.profiling_events[0].type == 'generate'
  assert proto.testcase.profiling_events[0].client == 'localhost'
  assert proto.testbed.toolchain == 'cpp'
  assert proto.testbed.name == 'clang'
  assert len(proto.testbed.opts) == 2
  assert proto.testbed.opts['arch'] == 'x86_64'
  assert proto.testbed.opts['build'] == 'debug+assert'
  assert len(proto.outputs) == 2
  assert proto.outputs['stdout'] == 'Hello, world!'
  assert proto.outputs['stderr'] == ''
  assert len(proto.testcase.profiling_events) == 2
  assert proto.profiling_events[0].client == 'localhost'
  assert proto.profiling_events[0].type == 'exec'
  assert proto.profiling_events[0].duration_ms == 500
  assert (proto.profiling_events[0].event_start_epoch_ms ==
          labdate.MillisecondsTimestamp(now))
  assert proto.profiling_events[1].client == 'localhost'
  assert proto.profiling_events[1].type == 'overhead'
  assert proto.profiling_events[1].duration_ms == 100
  assert (proto.profiling_events[1].event_start_epoch_ms ==
          labdate.MillisecondsTimestamp(now))
  assert proto.outcome == deepsmith_pb2.Result.PASS


def test_Generator_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Result(
      testcase=deepsmith_pb2.Testcase(
          toolchain='cpp',
          generator=deepsmith_pb2.Generator(name='generator'),
          harness=deepsmith_pb2.Harness(name='harness'),
          inputs={
              'src': 'void main() {}',
              'data': '[1,2]',
          },
          invariant_opts={
              'config': 'opt',
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='generate',
                  duration_ms=100,
                  event_start_epoch_ms=1123123123,
              ),
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='foo',
                  duration_ms=100,
                  event_start_epoch_ms=1123123123,
              ),
          ]),
      testbed=deepsmith_pb2.Testbed(
          toolchain='cpp',
          name='clang',
          opts={
              'arch': 'x86_64',
              'build': 'debug+assert',
          },
      ),
      returncode=0,
      outputs={
          'stdout': 'Hello, world!',
          'stderr': '',
      },
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='exec',
              duration_ms=500,
              event_start_epoch_ms=1123123123,
          ),
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='overhead',
              duration_ms=100,
              event_start_epoch_ms=1123123123,
          ),
      ],
      outcome=deepsmith_pb2.Result.PASS,
  )
  result = deeplearning.deepsmith.result.Result.GetOrAdd(session, proto_in)

  # NOTE: We have to flush so that SQLAlchemy resolves all of the object IDs.
  session.flush()
  proto_out = result.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField('outputs')
  assert proto_in != proto_out  # Sanity check.


def test_duplicate_testcase_testbed_ignored(session):
  """Test that result is ignored if testbed and testcase are not unique."""
  proto = deepsmith_pb2.Result(
      testcase=deepsmith_pb2.Testcase(
          toolchain='cpp',
          generator=deepsmith_pb2.Generator(name='generator'),
          harness=deepsmith_pb2.Harness(name='harness'),
          inputs={
              'src': 'void main() {}',
              'data': '[1,2]',
          },
          invariant_opts={
              'config': 'opt',
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='generate',
                  duration_ms=100,
                  event_start_epoch_ms=1123123123,
              ),
          ]),
      testbed=deepsmith_pb2.Testbed(
          toolchain='cpp',
          name='clang',
          opts={'arch': 'x86_64'},
      ),
      returncode=0,
      outputs={'stdout': 'Hello, world!'},
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='exec',
              duration_ms=100,
              event_start_epoch_ms=1123123123,
          ),
      ],
      outcome=deepsmith_pb2.Result.PASS,
  )
  r1 = deeplearning.deepsmith.result.Result.GetOrAdd(session, proto)
  session.add(r1)
  session.flush()

  # Attempt to add a new result which is identical to the first in all fields
  # except for the outputs.
  proto.outputs['stdout'] = '!'
  r2 = deeplearning.deepsmith.result.Result.GetOrAdd(session, proto)
  session.add(r2)
  session.flush()

  # Check that only one result was added.
  assert session.query(deeplearning.deepsmith.result.Result).count() == 1

  # Check that only the first result was added.
  r3 = session.query(deeplearning.deepsmith.result.Result).first()
  assert r3.outputs['stdout'] == 'Hello, world!'


def test_duplicate_results_ignored(session):
  """Test that results are only added if they are unique."""
  proto = deepsmith_pb2.Result(
      testcase=deepsmith_pb2.Testcase(
          toolchain='cpp',
          generator=deepsmith_pb2.Generator(name='generator'),
          harness=deepsmith_pb2.Harness(name='harness'),
          inputs={
              'src': 'void main() {}',
              'data': '[1,2]',
          },
          invariant_opts={
              'config': 'opt',
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='generate',
                  duration_ms=100,
                  event_start_epoch_ms=1123123123,
              ),
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='foo',
                  duration_ms=100,
                  event_start_epoch_ms=1123123123,
              ),
          ]),
      testbed=deepsmith_pb2.Testbed(
          toolchain='cpp',
          name='clang',
          opts={
              'arch': 'x86_64',
              'build': 'debug+assert',
          },
      ),
      returncode=0,
      outputs={
          'stdout': 'Hello, world!',
          'stderr': '',
      },
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='exec',
              duration_ms=100,
              event_start_epoch_ms=1123123123,
          ),
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='overhead',
              duration_ms=100,
              event_start_epoch_ms=1123123123,
          ),
      ],
      outcome=deepsmith_pb2.Result.PASS,
  )
  r1 = deeplearning.deepsmith.result.Result.GetOrAdd(session, proto)
  session.add(r1)
  session.flush()

  # Attempt to add a new result which is identical to the first in all fields
  # except for the profiling events.
  proto.profiling_events[0].duration_ms = -1
  r2 = deeplearning.deepsmith.result.Result.GetOrAdd(session, proto)
  session.add(r2)
  session.flush()

  # Check that only one result was added.
  assert session.query(deeplearning.deepsmith.result.Result).count() == 1

  # Check that only the first result was added.
  r3 = session.query(deeplearning.deepsmith.result.Result).first()
  assert r3.profiling_events[0].duration_ms == 100
  assert r3.profiling_events[1].duration_ms == 100


if __name__ == '__main__':
  test.Main()
