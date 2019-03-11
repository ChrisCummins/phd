"""Tests for //deeplearning/deepsmith:testcase."""
import random

import deeplearning.deepsmith.client
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.profiling_event
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import app
from labm8 import labdate
from labm8 import test

FLAGS = app.FLAGS


def test_Testcase_ToProto():
  now = labdate.GetUtcMillisecondsNow()

  testcase = deeplearning.deepsmith.testcase.Testcase(
      toolchain=deeplearning.deepsmith.toolchain.Toolchain(string='cpp'),
      generator=deeplearning.deepsmith.generator.Generator(name='generator'),
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
              value=deeplearning.deepsmith.testcase.TestcaseInvariantOptValue(
                  string='opt'),
          ),
      ],
      profiling_events=[
          deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
              client=deeplearning.deepsmith.client.Client(string='localhost'),
              type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                  string='generate',),
              duration_ms=100,
              event_start=now,
          ),
          deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
              client=deeplearning.deepsmith.client.Client(string='localhost'),
              type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                  string='foo',),
              duration_ms=100,
              event_start=now,
          ),
      ])
  proto = testcase.ToProto()
  assert proto.toolchain == 'cpp'
  assert proto.generator.name == 'generator'
  assert proto.harness.name == 'harness'
  assert len(proto.inputs) == 2
  assert proto.inputs['src'] == 'void main() {}'
  assert proto.inputs['data'] == '[1,2]'
  assert len(proto.invariant_opts) == 1
  assert proto.invariant_opts['config'] == 'opt'
  assert len(proto.profiling_events) == 2
  assert (proto.profiling_events[0].event_start_epoch_ms ==
          labdate.MillisecondsTimestamp(now))
  assert proto.profiling_events[0].client == 'localhost'
  assert proto.profiling_events[0].type == 'generate'
  assert proto.profiling_events[0].client == 'localhost'


def test_Testcase_GetOrAdd(session):
  proto = deepsmith_pb2.Testcase(
      toolchain='cpp',
      generator=deepsmith_pb2.Generator(name='generator',),
      harness=deepsmith_pb2.Harness(name='harness',),
      inputs={
          'src': 'void main() {}',
          'data': '[1,2]'
      },
      invariant_opts={'config': 'opt'},
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='generate',
              duration_ms=100,
              event_start_epoch_ms=1021312312,
          ),
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='foo',
              duration_ms=100,
              event_start_epoch_ms=1230812312,
          ),
      ])
  testcase = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(session, proto)

  # NOTE: We have to flush so that SQLAlchemy resolves all of the object IDs.
  session.flush()
  assert testcase.toolchain.string == 'cpp'
  assert testcase.generator.name == 'generator'
  assert testcase.harness.name == 'harness'
  assert len(testcase.inputset) == 2
  assert len(testcase.inputs) == 2
  assert testcase.inputs['src'] == 'void main() {}'
  assert testcase.inputs['data'] == '[1,2]'
  assert len(testcase.invariant_optset) == 1
  assert len(testcase.invariant_opts) == 1
  assert testcase.invariant_opts['config'] == 'opt'
  assert testcase.profiling_events[0].client.string == 'localhost'
  assert testcase.profiling_events[0].type.string == 'generate'
  assert testcase.profiling_events[0].duration_ms == 100
  assert (testcase.profiling_events[0].event_start ==
          labdate.DatetimeFromMillisecondsTimestamp(1021312312))
  assert testcase.profiling_events[1].client.string == 'localhost'
  assert testcase.profiling_events[1].type.string == 'foo'
  assert testcase.profiling_events[1].duration_ms == 100
  assert (testcase.profiling_events[1].event_start ==
          labdate.DatetimeFromMillisecondsTimestamp(1230812312))


def test_Generator_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Testcase(
      toolchain='cpp',
      generator=deepsmith_pb2.Generator(name='generator',),
      harness=deepsmith_pb2.Harness(name='harness',),
      inputs={
          'src': 'void main() {}',
          'data': '[1,2]'
      },
      invariant_opts={'config': 'opt'},
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='generate',
              duration_ms=100,
              event_start_epoch_ms=101231231,
          ),
      ])
  testcase = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session, proto_in)

  # NOTE: We have to flush so that SQLAlchemy resolves all of the object IDs.
  session.flush()
  proto_out = testcase.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField('toolchain')
  assert proto_in != proto_out  # Sanity check.


def test_duplicate_testcases_ignored(session):
  """Test that testcases are only added if they are unique."""
  proto = deepsmith_pb2.Testcase(
      toolchain='cpp',
      generator=deepsmith_pb2.Generator(name='generator'),
      harness=deepsmith_pb2.Harness(name='harness'),
      inputs={
          'src': 'void main() {}',
          'data': '[1,2]'
      },
      invariant_opts={'config': 'opt'},
      profiling_events=[
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='generate',
              duration_ms=100,
              event_start_epoch_ms=1021312312,
          ),
          deepsmith_pb2.ProfilingEvent(
              client='localhost',
              type='foo',
              duration_ms=100,
              event_start_epoch_ms=1230812312,
          ),
      ])
  t1 = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(session, proto)
  session.add(t1)
  session.flush()

  # Attempt to add a new testcase which is identical to the first in all fields
  # except for the profiling events.
  proto.profiling_events[0].duration_ms = -1
  t2 = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(session, proto)
  session.add(t2)
  session.flush()

  # Check that only one testcase was added.
  assert session.query(deeplearning.deepsmith.testcase.Testcase).count() == 1

  # Check that only the first testcase was added.
  t3 = session.query(deeplearning.deepsmith.testcase.Testcase).first()
  assert t3.profiling_events[0].duration_ms == 100
  assert t3.profiling_events[1].duration_ms == 100


# Benchmarks.


def _AddRandomNewTestcase(session):
  deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session,
      deepsmith_pb2.Testcase(
          toolchain=str(random.random()),
          generator=deepsmith_pb2.Generator(
              name=str(random.random()),
              opts={
                  str(random.random()): str(random.random()),
                  str(random.random()): str(random.random()),
                  str(random.random()): str(random.random()),
              },
          ),
          harness=deepsmith_pb2.Harness(
              name=str(random.random()),
              opts={
                  str(random.random()): str(random.random()),
                  str(random.random()): str(random.random()),
                  str(random.random()): str(random.random()),
              },
          ),
          inputs={
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
          },
          invariant_opts={
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
              str(random.random()): str(random.random()),
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client=str(random.random()),
                  type=str(random.random()),
                  duration_ms=int(random.random() * 1000),
                  event_start_epoch_ms=int(random.random() * 1000000),
              ),
          ]))
  session.flush()


def test_benchmark_Testcase_GetOrAdd_new(session, benchmark):
  benchmark(_AddRandomNewTestcase, session)


def _AddExistingTestcase(session):
  deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session,
      deepsmith_pb2.Testcase(
          toolchain='cpp',
          generator=deepsmith_pb2.Generator(
              name='name',
              opts={
                  'a': 'a',
                  'b': 'b',
                  'c': 'c',
              },
          ),
          harness=deepsmith_pb2.Harness(
              name='name',
              opts={
                  'a': 'a',
                  'b': 'b',
                  'c': 'c',
              },
          ),
          inputs={
              'src': 'void main() {}',
              'data': '[1,2]',
              'copt': '-DNDEBUG',
          },
          invariant_opts={
              'config': 'opt',
              'working_dir': '/tmp',
              'units': 'nanoseconds',
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client='localhost',
                  type='generate',
                  duration_ms=100,
                  event_start_epoch_ms=101231231,
              ),
          ]))
  session.flush()


def test_benchmark_Generator_GetOrAdd_existing(session, benchmark):
  benchmark(_AddExistingTestcase, session)


if __name__ == '__main__':
  test.Main()
