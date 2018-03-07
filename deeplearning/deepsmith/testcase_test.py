"""Tests for //deeplearning/deepsmith:testcase."""
import datetime
import random
import sys
import tempfile

import pytest
from absl import app

import deeplearning.deepsmith.client
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.profiling_event
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2


@pytest.fixture
def session() -> db.session_t:
  with tempfile.TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
    ds = datastore.DataStore(engine="sqlite", db_dir=tmpdir)
    with ds.Session() as session:
      yield session


def test_Testcase_ToProto():
  now = datetime.datetime.now()

  testcase = deeplearning.deepsmith.testcase.Testcase(
      toolchain=deeplearning.deepsmith.toolchain.Toolchain(name="cpp"),
      generator=deeplearning.deepsmith.generator.Generator(name="generator"),
      harness=deeplearning.deepsmith.harness.Harness(name="harness"),
      inputset=[
        deeplearning.deepsmith.testcase.TestcaseInput(
            name=deeplearning.deepsmith.testcase.TestcaseInputName(name="src"),
            value=deeplearning.deepsmith.testcase.TestcaseInputValue(string="void main() {}"),
        ),
        deeplearning.deepsmith.testcase.TestcaseInput(
            name=deeplearning.deepsmith.testcase.TestcaseInputName(name="data"),
            value=deeplearning.deepsmith.testcase.TestcaseInputValue(string="[1,2]"),
        ),
      ],
      invariant_optset=[
        deeplearning.deepsmith.testcase.TestcaseInvariantOpt(
            name=deeplearning.deepsmith.testcase.TestcaseInvariantOptName(name="config"),
            value=deeplearning.deepsmith.testcase.TestcaseInvariantOptValue(name="opt"),
        ),
      ],
      profiling_events=[
        deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
            client=deeplearning.deepsmith.client.Client(name="localhost"),
            type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                name="generate",
            ),
            duration_seconds=1.0,
            date=now,
        ),
        deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent(
            client=deeplearning.deepsmith.client.Client(name="localhost"),
            type=deeplearning.deepsmith.profiling_event.ProfilingEventType(
                name="foo",
            ),
            duration_seconds=1.0,
            date=now,
        ),
      ]
  )
  proto = testcase.ToProto()
  assert proto.toolchain == "cpp"
  assert proto.generator.name == "generator"
  assert proto.harness.name == "harness"
  assert len(proto.inputs) == 2
  assert proto.inputs["src"] == "void main() {}"
  assert proto.inputs["data"] == "[1,2]"
  assert len(proto.invariant_opts) == 1
  assert proto.invariant_opts["config"] == "opt"
  assert len(proto.profiling_events) == 2
  assert proto.profiling_events[0].client == "localhost"
  assert proto.profiling_events[0].type == "generate"
  assert proto.profiling_events[0].client == "localhost"


def test_Testcase_GetOrAdd(session):
  proto = deepsmith_pb2.Testcase(
      toolchain="cpp",
      generator=deepsmith_pb2.Generator(
          name="generator",
      ),
      harness=deepsmith_pb2.Harness(
          name="harness",
      ),
      inputs={
        "src": "void main() {}",
        "data": "[1,2]"
      },
      invariant_opts={
        "config": "opt"
      },
      profiling_events=[
        deepsmith_pb2.ProfilingEvent(
            client="localhost",
            type="generate",
            duration_seconds=1.0,
            date_epoch_seconds=1021312312,
        ),
        deepsmith_pb2.ProfilingEvent(
            client="localhost",
            type="foo",
            duration_seconds=1.0,
            date_epoch_seconds=1230812312,
        ),
      ]
  )
  testcase = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session, proto
  )

  # NOTE: We have to flush so that SQLAlchemy resolves all of the object IDs.
  session.flush()
  assert testcase.toolchain.name == "cpp"
  assert testcase.generator.name == "generator"
  assert testcase.harness.name == "harness"
  assert len(testcase.inputset) == 2
  assert len(testcase.inputs) == 2
  assert testcase.inputs["src"] == "void main() {}"
  assert testcase.inputs["data"] == "[1,2]"
  assert len(testcase.invariant_optset) == 1
  assert len(testcase.invariant_opts) == 1
  assert testcase.invariant_opts["config"] == "opt"


def test_Generator_GetOrAdd_ToProto_equivalence(session):
  proto_in = deepsmith_pb2.Testcase(
      toolchain="cpp",
      generator=deepsmith_pb2.Generator(
          name="generator",
      ),
      harness=deepsmith_pb2.Harness(
          name="harness",
      ),
      inputs={
        "src": "void main() {}",
        "data": "[1,2]"
      },
      invariant_opts={
        "config": "opt"
      },
      profiling_events=[
        deepsmith_pb2.ProfilingEvent(
            client="localhost",
            type="generate",
            duration_seconds=1.0,
            date_epoch_seconds=101231231,
        ),
      ]
  )
  testcase = deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session, proto_in
  )

  # NOTE: We have to flush so that SQLAlchemy resolves all of the object IDs.
  session.flush()
  proto_out = testcase.ToProto()
  assert proto_in == proto_out
  proto_out.ClearField("toolchain")
  assert proto_in != proto_out  # Sanity check.


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
                duration_seconds=random.random(),
                date_epoch_seconds=int(random.random() * 1000000),
            ),
          ]
      )
  )
  session.flush()


def test_benchmark_Testcase_GetOrAdd_new(session, benchmark):
  benchmark(_AddRandomNewTestcase, session)


def _AddExistingTestcase(session):
  deeplearning.deepsmith.testcase.Testcase.GetOrAdd(
      session,
      deepsmith_pb2.Testcase(
          toolchain="cpp",
          generator=deepsmith_pb2.Generator(
              name="name",
              opts={
                "a": "a",
                "b": "b",
                "c": "c",
              },
          ),
          harness=deepsmith_pb2.Harness(
              name="name",
              opts={
                "a": "a",
                "b": "b",
                "c": "c",
              },
          ),
          inputs={
            "src": "void main() {}",
            "data": "[1,2]",
            "copt": "-DNDEBUG",
          },
          invariant_opts={
            "config": "opt",
            "working_dir": "/tmp",
            "units": "nanoseconds",
          },
          profiling_events=[
            deepsmith_pb2.ProfilingEvent(
                client="localhost",
                type="generate",
                duration_seconds=1.0,
                date_epoch_seconds=101231231,
            ),
          ]
      )
  )
  session.flush()


def test_benchmark_Generator_GetOrAdd_existing(session, benchmark):
  benchmark(_AddExistingTestcase, session)


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
