"""Tests for //deeplearning/deepsmith:testcase."""
import hashlib
import random
import sys
import tempfile

import pytest
from absl import app

import deeplearning.deepsmith.harness
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.testcase
import deeplearning.deepsmith.profiling_event
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


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
