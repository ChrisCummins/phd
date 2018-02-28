import pytest
import random
import string
import sys
import typing

from absl import app

from tempfile import TemporaryDirectory

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2
from deeplearning.deepsmith.services import testing


@pytest.fixture
def ds():
  with TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
    yield datastore.DataStore(engine="sqlite", db_dir=tmpdir)


def random_generator() -> deepsmith_pb2.Generator:
  return deepsmith_pb2.Generator(
      name=random.choice(["foo", "bar", "baz"]),
      version=random.choice(["1", "1", "1", "2"]),
  )


def random_harness():
  return deepsmith_pb2.Harness(
      name=random.choice(["a", "b", "c"]),
      version=random.choice(["1", "1", "1", "2"]),
  )


def random_str(n: int) -> str:
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def random_input() -> str:
  return random_str(int(random.random() * 1000) + 1)


def random_inputs():
  return {"src": random_input()}


def random_opt():
  return random_str(32)


def random_opts():
  return [random_opt()] * (int(random.random() * 5) + 1)


def random_testcase() -> deepsmith_pb2.Testcase:
  client = random_str(16)
  return deepsmith_pb2.Testcase(
      generator=random_generator(),
      harness=random_harness(),
      inputs=random_inputs(),
      opts=random_opts(),
      timings=[
        deepsmith_pb2.ProfilingEvent(client=client, name="a", duration_seconds=random.random()),
        deepsmith_pb2.ProfilingEvent(client=client, name="b", duration_seconds=random.random()),
        deepsmith_pb2.ProfilingEvent(client=client, name="c", duration_seconds=random.random()),
      ],
  )


def test_service_empty(ds):
  service = testing.TestingService(ds)
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=[])
  response = service.SubmitTestcases(request, None)
  assert type(response) == deepsmith_pb2.SubmitTestcasesResponse

  with ds.session() as s:
    assert s.query(db.Client).count() == 0
    assert s.query(db.ProfilingEventName).count() == 0
    assert s.query(db.Testcase).count() == 0
    assert s.query(db.TestcaseInput).count() == 0
    assert s.query(db.TestcaseOpt).count() == 0
    assert s.query(db.TestcaseTiming).count() == 0


def test_service_add_one(ds):
  service = testing.TestingService(ds)
  testcases = [
    deepsmith_pb2.Testcase(
        generator=deepsmith_pb2.Generator(name="foo", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "foo"},
        opts=["1", "2", "3"],
        timings=[
          deepsmith_pb2.ProfilingEvent(client="c", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="c", name="b", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="c", name="c", duration_seconds=3),
        ],
    ),
  ]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  service.SubmitTestcases(request, None)

  with ds.session() as s:
    assert s.query(db.Client).count() == 1
    assert s.query(db.ProfilingEventName).count() == 3
    assert s.query(db.Testcase).count() == 1
    assert s.query(db.TestcaseInput).count() == 1
    assert s.query(db.TestcaseOpt).count() == 3
    assert s.query(db.TestcaseTiming).count() == 3


def test_service_add_two(ds):
  service = testing.TestingService(ds)
  testcases = [
    deepsmith_pb2.Testcase(
        generator=deepsmith_pb2.Generator(name="foo", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "foo"},
        opts=["1", "2", "3"],
        timings=[
          deepsmith_pb2.ProfilingEvent(client="c", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="c", name="b", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="c", name="c", duration_seconds=3),
        ],
    ),
    deepsmith_pb2.Testcase(
        generator=deepsmith_pb2.Generator(name="bar", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "abc"},
        opts=["1", "2", "3", "4"],
        timings=[
          deepsmith_pb2.ProfilingEvent(client="d", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="d", name="d", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="d", name="c", duration_seconds=3),
        ],
    ),
  ]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  service.SubmitTestcases(request, None)

  with ds.session() as s:
    assert s.query(db.Client).count() == 2
    assert s.query(db.ProfilingEventName).count() == 4
    assert s.query(db.Testcase).count() == 2
    assert s.query(db.TestcaseInput).count() == 2
    assert s.query(db.TestcaseOpt).count() == 4
    assert s.query(db.TestcaseTiming).count() == 6


def serve_request(service, request):
  service.SubmitTestcases(request, None)

# Benchmarks.

def test_benchmark_add_1(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = [random_testcase()]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(serve_request, service, request)


def test_benchmark_add_2(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = [random_testcase(), random_testcase()]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(serve_request, service, request)


def test_benchmark_add_100(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = []
  for _ in range(100):
    testcases.append(random_testcase())
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(serve_request, service, request)


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
