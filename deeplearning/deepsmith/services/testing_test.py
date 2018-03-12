"""Tests for //deeplearning/deepsmith/services:testing."""
import random
import sys

import pytest
import string
from absl import app

import deeplearning.deepsmith.client
import deeplearning.deepsmith.profiling_event
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.services import testing


def CreateRandomGenerator() -> deepsmith_pb2.Generator:
  return deepsmith_pb2.Generator(
      name=random.choice(["foo", "bar", "baz"]),
      version=random.choice(["1", "1", "1", "2"]),
  )


def CreateRandomHarness():
  return deepsmith_pb2.Harness(
      name=random.choice(["a", "b", "c"]),
      version=random.choice(["1", "1", "1", "2"]),
  )


def CreateRandomToolchain() -> str:
  return random.choice(["cpp", "java", "python"])


def CreateRandomStr(n: int) -> str:
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def CreateRandomInput() -> str:
  return CreateRandomStr(int(random.random() * 1000) + 1)


def CreateRandomInputs():
  return {"src": CreateRandomInput(), "opt": CreateRandomInput()}


def CreateRandomOpt():
  return {CreateRandomStr(3): CreateRandomStr(32)}


def CreateRandomOpts():
  return [CreateRandomOpt() for _ in range(int(random.random() * 5) + 1)]


def CreateRandomTestcase() -> deepsmith_pb2.Testcase:
  client = CreateRandomStr(16)
  return deepsmith_pb2.Testcase(
      toolchain=CreateRandomToolchain(),
      generator=CreateRandomGenerator(),
      harness=CreateRandomHarness(),
      inputs=CreateRandomInputs(),
      profiling_events=[
        deepsmith_pb2.ProfilingEvent(client=client, name="a", duration_seconds=random.random()),
        deepsmith_pb2.ProfilingEvent(client=client, name="b", duration_seconds=random.random()),
        deepsmith_pb2.ProfilingEvent(client=client, name="c", duration_seconds=random.random()),
      ],
  )


def test_TestingService_empty_datastore(ds):
  service = testing.TestingService(ds)
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=[])
  response = service.SubmitTestcases(request, None)
  assert type(response) == deepsmith_pb2.SubmitTestcasesResponse
  assert response.status == deepsmith_pb2.SubmitTestcasesResponse.SUCCESS

  with ds.Session() as s:
    assert s.query(deeplearning.deepsmith.client.Client).count() == 0
    assert s.query(deeplearning.deepsmith.profiling_event.ProfilingEventType).count() == 0
    assert s.query(deeplearning.deepsmith.testcase.Testcase).count() == 0
    assert s.query(deeplearning.deepsmith.testcase.TestcaseInput).count() == 0
    assert s.query(deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent).count() == 0


def test_TestingService_SubmitTestcases_one(ds):
  service = testing.TestingService(ds)
  testcases = [
    deepsmith_pb2.Testcase(
        toolchain="cpp",
        generator=deepsmith_pb2.Generator(name="foo", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "foo"},
        timings=[
          deepsmith_pb2.ProfilingEvent(client="c", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="c", name="b", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="c", name="c", duration_seconds=3),
        ],
    ),
  ]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  service.SubmitTestcases(request, None)

  with ds.Session() as s:
    assert s.query(deeplearning.deepsmith.client.Client).count() == 1
    assert s.query(deeplearning.deepsmith.profiling_event.ProfilingEventType).count() == 3
    assert s.query(deeplearning.deepsmith.testcase.Testcase).count() == 1
    assert s.query(deeplearning.deepsmith.testcase.TestcaseInput).count() == 1
    assert s.query(deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent).count() == 3


def test_TestingService_SubmitTestcases_two(ds):
  service = testing.TestingService(ds)
  testcases = [
    deepsmith_pb2.Testcase(
        toolchain="cpp",
        generator=deepsmith_pb2.Generator(name="foo", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "foo"},
        timings=[
          deepsmith_pb2.ProfilingEvent(client="c", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="c", name="b", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="c", name="c", duration_seconds=3),
        ],
    ),
    deepsmith_pb2.Testcase(
        toolchain="cpp",
        generator=deepsmith_pb2.Generator(name="bar", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "abc"},
        timings=[
          deepsmith_pb2.ProfilingEvent(client="d", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="d", name="d", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="d", name="c", duration_seconds=3),
        ],
    ),
  ]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  service.SubmitTestcases(request, None)

  with ds.Session() as s:
    assert s.query(deeplearning.deepsmith.client.Client).count() == 2
    assert s.query(deeplearning.deepsmith.profiling_event.ProfilingEventType).count() == 4
    assert s.query(deeplearning.deepsmith.testcase.Testcase).count() == 2
    assert s.query(deeplearning.deepsmith.testcase.TestcaseInput).count() == 2
    assert s.query(deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent).count() == 6


# RequestTestcases

@pytest.mark.skip(reason="FIXME(cec):")
def test_TestingService_RequestTestcases_one(ds):
  service = testing.TestingService(ds)
  testcases = [
    deepsmith_pb2.Testcase(
        toolchain="cpp",
        generator=deepsmith_pb2.Generator(name="foo", version="foo"),
        harness=deepsmith_pb2.Harness(name="foo", version="bar"),
        inputs={"src": "foo"},
        timings=[
          deepsmith_pb2.ProfilingEvent(client="c", name="a", duration_seconds=1),
          deepsmith_pb2.ProfilingEvent(client="c", name="b", duration_seconds=2),
          deepsmith_pb2.ProfilingEvent(client="c", name="c", duration_seconds=3),
        ],
    ),
  ]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  service.SubmitTestcases(request, None)

  with ds.Session() as s:
    assert s.query(deeplearning.deepsmith.client.Client).count() == 1
    assert s.query(deeplearning.deepsmith.profiling_event.ProfilingEventType).count() == 3
    assert s.query(deeplearning.deepsmith.testcase.Testcase).count() == 1
    assert s.query(deeplearning.deepsmith.testcase.TestcaseInput).count() == 1
    assert s.query(deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent).count() == 3

  request = deepsmith_pb2.RequestTestcasesRequest(
      toolchain="cpp",
  )
  response = service.RequestTestcases(request, None)

  assert response.status == deepsmith_pb2.RequestTestcasesResponse.SUCCESS
  assert len(response.testcases) == 1
  # assert response.testcases[0] == testcases[0]


def test_TestingService_RequestTestcases_invalid_request(ds):
  service = testing.TestingService(ds)

  # max_num_testcases must be > 1.
  request = deepsmith_pb2.RequestTestcasesRequest(
      max_num_testcases=-1,
  )

  response = service.RequestTestcases(request, None)
  assert response.status == deepsmith_pb2.RequestTestcasesResponse.INVALID_REQUEST
  assert response.error == "max_num_testcases must be >= 1, not -1"


# Benchmarks.

def _SubmitTestcasesRequest(service, request):
  service.SubmitTestcases(request, None)


def test_benchmark_TestingService_SubmitTestcases_one(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = [CreateRandomTestcase()]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(_SubmitTestcasesRequest, service, request)


def test_benchmark_TestingService_SubmitTestcases_two(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = [CreateRandomTestcase(), CreateRandomTestcase()]
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(_SubmitTestcasesRequest, service, request)


def test_benchmark_TestingService_SubmitTestcases_100(ds, benchmark):
  service = testing.TestingService(ds)
  testcases = []
  for _ in range(100):
    testcases.append(CreateRandomTestcase())
  request = deepsmith_pb2.SubmitTestcasesRequest(testcases=testcases)
  benchmark(_SubmitTestcasesRequest, service, request)


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
