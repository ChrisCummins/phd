"""Unit tests for //experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz.py."""

import pytest
import sys
import typing
from absl import app
from absl import flags

from deeplearning.deepsmith.difftests import difftests
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.harnesses import harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from experimental.deeplearning.deepsmith.opencl_fuzz import opencl_fuzz
from gpu.oclgrind import oclgrind


FLAGS = flags.FLAGS


# Test fixtures.

@pytest.fixture(scope='function')
def cldrive_harness():
  """Test fixture to return an Cldrive test harness."""
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind.OpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  return cldrive.CldriveHarness(config)


# Mock classes.

class MockFilters(difftests.FiltersBase):
  """A mock class for simple filters."""

  def __init__(self, return_val: bool = True):
    super(MockFilters, self).__init__()
    self.return_val = return_val
    self.PreExec_call_args = []
    self.PostExec_call_args = []
    self.PreDifftest_call_args = []
    self.PostDifftest_call_args = []

  def PreExec(self, testcase: deepsmith_pb2.Testcase
              ) -> typing.Optional[deepsmith_pb2.Testcase]:
    self.PreExec_call_args.append(testcase)
    return testcase if self.return_val else None

  def PostExec(self, result: deepsmith_pb2.Result
               ) -> typing.Optional[deepsmith_pb2.Result]:
    self.PostExec_call_args.append(result)
    return result if self.return_val else None

  def PreDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    self.PreDifftest_call_args.append(difftest)
    return difftest if self.return_val else None

  def PostDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                   ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    self.PostDifftest_call_args.append(difftest)
    return difftest if self.return_val else None


class MockUnaryTester(difftests.UnaryTester):
  """A mock unary tester."""

  def __init__(self, return_val: typing.List[int] = None):
    super(MockUnaryTester, self).__init__()
    self.call_args = []
    self.return_val = return_val

  def __call__(self,
               results: typing.List[deepsmith_pb2.Result]) -> typing.List[int]:
    self.call_args.append(results)
    return self.return_val


class MockGoldStandardDiffTester(difftests.GoldStandardDiffTester):
  """A mock gold standard difftester."""

  def __init__(self, return_val: typing.List[int] = None):
    super(MockGoldStandardDiffTester, self).__init__(
        difftests.OutputsEqualityTest())
    self.call_args = []
    self.return_val = return_val

  def __call__(self,
               results: typing.List[deepsmith_pb2.Result]) -> typing.List[int]:
    self.call_args.append(results)
    return self.return_val


class MockHarness(harness.HarnessBase):
  """A mock harness."""

  def __init__(self, return_val: harness_pb2.RunTestcasesResponse = None):
    super(MockHarness, self).__init__(None)
    self.return_val = return_val
    self.RunTestcases_call_requests = []

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    """Mock method which returns return_val."""
    del context
    self.RunTestcases_call_requests.append(request)
    return self.return_val


# ResultIsInteresting() tests.


def test_ResultIsInteresting_unknown():
  """An unknown outcome is not interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
      deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.UNKNOWN),
      difftests.UnaryTester(),
      difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual('stdout')),
      gs_harness,
      filters)
  assert not result
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


def test_ResultIsInteresting_build_crash():
  """A build crash is interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
      deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.BUILD_CRASH),
      difftests.UnaryTester(),
      difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual('stdout')),
      gs_harness,
      filters)
  assert result
  assert result.outputs['difftest_outcome'] == 'ANOMALOUS_BUILD_FAILURE'
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


def test_ResultIsInteresting_build_timeout():
  """A build timeout is interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
      deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.BUILD_TIMEOUT),
      difftests.UnaryTester(),
      difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual('stdout')),
      gs_harness,
      filters)
  assert result
  assert result.outputs['difftest_outcome'] == 'ANOMALOUS_BUILD_FAILURE'
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
