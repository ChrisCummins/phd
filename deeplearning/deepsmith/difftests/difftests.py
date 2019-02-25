"""This module defines differential tests for results."""
import typing

from absl import flags

from deeplearning.deepsmith.proto import deepsmith_pb2

FLAGS = flags.FLAGS


class DiffTesterBase(object):
  """Base class for differential testers."""

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
              ) -> typing.List['deepsmith_pb2.DifferentialTest.Outcome']:
    """Differential test results and return their outcomes.

    Args:
      results: A list of Result protos.

    Returns:
      A list of differential test outcomes, one for each input result.
    """
    raise NotImplementedError


class OutputsEqualityTest(object):
  """An object which compares result outputs."""

  def __call__(self, results: typing.List[deepsmith_pb2.Result]) -> bool:
    raise NotImplementedError


class OutputsAreEqual(OutputsEqualityTest):
  """An outputs equality test which compares all outputs."""

  def __call__(self, results: typing.List[deepsmith_pb2.Result]) -> bool:
    return len(set(r.outputs for r in results)) == 1


class NamedOutputIsEqual(OutputsEqualityTest):
  """An outputs equality test which compares a single named output."""

  def __init__(self, output_name: str):
    self.output_name = output_name

  def __call__(self, results: typing.List[deepsmith_pb2.Result]) -> bool:
    """Test that a named output is equal in all results.

    Args:
      results: A list of results to compare the named output of.
      output_name: The name of the output in the result's outputs map.

    Returns:
      True if all named outputs are equal, else False.

    Raises:
      ValueError: if the named output is missing from any of the results.
    """
    if any(self.output_name not in r.outputs for r in results):
      raise ValueError(f"'{self.output_name}' missing in one or more results.")
    return len(set(r.outputs[self.output_name] for r in results)) == 1


class UnaryTester(DiffTesterBase):

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
              ) -> typing.List['deepsmith_pb2.DifferentialTest.Outcome']:
    """Unary test a result.

    Args:
      results: A list containing a single Result proto.

    Returns:
      A list containing one differential test outcome.

    Raises:
      ValueError: If called with more than or less than one Result proto.
    """
    if len(results) != 1:
      raise ValueError('UnaryTester must be called with exactly one result.')

    return [{
        deepsmith_pb2.Result.UNKNOWN:
        deepsmith_pb2.DifferentialTest.UNKNOWN,
        deepsmith_pb2.Result.BUILD_FAILURE:
        deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.BUILD_CRASH:
        deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE,
        deepsmith_pb2.Result.BUILD_TIMEOUT:
        deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE,
        deepsmith_pb2.Result.RUNTIME_CRASH:
        deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.RUNTIME_TIMEOUT:
        deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.PASS:
        deepsmith_pb2.DifferentialTest.PASS,
    }[results[0].outcome]]


class GoldStandardDiffTester(DiffTesterBase):
  """A difftest which compares all results against the first result."""

  def __init__(self, outputs_equality_test: OutputsEqualityTest):
    self.outputs_equality_test = outputs_equality_test

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
              ) -> typing.List['deepsmith_pb2.DifferentialTest.Outcome']:
    """Perform a difftest.

    Args:
      results: A list of Result protos.

    Returns:
      The differential test outcomes.
    """
    gs_result, *results = results
    if not results:
      raise ValueError('GoldStandardDiffTester called with only one result')

    # Determine the outcome of the gold standard.
    outcomes = [self.DiffTestOne(gs_result, gs_result)]

    # Difftest the results against the gold standard.
    for result in results:
      outcomes.append(self.DiffTestOne(gs_result, result))

    return outcomes

  def DiffTestOne(
      self,
      gs_result: deepsmith_pb2.Result,
      result: deepsmith_pb2.Result,
  ) -> deepsmith_pb2.DifferentialTest.Outcome:
    """Difftest one result against a golden standard.

    Args:
      gs_result: The golden standard (i.e. ground truth) result.
      result: The result to compare against the ground truth.

    Returns:
      The difftest outcome of the result.
    """

    # Short hand variables.
    result_outcome = deepsmith_pb2.Result
    difftest_outcome = deepsmith_pb2.DifferentialTest

    # We can't difftest an unknown outcome.
    if result.outcome == result_outcome.UNKNOWN:
      return difftest_outcome.UNKNOWN

    # Outcomes which are uninteresting if they match.
    uninteresting_equal_outcomes = {
        result_outcome.UNKNOWN,
        result_outcome.BUILD_FAILURE,
        result_outcome.RUNTIME_CRASH,
        result_outcome.RUNTIME_TIMEOUT,
    }

    # Outcomes which signal build failures.
    build_failures = {
        result_outcome.UNKNOWN,
        result_outcome.BUILD_FAILURE,
        result_outcome.BUILD_CRASH,
        result_outcome.BUILD_TIMEOUT,
    }

    # Outcomes which signal runtime failures.
    runtime_failures = {
        result_outcome.RUNTIME_CRASH,
        result_outcome.RUNTIME_TIMEOUT,
    }

    # Outcomes which are not interesting if they are equal.
    if (gs_result.outcome in uninteresting_equal_outcomes and
        gs_result.outcome == result.outcome):
      return difftest_outcome.PASS
    # Build failures which are always interesting.
    elif (result.outcome in {
        result_outcome.BUILD_CRASH, result_outcome.BUILD_TIMEOUT
    }):
      return difftest_outcome.ANOMALOUS_BUILD_FAILURE
    # Gold standard completed testcase, device under test failed to build OR
    # gold standard failed to build, device under test completed test.
    elif (gs_result.outcome not in build_failures and
          result.outcome in build_failures):
      return deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE
    elif (gs_result.outcome == result_outcome.BUILD_FAILURE and
          result.outcome not in build_failures):
      return deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_PASS
    # Gold standard completed testcase, device under test crashed OR
    # gold standard crashed, device under test completed testcase.
    elif (gs_result.outcome == result_outcome.PASS and
          result.outcome == result_outcome.RUNTIME_CRASH):
      return deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_CRASH
    elif (gs_result.outcome in runtime_failures and
          result.outcome == result_outcome.PASS):
      return deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_PASS
    # Gold standard crashed, device under test times out OR
    # gold standard times out, device under test crashes.
    elif ((gs_result.outcome == result_outcome.RUNTIME_CRASH and
           result.outcome == result_outcome.RUNTIME_TIMEOUT) or
          (gs_result.outcome == result_outcome.RUNTIME_TIMEOUT and
           result.outcome == result_outcome.RUNTIME_CRASH)):
      return deepsmith_pb2.DifferentialTest.PASS
    # Gold standard passes, device under test times out.
    elif (gs_result.outcome == result_outcome.PASS and
          result.outcome == result_outcome.RUNTIME_TIMEOUT):
      return deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_TIMEOUT
    # Both devices completed testcase, compare outputs.
    elif (gs_result.outcome == result_outcome.PASS and
          result.outcome == result_outcome.PASS):
      return (difftest_outcome.PASS if self.outputs_equality_test(
          [gs_result, result]) else difftest_outcome.ANOMALOUS_WRONG_OUTPUT)

    return difftest_outcome.UNKNOWN


class FiltersBase(object):
  """Base class for DeepSmith filters."""

  def PreExec(self, testcase: deepsmith_pb2.Testcase
             ) -> typing.Optional[deepsmith_pb2.Testcase]:
    """A filter callback to determine whether a testcase should be discarded.

    Args:
      testcase: The testcase to filter.

    Returns:
      True if testcase should be discarded, else False.
    """
    return testcase

  def PostExec(self, result: deepsmith_pb2.Result
              ) -> typing.Optional[deepsmith_pb2.Result]:
    """A filter callback to determine whether a result should be discarded.

    Args:
      result: The result to filter.

    Returns:
      True if result should be discarded, else False.
    """
    return result

  def PreDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                 ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """A filter callback to determine whether a difftest should be discarded.

    Args:
      difftest: The difftest to filter.

    Returns:
      True if difftest should be discarded, else False.
    """
    return difftest

  def PostDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """A filter callback to determine whether a difftest should be discarded.

    Args:
      difftest: The difftest to filter.

    Returns:
      True if difftest should be discarded, else False.
    """
    return difftest
