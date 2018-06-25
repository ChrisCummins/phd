import pathlib
import sys
import time
import typing

import humanize
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.services import cldrive
from deeplearning.deepsmith.services import clgen
from deeplearning.deepsmith.services import generator as base_generator
from deeplearning.deepsmith.services import harness as base_harness
from gpu.oclgrind import oclgrind
from lib.labm8 import labdate
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'generator', None,
    'The path of the generator config proto.')
flags.DEFINE_string(
    'dut', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The name of the device under test, as described by cldrive. Run '
    '//gpu/cldrive --ls_env to see a list of available devices.')
flags.DEFINE_string(
    'interesting_results_dir', '/tmp/',
    'Directory to write interesting results to.')
flags.DEFINE_integer(
    'min_interesting_results', 1,
    'The minimum number of interesting testcases to discover before stopping.')
flags.DEFINE_integer(
    'max_testing_time_seconds', 60,
    'The maximum number of time to run in seconds. The actual runtime may be '
    'higher, as the in-progress batch must complete.')
flags.DEFINE_integer(
    'batch_size', 128,
    'The number of test cases to generate and execute in a single batch.')


def RunBatch(generator: base_generator.GeneratorBase,
             dut_harness: base_harness.HarnessBase,
             gs_harness: base_harness.HarnessBase,
             batch_size: int) -> typing.List[deepsmith_pb2.Result]:
  """Run one batch of testing.

  A batch of testing involves generating a set of testcases, executing them on
  the device under test, then determining for each whether the result is
  interesting. The interesting-ness test may involve executing testcases on the
  gold-standard device and comparing the outputs.

  Args:
    generator: The generator for testcases.
    dut_harness: The device under test.
    gs_harness: The gold-standard device, used to compare outputs against the
      device under test.
    batch_size: The number of testcases to generate and evaluate.

  Returns:
    A list of results which are determined to be interesting.
  """
  interesting_results = []

  # Generate testcases.
  logging.info('Generating %d testcases ...', batch_size)
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = batch_size
  res = generator.GenerateTestcases(req, None)

  # TODO(cec): Pre-exec testcase filters.

  # Evaluate testcases.
  logging.info('Evaluating %d testcases on %s ...', len(res.testcases),
               dut_harness.testbeds[0].opts['platform'][:12])
  results = RunTestcases(dut_harness, res.testcases)

  # TODO(cec): Pre-difftest result filters.

  for i, result in enumerate(results):
    outcome = ResultIsInteresting(result, gs_harness)
    if (outcome != deepsmith_pb2.DifferentialTest.Outcome.PASS):
      interesting_results.append(result)

  # TODO(cec): Post-difftest result filters.

  return interesting_results


def PreDifftestFilter(result: deepsmith_pb2.Result,
                      outcome: deepsmith_pb2.DifferentialTest.Outcome
                      ) -> bool:
  # TODO(cec): Complete port of dsmith difftest filters to new format.
  stderr = result.outputs['stderr']
  if result.outcome == deepsmith_pb2.Result.BUILD_FAILURE:
    if (("use of type 'double' requires cl_khr_fp64 extension" or
         'implicit declaration of function' or
         'function cannot have argument whose type is, or contains, type size_t' or
         'unresolved extern function' or
         'error: cannot increment value of type%' or
         'subscripted access is not allowed for OpenCL vectors' or
         'Images are not supported on given device' or
         'error: variables in function scope cannot be declared' or
         'error: implicit conversion ' or
         'Could not find a definition ') in stderr):
      return False
    if result.testbed.opts['opencl_version'] == '1.2':
      return False
  elif result.outcome == deepsmith_pb2.Result.RUNTIME_CRASH:
    if ((
        'clFinish CL_INVALID_COMMAND_QUEUE'
        'incompatible pointer to integer conversion' or
        'comparison between pointer and integer' or
        'warning: incompatible' or
        'warning: division by zero is undefined' or
        'warning: comparison of distinct pointer types' or
        'is past the end of the array' or
        'warning: comparison between pointer and' or
        'warning: array index' or
        'warning: implicit conversion from' or
        'array index -1 is before the beginning of the array' or
        'incompatible pointer' or
        'incompatible integer to pointer ') in stderr):
      return False
    # TODO(cec): oclgrind.verify_dsmith_testcase(testcase)
  elif outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_WRONG_OUTPUT:
    if 'float' or 'double' in result.testcase.inputs['src']:
      return False
    if program_meta.get_vector_inputs(s):
      return False
    if program_meta.get_compiler_warnings(s):
      return False
    if not self.get_gpuverified(s):
      return False
    if not self.get_oclverified(s):
      return False

  return True


def RunTestcases(harness: base_harness.HarnessBase,
                 testcases: typing.List[deepsmith_pb2.Testcase]
                 ) -> typing.List[deepsmith_pb2.Result]:
  """Run a set of testcases on a harness.

  Args:
    harness: The harness to drive the testcases with.
    testcases: The testcases to run.

  Returns:
    A list of results.
  """
  req = harness_pb2.RunTestcasesRequest()
  req.testbed.CopyFrom(harness.testbeds[0])
  req.testcases.extend(testcases)
  res = harness.RunTestcases(req, None)
  return res.results


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


class DiffTesterBase(object):
  """Base class for differential testers."""

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
               ) -> typing.List[deepsmith_pb2.DifferentialTest.Outcome]:
    """Differential test results and return their outcomes.

    Args:
      results: A list of Result protos.

    Returns:
      A list of differential test outcomes, one for each input result.
    """
    raise NotImplementedError


class UnaryTester(DiffTesterBase):

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
               ) -> typing.List[deepsmith_pb2.DifferentialTest.Outcome]:
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

    return [
      {
        deepsmith_pb2.Result.UNKNOWN: deepsmith_pb2.DifferentialTest.UNKNOWN,
        deepsmith_pb2.Result.BUILD_FAILURE: deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.BUILD_CRASH:
          deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE,
        deepsmith_pb2.Result.BUILD_TIMEOUT:
          deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE,
        deepsmith_pb2.Result.RUNTIME_CRASH: deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.RUNTIME_TIMEOUT:
          deepsmith_pb2.DifferentialTest.PASS,
        deepsmith_pb2.Result.PASS: deepsmith_pb2.DifferentialTest.PASS,
      }[results[0]]]


class GoldStandardDiffTester(DiffTesterBase):
  """A difftest which compares all results against the first result."""

  def __init__(self, outputs_equality_test: OutputsEqualityTest):
    self.outputs_equality_test = outputs_equality_test

  def __call__(self, results: typing.List[deepsmith_pb2.Result]
               ) -> typing.List[deepsmith_pb2.DifferentialTest.Outcome]:
    """Perform a difftest.

    Args:
      results: A list of Result protos.

    Returns:
      The differential test outcomes.
    """
    gs_result, *results = results

    # Determine the outcome of the gold standard.
    outcomes = [self.DiffTestOne(gs_result, gs_result)]

    # Difftest the results against the gold standard.
    for result in results:
      outcomes.append(self.DiffTestOne(gs_result, result))

    return outcomes

  def DiffTestOne(self, gs_result: deepsmith_pb2.Result,
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
    elif (result.outcome in {result_outcome.BUILD_CRASH,
                             result_outcome.BUILD_TIMEOUT}):
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
      return (
        difftest_outcome.PASS if
        self.outputs_equality_test([gs_result, result]) else
        difftest_outcome.ANOMALOUS_WRONG_OUTPUT
      )

    return difftest_outcome.UNKNOWN


def ResultIsInteresting(result: deepsmith_pb2.Result,
                        gs_harness: base_harness.HarnessBase
                        ) -> deepsmith_pb2.DifferentialTest.Outcome:
  """Determine if a result is interesting.

  If the result is interesting, an output 'notes' is added to explain why.

  Args:
    result: The result to check.
    gs_harness: A harness for a gold-standard device, which is used to compare
      output against.

  Returns:
    True if the result is interesting, else False.
  """
  difftester = UnaryTester()
  outcome = difftester([result])[0]

  if (outcome != deepsmith_pb2.DifferentialTest.PASS and
      outcome != deepsmith_pb2.DifferentialTest.UNKNOWN):
    return outcome

  if not (outcome == deepsmith_pb2.DifferentialTest.PASS and
          result.outcome == deepsmith_pb2.Result.PASS):
    return deepsmith_pb2.DifferentialTest.PASS

  # Run testcases against gold standard devices and apply differential testing.
  gs_result = RunTestcases(gs_harness, [result.testcase])[0]

  difftester = GoldStandardDiffTester(NamedOutputIsEqual('stdout'))
  _, dt_outcome = difftester([gs_result, result])
  logging.info('Differential test outcome: %s.',
               deepsmith_pb2.DifferentialTest.Outcome.Name(dt_outcome))
  # Add the differential test outcome to the result.
  result.outputs[
    'difftest_outcome'] = deepsmith_pb2.DifferentialTest.Outcome.Name(
      dt_outcome)
  result.outputs['gs_stdout'] = gs_result.outputs['stdout']
  result.outputs['gs_stderr'] = gs_result.outputs['stderr']
  return dt_outcome


def TestingLoop(min_interesting_results: int, max_testing_time_seconds: int,
                batch_size: int, generator: base_generator.GeneratorBase,
                dut_harness: base_harness.HarnessBase,
                gs_harness: base_harness.HarnessBase,
                interesting_results_dir: pathlib.Path,
                start_time: float = None) -> None:
  """The main fuzzing loop.

  Args:
    min_interesting_results: The minimum number of interesting results to find.
    max_testing_time_seconds: The maximum time allowed to find interesting
      results.
    batch_size: The number of testcases to generate and execute in each batch.
    generator: A testcase generator.
    dut_harness: The device under test.
    gs_harness: The device to compare outputs against.
    interesting_results_dir: The directory to write interesting results to.
    start_time: The starting time, as returned by time.time().
  """
  start_time = start_time or time.time()
  interesting_results_dir.mkdir(parents=True, exist_ok=True)
  num_interesting_results = 0
  batch_num = 0
  while (num_interesting_results < min_interesting_results and
         time.time() < start_time + max_testing_time_seconds):
    batch_num += 1
    logging.info('Starting generate / test / eval batch %d ...', batch_num)
    interesting_results = RunBatch(
        generator, dut_harness, gs_harness, batch_size)
    for result in interesting_results:
      pbutil.ToFile(result,
                    interesting_results_dir /
                    (str(labdate.MillisecondsTimestamp()) + '.pbtxt'))
      num_interesting_results += 1

  logging.info(
      'Stopping after %.2f seconds and %s batches (%.0fms / testcase).\n'
      'Found %s interesting results.', time.time() - start_time,
      humanize.intcomma(batch_num),
      (((time.time() - start_time) / (batch_num * batch_size)) * 1000),
      len(interesting_results))
  logging.flush()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  start_time = time.time()

  # Parse flags and instantiate testing objects.
  if not FLAGS.interesting_results_dir:
    raise app.UsageError('--interesting_results_dir must be set')
  interesting_results_dir = pathlib.Path(FLAGS.interesting_results_dir)
  if interesting_results_dir.exists() and not interesting_results_dir.is_dir():
    raise app.UsageError('--interesting_results_dir must be a directory')
  logging.info('Recording interesting results in %s.', interesting_results_dir)

  logging.info('Preparing generator.')
  if not FLAGS.generator:
    raise app.UsageError('--generator must be set')
  config = pathlib.Path(FLAGS.generator)
  if not pbutil.ProtoIsReadable(config, generator_pb2.ClgenGenerator()):
    raise app.UsageError('--generator is not a Generator proto')
  generator_config = pbutil.FromFile(config, generator_pb2.ClgenGenerator())
  generator = clgen.ClgenGenerator(generator_config)

  logging.info('Preparing device under test.')
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([FLAGS.dut])
  dut_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) == 1

  logging.info('Preparing gold standard testbed.')
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind.OpenCLEnvironment().name])
  gs_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) >= 1

  TestingLoop(FLAGS.min_interesting_results, FLAGS.max_testing_time_seconds,
              FLAGS.batch_size, generator, dut_harness, gs_harness,
              interesting_results_dir, start_time=start_time)


if __name__ == '__main__':
  try:
    app.run(main)
  except KeyboardInterrupt:
    logging.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    print('keyboard interrupt')
    sys.exit(1)
