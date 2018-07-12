import pathlib
import sys
import time
import typing

import humanize
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith import difftests
from deeplearning.deepsmith import filters as filters_base
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.generators import generator as base_generator
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.harnesses import harness as base_harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import harness_pb2
from gpu import cldrive as cldrive_lib
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
  # Our differential testers and result filters.
  unary_difftester = difftests.UnaryTester()
  gs_difftester = difftests.GoldStandardDiffTester(
      difftests.NamedOutputIsEqual('stdout'))
  filters = ClgenOpenClFilters()

  interesting_results = []

  # Generate testcases.
  logging.info('Generating %d testcases ...', batch_size)
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = batch_size
  res = generator.GenerateTestcases(req, None)
  testcases = [testcase for testcase in res.testcases if
               filters.PreExec(testcase)]
  logging.info('Discarded %d testcases prior to execution.',
               len(res.testcases) - len(testcases))

  # Evaluate testcases.
  logging.info('Evaluating %d testcases on %s ...', len(testcases),
               dut_harness.testbeds[0].opts['platform'][:12])
  unfiltered_results = RunTestcases(dut_harness, testcases)
  results = [result for result in unfiltered_results
             if filters.PostExec(result)]
  logging.info('Discarded %d results.', len(unfiltered_results) - len(results))

  for i, result in enumerate(results):
    # First perform a unary difftest to see the result is interesting without
    # needing to difftest, such as a compiler crash.
    unary_dt_outcome = unary_difftester([result])[0]

    if (unary_dt_outcome != deepsmith_pb2.DifferentialTest.PASS and
        unary_dt_outcome != deepsmith_pb2.DifferentialTest.UNKNOWN):
      interesting_results.append(result)
      continue

    if not (unary_dt_outcome == deepsmith_pb2.DifferentialTest.PASS and
            result.outcome == deepsmith_pb2.Result.PASS):
      continue

    # Determine whether we can difftest the testcase.
    dt = filters.PreDifftest(deepsmith_pb2.DifferentialTest(result=[result]))
    if not dt:
      logging.info('Cannot difftest result.')
      continue

    # Run testcases against gold standard devices and difftest.
    gs_result = RunTestcases(gs_harness, dt.result[0].testcase)[0]

    dt_outcomes = gs_difftester([gs_result, result])
    dt_outcome = dt.outcome[1]
    logging.info('Differential test outcome: %s.',
                 deepsmith_pb2.DifferentialTest.Outcome.Name(dt_outcome))

    # Determine whether we can use the difftest result.
    dt = filters.PostDifftest(deepsmith_pb2.DifferentialTest(
        result=[gs_result, result],
        outcome=dt_outcomes))
    if not dt:
      logging.info('Cannot use gold standard difftest result.')
      continue

    if dt_outcome != deepsmith_pb2.DifferentialTest.PASS:
      # Add the differential test outcome to the result.
      dt.result[1].outputs[
        'difftest_outcome'] = deepsmith_pb2.DifferentialTest.Outcome.Name(
          dt_outcome)
      dt.result[1].outputs['gs_stdout'] = dt.result[0].outputs['stdout']
      dt.result[1].outputs['gs_stderr'] = dt.result[0].outputs['stderr']
      interesting_results.append(dt.result[1])

  return interesting_results


def RedFlagCompilerWarnings(result: deepsmith_pb2.Result) -> bool:
  return ('clFinish CL_INVALID_COMMAND_QUEUE' or
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
          'incompatible integer to pointer ') in result.outputs['stderr']


def LegitimateBuildFailure(result: deepsmith_pb2.Result) -> bool:
  return ("use of type 'double' requires cl_khr_fp64 extension" or
          'implicit declaration of function' or
          ('function cannot have argument whose type is, or '
           'contains, type size_t') or
          'unresolved extern function' or
          'error: cannot increment value of type' or
          'subscripted access is not allowed for OpenCL vectors' or
          'Images are not supported on given device' or
          'error: variables in function scope cannot be declared' or
          'error: implicit conversion ' or
          'Could not find a definition ') in result.outputs['stderr']


def ContainsFloatingPoint(testcase: deepsmith_pb2.Testcase) -> bool:
  """Return whether source code contains floating points."""
  return 'float' or 'double' in testcase.inputs['src']


def HasVectorInputs(testcase: deepsmith_pb2.Testcase) -> bool:
  """Return whether any of the kernel arguments are vector types."""
  for arg in cldrive_lib.extract_args(testcase.inputs['src']):
    if arg.is_vector:
      return True
  return False


class ClgenOpenClFilters(filters_base.FiltersBase):

  def PreDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """Determine whether a difftest should be discarded."""
    # We cannot difftest the output of OpenCL kernels which contain vector
    # inputs or floating points. We *can* still difftest these kernels if we're
    # not comparing the outputs.
    if difftest.result[0].outcome == deepsmith_pb2.Result.PASS:
      if ContainsFloatingPoint(difftest.result[0].testcase):
        return None
      if HasVectorInputs(difftest.result[0].testcase):
        return None
    return difftest

  def PostDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                   ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """Determine whether a difftest should be discarded."""
    for result, outcome in zip(difftest.result, difftest.outcome):
      # An OpenCL kernel which legitimately failed to build on any testbed
      # automatically disqualifies from difftesting.
      if (result.outcome == deepsmith_pb2.Result.BUILD_FAILURE and
          LegitimateBuildFailure(result)):
        return None
      # An anomalous build failure for an earlier OpenCL version can't be
      # difftested, since we don't know if the failure is legitimate.
      if (outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE and
          result.testbed.opts['opencl_version'] == '1.2'):
        # TODO(cec): Determine if build succeeded on any 1.2 testbed before
        # discarding.
        return None
      # An anomalous runtime outcome requires more vigurous examination of the
      # testcase.
      if (outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_CRASH or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_PASS or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_WRONG_OUTPUT or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_TIMEOUT):
        if RedFlagCompilerWarnings(result):
          return None
        # TODO(cec): Verify testcase with oclgrind.
        # TODO(cec): Verify testcase with gpuverify.
    return difftest


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
    num_interesting_results += len(interesting_results)
    for result in interesting_results:
      pbutil.ToFile(result,
                    interesting_results_dir /
                    (str(labdate.MillisecondsTimestamp()) + '.pbtxt'))

  logging.info(
      'Stopping after %.2f seconds and %s batches (%.0fms / testcase).\n'
      'Found %s interesting results.', time.time() - start_time,
      humanize.intcomma(batch_num),
      (((time.time() - start_time) / (batch_num * batch_size)) * 1000),
      num_interesting_results)
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
  generator = clgen_pretrained.ClgenGenerator(generator_config)

  logging.info('Preparing device under test.')
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([FLAGS.dut])
  dut_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) == 1

  logging.info('Preparing gold standard testbed.')
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind.OpenCLEnvironment().name])
  gs_harness = cldrive.CldriveHarness(config)
  assert len(gs_harness.testbeds) >= 1

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
