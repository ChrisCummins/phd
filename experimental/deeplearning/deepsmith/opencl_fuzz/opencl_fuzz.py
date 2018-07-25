import pathlib
import sys
import time
import typing

import humanize
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.difftests import difftests
from deeplearning.deepsmith.difftests import opencl as opencl_difftests
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.generators import generator as base_generator
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.harnesses import harness as base_harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import harness_pb2
from gpu.cldrive import env
from gpu.oclgrind import oclgrind
from lib.labm8 import labdate
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'generator', None,
    'The path of the generator config proto.')
flags.DEFINE_string(
    'base_harness', None,
    'The path to an optional base harness config proto. If set, the harness '
    'configs are copied from this. Else, the default config is used.')
flags.DEFINE_string(
    'dut', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The name of the device under test, as described by cldrive. Run '
    '//gpu/cldrive --ls_env to see a list of available devices.')
flags.DEFINE_bool(
    'opencl_opt', True,
    'If --noopencl_opt, OpenCL optimizations are disabled.')
flags.DEFINE_string(
    'interesting_results_dir',
    '/tmp/phd/experimental/deeplearning/opencl_fuzz/interesting_results',
    'Directory to write interesting results to.')
flags.DEFINE_bool(
    'all_results_are_interesting', False,
    'If set, all results are written to interesting_results_dir.')
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
  filters = opencl_difftests.ClgenOpenClFilters()

  interesting_results = []

  # Generate testcases.
  logging.info('Generating %d testcases ...', batch_size)
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = batch_size
  res = generator.GenerateTestcases(req, None)
  testcases = [testcase for testcase in res.testcases if
               filters.PreExec(testcase)]
  if len(res.testcases) - len(testcases):
    logging.info('Discarded %d testcases prior to execution.',
                 len(res.testcases) - len(testcases))

  # Evaluate testcases.
  logging.info('Evaluating %d testcases on %s ...', len(testcases),
               dut_harness.testbeds[0].opts['platform'][:12])
  unfiltered_results = RunTestcases(dut_harness, testcases)
  results = [result for result in unfiltered_results
             if filters.PostExec(result)]
  if len(unfiltered_results) - len(results):
    logging.info('Discarded %d results.',
                 len(unfiltered_results) - len(results))

  for i, result in enumerate(results):
    interesting_result = ResultIsInteresting(
        result, unary_difftester, gs_difftester, gs_harness, filters)
    if interesting_results:
      interesting_results.append(interesting_result)

  return interesting_results


def NotInteresting(
    result: deepsmith_pb2) -> typing.Optional[deepsmith_pb2.Result]:
  """An uninteresting result wrapper.

  Args:
    result: The result which is not interesting.

  Returns:
    The input result if --all_results_are_interesting flag set, else None.
  """
  if FLAGS.all_results_are_interesting:
    return result
  else:
    return None


def ResultIsInteresting(result: deepsmith_pb2.Result,
                        unary_difftester: difftests.UnaryTester,
                        gs_difftester: difftests.GoldStandardDiffTester,
                        gs_harness: base_harness.HarnessBase,
                        filters: difftests.FiltersBase
                        ) -> typing.Optional[deepsmith_pb2.Result]:
  """Determine if a result is interesting, and return it if it is.

  Args:
    result: The result to test.
    unary_difftester: A unary difftester.
    gs_difftester: A golden standard difftester.
    gs_harness: A golden standard test harness.
    filters: A set of difftest filters.

  Returns:
    The result if it is interesting, else None.
  """
  # First perform a unary difftest to see if the result is interesting without
  # needing to difftest, such as a compiler crash.
  unary_dt_outcome = unary_difftester([result])[0]
  if (unary_dt_outcome != deepsmith_pb2.DifferentialTest.PASS and
      unary_dt_outcome != deepsmith_pb2.DifferentialTest.UNKNOWN):
    result.outputs['difftest_outcome'] = (
      deepsmith_pb2.DifferentialTest.Outcome.Name(unary_dt_outcome))
    return result

  if not (unary_dt_outcome == deepsmith_pb2.DifferentialTest.PASS and
          result.outcome == deepsmith_pb2.Result.PASS):
    return NotInteresting(result)

  # Determine whether we can difftest the testcase.
  dt = filters.PreDifftest(deepsmith_pb2.DifferentialTest(result=[result]))
  if not dt:
    return NotInteresting(result)
  result = dt.result[0]

  # Run testcases against gold standard devices and difftest.
  gs_result = RunTestcases(gs_harness, result.testcase)[0]

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
    return NotInteresting(result)
  result = dt.result[1]

  if dt_outcome != deepsmith_pb2.DifferentialTest.PASS:
    # Add the differential test outcome to the result.
    result.outputs[
      'difftest_outcome'] = deepsmith_pb2.DifferentialTest.Outcome.Name(
        dt_outcome)
    result.outputs['gs_stdout'] = dt.result[0].outputs['stdout']
    result.outputs['gs_stderr'] = dt.result[0].outputs['stderr']
    return result


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


def GetBaseHarnessConfig() -> harness_pb2.CldriveHarness:
  """Load the base Cldrive harness configuration.

  If --base_harness is set, the config is read from this path. This allows
  overriding options such as the driver_cflags field.

  Returns:
    A CldriveHarness proto instance.
  """
  if FLAGS.base_harness:
    config = pbutil.FromFile(pathlib.Path(FLAGS.base_harness),
                             harness_pb2.CldriveHarness())
    config.ClearField('opencl_env')
    config.ClearField('opencl_opt')
    return config
  else:
    return harness_pb2.CldriveHarness()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  if FLAGS.ls_env:
    env.PrintOpenClEnvironments()
    return

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
  config = GetBaseHarnessConfig()
  config.opencl_env.extend([FLAGS.dut])
  config.opencl_opt.extend([FLAGS.opencl_opt])
  dut_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) == 1

  logging.info('Preparing gold standard testbed.')
  config = GetBaseHarnessConfig()
  config.opencl_env.extend([oclgrind.OpenCLEnvironment().name])
  config.opencl_opt.extend([True])
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
