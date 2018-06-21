import pathlib
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
    'max_testing_time_seconds', 3600,
    'The maximum number of time to run in seconds. The actual runtime may be '
    'higher, as the in-progress batch must complete.')
flags.DEFINE_integer(
    'batch_size', 100,
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

  # Evaluate testcases.
  logging.info('Evaluating %d testcases on %s ...', len(res.testcases),
               dut_harness.testbeds[0].opts['platform'][:12])
  results = RunTestcases(dut_harness, res.testcases)

  for i, result in enumerate(results):
    if ResultIsInteresting(result, gs_harness):
      interesting_results.append(result)

  return interesting_results


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


def ResultIsInteresting(result: deepsmith_pb2.Result,
                        gs_harness: base_harness.HarnessBase) -> bool:
  """Determine if a result is interesting.

  If the result is interesting, an output 'notes' is added to explain why.

  Args:
    result: The result to check.
    gs_harness: A harness for a gold-standard device, which is used to compare
      output against.

  Returns:
    True if the result is interesting, else False.
  """
  # We don't extract anything of interest from runtime timeouts or build
  # failures. We *could* see if the outcome differs on the gold standard
  # harness.
  if (result.outcome == deepsmith_pb2.Result.BUILD_FAILURE or
      result.outcome == deepsmith_pb2.Result.RUNTIME_TIMEOUT):
    return False

  # A static failure is of immediate interest.
  if (result.outcome == deepsmith_pb2.Result.BUILD_CRASH or
      result.outcome == deepsmith_pb2.Result.BUILD_TIMEOUT):
    result.outputs['notes'] = 'OpenCL kernel failed to compile'
    return True

  # Remaining outcomes: {Runtime Crash, Pass}.
  # Run testcases against gold standard devices and apply differential testing.
  gs_result = RunTestcases(gs_harness, [result.testcase])[0]
  logging.info('Gold standard outcome: %s.', gs_result.outcome)

  # Gold standard crashes. Nothing interesting here.
  if (result.outcome == deepsmith_pb2.Result.RUNTIME_CRASH and
      gs_result.outcome == deepsmith_pb2.Result.RUNTIME_CRASH):
    return False

  # Gold standard device fails to build. This is potentially interesting if the
  # device under test was incorrect in successfully building the kernel.
  if (gs_result.outcome == deepsmith_pb2.Result.BUILD_CRASH or
      gs_result.outcome == deepsmith_pb2.Result.BUILD_TIMEOUT):
    result.outputs['notes'] = (
      'OpenCL kernel failed to compile on gold standard device')
    return True

  if (result.outcome == deepsmith_pb2.Result.RUNTIME_CRASH and
      gs_result.outcome == deepsmith_pb2.Result.PASS):
    result.outputs['notes'] = (
      'OpenCL kernel crashed on device under test, but not on gold '
      'standard device')
    return True

  if (result.outcome == deepsmith_pb2.Result.PASS and
      gs_result.outcome == deepsmith_pb2.Result.PASS and
      gs_result.outputs['stdout'] != result.outputs['stdout']):
    result.output['notes'] = (
      'OpenCL kernel produces different output on gold standard device')
    result.outputs['gs_stdout'] = gs_result.outputs['stdout']
    result.outputs['gs_stderr'] = gs_result.outputs['stderr']
    return True


def TestingLoop(min_interesting_results: int, max_testing_time_seconds: int,
                batch_size: int, generator: base_generator.GeneratorBase,
                dut_harness: base_harness.HarnessBase,
                gs_harness: base_harness.HarnessBase,
                interesting_results_dir: pathlib.Path) -> None:
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
  """
  interesting_results_dir.mkdir(parents=True, exist_ok=True)
  start_time = time.time()
  num_interesting_results = 0
  batch_num = 0
  while (num_interesting_results < min_interesting_results and
         time.time() < start_time + max_testing_time_seconds):
    batch_num += 1
    logging.info('Starting generate / test / eval batch %d ...', batch_num)
    interesting_results = RunBatch(
        generator, dut_harness, gs_harness, batch_size)
    for result in interesting_results:
      pbutil.ToFile(
          result,
          interesting_results_dir / f'{num_interesting_results:04d}.pbtxt')
      num_interesting_results += 1

  logging.info('Stopping after %.2f seconds and %s batches (%.2fs / batch).\n'
               'Found %s interesting results.', time.time() - start_time,
               humanize.intcomma(batch_num),
               (time.time() - start_time) / batch_num, len(interesting_results))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

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
              interesting_results_dir)


if __name__ == '__main__':
  app.run(main)
