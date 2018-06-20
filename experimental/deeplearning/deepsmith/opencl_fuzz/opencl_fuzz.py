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
from deeplearning.deepsmith.services import generator as base_generator
from deeplearning.deepsmith.services import harness as base_harness
from deeplearning.deepsmith.services import randchar
# from deeplearning.deepsmith.services import clgen
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'interesting_results_dir', '/tmp/',
    'Directory to write interesting results to.')
flags.DEFINE_integer(
    'min_interesting_results', 1,
    'The minimum number of interesting testcases to discover before stopping.')
flags.DEFINE_integer(
    'max_testing_time_seconds', 1,
    'The maximum number of time to run in seconds. The actual runtime may be '
    'higher, as the in-progress batch must complete.')
flags.DEFINE_integer(
    'batch_size', 100,
    'The number of test cases to generate and execute in a single batch.')
flags.DEFINE_string(
    'oclgrind', 'oclgrind',
    'Path to the oclgrind binary. This does not need to be set if oclgring is '
    'in the system $PATH.')

# TODO(cec): Use a ClgenGenerator, not a RandcharGenerator.
CLGEN_CONFIG = """
# File: //deeplearning/deepsmith/proto/generator.proto
# Proto: deepsmith.RandCharGenerator
toolchain: "opencl"
string_min_len: 100
string_max_len: 200
harness {
  name: "cldrive"
}
"""

DUT_HARNESS_CONFIG = """
# File: //deeplearning/deepsmith/proto/harness.proto
# Proto: deepsmith.CldriveHarness
opencl_env: "CPU|Apple|Intel(R)_Core(TM)_i5-3570K_CPU_@_3.40GHz|1.1|1.2"
"""

GOLDEN_STANDARD_HARNESS_CONFIG = """
# File: //deeplearning/deepsmith/proto/harness.proto
# Proto: deepsmith.CldriveHarness
opencl_env: "Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_16.10|1.2"
"""


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
  logging.info('Evaluating %d testcases on %s ...', batch_size,
               dut_harness.testbeds[0].opts['platform'][:12])
  results = RunTestcases(dut_harness, res.testcases)

  for i, result in enumerate(results):
    logging.info('Result %d: %s.', i + 1, result.outputs['outcome'])
    if ResultIsInteresting(result, gs_harness):
      interesting_results.append(result)

  return interesting_results


def RunTestcases(harness: base_harness.HarnessBase,
                 testcases: typing.List[deepsmith_pb2.Testcase]
                 ) -> typing.List[deepsmith_pb2.Result]:
  """Wrapper around RPC calls.

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
  outcome = result.outputs['outcome']

  if outcome == 'Build Failure':
    # We assume that a
    return False

  # A static failure is of immediate interest.
  if outcome == 'Build Crash' or outcome == 'Build Timeout':
    result.outputs['notes'] = 'OpenCL kernel failed to compile'
    return True

  # Remaining outcomes: {Runtime Crash, Pass}.
  # Run testcases against gold standard devices and apply differential testing.
  gs_req = harness_pb2.RunTestcasesRequest()
  gs_req.testbed.CopyFrom(gs_harness.testbeds[0])
  gs_req.testcases.extend([result.testcase])
  gs_res = gs_harness.RunTestcases(gs_req, None)
  gs_result = gs_res.results[0]
  gs_outcome = gs_result.outputs['outcome']
  logging.info('Gold standard outcome: %s.', gs_outcome)

  if gs_outcome == 'Build Crash' or gs_outcome == 'Build Timeout':
    result.outputs['notes'] = (
      'OpenCL kernel failed to compile on gold standard device')
    return True

  if outcome == 'Runtime Crash' and gs_outcome == 'Runtime Crash':
    # Gold standard crashes. Nothing interesting here.
    return False

  if outcome == 'Runtime Crash' and gs_outcome == 'Pass':
    result.outputs['notes'] = (
      'OpenCL kernel crashed on device under test, but not on gold '
      'standard device')
    return True

  if outcome == 'Pass' and gs_outcome == 'Pass':
    if gs_result.outputs['stdout'] != result.outputs['stdout']:
      result.output['notes'] = (
        'OpenCL kernel produces different output on gold standard device')
      result.outputs['gs_stdout'] = gs_result.outputs['stdout']
      result.outputs['gs_stderr'] = gs_result.outputs['stderr']
      return True


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  # Get the directory to write interesting results to.
  if not FLAGS.interesting_results_dir:
    raise app.UsageError('--interesting_results_dir must be set')
  interesting_results_dir = pathlib.Path(FLAGS.interesting_results_dir)
  if interesting_results_dir.exists() and not interesting_results_dir.is_dir():
    raise app.UsageError('--interesting_results_dir must be a directory')
  interesting_results_dir.mkdir(parents=True, exist_ok=True)
  logging.info('Recording interesting results in %s.', interesting_results_dir)

  logging.info('Preparing generator.')
  # TODO(cec): Use a ClgenGenerator, not a RandcharGenerator.
  clgen_config = pbutil.FromString(
      CLGEN_CONFIG, generator_pb2.RandCharGenerator())
  generator = randchar.RandCharGenerator(clgen_config)

  logging.info('Preparing device under test.')
  config = pbutil.FromString(DUT_HARNESS_CONFIG, harness_pb2.CldriveHarness())
  dut_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) == 1

  logging.info('Preparing gold standard testbed.')
  config = pbutil.FromString(
      GOLDEN_STANDARD_HARNESS_CONFIG, harness_pb2.CldriveHarness())
  gs_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) >= 1

  start_time = time.time()
  num_interesting_results = 0
  batch_num = 0
  while (num_interesting_results < FLAGS.min_interesting_results and
         time.time() < start_time + FLAGS.max_testing_time_seconds):
    batch_num += 1
    logging.info('Starting generate / test / eval batch %d ...', batch_num)
    interesting_results = RunBatch(
        generator, dut_harness, gs_harness, FLAGS.batch_size)
    for result in interesting_results:
      pbutil.ToFile(
          result,
          interesting_results_dir / f'{num_interesting_results:04d}.pbtxt')
      num_interesting_results += 1

  logging.info(
      'Stopping after %.2f seconds and %s batches (%.2fs / batch).\n'
      'Found %s interesting results.', time.time() - start_time,
      humanize.intcomma(batch_num), (time.time() - start_time) / batch_num,
      len(interesting_results))


if __name__ == '__main__':
  app.run(main)
