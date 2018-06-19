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
from deeplearning.deepsmith.services import randchar
# from deeplearning.deepsmith.services import clgen
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'batch_size', 100,
    'The number of test cases to generate and execute in a single batch.')
flags.DEFINE_integer(
    'min_interesting_results', 1,
    'The minimum number of interesting testcases to discover before stopping.')
flags.DEFINE_integer(
    'max_testing_time_seconds', 1,
    'The maximum number of time to run in seconds. The actual runtime may be '
    'higher, as the in-progress batch must complete.')

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
opencl_env: "CPU|Apple|Intel(R)_Core(TM)_i5-3570K_CPU_@_3.40GHz|1.1|1.2"
"""


def RunBatch(generator, dut_harness, gs_harness,
             batch_size: int) -> typing.List[deepsmith_pb2.Result]:
  interesting_results = []
  logging.info('Starting generate / test / eval batch ...')

  # Generate testcases.
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = batch_size
  res = generator.GenerateTestcases(req, None)

  # Evaluate testcases.
  req = harness_pb2.RunTestcasesRequest()
  req.testbed.CopyFrom(dut_harness.testbeds[0])
  req.testcases.extend(res.testcases)
  res = dut_harness.RunTestcases(req, None)

  for result in res.results:
    # TODO(cec):
    # if result.outputs['outcome'] == 'Build failure':
    #   continue

    if (result.outputs['outcome'] == 'Build Crash' or
        result.outputs['outome'] == 'Build Timeout'):
      # A static failure is of immediate interest.
      result.outputs['notes'] = 'OpenCL kernel failed to compile'
      interesting_results.append(result)
      continue

    # Remaining outcomes: {Runtime Crash, Pass}.
    # Run testcases against gold standard devices and apply differential
    # testing.
    logging.info("Comparing to gold standard")
    gs_req = harness_pb2.RunTestcasesRequest()
    gs_req.testbed.CopyFrom(gs_harness.testbeds[0])
    gs_req.testcases.extend([result.testcase])
    gs_res = gs_harness.RunTestcases(gs_req, None)
    gs_result = gs_res.results[0]
    if (gs_result.outputs['outcome'] == 'Build Crash' or
        gs_result.outputs['outcome'] == 'Build Timeout'):
      result.outputs['notes'] = (
        'OpenCL kernel failed to compile on gold standard device')
      interesting_results.append(result)
      continue
    elif gs_result.outputs['outcome'] == 'Runtime Crash':
      # Gold standard crashes. Nothing interesting here.
      continue
    elif gs_result.outputs['outcome'] == result.outputs['Runtime Crash']:
      result.outputs['notes'] = (
        'OpenCL kernel crashed on device under test, but not on gold '
        'standard device')
      interesting_results.append(result)
    elif (gs_result.outputs['outcome'] == 'Pass' and
          result.outputs['outcome'] == 'Pass'):
      if gs_result.outputs['stdout'] != result.outputs['stdout']:
        result.output['notes'] = (
          'OpenCL kernel produces different output on gold standard device')
        result.outputs['gs_stdout'] = gs_result.outputs['stdout']
        result.outputs['gs_stderr'] = gs_result.outputs['stderr']
        interesting_results.append(result)

  return interesting_results


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  logging.info('CLgen')
  # TODO(cec): Use a ClgenGenerator, not a RandcharGenerator.
  clgen_config = pbutil.FromString(
      CLGEN_CONFIG, generator_pb2.RandCharGenerator())
  generator = randchar.RandCharGenerator(clgen_config)

  config = pbutil.FromString(DUT_HARNESS_CONFIG, harness_pb2.CldriveHarness())
  dut_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) == 1

  config = pbutil.FromString(
      GOLDEN_STANDARD_HARNESS_CONFIG, harness_pb2.CldriveHarness())
  gs_harness = cldrive.CldriveHarness(config)
  assert len(dut_harness.testbeds) >= 1

  start_time = time.time()
  interesting_results = []
  batch_num = 0
  while (len(interesting_results) < FLAGS.min_interesting_results and
         time.time() < start_time + FLAGS.max_testing_time_seconds):
    interesting_results += RunBatch(
        generator, dut_harness, gs_harness, FLAGS.batch_size)
    batch_num += 1

  logging.info(
      'Stopping after %.2f seconds and %s batches (%.2fs / batch).\n'
      'Found %s interesting results.', time.time() - start_time,
      humanize.intcomma(batch_num), (time.time() - start_time) / batch_num,
      len(interesting_results))
  for result in interesting_results:
    print(result)


if __name__ == '__main__':
  app.run(main)
