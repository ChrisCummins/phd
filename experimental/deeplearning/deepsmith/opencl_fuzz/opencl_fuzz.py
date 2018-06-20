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
flags.DEFINE_string(
    'oclgrind', 'oclgrind',
    'Path to the oclgrind binary. This does not need to be set if oclgring is '
    'in the system $PATH.')

CLGEN_CONFIG = """
# File: //deeplearning/deepsmith/proto/generator.proto
# Proto: deepsmith.ClgenGenerator
instance {
  working_dir: "/mnt/cc/data/experimental/deeplearning/polyglot/clgen"
  model {
    corpus {
      content_id: "ca98cc9e59c73b712d50f099d033ed3dcc6fa10a"
      ascii_character_atomizer: true
      preprocessor: "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:Compile"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes"
      preprocessor: "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype"
      preprocessor: "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:ClangFormat"
      preprocessor: "deeplearning.clgen.preprocessors.common:MinimumLineCount3"
      preprocessor: "deeplearning.clgen.preprocessors.opencl:Compile"
      contentfile_separator: "\n\n"
    }
    architecture {
      backend: TENSORFLOW
      neuron_type: LSTM
      neurons_per_layer: 256
      num_layers: 2
      post_layer_dropout_micros: 0
    }
    training {
      num_epochs: 50
      sequence_length: 64
      shuffle_corpus_contentfiles_between_epochs: true
      batch_size: 64
      adam_optimizer {
        initial_learning_rate_micros: 2000
        learning_rate_decay_per_epoch_micros: 50000
        beta_1_micros: 900000
        beta_2_micros: 999000
        normalized_gradient_clip_micros: 5000000
      }
    }
  }
  sampler {
    start_text: "kernel void "
    batch_size: 1
    temperature_micros: 750000  # real value = 0.75
    termination_criteria {
      symtok {
        depth_increase_token: "{"
        depth_decrease_token: "}"
      }
    }
    termination_criteria {
      maxlen {
        maximum_tokens_in_sample: 5000
      }
    }
  }
}
testcase_skeleton {
  toolchain: "opencl"
  harness {
    name: "cldrive"
  }
  inputs {
    key: "gsize"
    value: "1,1,1"
  }
  inputs {
    key: "lsize"
    value: "1,1,1"
  }
}
"""

DUT_HARNESS_CONFIG = """
# File: //deeplearning/deepsmith/proto/harness.proto
# Proto: deepsmith.CldriveHarness
opencl_env: "Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2"
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
  logging.debug('Stdout: %s', result.outputs['stdout'])
  logging.debug('Stderr: %s', result.outputs['stderr'])

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
  clgen_config = pbutil.FromString(
      CLGEN_CONFIG, generator_pb2.ClgenGenerator())
  generator = clgen.ClgenGenerator(clgen_config)

  logging.info('Preparing device under test.')
  config = pbutil.FromString(DUT_HARNESS_CONFIG, harness_pb2.CldriveHarness())
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
