"""A gold-standard fuzzing loop.

This program generates testcases and executes them against them on a device
under test. If the program produces an output, it is compared against a gold
standard Oclgrind output and used to determine if the result is interesting.

CLgen and CLSmith generators and harnesses are supported.
"""
import pathlib
import shutil
import sys
import time
import typing

from deeplearning.deepsmith.difftests import difftests
from deeplearning.deepsmith.difftests import opencl as opencl_filters
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.generators import clsmith
from deeplearning.deepsmith.generators import generator as base_generator
from deeplearning.deepsmith.harnesses import cl_launcher
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.harnesses import harness as base_harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import harness_pb2
from gpu.cldrive.legacy import env
from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import labdate
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_boolean('ls_env', False,
                   'List the available OpenCL devices and exit.')
app.DEFINE_string('generator', 'clgen',
                  'The type of generator to use. One of: {clgen,clsmith}.')
app.DEFINE_string(
    'generator_config', None,
    'The path of the generator config proto. If --generator=clgen, this must '
    'be a ClgenGenerator proto. If --generator=clsmith, this must be a '
    'ClSmithGenerator proto.')
app.DEFINE_string(
    'base_harness', None,
    'The path to an optional base harness config proto. If set, the harness '
    'configs are copied from this. Else, the default config is used. If '
    '--clgen_generator is set, this must be a cldrive harness proto. If '
    '--clsmith_generator is set, this must be a cl_launcher harness proto.')
app.DEFINE_string(
    'dut', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The name of the device under test, as described by cldrive. Run '
    '//gpu/cldrive --ls_env to see a list of available devices.')
app.DEFINE_boolean('opencl_opt', True,
                   'If --noopencl_opt, OpenCL optimizations are disabled.')
app.DEFINE_string(
    'interesting_results_dir',
    '/tmp/phd/experimental/deeplearning/deepsmith/opencl_fuzz/'
    'interesting_results', 'Directory to write interesting results to.')
app.DEFINE_boolean(
    'all_results_are_interesting', False,
    'If set, all results are written to interesting_results_dir.')
app.DEFINE_integer(
    'min_interesting_results', 1,
    'The minimum number of interesting testcases to discover before stopping.')
app.DEFINE_integer(
    'max_testing_time_seconds', 60,
    'The maximum number of time to run in seconds. The actual runtime may be '
    'higher, as the in-progress batch must complete.')
app.DEFINE_integer(
    'batch_size', 128,
    'The number of test cases to generate and execute in a single batch.')
app.DEFINE_string(
    'rerun_result', None,
    'If --rerun_result points to the path of a Result proto, the result '
    'testcase is executed under the specified --dut, and the new Result is '
    'printed to stdout.')
app.DEFINE_string(
    'unpack_result', None,
    'If --unpack_result points to the path of a Result proto, the result '
    'proto is unpacked into the testcase OpenCL kernel, and C source code. '
    'Files are written to the directory containing the result.')

# The path of the CLSmith cl_launcher C program source.
CL_LAUNCHER_SRC = bazelutil.DataPath('CLSmith/src/CLSmith/cl_launcher.c')


def RunBatch(generator: base_generator.GeneratorServiceBase,
             dut_harness: base_harness.HarnessBase,
             gs_harness: base_harness.HarnessBase,
             filters: difftests.FiltersBase,
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
    filters: A testcase filters instance.
    batch_size: The number of testcases to generate and evaluate.

  Returns:
    A list of results which are determined to be interesting.
  """
  # Our differential testers and result filters.
  unary_difftester = difftests.UnaryTester()
  gs_difftester = difftests.GoldStandardDiffTester(
      difftests.NamedOutputIsEqual('stdout'))

  interesting_results = []

  # Generate testcases.
  app.Log(1, 'Generating %d testcases ...', batch_size)
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = batch_size
  res = generator.GenerateTestcases(req, None)
  testcases = [
      testcase for testcase in res.testcases if filters.PreExec(testcase)
  ]
  if len(res.testcases) - len(testcases):
    app.Log(1, 'Discarded %d testcases prior to execution.',
            len(res.testcases) - len(testcases))

  # Evaluate testcases.
  app.Log(1, 'Evaluating %d testcases on %s ...', len(testcases),
          dut_harness.testbeds[0].opts['platform'][:12])
  unfiltered_results = RunTestcases(dut_harness, testcases)
  results = [
      result for result in unfiltered_results if filters.PostExec(result)
  ]
  if len(unfiltered_results) - len(results):
    app.Log(1, 'Discarded %d results.', len(unfiltered_results) - len(results))

  for i, result in enumerate(results):
    interesting_result = ResultIsInteresting(result, unary_difftester,
                                             gs_difftester, gs_harness, filters)
    if interesting_result:
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


def ResultIsInteresting(
    result: deepsmith_pb2.Result, unary_difftester: difftests.UnaryTester,
    gs_difftester: difftests.GoldStandardDiffTester,
    gs_harness: base_harness.HarnessBase,
    filters: difftests.FiltersBase) -> typing.Optional[deepsmith_pb2.Result]:
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
  gs_result = RunTestcases(gs_harness, [result.testcase])[0]

  dt_outcomes = gs_difftester([gs_result, result])
  dt_outcome = dt_outcomes[1]
  app.Log(1, 'Differential test outcome: %s.',
          deepsmith_pb2.DifferentialTest.Outcome.Name(dt_outcome))

  # Determine whether we can use the difftest result.
  dt = filters.PostDifftest(
      deepsmith_pb2.DifferentialTest(
          result=[gs_result, result], outcome=dt_outcomes))
  if not dt:
    app.Log(1, 'Cannot use gold standard difftest result.')
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

  return NotInteresting(result)


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


def TestingLoop(min_interesting_results: int,
                max_testing_time_seconds: int,
                batch_size: int,
                generator: base_generator.GeneratorServiceBase,
                dut_harness: base_harness.HarnessBase,
                gs_harness: base_harness.HarnessBase,
                filters: difftests.FiltersBase,
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
    filters: A filters instance for testcases.
    interesting_results_dir: The directory to write interesting results to.
    start_time: The starting time, as returned by time.time(). If not provided,
      the starting time will be the moment that this function is called. Set
      this value if you would like to include initialization overhead in the
      calculated testing time.
  """
  start_time = start_time or time.time()
  interesting_results_dir.mkdir(parents=True, exist_ok=True)
  num_interesting_results = 0
  batch_num = 0
  while (num_interesting_results < min_interesting_results and
         time.time() < start_time + max_testing_time_seconds):
    batch_num += 1
    app.Log(1, 'Starting generate / test / eval batch %d ...', batch_num)
    interesting_results = RunBatch(generator, dut_harness, gs_harness, filters,
                                   batch_size)
    num_interesting_results += len(interesting_results)
    for result in interesting_results:
      pbutil.ToFile(
          result, interesting_results_dir /
          (str(labdate.MillisecondsTimestamp()) + '.pbtxt'))

  app.Log(
      1, 'Stopping after %.2f seconds and %s batches (%.0fms / testcase).\n'
      'Found %s interesting results.',
      time.time() - start_time, humanize.Commas(batch_num),
      (((time.time() - start_time) / (batch_num * batch_size)) * 1000),
      num_interesting_results)
  app.FlushLogs()


def GetBaseHarnessConfig(config_class):
  """Load the base Cldrive harness configuration.

  If --base_harness is set, the config is read from this path. This allows
  overriding options such as the driver_cflags field.

  Returns:
    A CldriveHarness proto instance.
  """
  if FLAGS.base_harness:
    config = pbutil.FromFile(pathlib.Path(FLAGS.base_harness), config_class())
    config.ClearField('opencl_env')
    config.ClearField('opencl_opt')
    return config
  else:
    return config_class()


def GeneratorFromFlag(config_class,
                      generator_class) -> base_generator.GeneratorServiceBase:
  """Instantiate a generator from the --generator_config flag."""
  if not pbutil.ProtoIsReadable(FLAGS.generator_config, config_class()):
    raise app.UsageError(
        f'--generator_config is not a {config_class.__name__} proto')
  config = pbutil.FromFile(pathlib.Path(FLAGS.generator_config), config_class())
  return generator_class(config)


def GetDeviceUnderTestHarness() -> base_harness.HarnessBase:
  """Instantiate the device under test harness.

  Uses the global FLAGS to determine the harness to instantiate.

  Returns:
    A Harness instance.
  """
  if FLAGS.rerun_result:
    result = pbutil.FromFile(
        pathlib.Path(FLAGS.rerun_result), deepsmith_pb2.Result())
    if result.testcase.harness.name == 'cldrive':
      harness_class = cldrive.CldriveHarness
      config_class = harness_pb2.CldriveHarness
    elif result.testcase.harness.name == 'cl_launcher':
      harness_class = cl_launcher.ClLauncherHarness
      config_class = harness_pb2.ClLauncherHarness
    else:
      raise app.UsageError(
          f"Unrecognized harness: '{result.testcase.harness.name}'")
  elif FLAGS.generator == 'clgen':
    harness_class = cldrive.CldriveHarness
    config_class = harness_pb2.CldriveHarness
  elif FLAGS.generator == 'clsmith':
    harness_class = cl_launcher.ClLauncherHarness
    config_class = harness_pb2.ClLauncherHarness
  else:
    raise app.UsageError(
        f"Unrecognized value for --generator: '{FLAGS.generator}'")
  app.Log(1, 'Preparing device under test.')
  config = GetBaseHarnessConfig(config_class)
  config.opencl_env.extend([FLAGS.dut])
  config.opencl_opt.extend([FLAGS.opencl_opt])
  dut_harness = harness_class(config)
  assert len(dut_harness.testbeds) == 1
  return dut_harness


def GetGoldStandardTestHarness() -> base_harness.HarnessBase:
  """Instantiate the gold standard test harness.

  Uses the global FLAGS to determine the harness to instantiate.

  Returns:
    A Harness instance.
  """
  if FLAGS.generator == 'clgen':
    harness_class = cldrive.CldriveHarness
    config_class = harness_pb2.CldriveHarness
  elif FLAGS.generator == 'clsmith':
    harness_class = cl_launcher.ClLauncherHarness
    config_class = harness_pb2.ClLauncherHarness
  else:
    raise app.UsageError(
        f"Unrecognized value for --generator: '{FLAGS.generator}'")
  app.Log(1, 'Preparing gold standard testbed.')
  config = GetBaseHarnessConfig(config_class)
  config.opencl_env.extend([env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  gs_harness = harness_class(config)
  assert len(gs_harness.testbeds) >= 1
  return gs_harness


def GetGenerator() -> base_generator.GeneratorServiceBase:
  app.Log(1, 'Preparing generator.')
  if FLAGS.generator == 'clgen':
    generator = GeneratorFromFlag(generator_pb2.ClgenGenerator,
                                  clgen_pretrained.ClgenGenerator)
  elif FLAGS.generator == 'clsmith':
    generator = GeneratorFromFlag(generator_pb2.ClsmithGenerator,
                                  clsmith.ClsmithGenerator)
  else:
    raise app.UsageError(
        f"Unrecognized value for --generator: '{FLAGS.generator}'")
  app.Log(1, '%s:\n %s', type(generator).__name__, generator.config)
  return generator


def GetFilters() -> difftests.FiltersBase:
  if FLAGS.generator == 'clgen':
    return opencl_filters.ClgenOpenClFilters()
  elif FLAGS.generator == 'clsmith':
    # TODO(cec): Replace with CLSmith filters.
    return difftests.FiltersBase()
  else:
    raise app.UsageError(
        f"Unrecognized value for --generator: '{FLAGS.generator}'")


def ReRunResult(result: deepsmith_pb2.Result,
                dut_harness: base_harness.HarnessBase) -> deepsmith_pb2.Result:
  """Re-run a result.

  Args:
    result: The result to re-run.
    dut_harness: The device harness to re-run the result using.
  """
  if dut_harness.testbeds[0] != result.testbed:
    app.Warning('Re-running result on a different testbed!')
  results = RunTestcases(dut_harness, [result.testcase])
  assert len(results) == 1
  new_result = results[0]
  app.Warning(f'Re-run result has same outcome: '
              f'{new_result.outcome == result.outcome}.')
  app.Warning(f'Re-run result has same returncode: '
              f'{new_result.returncode == result.returncode}.')
  app.Warning('Re-run result has same stdout: '
              f"{new_result.outputs['stdout'] == result.outputs['stdout']}.")
  app.Warning('Re-run result has same stderr: '
              f"{new_result.outputs['stderr'] == result.outputs['stderr']}.")
  return result


def ResultProtoFromFlag(flag: typing.Optional[str]) -> deepsmith_pb2.Result:
  """Read a result proto from a --flag path.

  Args:
    flag: The value of the flag which points to a result proto.

  Returns:
    The Result proto.

  Raises:
    UsageError: If the flag is not set or the flag does not point to a Result
      proto.
  """
  if not flag:
    raise app.UsageError('Path is not set.')
  path = pathlib.Path(flag)
  if not path.is_file():
    raise app.UsageError(f"File not found: '{path}'.")
  if not pbutil.ProtoIsReadable(path, deepsmith_pb2.Result()):
    raise app.UsageError(f"Cannot read Result proto: '{path}'.")
  return pbutil.FromFile(path, deepsmith_pb2.Result())


def WriteFile(path: pathlib.Path, text: str) -> None:
  """Write the given text to a file.

  Args:
    path: The path of the file to write.
    text: The file contents.
  """
  with open(path, 'w') as f:
    f.write(text)
  app.Log(1, 'Wrote %s', path)


def UnpackResult(result_path: typing.Optional[str]) -> None:
  """Unpack a result proto into its components.

  Args:
    result_path: The path of the result to unpack.

  Raises:
    UsageError: In case of error.
  """
  result_to_unpack = ResultProtoFromFlag(result_path)
  unpack_dir = pathlib.Path(result_path).parent
  WriteFile(unpack_dir / 'kernel.cl', result_to_unpack.testcase.inputs['src'])
  WriteFile(unpack_dir / 'stdout.txt', result_to_unpack.outputs['stdout'])
  WriteFile(unpack_dir / 'stderr.txt', result_to_unpack.outputs['stderr'])

  if result_to_unpack.testcase.harness.name == 'cldrive':
    WriteFile(unpack_dir / 'driver.c',
              cldrive.MakeDriver(result_to_unpack.testcase, FLAGS.opencl_opt))
  elif result_to_unpack.testcase.harness.name == 'cl_launcher':
    shutil.copyfile(CL_LAUNCHER_SRC, unpack_dir / 'driver.c')
  else:
    raise app.UsageError(
        f"Unrecognized harness: '{result.testcase.harness.name}'")


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  if FLAGS.ls_env:
    env.PrintOpenClEnvironments()
    return

  # Unpack a result.
  if FLAGS.unpack_result:
    UnpackResult(FLAGS.unpack_result)
    sys.exit(0)

  # Re-run a result.
  if FLAGS.rerun_result:
    result_to_rerun = ResultProtoFromFlag(FLAGS.rerun_result)
    dut_harness = GetDeviceUnderTestHarness()
    print(ReRunResult(result_to_rerun, dut_harness))
    sys.exit(0)

  # "Normal" fuzzing routine.
  start_time = time.time()
  if not FLAGS.interesting_results_dir:
    raise app.UsageError('--interesting_results_dir must be set')
  interesting_results_dir = pathlib.Path(FLAGS.interesting_results_dir)
  if interesting_results_dir.exists() and not interesting_results_dir.is_dir():
    raise app.UsageError('--interesting_results_dir must be a directory')
  app.Log(1, 'Recording interesting results in %s.', interesting_results_dir)

  generator = GetGenerator()
  filters = GetFilters()
  dut_harness = GetDeviceUnderTestHarness()
  gs_harness = GetGoldStandardTestHarness()
  TestingLoop(
      FLAGS.min_interesting_results,
      FLAGS.max_testing_time_seconds,
      FLAGS.batch_size,
      generator,
      dut_harness,
      gs_harness,
      filters,
      interesting_results_dir,
      start_time=start_time)


if __name__ == '__main__':
  try:
    app.RunWithArgs(main)
  except KeyboardInterrupt:
    app.FlushLogs()
    sys.stdout.flush()
    sys.stderr.flush()
    print('keyboard interrupt')
    sys.exit(1)
