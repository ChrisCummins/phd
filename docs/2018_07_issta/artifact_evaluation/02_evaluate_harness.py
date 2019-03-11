"""Run testcases on an OpenCL testbed.

This program runs DeepSmith testcases on an OpenCL testbed, and records the
results.
"""
import pathlib

from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from gpu.cldrive.legacy import env
from labm8 import app
from labm8 import bazelutil
from labm8 import crypto
from labm8 import fs
from labm8 import labtypes
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_list('input_directories', [
    str(
        bazelutil.DataPath(
            'phd/docs/2018_07_issta/artifact_evaluation/data/testcases')),
    '/tmp/phd/docs/2018_07_issta/artifact_evaluation/generated_testcases',
], 'Directories to read testcases from.')
app.DEFINE_string('output_directory',
                  '/tmp/phd/docs/2018_07_issta/artifact_evaluation/results',
                  'Directory to write results to.')
app.DEFINE_boolean('opencl_opt', True,
                   'If --noopencl_opt is set, disable OpenCL optimizations.')


def main(argv):
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f"Unknown arguments {unknown_args}")

  app.Log(1, 'Preparing OpenCL testbed.')
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([FLAGS.opencl_opt])
  harness = cldrive.CldriveHarness(config)
  assert len(harness.testbeds) >= 1

  input_directories = FLAGS.input_directories
  app.Log(1, 'Reading testcases from: %s', ' '.join(input_directories))

  output_directory = pathlib.Path(FLAGS.output_directory)
  app.Log(1, 'Writing results to %s', output_directory)
  output_directory.mkdir(parents=True, exist_ok=True)

  # Load testcases.
  testcase_dirs = [
      pathlib.Path(x) for x in input_directories if pathlib.Path(x).is_dir()
  ]
  if not testcase_dirs:
    raise app.UsageError('No --input_directories found.')
  testcase_paths = labtypes.flatten([
      [pathlib.Path(y) for y in fs.ls(x, abspaths=True)] for x in testcase_dirs
  ])
  testcases = [
      pbutil.FromFile(path, deepsmith_pb2.Testcase()) for path in testcase_paths
  ]
  app.Log(1, 'Read %d testcases.', len(testcases))
  if not len(testcases):
    raise app.UsageError("No testcases found: '%s'",
                         ' '.join(input_directories))

  # Execute testcases.
  req = harness_pb2.RunTestcasesRequest()
  req.testbed.CopyFrom(harness.testbeds[0])
  req.testcases.extend(testcases)
  res = harness.RunTestcases(req, None)

  # Write results to file.
  for testcase, result in zip(testcases, res.results):
    result_id = crypto.md5_str(str(testcase))
    pbutil.ToFile(result, output_directory / f'{result_id}.pbtxt')

  app.Log(1, 'Executed %d testcases and wrote results to %s', len(res.results),
          output_directory)
  execution_times = [
      result.profiling_events[0].duration_ms for result in res.results
  ]
  app.Log(1, 'Average time to evaluate testcase: %.2f ms',
          sum(execution_times) / len(execution_times))


if __name__ == '__main__':
  app.RunWithArgs(main)
