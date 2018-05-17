"""Run testcases on a local OpenCL device.

This script runs DeepSmith testcases on an OpenCL testbed, and records the
results.

Usage:

    $ python experiments.py --testbed_id <device> --testcase_dirs <dirs> \
        --output_dir <dir>
"""
import collections
import pathlib
import subprocess
import tempfile
import typing

import humanize
from absl import app
from absl import flags

from deeplearning.deepsmith.proto import deepsmith_pb2
from gpu import cldrive
from lib.labm8 import fmt
from lib.labm8 import fs
from lib.labm8 import labdate
from lib.labm8 import labtypes
from lib.labm8 import pbutil
from lib.labm8 import system
from lib.labm8 import text

FLAGS = flags.FLAGS

flags.DEFINE_string('testbed_id', '1+',
                    'The OpenCL device to run the experiments on.')
flags.DEFINE_list('testcase_dirs',
                  ['./02_evaluate_harness/data/testcases'],
                  'Directories to read testcases from.')
flags.DEFINE_string('output_dir',
                    './02_evaluate_harness/output/results',
                    'Directory to write results to.')


def CldriveEnvToTestbed(env: cldrive.OpenCLEnvironment,
                        optimizations: bool) -> deepsmith_pb2.Testbed:
  """Build a DeepSmith testbed from a cldrive OpenCL environment."""
  testbed = deepsmith_pb2.Testbed()

  def _Escape(x): return '.'.join(x.lower().split())

  testbed.name = '_'.join([
    _Escape(env.platform_name), _Escape(env.device_type),
    _Escape(env.device_name), _Escape(env.driver_version)
  ])
  testbed.toolchain = 'opencl'
  testbed.opts['driver_version'] = env.driver_version
  testbed.opts['host'] = system.HOSTNAME
  testbed.opts['testbed_id'] = env.device_name
  testbed.opts['opencl_devtype'] = env.device_type
  testbed.opts['opencl_opt'] = 'enabled' if optimizations else 'disabled'
  testbed.opts['opencl_platform'] = env.platform_name
  testbed.opts['opencl_version'] = env.opencl_version
  return testbed


TestbedAndEnv = collections.namedtuple('TestbedAndEnv', ['env', 'testbed'])


def GetTestbedAndEnvironment(arg: str) -> TestbedAndEnv:
  """Lookup and return the requested testbed."""
  try:
    _num, _opt = arg[0], arg[1]
    num = int(_num)
    if _opt == '+':
      optimizations = True
    elif _opt == '-':
      optimizations = False
    else:
      raise Exception
    env = list(cldrive.all_envs())[num - 1]
    return TestbedAndEnv(env, CldriveEnvToTestbed(env, optimizations))
  except:
    raise app.UsageError(f"Unknown OpenCL device '{FLAGS.testbed_id}''")


def MakeDriver(testcase: deepsmith_pb2.Testcase) -> str:
  """Generate a self-contained C program for the given test case."""
  try:
    # Generate a compile-and-execute test harness.
    gsize = cldrive.NDRange(
      *[int(x) for x in testcase.inputs['gsize'].split(',')])
    lsize = cldrive.NDRange(
      *[int(x) for x in testcase.inputs['lsize'].split(',')])
    size = max(gsize.product * 2, 256)
    inputs = cldrive.make_data(
      src=testcase.inputs['src'], size=size,
      data_generator=cldrive.Generator.ARANGE,
      scalar_val=size)
    src = cldrive.emit_c(
      src=testcase.inputs['src'], inputs=inputs, gsize=gsize, lsize=lsize)
  except Exception:
    # Create a compile-only stub if not possible.
    try:
      src = cldrive.emit_c(
        src=testcase.inputs['src'], inputs=None, gsize=None, lsize=None,
        compile_only=True)
    except Exception:
      # Create a compiler-only stub without creating kernel.
      src = cldrive.emit_c(
        src=testcase.inputs['src'], inputs=None, gsize=None, lsize=None,
        compile_only=True, create_kernel=False)
  return src


def CompileDriver(src: str, path: str, platform_id,
                  device_id, cc: str = 'gcc',
                  cflags: typing.Optional[typing.List[str]] = None,
                  timeout: int = 60) -> None:
  """Compile driver binary from source."""
  # Assign default compilation flags.
  cflags = cflags or ["-std=c99", "-Wno-deprecated-declarations"]
  if system.is_linux():
    cflags.append('-lOpenCL')
  elif system.is_mac():
    cflags += ['-framework', 'OpenCL']

  cmd = [
          'timeout', '-s9', str(timeout),
          cc, '-xc', '-',
          '-o', str(path),
          f'-DPLATFORM_ID={platform_id}',
          f'-DDEVICE_ID={device_id}',
        ] + cflags
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  proc.communicate(src.encode('utf-8'))
  if not proc.returncode == 0:
    raise EnvironmentError(
      f'Driver compilation failed with returncode {proc.returncode}.')
  return path


def RunTestcase(opencl: TestbedAndEnv,
                testcase: deepsmith_pb2.Testcase) -> deepsmith_pb2.Result:
  """Run a testcase."""
  assert testcase.toolchain == 'opencl'
  result = deepsmith_pb2.Result()
  result.testcase.CopyFrom(testcase)
  result.testbed.CopyFrom(opencl.testbed)
  platform_id, device_id = opencl.env.ids()
  driver = MakeDriver(testcase)
  with tempfile.NamedTemporaryFile(prefix='dsmith-cldrive-', delete=False) as f:
    path = f.name
  CompileDriver(driver, path, platform_id, device_id)
  try:
    cmd = ['timeout', '-s9', testcase.harness.opts['timeout_seconds'], f.name]
    start_time = labdate.GetUtcMillisecondsNow()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    stdout, stderr = proc.communicate()
    end_time = labdate.GetUtcMillisecondsNow()
    # Build result message.
    result.returncode = proc.returncode
    result.outputs['stdout'] = stdout
    result.outputs['stderr'] = stderr
    runtime = result.profiling_events.add()
    runtime.client = system.HOSTNAME
    runtime.type = 'runtime'
    runtime.duration_ms = int(round(
      (end_time - start_time).total_seconds() * 1000))
    runtime.event_start_epoch_ms = labdate.MillisecondsTimestamp(start_time)
  finally:
    fs.rm(path)
  return result


def main(argv):
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f"Unknown arguments {unknown_args}")
  opencl = GetTestbedAndEnvironment(FLAGS.testbed_id)
  testcase_dirs = [pathlib.Path(x) for x in FLAGS.testcase_dirs]
  if not all(testcase_dirs):
    raise app.UsageError('--testcase_dirs must be directories.')
  testcase_paths = labtypes.flatten(
    [[pathlib.Path(y) for y in fs.ls(x, abspaths=True)]
     for x in testcase_dirs])
  print('Running', len(testcase_paths), 'testcases on OpenCL device:')
  print(fmt.Indent(2, opencl.testbed))
  results_dir = pathlib.Path(FLAGS.output_dir) / FLAGS.testbed_id
  results_dir.mkdir(parents=True, exist_ok=True)
  for path in testcase_paths:
    print('Running testcase', path, '...')
    testcase = pbutil.FromFile(path, deepsmith_pb2.Testcase())
    result = RunTestcase(opencl, testcase)
    runtime = humanize.intcomma(result.profiling_events[0].duration_ms)
    print(f'Testcase completed in {runtime} ms '
          f'with returncode {result.returncode}.')
    if result.returncode:
      print(fmt.Indent(
        2, text.truncate(result.outputs['stderr'].rstrip(), 600)))
    pbutil.ToFile(result, results_dir / fs.basename(path))
    print('Wrote result', (results_dir / fs.basename(path)).absolute())


if __name__ == '__main__':
  app.run(main)
