"""This file defines TODO:

TODO: Detailed explanation of this file.
"""
import collections
import pathlib
import re
import subprocess
import tempfile
import typing

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

FLAGS = flags.FLAGS

flags.DEFINE_string('opencl_device', '1+',
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
  testbed.opts['opencl_device'] = env.device_name
  testbed.opts['opencl_devtype'] = env.device_type
  testbed.opts['opencl_opt'] = 'true' if optimizations else 'false'
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
    raise app.UsageError(f"Unknown OpenCL device '{FLAGS.opencl_device}''")


def __VerifyParams(platform: str, device: str, optimizations: bool,
                   global_size: typing.Tuple[int, int, int],
                   local_size: typing.Tuple[int, int, int], stderr: str) -> None:
  """ verify that expected params match actual as reported by cldrive """
  optimizations = "on" if optimizations else "off"

  actual_platform = None
  actual_device = None
  actual_optimizations = None
  actual_global_size = None
  actual_local_size = None
  for line in stderr.split('\n'):
    if line.startswith("[cldrive] Platform: "):
      actual_platform_name = re.sub(r"^\[cldrive\] Platform: ", "", line).rstrip()
    elif line.startswith("[cldrive] Device: "):
      actual_device_name = re.sub(r"^\[cldrive\] Device: ", "", line).rstrip()
    elif line.startswith("[cldrive] OpenCL optimizations: "):
      actual_optimizations = re.sub(r"^\[cldrive\] OpenCL optimizations: ", "", line).rstrip()

    # global size
    match = re.match('^\[cldrive\] 3-D global size \d+ = \[(\d+), (\d+), (\d+)\]', line)
    if match:
      actual_global_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # local size
    match = re.match('^\[cldrive\] 3-D local size \d+ = \[(\d+), (\d+), (\d+)\]', line)
    if match:
      actual_local_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # check if we've collected everything:
    if (actual_platform and actual_device and actual_optimizations and
        actual_global_size and actual_local_size):
      assert (actual_platform_name == platform)
      assert (actual_device_name == device)
      assert (actual_optimizations == optimizations)
      assert (actual_global_size == global_size)
      assert (actual_local_size == local_size)
      return


def MakeCDriver(testcase: deepsmith_pb2.Testcase) -> str:
  """ generate a self-contained C program for the given test case """
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


def CompileHarness(src: str, path: str = 'a.out', platform_id=None,
                   device_id=None, cc: str = 'gcc',
                   cflags: typing.Optional[typing.List[str]] = None,
                   timeout: int = 60) -> None:
  """ compile harness binary from source """
  # Assign default compilation flags.
  cflags = cflags or ["-std=c99", "-Wno-deprecated-declarations"]
  if system.is_linux():
    cflags.append('-lOpenCL')
  elif system.is_mac():
    cflags += ['-framework', 'OpenCL']

  cmd = ['timeout', '-s9', str(timeout), cc,
         '-xc', '-', '-o', str(path)] + cflags
  if platform_id is not None:
    cmd.append(f'-DPLATFORM_ID={platform_id}')
  if device_id is not None:
    cmd.append(f'-DDEVICE_ID={device_id}')

  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  proc.communicate(src.encode('utf-8'))
  if not proc.returncode == 0:
    raise EnvironmentError(
      f'harness compilation failed with returncode {proc.returncode}')
  return path


def RunTestcase(opencl: TestbedAndEnv,
                testcase: deepsmith_pb2.Testcase) -> deepsmith_pb2.Result:
  """Run a testcase."""
  result = deepsmith_pb2.Result()
  result.testcase.CopyFrom(testcase)
  result.testbed.CopyFrom(opencl.testbed)
  platform_id, device_id = opencl.env.ids()
  driver = MakeCDriver(testcase)

  with tempfile.NamedTemporaryFile(prefix='dsmith-cldrive-', delete=False) as f:
    path = f.name
  try:
    CompileHarness(driver, path, platform_id=platform_id, device_id=device_id)

    cmd = ['timeout', '-s9', str(testcase.inputs['timeout']), f.name]
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

  opencl = GetTestbedAndEnvironment(FLAGS.opencl_device)
  testcase_dirs = [pathlib.Path(x) for x in FLAGS.testcase_dirs]
  if not all(testcase_dirs):
    raise app.UsageError('--testcase_dirs must be directories.')
  testcase_paths = labtypes.flatten(
    [[pathlib.Path(y) for y in fs.ls(x, abspaths=True)]
     for x in testcase_dirs])

  print('Running', len(testcase_paths), 'testcases on OpenCL device:')
  print(fmt.Indent(2, opencl.testbed))

  results_dir = pathlib.Path(FLAGS.output_dir) / FLAGS.opencl_device
  results_dir.mkdir(parents=True, exist_ok=True)

  for path in testcase_paths:
    print('Running testcase', path, '...')
    testcase = pbutil.FromFile(path, deepsmith_pb2.Testcase())
    result = RunTestcase(opencl, testcase)
    pbutil.ToFile(result, results_dir / fs.basename(path))
    print('Wrote result', (results_dir / fs.basename(path)).absolute())


if __name__ == '__main__':
  app.run(main)
