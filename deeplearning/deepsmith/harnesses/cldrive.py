import copy
import pathlib
import subprocess
import tempfile
import time
import typing
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith import services
from deeplearning.deepsmith.harnesses import harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from deeplearning.deepsmith.proto import service_pb2
from gpu.cldrive import cgen
from gpu.cldrive import data
from gpu.cldrive import driver
from gpu.cldrive import env
from labm8 import bazelutil
from labm8 import fs
from labm8 import labdate
from labm8 import system


FLAGS = flags.FLAGS

_UNAME = 'linux' if system.is_linux() else 'mac'
# Path to clang binary.
CLANG_PATH = bazelutil.DataPath(f'llvm_{_UNAME}/bin/clang')
# Flags for compiling with libcxx.
LIBCXX_LIB_DIR = bazelutil.DataPath(f'llvm_{_UNAME}/lib')
# Path to OpenCL headers.
OPENCL_HEADERS_DIR = bazelutil.DataPath('opencl_120_headers')
if system.is_linux():
  LIBOPENCL_DIR = bazelutil.DataPath('libopencl')


class DriverCompilationError(OSError):
  """Exception raised in case driver compilation fails."""
  pass


class CldriveHarness(harness.HarnessBase,
                     harness_pb2_grpc.HarnessServiceServicer):
  """A harness for running OpenCL testcases using cldrive."""

  def __init__(self, config: harness_pb2.CldriveHarness):
    """Instantiate a CLdrive harness service.

    Args:
      config: A CldriveHarness proto.

    Raises:
      LookupError: If a requested 'opencl_env' is not available.
      EnvironmentError: If no 'opencl_env' were requested, and none are
        available on the host.
    """
    super(CldriveHarness, self).__init__(config)

    if len(self.config.opencl_env) != len(self.config.opencl_opt):
      raise ValueError(
          'CldriveHarness.opencl_env and CldriveHarness.opencl_opt lists are '
          'not the same length:\n'
          f'    CldriveHarness.opencl_env = {config.opencl_env}\n'
          f'    CldriveHarness.opencl_opt = {config.opencl_opt}')

    # Match and instantiate the OpenCL environments.
    all_envs = {env.name: env for env in env.GetOpenClEnvironments()}
    envs = []
    if self.config.opencl_env:
      for opencl_env, opt in zip(
          self.config.opencl_env, self.config.opencl_opt):
        if opencl_env in all_envs:
          env_ = copy.copy(all_envs[opencl_env])
          env_.opencl_opt = opt
          envs.append(env_)
        else:
          available = '\n'.join(f'    {n}' for n in sorted(all_envs.keys()))
          raise LookupError(
              f"Requested OpenCL environment not available: '{opencl_env}'.\n"
              f"Available OpenCL devices:\n{available}")
    else:
      # Use all available OpenCL environments.
      for env_ in all_envs.values():
        # Create environment with optimizations enabled.
        env_opt = copy.copy(env_)
        env_opt.opencl_opt = True
        envs.append(env_opt)
        # Create environment with optimizations disabled.
        env_noopt = copy.copy(env_)
        env_noopt.opencl_opt = False
        envs.append(env_noopt)
    if not envs:
      raise EnvironmentError('No OpenCL environments available')

    self.envs = envs
    self.testbeds = [OpenClEnvironmentToTestbed(e) for e in envs]
    self.ids = [e.ids() for e in envs]

    # Logging output.
    for testbed in self.testbeds:
      logging.info('OpenCL testbed:\n%s', testbed)

  def GetHarnessCapabilities(self,
                             request: harness_pb2.GetHarnessCapabilitiesRequest,
                             context) -> harness_pb2.GetHarnessCapabilitiesResponse:
    """Get the harness capabilities."""
    del context
    response = services.BuildDefaultResponse(
        harness_pb2.GetHarnessCapabilitiesRequest)
    response.harness.name = 'cldrive'
    response.testbed.extend(self.testbeds)
    return response

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    del context
    response = services.BuildDefaultResponse(harness_pb2.RunTestcasesResponse)
    if request.testbed not in self.testbeds:
      response.status.returncode = service_pb2.ServiceStatus.INVALID_REQUEST_PARAMETERS
      response.status.error_message = 'Requested testbed not found.'
      return response

    testbed_idx = self.testbeds.index(request.testbed)
    for i, testcase in enumerate(request.testcases):
      result = RunTestcase(
          self.envs[testbed_idx], self.testbeds[testbed_idx], testcase,
          self.config.driver_cflag)
      logging.info('Testcase %d: %s.', i + 1,
                   deepsmith_pb2.Result.Outcome.Name(result.outcome))
      response.results.extend([result])

    return response


def OpenClEnvironmentToTestbed(
    opencl_environment: env.OpenCLEnvironment) -> deepsmith_pb2.Testbed:
  """Instantiate a DeepSmith testbed from an OpenCL environment.

  Args:
    opencl_environment: A cldrive OpenCLEnvironment instance.

  Returns:
    A Testbed proto instance.
  """
  testbed = deepsmith_pb2.Testbed()
  testbed.toolchain = 'opencl'
  testbed.name = opencl_environment.name
  testbed.opts['platform'] = opencl_environment.platform_name
  testbed.opts['device'] = opencl_environment.device_name
  testbed.opts['driver'] = opencl_environment.driver_version
  testbed.opts['device_type'] = opencl_environment.device_type
  testbed.opts['opencl_version'] = opencl_environment.opencl_version
  testbed.opts['opencl_opt'] = ('enabled' if opencl_environment.opencl_opt
                                else 'disabled')
  return testbed


def RunTestcase(opencl_environment: env.OpenCLEnvironment,
                testbed: deepsmith_pb2.Testbed,
                testcase: deepsmith_pb2.Testcase,
                cflags: typing.List[str]) -> deepsmith_pb2.Result:
  """Run a testcase."""
  if testcase.toolchain != 'opencl':
    raise ValueError(f"Unsupported testcase toolchain: '{testcase.toolchain}'")
  if testcase.harness.name != 'cldrive':
    raise ValueError(f"Unsupported testcase harness: '{testcase.harness.name}'")
  result = deepsmith_pb2.Result()
  result.testbed.CopyFrom(testbed)
  platform_id, device_id = opencl_environment.ids()
  driver = MakeDriver(
      testcase, True if testbed.opts['opencl_opt'] == 'enabled' else False)
  # MakeDriver() annotates the testcase, so we must only set the testcase field
  # of the output result after we have called it.
  result.testcase.CopyFrom(testcase)
  # Get a temporary file to write and run the driver from.
  with tempfile.NamedTemporaryFile(prefix='deepsmith_', delete=False) as f:
    path = pathlib.Path(f.name)
  try:
    CompileDriver(driver, path, platform_id, device_id, cflags=cflags)
    timeout = testcase.harness.opts.get('timeout_seconds', '60')
    cmd = ['timeout', '-s9', timeout, f.name]
    start_time = labdate.GetUtcMillisecondsNow()
    proc = opencl_environment.Exec(cmd)
    end_time = labdate.GetUtcMillisecondsNow()
    # Build result message.
    result.returncode = proc.returncode
    result.outputs['stdout'] = proc.stdout
    result.outputs['stderr'] = proc.stderr
    runtime = result.profiling_events.add()
    runtime.client = system.HOSTNAME
    runtime.type = 'runtime'
    runtime.duration_ms = int(round(
        (end_time - start_time).total_seconds() * 1000))
    runtime.event_start_epoch_ms = labdate.MillisecondsTimestamp(start_time)
    result.outcome = GetResultOutcome(result)
  except DriverCompilationError as e:
    logging.warning('%s', e)
    result.outcome = deepsmith_pb2.Result.UNKNOWN
  finally:
    fs.rm(path)
  return result


def MakeDriver(testcase: deepsmith_pb2.Testcase,
               optimizations: bool) -> str:
  """Generate a self-contained C program for the given test case.

  Args:
    testcase: The testcase to generate a driver for. Requires three inputs:
      'src', 'gsize', and 'lsize'.

  Returns:
    A string of C code.

  Raises:
    ValueError: In case the testcase is missing the required gsize, lsize, and
      src inputs.
  """
  if 'gsize' not in testcase.inputs:
    raise ValueError("Field not set: 'Testcase.inputs[\"gsize\"]'")
  if 'lsize' not in testcase.inputs:
    raise ValueError("Field not set: 'Testcase.inputs[\"lsize\"]'")
  if 'src' not in testcase.inputs:
    raise ValueError("Field not set: 'Testcase.inputs[\"src\"]'")

  gsize = driver.NDRange(
      *[int(x) for x in testcase.inputs['gsize'].split(',')])
  lsize = driver.NDRange(
      *[int(x) for x in testcase.inputs['lsize'].split(',')])
  size = max(gsize.product * 2, 256)
  src = testcase.inputs['src']

  # Optionally support custom data generators.
  data_generator = data.Generator.ARANGE
  if 'data_generator' in testcase.inputs:
    data_generator = {
      'arange': data.Generator.ARANGE,
      'ones': data.Generator.ONES,
    }.get(testcase.inputs['data_generator'])
    if not data_generator:
      raise ValueError(
          "Unknown value for 'Testcase.inputs[\"data_generator\"]': "
          f"'{testcase.inputs['data_generator']}'")

  try:
    # Generate a compile-and-execute test harness.
    inputs = data.MakeData(
        src=src, size=size,
        data_generator=data_generator,
        scalar_val=size)
    src = cgen.emit_c(
        src=src, inputs=inputs, gsize=gsize, lsize=lsize,
        optimizations=optimizations)
    testcase.invariant_opts['driver_type'] = 'compile_and_run'
  except Exception:
    # Create a compile-only stub if not possible.
    try:
      src = cgen.emit_c(
          src=src, inputs=None, gsize=None, lsize=None,
          compile_only=True, optimizations=optimizations)
      testcase.invariant_opts['driver_type'] = 'compile_and_create_kernel'
    except Exception:
      # Create a compiler-only stub without creating kernel.
      src = cgen.emit_c(
          src=src, inputs=None, gsize=None, lsize=None,
          compile_only=True, create_kernel=False, optimizations=optimizations)
      testcase.invariant_opts['driver_type'] = 'compile_only'
  return src


def CompileDriver(src: str, output_path: pathlib.Path,
                  platform_id: int, device_id: int,
                  timeout_seconds: int = 60,
                  cflags: typing.List[str] = None) -> pathlib.Path:
  """Compile driver binary from source.

  Args:
    src: The C source code to compile.
    output_path: The path to the binary to generate.
    platform_id: The OpenCL platform ID.
    device_id: The OpenCL device ID.
    timeout_seconds: The number of seconds to allow for compilation.

  Returns:
    The path to the generated binary, same as output_path.

  Raises:
    DriverCompilationError: In case compilation fails.
  """
  cmd = [
    'timeout', '-s9', str(timeout_seconds),
    str(CLANG_PATH), '-xc', '-', '-o', str(output_path),
    f'-DPLATFORM_ID={platform_id}', f'-DDEVICE_ID={device_id}',
    '-ferror-limit=1', '-std=c99', '-Wno-deprecated-declarations',
    # Add OpenCL headers.
    '-isystem', str(OPENCL_HEADERS_DIR),
    # Link against libcxx.
    f'-L{LIBCXX_LIB_DIR}', f'-Wl,-rpath,{LIBCXX_LIB_DIR}',
    '-nodefaultlibs', '-stdlib=libc++', '-lc++', '-lc++abi', '-lm', '-lc',
  ]
  if system.is_linux():
    cmd += [
      # Additional libraries required to link against libcxxx.
      '-lgcc_s', '-lgcc', '-ldl', '-lpthread',
      # Link against libOpenCL.
      f'-L{LIBOPENCL_DIR}', f'-Wl,-rpath,{LIBOPENCL_DIR}', '-lOpenCL'
    ]
  elif system.is_mac():
    cmd += ['-framework', 'OpenCL']
  # Add any additional cflags.
  if cflags:
    cmd += cflags

  # logging.debug('$ %s', ' '.join(cmd))
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
  stdout, stderr = proc.communicate(src)
  if not proc.returncode == 0:
    argv = ' '.join(cmd)
    raise DriverCompilationError(
        f'Driver compilation failed with returncode {proc.returncode}.\n'
        f'Command: {argv}\n'
        f'Stdout:\n{stdout}\n'
        f'Stderr:\n{stderr}\n'
        f'Driver source:\n{src}')
  return output_path


def GetResultRuntimeMs(result: deepsmith_pb2.Result) -> int:
  for event in result.profiling_events:
    if str(event.type) == 'runtime':
      return event.duration_ms
  return 0


def GetResultOutcome(
    result: deepsmith_pb2.Result) -> deepsmith_pb2.Result.Outcome:
  """Determine the output class of a testcase.

  Args:
    result: The result to determine the output class of.

  Returns:
    The result outcome.

  Raises:
    ValueError: If the outcome class could not be determined.
  """

  def RuntimeCrashOrBuildFailure():
    if "[cldrive] Kernel: " in result.outputs['stderr']:
      return deepsmith_pb2.Result.BUILD_FAILURE
    else:
      return deepsmith_pb2.Result.BUILD_CRASH

  def RuntimeCrashOrBuildCrash():
    if "[cldrive] Kernel: " in result.outputs['stderr']:
      return deepsmith_pb2.Result.RUNTIME_CRASH
    else:
      return deepsmith_pb2.Result.BUILD_CRASH

  def RuntimeTimeoutOrBuildTimeout():
    if "[cldrive] Kernel: " in result.outputs['stderr']:
      return deepsmith_pb2.Result.RUNTIME_TIMEOUT
    else:
      return deepsmith_pb2.Result.BUILD_TIMEOUT

  runtime_ms = GetResultRuntimeMs(result)
  timeout_ms = int(
      result.testcase.harness.opts.get('timeout_seconds', 60)) * 1000

  if result.returncode == 0:
    return deepsmith_pb2.Result.PASS
  elif result.returncode == 139 or result.returncode == -11:
    # SIGSEV.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -5:
    # SIGTRAP.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -9 and runtime_ms >= timeout_ms:
    # SIGKILL.
    return RuntimeTimeoutOrBuildTimeout()
  elif result.returncode == -9:
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -4:
    # SIGILL.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -8:
    # SIGFPE.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -7:
    # SIGBUS.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -6:
    # SIGABRT.
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == 1 and runtime_ms >= timeout_ms:
    return RuntimeTimeoutOrBuildTimeout()
  elif result.returncode == 1:
    return RuntimeCrashOrBuildFailure()
  elif result.returncode == 127:
    return RuntimeCrashOrBuildFailure()
  elif (result.returncode == 3 and
        "Undefined external function" in result.outputs['stderr']):
    return deepsmith_pb2.Result.BUILD_FAILURE
  raise ValueError(f'Failed to output class of result: {str(result)[:1024]}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  harness_config = services.ServiceConfigFromFlag(
      'harness_config', harness_pb2.CldriveHarness())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(harness_config.service)
  service = CldriveHarness(harness_config)
  harness_pb2_grpc.add_HarnessServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{harness_config.service.port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               harness_config.service.hostname,
               harness_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
