import subprocess
import tempfile
import time
import typing
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from deeplearning.deepsmith.proto import service_pb2
from deeplearning.deepsmith.services import harness
from deeplearning.deepsmith.services import services
from gpu import cldrive as cldrive_lib
from gpu.cldrive import cldrive
from gpu.cldrive import env
from lib.labm8 import bazelutil
from lib.labm8 import fs
from lib.labm8 import labdate
from lib.labm8 import system


FLAGS = flags.FLAGS

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'
# Path to clang binary.
CLANG_PATH = bazelutil.DataPath(f'{_LLVM_REPO}/bin/clang')
# Path to OpenCL headers.
OPENCL_HEADERS_INCLUDE = bazelutil.DataPath('opencl_headers')


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

    # Match and instantiate the OpenCL environments.
    all_envs = {env.name: env for env in cldrive.GetOpenClEnvironments()}
    if self.config.opencl_env:
      envs = []
      for opencl_env in sorted(set(self.config.opencl_env)):
        if opencl_env in all_envs:
          envs.append(all_envs[opencl_env])
        else:
          available = '\n'.join(f'    {n}' for n in sorted(all_envs.keys()))
          raise LookupError(
              f"Requested OpenCL environment not available: '{opencl_env}'.\n"
              f"Available OpenCL devices:\n{available}")
    else:
      envs = all_envs.values()
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
          self.envs[testbed_idx], self.testbeds[testbed_idx], testcase)
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
  return testbed


def RunTestcase(opencl_environment: env.OpenCLEnvironment,
                testbed: deepsmith_pb2.Testbed,
                testcase: deepsmith_pb2.Testcase) -> deepsmith_pb2.Result:
  """Run a testcase."""
  assert testcase.toolchain == 'opencl'
  result = deepsmith_pb2.Result()
  result.testcase.CopyFrom(testcase)
  result.testbed.CopyFrom(testbed)
  platform_id, device_id = opencl_environment.ids()
  driver = MakeDriver(testcase)
  # Get a temporary file to write and run the driver from.
  with tempfile.NamedTemporaryFile(prefix='deepsmith_', delete=False) as f:
    path = f.name
  try:
    CompileDriver(driver, path, platform_id, device_id)
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
    result.outcome = GetResultOutputClass(result)
  finally:
    fs.rm(path)
  return result


def MakeDriver(testcase: deepsmith_pb2.Testcase) -> str:
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
  gsize = cldrive_lib.NDRange(
      *[int(x) for x in testcase.inputs['gsize'].split(',')])
  lsize = cldrive_lib.NDRange(
      *[int(x) for x in testcase.inputs['lsize'].split(',')])
  size = max(gsize.product * 2, 256)
  src = testcase.inputs['src']
  try:
    # Generate a compile-and-execute test harness.
    inputs = cldrive_lib.make_data(
        src=src, size=size,
        data_generator=cldrive_lib.Generator.ARANGE,
        scalar_val=size)
    src = cldrive_lib.emit_c(
        src=src, inputs=inputs, gsize=gsize, lsize=lsize)
  except Exception:
    # Create a compile-only stub if not possible.
    try:
      src = cldrive_lib.emit_c(
          src=src, inputs=None, gsize=None, lsize=None,
          compile_only=True)
    except Exception:
      # Create a compiler-only stub without creating kernel.
      src = cldrive_lib.emit_c(
          src=src, inputs=None, gsize=None, lsize=None,
          compile_only=True, create_kernel=False)
  return src


def CompileDriver(src: str, path: str, platform_id: int,
                  device_id: int, timeout: int = 60) -> None:
  """Compile driver binary from source."""
  cmd = ['timeout', '-s9', str(timeout), str(CLANG_PATH), '-xc', '-', '-o',
         str(path), f'-DPLATFORM_ID={platform_id}', f'-DDEVICE_ID={device_id}',
         '-std=c99', '-Wno-deprecated-declarations',
         f'-I{OPENCL_HEADERS_INCLUDE}']
  if system.is_linux():
    cmd.append('-lOpenCL')
  elif system.is_mac():
    cmd += ['-framework', 'OpenCL']

  # logging.debug('$ %s', ' '.join(cmd))
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
  stdout, stderr = proc.communicate(src)
  if not proc.returncode == 0:
    argv = ' '.join(cmd)
    raise EnvironmentError(
        f'Driver compilation failed with returncode {proc.returncode}.\n'
        f'Command: {argv}\n'
        f'Stdout:\n{stdout}\n'
        f'Stderr:\n{stderr}\n'
        f'Driver source:\n{src}')
  return path


def GetResultRuntimeMs(result: deepsmith_pb2.Result) -> int:
  for event in result.profiling_events:
    if str(event.type) == 'runtime':
      return event.duration_ms
  return 0


def GetResultOutputClass(
    result: deepsmith_pb2.Result) -> deepsmith_pb2.Result.Outcome:
  """Determine the output class of a testcase."""

  def RuntimeCrashOrBuildFailure():
    if "[cldrive] Kernel: " in result.outputs['stderr']:
      return deepsmith_pb2.Result.BUILD_CRASH
    else:
      return deepsmith_pb2.Result.BUILD_FAILURE

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
  # SIGSEV.
  elif result.returncode == 139 or result.returncode == -11:
    return RuntimeCrashOrBuildCrash()
  # SIGTRAP.
  elif result.returncode == -5:
    return RuntimeCrashOrBuildCrash()
  # SIGKILL.
  elif result.returncode == -9 and runtime_ms >= timeout_ms:
    return RuntimeTimeoutOrBuildTimeout()
  elif result.returncode == -9:
    return RuntimeCrashOrBuildCrash()
  # SIGILL.
  elif result.returncode == -4:
    return RuntimeCrashOrBuildCrash()
  # SIGFPE.
  elif result.returncode == -8:
    return RuntimeCrashOrBuildCrash()
  # SIGBUS.
  elif result.returncode == -7:
    return RuntimeCrashOrBuildCrash()
  # SIGABRT.
  elif result.returncode == -6:
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == 1 and runtime_ms >= timeout_ms:
    return RuntimeTimeoutOrBuildTimeout()
  elif result.returncode == 1:
    return RuntimeCrashOrBuildFailure()
  elif result.returncode == 127:
    return RuntimeCrashOrBuildFailure()
  raise LookupError('Failed to output class of result.')


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
