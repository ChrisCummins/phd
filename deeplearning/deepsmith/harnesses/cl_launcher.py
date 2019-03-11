"""The cl_launcher harness for CLSmith."""
import copy
import socket
import time
import typing
from concurrent import futures

import grpc

from compilers.clsmith import cl_launcher
from deeplearning.deepsmith import services
from deeplearning.deepsmith.harnesses import harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from deeplearning.deepsmith.proto import service_pb2
from gpu.cldrive.legacy import driver
from gpu.cldrive.legacy import env
from labm8 import app
from labm8 import labdate

FLAGS = app.FLAGS


class ClLauncherHarness(harness.HarnessBase,
                        harness_pb2_grpc.HarnessServiceServicer):
  """A harness for running CLSmith-generated programs."""

  def __init__(self, config: harness_pb2.ClLauncherHarness):
    """Instantiate a ClLauncherHarness harness service.

    Args:
      config: A config proto.

    Raises:
      LookupError: If a requested 'opencl_env' is not available.
      EnvironmentError: If no 'opencl_env' were requested, and none are
        available on the host.
    """
    super(ClLauncherHarness, self).__init__(config)

    if len(self.config.opencl_env) != len(self.config.opencl_opt):
      raise ValueError(
          'ClLauncherHarness.opencl_env and ClLauncherHarness.opencl_opt lists '
          'are not the same length:\n'
          f'    ClLauncherHarness.opencl_env = {config.opencl_env}\n'
          f'    ClLauncherHarness.opencl_opt = {config.opencl_opt}')

    # Match and instantiate the OpenCL environments.
    all_envs = {env_.name: env_ for env_ in env.GetOpenClEnvironments()}
    envs = []
    if self.config.opencl_env:
      for opencl_env, opt in zip(self.config.opencl_env,
                                 self.config.opencl_opt):
        if opencl_env in all_envs:
          env_ = copy.deepcopy(all_envs[opencl_env])
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
      app.Log(1, 'OpenCL testbed:\n%s', testbed)

  def GetHarnessCapabilities(
      self, request: harness_pb2.GetHarnessCapabilitiesRequest,
      context) -> harness_pb2.GetHarnessCapabilitiesResponse:
    """Get the harness capabilities."""
    del context
    response = services.BuildDefaultResponse(
        harness_pb2.GetHarnessCapabilitiesRequest)
    response.harness.name = 'cl_launcher'
    response.testbed.extend(self.testbeds)
    return response

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    del context
    response = services.BuildDefaultResponse(harness_pb2.RunTestcasesResponse)
    if request.testbed not in self.testbeds:
      response.status.returncode = (
          service_pb2.ServiceStatus.INVALID_REQUEST_PARAMETERS)
      response.status.error_message = 'Requested testbed not found.'
      return response

    testbed_idx = self.testbeds.index(request.testbed)
    for i, testcase in enumerate(request.testcases):
      result = RunTestcase(self.envs[testbed_idx], self.testbeds[testbed_idx],
                           testcase, list(self.config.opts))
      app.Log(1, 'Testcase %d: %s.', i + 1,
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
  testbed.opts['opencl_opt'] = ('enabled' if opencl_environment.opencl_opt else
                                'disabled')
  return testbed


def RunTestcase(opencl_environment: env.OpenCLEnvironment,
                testbed: deepsmith_pb2.Testbed,
                testcase: deepsmith_pb2.Testcase,
                opts: typing.List[str]) -> deepsmith_pb2.Result:
  """Run a testcase."""
  if testcase.toolchain != 'opencl':
    raise ValueError(f"Unsupported testcase toolchain: '{testcase.toolchain}'")
  if testcase.harness.name != 'cl_launcher':
    raise ValueError(f"Unsupported testcase harness: '{testcase.harness.name}'")
  result = deepsmith_pb2.Result()
  result.testbed.CopyFrom(testbed)
  result.testcase.CopyFrom(testcase)

  # Set up additional command line flags for cl_launcher. We always run with
  # debugging output enabled.
  opts.append('---debug')
  if testbed.opts['opencl_opt'] == 'disabled':
    opts.append('---disable_opts')

  start_time_epoch_ms = labdate.MillisecondsTimestamp()
  process = cl_launcher.ExecClsmithSource(
      opencl_environment,
      testcase.inputs['src'],
      driver.NDRange.FromString(testcase.inputs['gsize']),
      driver.NDRange.FromString(testcase.inputs['lsize']),
      *opts,
      timeout_seconds=testcase.harness.opts.get('timeout_seconds', '60'))

  wall_time = labdate.MillisecondsTimestamp() - start_time_epoch_ms
  result = deepsmith_pb2.Result()
  result.testcase.CopyFrom(testcase)
  result.testbed.CopyFrom(testbed)
  result.returncode = process.returncode
  result.outputs['stdout'] = process.stdout
  result.outputs['stderr'] = process.stderr
  prof = result.profiling_events.add()
  prof.client = socket.gethostname()
  prof.type = 'runtime'
  prof.duration_ms = wall_time
  prof.event_start_epoch_ms = start_time_epoch_ms
  result.outcome = GetResultOutcome(result)
  return result


def GetResultRuntimeMs(result: deepsmith_pb2.Result) -> int:
  for event in result.profiling_events:
    if str(event.type) == 'runtime':
      return event.duration_ms
  raise ValueError('Result contains no runtime profiling event!')


def GetResultOutcome(
    result: deepsmith_pb2.Result) -> deepsmith_pb2.Result.Outcome:
  """Determine the output class of a result.

  Args:
    result: The result to determine the output class of.

  Returns:
    The result outcome.

  Raises:
    ValueError: If the outcome class could not be determined.
  """

  def RuntimeCrashOrBuildCrash():
    if "Compilation terminated successfully..." in result.outputs['stderr']:
      return deepsmith_pb2.Result.RUNTIME_CRASH
    else:
      return deepsmith_pb2.Result.BUILD_CRASH

  def RuntimeTimoutOrBuildTimeout():
    if "Compilation terminated successfully..." in result.outputs['stderr']:
      return deepsmith_pb2.Result.RUNTIME_TIMEOUT
    else:
      return deepsmith_pb2.Result.BUILD_TIMEOUT

  runtime_ms = GetResultRuntimeMs(result)
  timeout_ms = int(result.testcase.harness.opts.get('timeout_seconds',
                                                    60)) * 1000

  if result.returncode == 0:
    return deepsmith_pb2.Result.PASS
  elif result.returncode == 139 or result.returncode == -11:
    # 139 is SIGSEV
    result.returncode = 139
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -5:
    # SIGTRAP
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -9 and runtime_ms >= timeout_ms:
    # SIGKILL
    return RuntimeTimoutOrBuildTimeout()
  elif result.returncode == -9:
    app.Warning('SIGKILL, but only ran for %d ms', runtime_ms)
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -4:
    # SIGILL
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -6:
    # SIGABRT
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -8:
    # SIGFPE
    return RuntimeCrashOrBuildCrash()
  elif result.returncode == -7:
    # SIGBUS
    return RuntimeCrashOrBuildCrash()
  elif (result.returncode == 1 and
        'Error building program:' in result.outputs['stderr']):
    return deepsmith_pb2.Result.BUILD_FAILURE
  elif result.returncode == 1:
    # cl_launcher error
    return deepsmith_pb2.Result.UNKNOWN
  raise ValueError(f'Failed to output class of result: {result}')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  config = services.ServiceConfigFromFlag('harness_config',
                                          harness_pb2.ClLauncherHarness())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(config.service)
  service = ClLauncherHarness(config)
  harness_pb2_grpc.add_HarnessServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{config.service.port}')
  app.Log(1, '%s listening on %s:%s',
          type(service).__name__, config.service.hostname, config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.RunWithArgs(main)
