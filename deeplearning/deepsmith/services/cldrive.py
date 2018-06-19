import time
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
from gpu.cldrive import cldrive
from gpu.cldrive import env


FLAGS = flags.FLAGS


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
      self.envs = []
      for opencl_env in sorted(set(self.config.opencl_env)):
        if opencl_env in all_envs:
          self.envs.append(all_envs[opencl_env])
        else:
          raise LookupError(
              f"Requested OpenCL environment not available: '{opencl_env}'")
    else:
      self.envs = all_envs.values()
    if not self.envs:
      raise EnvironmentError('No OpenCL environments available')

    self.testbeds = [OpenClEnvironmentToTestbed(env) for env in self.envs]

    # Logging output.
    for testbed in self.testbeds:
      logging.info('OpenCL testbed:\n%s', testbed)

  def GetHarnessCapabilities(self,
                             request: harness_pb2.GetHarnessCapabilitiesRequest,
                             context) -> harness_pb2.GetHarnessCapabilitiesResponse:
    """Get the harness capabilities."""
    del context
    logging.info('GetHarnessCapabilities() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
        harness_pb2.GetHarnessCapabilitiesRequest)
    response.harness.name = 'cldrive'
    response.testbed.extend(self.testbeds)
    return response

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    del context
    logging.info('RunTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(harness_pb2.RunTestcasesResponse)
    if request.testbed not in self.testbeds:
      response.status.returncode = service_pb2.ServiceStatus.INVALID_REQUEST_PARAMETERS
      response.status.error_message = 'Requested testbed not found.'
      return response
    
    # TODO(cec): Implement!
    return response


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
