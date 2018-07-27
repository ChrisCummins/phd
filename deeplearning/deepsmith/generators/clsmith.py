"""A CLSmith program generator."""
import grpc
import math
import os
import pathlib
import subprocess
import tempfile
import time
import typing
from absl import app
from absl import flags
from absl import logging
from concurrent import futures
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import labdate

from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.proto import service_pb2


FLAGS = flags.FLAGS


class CLSmithError(EnvironmentError):
  """Error thrown in case CLSmith fails."""
  pass


def ConfigToGenerator(
    config: generator_pb2.ClsmithGenerator) -> deepsmith_pb2.Generator:
  """Convert a config proto to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = 'clsmith'
  g.opts['opts'] = ' '.join(config.opt)
  return g


class ClsmithGenerator(generator.GeneratorBase,
                       generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClsmithGenerator, self).__init__(config)
    self.config = config
    self.generator = ConfigToGenerator(self.config)
    if not self.config.testcase_skeleton:
      raise ValueError('No testcase skeletons provided')
    for skeleton in self.config.testcase_skeleton:
      skeleton.generator.CopyFrom(self.generator)

  def GetGeneratorCapabilities(
      self, request: generator_pb2.GetGeneratorCapabilitiesRequest,
      context) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GetGeneratorCapabilitiesRequest)
    response.toolchain = 'opencl'
    response.generator = self.generator
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    num_programs = math.ceil(
        request.num_testcases / len(self.config.testcase_skeleton))
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)
    with tempfile.TemporaryDirectory(prefix='clsmith_') as d:
      os.chdir(d)
      try:
        for i in range(num_programs):
          response.testcases.extend(
              self.FileToTestcases(*self.GenerateOneFile()))
          logging.info('Generated file %d.', i + 1)
      except CLSmithError as e:
        response.status.returncode = service_pb2.ServiceStatus.ERROR
        response.status.error_message = str(e)
    return response

  def GenerateOneFile(self) -> typing.Tuple[pathlib.Path, int, int]:
    start_epoch_ms_utc = labdate.MillisecondsTimestamp()
    proc = subprocess.Popen(
        [bazelutil.DataPath('CLSmith/CLSmith')] + list(self.config.opt))
    proc.communicate()
    if proc.returncode:
      raise CLSmithError(f'CLSmith exited with returncode: {proc.returncode}')
    wall_time_ms = labdate.MillisecondsTimestamp() - start_epoch_ms_utc
    return (
      pathlib.Path(os.getcwd()) / 'CLProg.c', wall_time_ms, start_epoch_ms_utc)

  def FileToTestcases(self, path: pathlib.Path,
                      wall_time_ms: int,
                      start_epoch_ms_utc: int) -> typing.List[
    deepsmith_pb2.Testcase]:
    """Make testcases from a CLSmith generated file."""
    testcases = []
    for skeleton in self.config.testcase_skeleton:
      t = deepsmith_pb2.Testcase()
      t.CopyFrom(skeleton)
      p = t.profiling_events.add()
      p.type = 'generation'
      p.duration_ms = wall_time_ms
      p.event_start_epoch_ms = start_epoch_ms_utc
      with open(path) as f:
        t.inputs['src'] = f.read().strip()
      testcases.append(t)
    return testcases


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
      'generator_config', generator_pb2.ClsmithGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config.service)
  service = ClsmithGenerator(generator_config)
  generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{generator_config.service.port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               generator_config.service.hostname, generator_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
