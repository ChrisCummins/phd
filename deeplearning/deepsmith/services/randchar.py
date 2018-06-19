"""A very basic "generator" which produces strings of random characters."""
import random
import string
import time
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.services import generator
from deeplearning.deepsmith.services import services
from lib.labm8 import labdate
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


class RandCharGenerator(generator.GeneratorBase,
                        generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.RandCharGenerator):
    super(RandCharGenerator, self).__init__(config)
    self.generator = deepsmith_pb2.Generator()
    self.generator.name = 'randchar'
    self.generator.opts['toolchain'] = str(pbutil.AssertFieldConstraint(
        self.config, 'toolchain', lambda x: len(x)))
    self.generator.opts['min_len'] = str(pbutil.AssertFieldConstraint(
        self.config, 'string_min_len', lambda x: x > 0))
    self.generator.opts['max_len'] = str(pbutil.AssertFieldConstraint(
        self.config, 'string_max_len',
        lambda x: x > 0 and x >= self.config.string_min_len))
    logging.info('RandChar generator:\n%s', self.generator)

  def GetGeneratorCapabilities(
      self, request: generator_pb2.GetGeneratorCapabilitiesRequest,
      context) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    """Get the generator capabilities."""
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GetGeneratorCapabilitiesRequest)
    response.toolchain = self.config.model.corpus.language
    response.generator = self.generator
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    """Generate testcases."""
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)

    # Generate random strings.
    for _ in range(request.num_testcases):
      # Pick a length for the random string.
      n = random.randint(self.config.string_min_len,
                         self.config.string_max_len + 1)
      # Instantiate a testcase.
      testcase = response.testcases.add()
      testcase.toolchain = self.config.toolchain
      testcase.generator.CopyFrom(self.generator)
      start_time = labdate.MillisecondsTimestamp()
      testcase.inputs['src'] = ''.join(
          random.choice(string.ascii_lowercase) for _ in range(n))
      end_time = labdate.MillisecondsTimestamp()
      p = testcase.profiling_events.add()
      p.type = 'generation'
      p.event_start_epoch_ms = start_time
      p.duration_ms = end_time - start_time

    return response


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
      'generator_config', generator_pb2.RandCharGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config.service)
  service = RandCharGenerator(generator_config)
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
