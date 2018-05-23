import time
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.services import generator
from deeplearning.deepsmith.services import services


FLAGS = flags.FLAGS


class ClgenGenerator(generator.GeneratorBase,
                     generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    self.config = config

  def GetGeneratorCapabilities(self,
                               request:
                               generator_pb2.GetGeneratorCapabilitiesRequest,
                               context) -> \
      generator_pb2.GetGeneratorCapabilitiesResponse:
    del context
    logging.info('GetGeneratorCapabilities() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
      generator_pb2.GetGeneratorCapabilitiesRequest)
    # TODO(cec): Implement!
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    logging.info('GenerateTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
      generator_pb2.GenerateTestcasesResponse)
    # TODO(cec): Implement!
    return response


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag('generator_config',
                                                    generator_pb2.ClgenGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config.service)
  service = ClgenGenerator(generator_config)
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
