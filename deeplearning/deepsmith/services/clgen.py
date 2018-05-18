import time
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.services import services

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'generator_config', None,
  'Path to a ClgenGenerator message.')


class ClgenGenerator(generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    self.config = config

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    logging.info("GenerateTestcases() client=%s", request.client)
    response = generator_pb2.GenerateTestcasesResponse()
    response.status = generator_pb2.GenerateTestcasesResponse.SUCCESS
    return response


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
    'generator_config', generator_pb2.ClgenGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config)
  service = ClgenGenerator(generator_config)
  generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{generator_config.generator.service_port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               generator_config.generator.service_hostname,
               generator_config.generator.service_port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
