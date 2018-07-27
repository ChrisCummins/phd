import grpc
import time
from absl import app
from absl import flags
from absl import logging
from concurrent import futures

from deeplearning.clgen import clgen
from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc


FLAGS = flags.FLAGS


def ClgenInstanceToGenerator(
    instance: clgen.Instance) -> deepsmith_pb2.Generator:
  """Convert a CLgen instance to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = 'clgen'
  g.opts['model'] = instance.model.hash
  g.opts['sampler'] = instance.sampler.hash
  return g


class ClgenGenerator(clgen_pretrained.ClgenGenerator):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClgenGenerator, self).__init__(config, no_init=True)
    self.instance = clgen.Instance(self.config.instance)
    self.generator = ClgenInstanceToGenerator(self.instance)
    if not self.config.testcase_skeleton:
      raise ValueError('No testcase skeletons provided')
    for skeleton in self.config.testcase_skeleton:
      skeleton.generator.CopyFrom(self.generator)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
      'generator_config', generator_pb2.ClgenGenerator())
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
