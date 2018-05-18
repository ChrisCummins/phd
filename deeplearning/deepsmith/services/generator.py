from absl import flags

from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.services import services

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'generator_config', None,
  'Path to a ClgenGenerator message.')


class GeneratorBase(services.ServiceBase):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    self.config = config

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del request
    del context
    raise NotImplementedError('abstract class')
