from absl import flags

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import generator_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'generator_config', None,
    'Path to a generator config proto.')


class GeneratorBase(services.ServiceBase):

  def __init__(self, config: pbutil.ProtocolBuffer):
    self.config = config

  def GetGeneratorCapabilities(
      self, request: generator_pb2.GetGeneratorCapabilitiesRequest,
      context) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    del request
    del context
    raise NotImplementedError('abstract class')

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del request
    del context
    raise NotImplementedError('abstract class')
