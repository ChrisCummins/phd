from absl import flags
from phd.lib.labm8 import pbutil

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import generator_pb2


FLAGS = flags.FLAGS


class GeneratorBase(services.ServiceBase):

  def __init__(self, config: pbutil.ProtocolBuffer):
    super(GeneratorBase, self).__init__(config)

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
