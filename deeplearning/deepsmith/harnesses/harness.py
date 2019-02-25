from absl import flags

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import harness_pb2
from labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string('harness_config', None, 'Path to a harness config proto.')


class HarnessBase(services.ServiceBase):

  def __init__(self, config: pbutil.ProtocolBuffer):
    self.config = config
    self.testbeds = []

  def GetHarnessCapabilities(
      self, request: harness_pb2.GetHarnessCapabilitiesRequest,
      context) -> harness_pb2.GetHarnessCapabilitiesResponse:
    raise NotImplementedError('abstract class')

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    raise NotImplementedError('abstract class')
