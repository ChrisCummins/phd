"""This file defines the base class for generator implementations."""
import grpc
import time
import typing
from absl import app
from absl import flags
from concurrent import futures
from phd.lib.labm8 import pbutil

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc


FLAGS = flags.FLAGS


class GeneratorBase(services.ServiceBase):
  """Base class for generator implementations."""

  def __init__(self, config: pbutil.ProtocolBuffer):
    super(GeneratorBase, self).__init__(config)
    self.generator = None

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
    del request
    del context
    raise NotImplementedError('abstract class')

  @classmethod
  def Main(cls, config_proto_class: pbutil.ProtocolBuffer
           ) -> typing.Callable[[typing.List[str]], None]:
    """

    Args:
      config_proto_class: The generator configuration class.

    Returns:
      A callable main method.
    """

    def RunMain(argv: typing.List[str]) -> None:
      if len(argv) > 1:
        raise app.UsageError('Unrecognized arguments')
      generator_config = services.ServiceConfigFromFlag(
          'generator_config', config_proto_class())
      server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
      services.AssertLocalServiceHostname(generator_config.service)
      service = cls(generator_config)
      generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
      server.add_insecure_port(f'[::]:{generator_config.service.port}')
      logging.info('%s listening on %s:%s', type(service).__name__,
                   generator_config.service.hostname,
                   generator_config.service.port)
      server.start()
      try:
        while True:
          time.sleep(3600 * 24)
      except KeyboardInterrupt:
        server.stop(0)

    return RunMain
