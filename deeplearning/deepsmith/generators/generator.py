# Copyright (c) 2017-2020 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines the base class for generator implementations."""
import time
import typing
from concurrent import futures

import grpc

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


class GeneratorServiceBase(services.ServiceBase):
  """Base class for generator implementations."""

  def __init__(self, config: pbutil.ProtocolBuffer):
    super(GeneratorServiceBase, self).__init__(config)
    # Inheriting subclasses are required to set these parameters.
    self.toolchain: str = None
    self.generator: deepsmith_pb2.Generator = None

  def GetGeneratorCapabilities(
    self, request: generator_pb2.GetGeneratorCapabilitiesRequest, context
  ) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    """Get the capabilities of the generator service.

    Args:
      request: A request proto.
      context: RPC context. Unused.

    Returns:
      A response proto.
    """
    del context
    response = services.BuildDefaultResponse(
      generator_pb2.GetGeneratorCapabilitiesRequest
    )
    response.toolchain = self.toolchain
    response.generator = self.generator
    return response

  def GenerateTestcases(
    self, request: generator_pb2.GenerateTestcasesRequest, context
  ) -> generator_pb2.GenerateTestcasesResponse:
    """Generate testcases using the generator.

    Args:
      request: A request proto.
      context: RPC context. Unused.

    Returns:
      A response proto.
    """
    del request
    del context
    raise NotImplementedError("abstract class")

  @classmethod
  def Main(
    cls, config_proto_class: pbutil.ProtocolBuffer
  ) -> typing.Callable[[typing.List[str]], None]:
    """Return a main method for running this service.

    Args:
      config_proto_class: The generator configuration class.

    Returns:
      A callable main method.
    """

    def RunMain(argv: typing.List[str]) -> None:
      if len(argv) > 1:
        raise app.UsageError("Unrecognized arguments")
      generator_config = services.ServiceConfigFromFlag(
        "generator_config", config_proto_class()
      )
      server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
      services.AssertLocalServiceHostname(generator_config.service)
      service = cls(generator_config)
      generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
      server.add_insecure_port(f"[::]:{generator_config.service.port}")
      app.Log(
        1,
        "%s listening on %s:%s",
        type(service).__name__,
        generator_config.service.hostname,
        generator_config.service.port,
      )
      server.start()
      try:
        while True:
          time.sleep(3600 * 24)
      except KeyboardInterrupt:
        server.stop(0)

    return RunMain
