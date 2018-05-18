import time
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from deeplearning.deepsmith.services import harness
from deeplearning.deepsmith.services import services

FLAGS = flags.FLAGS


class CldriveHarness(harness.HarnessBase,
                     harness_pb2_grpc.HarnessServiceServicer):

  def GetHarnessCapabilities(self, request: harness_pb2.GetHarnessCapabilitiesRequest,
                             context) -> harness_pb2.GetHarnessCapabilitiesResponse:
    pass

  def RunTestcases(self, request: harness_pb2.RunTestcasesRequest,
                   context) -> harness_pb2.RunTestcasesResponse:
    pass


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  harness_config = services.ServiceConfigFromFlag(
    'harness_config', harness_pb2.CldriveHarness())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(harness_config.service)
  service = CldriveHarness(harness_config)
  harness_pb2_grpc.add_HarnessServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{harness_config.service.port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               harness_config.service.hostname,
               harness_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
