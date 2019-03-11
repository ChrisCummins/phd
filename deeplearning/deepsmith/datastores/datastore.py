import time
from concurrent import futures

import grpc

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import datastore_pb2_grpc
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_string('datastore_config', None, 'Path to a DataStore message.')


class DataStore(services.ServiceBase,
                datastore_pb2_grpc.DataStoreServiceServicer):

  def __init__(self, config: datastore_pb2.DataStore):
    self.config = config

  def GetTestcases(self, request: datastore_pb2.GetTestcasesRequest,
                   context) -> datastore_pb2.GetTestcasesResponse:
    del context
    app.Info('GetTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(datastore_pb2.GetTestcasesResponse)
    # TODO(cec): Implement!
    return response

  def SubmitTestcases(self, request: datastore_pb2.SubmitTestcasesRequest,
                      context) -> datastore_pb2.SubmitTestcasesResponse:
    del context
    app.Info('SubmitTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
        datastore_pb2.SubmitTestcasesResponse)
    # TODO(cec): Implement!
    return response

  def SubmitResults(self, request: datastore_pb2.SubmitResultsRequest,
                    context) -> datastore_pb2.SubmitResultsResponse:
    del context
    app.Info('SubmitResults() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
        datastore_pb2.SubmitResultsResponse)
    # TODO(cec): Implement!
    return response


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  datastore_config = services.ServiceConfigFromFlag('datastore_config',
                                                    datastore_pb2.DataStore())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(datastore_config.service)
  service = DataStore(datastore_config)
  datastore_pb2_grpc.add_DataStoreServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{datastore_config.service.port}')
  app.Info('%s listening on %s:%s',
           type(service).__name__, datastore_config.service.hostname,
           datastore_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.RunWithArgs(main)
