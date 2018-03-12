"""
This file defines the TestingService object.
"""
import time

import grpc
from absl import app
from absl import flags
from absl import logging
from concurrent import futures

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import deepsmith_pb2_grpc

FLAGS = flags.FLAGS

flags.DEFINE_integer("testing_service_port", 50051, "")


class TestingService(deepsmith_pb2_grpc.TestingServiceServicer):
  """An RPC service for testing data.

  A TestingService maintains a reference to a datastore, and acts as a thin
  wrapper around a subset of its functions. The added value provided by
  this class is to handle errors thrown by the datastore, and to set
  a response.
  """

  def __init__(self, ds: datastore.DataStore):
    self.ds = ds

  def SubmitTestcases(self, request: deepsmith_pb2.SubmitTestcasesRequest,
                      context) -> deepsmith_pb2.SubmitTestcasesResponse:
    """Submit Testcases.
    """
    logging.info("SubmitTestcases() client=%s", request.client)
    response = deepsmith_pb2.SubmitTestcasesResponse()
    try:
      self.ds.SubmitTestcases(request, response)
    except:
      response.status = deepsmith_pb2.SubmitTestcasesResponse.FAILURE
    return response

  def RequestTestcases(self, request: deepsmith_pb2.RequestTestcasesRequest,
                       context) -> deepsmith_pb2.RequestTestcasesResponse:
    """Request Testcases.
    """
    logging.info("RequestTestcases() client=%s", request.client)
    response = deepsmith_pb2.RequestTestcasesResponse()

    try:
      self.ds.RequestTestcases(request, response)
    except datastore.InvalidRequest as e:
      response.status = deepsmith_pb2.RequestTestcasesResponse.INVALID_REQUEST
      response.error = str(e)
    except:
      response.status = deepsmith_pb2.RequestTestcasesResponse.FAILURE

    return response


def main(argv):
  del argv
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  ds = datastore.DataStore.FromFlags()
  service = TestingService(ds)
  deepsmith_pb2_grpc.add_TestingServiceService_to_server(service, server)
  server.add_insecure_port(f"[::]:{FLAGS.port}")
  server.start()

  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
