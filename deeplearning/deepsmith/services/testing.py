"""
The testing service.
"""
import time

from concurrent import futures

import grpc
from absl import flags
from absl import app

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2
from deeplearning.deepsmith.protos import deepsmith_pb2_grpc

FLAGS = flags.FLAGS

flags.DEFINE_integer("testing_service_port", 50051, "")


class TestingService(deepsmith_pb2_grpc.TestingServiceServicer):
  """An RPC service for testing data."""

  def __init__(self, ds: datastore.DataStore):
    self.ds = ds

  def SubmitTestcases(self, request: deepsmith_pb2.SubmitTestcasesRequest,
                      context) -> deepsmith_pb2.SubmitTestcasesResponse:
    """ Submit test cases. """
    self.ds.add_testcases(request.testcases)
    return deepsmith_pb2.SubmitTestcasesResponse()


def main():  # pylint: disable=missing-docstring
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service = TestingService(db.DatabaseContext())
  deepsmith_pb2_grpc.add_TestingServiceServicer_to_server(service, server)
  server.add_insecure_port(f"[::]:{FLAGS.port}")
  server.start()

  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
