"""
The testing service exposes.
"""
import time

from concurrent import futures

import grpc
from absl import flags
from absl import app

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2 as pb
from deeplearning.deepsmith.protos import deepsmith_pb2_grpc as rpc


FLAGS = flags.FLAGS

flags.DEFINE_integer("testing_service_port", 50051, "")


class TestingService(rpc.TestingServiceServicer):
  """ """

  def __init__(self, ds: datastore.DataStore):
    self.ds = ds

  def SubmitTestcases(self, request: pb.SubmitTestcasesRequest,
                      context) -> pb.SubmitTestcasesResponse:
    """ Submit test cases. """
    self.ds.add_testcases(request.testcases)
    return pb.SubmitTestcasesResponse()


def main():
  """ Spool up a local server. """
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service = TestingService(db.DatabaseContext())
  rpc.add_TestingServiceServicer_to_server(service, server)
  server.add_insecure_port(f"[::]:{port}")
  server.start()

  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
