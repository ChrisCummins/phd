"""
The testing service exposes.
"""
import grpc
import time

from concurrent import futures

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2 as pb
from deeplearning.deepsmith.protos import deepsmith_pb2_grpc as rpc


class TestingService(rpc.TestingServiceServicer):
    """ """
    def __init__(self, ds: datastore.DataStore):
        self.ds = ds

    def SubmitTestcases(self, request: pb.SubmitTestcasesRequest,
                        context) -> pb.SubmitTestcasesResponse:
        """ Submit test cases. """
        self.ds.add_testcases(request.testcases)
        return pb.SubmitTestcasesResponse()


def main(port: int=50051):
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
    main()
