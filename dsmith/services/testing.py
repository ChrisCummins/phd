#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
import grpc
import time

from concurrent import futures

import dsmith
from dsmith import datastore
from dsmith import db
from dsmith import dsmith_pb2 as pb
from dsmith import dsmith_pb2_grpc as rpc


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
