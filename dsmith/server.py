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
    def SubmitTestcases(self, request: pb.SubmitTestcasesRequest,
                        context) -> pb.SubmitTestcasesResponse:
        """ Submit test cases. """
        with db.Session(db.init()) as session:
            for testcase in request.testcases:
                datastore.add_testcase(session, testcase)
            session.commit()

        return pb.SubmitTestcasesResponse()


def serve(port: int=50051):
    """ Spool up a local server. """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_TestingServiceServicer_to_server(TestingService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    try:
        while True:
            time.sleep(3600 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
