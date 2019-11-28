#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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

from experimental.dsmith import dsmith_pb2 as pb
from experimental.dsmith import dsmith_pb2_grpc as rpc


def run():
  channel = grpc.insecure_channel("localhost:50051")
  stub = rpc.TestingServiceStub(channel)

  request = pb.SubmitTestcasesRequest(testcases=[])

  response: pb.SubmitTestcasesResponse = stub.SubmitTestcases(request)
  print("TestingService client received: ", type(response).__name__)


if __name__ == "__main__":
  run()
