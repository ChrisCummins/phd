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
import time
from concurrent import futures

import grpc

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import datastore_pb2_grpc
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string("datastore_config", None, "Path to a DataStore message.")


class DataStore(
  services.ServiceBase, datastore_pb2_grpc.DataStoreServiceServicer
):
  def __init__(self, config: datastore_pb2.DataStore):
    self.config = config

  def GetTestcases(
    self, request: datastore_pb2.GetTestcasesRequest, context
  ) -> datastore_pb2.GetTestcasesResponse:
    del context
    app.Log(1, "GetTestcases() client=%s", request.status.client)
    response = services.BuildDefaultResponse(datastore_pb2.GetTestcasesResponse)
    # TODO(cec): Implement!
    return response

  def SubmitTestcases(
    self, request: datastore_pb2.SubmitTestcasesRequest, context
  ) -> datastore_pb2.SubmitTestcasesResponse:
    del context
    app.Log(1, "SubmitTestcases() client=%s", request.status.client)
    response = services.BuildDefaultResponse(
      datastore_pb2.SubmitTestcasesResponse
    )
    # TODO(cec): Implement!
    return response

  def SubmitResults(
    self, request: datastore_pb2.SubmitResultsRequest, context
  ) -> datastore_pb2.SubmitResultsResponse:
    del context
    app.Log(1, "SubmitResults() client=%s", request.status.client)
    response = services.BuildDefaultResponse(
      datastore_pb2.SubmitResultsResponse
    )
    # TODO(cec): Implement!
    return response


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Unrecognized arguments")
  datastore_config = services.ServiceConfigFromFlag(
    "datastore_config", datastore_pb2.DataStore()
  )
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(datastore_config.service)
  service = DataStore(datastore_config)
  datastore_pb2_grpc.add_DataStoreServiceServicer_to_server(service, server)
  server.add_insecure_port(f"[::]:{datastore_config.service.port}")
  app.Log(
    1,
    "%s listening on %s:%s",
    type(service).__name__,
    datastore_config.service.hostname,
    datastore_config.service.port,
  )
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == "__main__":
  app.RunWithArgs(main)
