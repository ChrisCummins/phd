# Copyright (c) 2017, 2018, 2019 Chris Cummins.
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
import typing

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import datastore_pb2_grpc
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string("datastore_config", None, "Path to a DataStore message.")
app.DEFINE_string("generator_config", None, "Path to a ClgenGenerator message.")
app.DEFINE_integer(
  "target_total_testcases",
  -1,
  "The number of testcases to generate. Testcases already in the datastore "
  "contribute towards this total. If --target_total_testcases is negative, "
  "testcases are generated indefinitely.",
)
app.DEFINE_integer(
  "generator_batch_size",
  1000,
  "The number of testcases to generate in each batch.",
)


def GetGeneratorCapabilities(
  generator_stub: generator_pb2_grpc.GeneratorServiceStub,
) -> generator_pb2.GetCapabilitiesResponse:
  request = services.BuildDefaultRequest(generator_pb2.GetCapabilitiesRequest)
  response = generator_stub.GetGeneratorCapabilities(request)
  services.AssertResponseStatus(response.status)
  return response


def GetNumberOfTestcasesInDataStore(
  datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
  capabilities: generator_pb2.GetCapabilitiesResponse,
) -> int:
  request = services.BuildDefaultRequest(datastore_pb2.GetTestcasesRequest)
  request.return_total_matching_count = True
  request.return_testcases = False
  request.mark_pending_results = False
  request.include_testcases_with_results = True
  request.include_testcases_with_pending_results = True
  request.toolchain = capabilities.toolchain
  request.generator.CopyFrom(capabilities.generator)
  response = datastore_stub.GetTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.total_matching_count


def GenerateTestcases(
  generator_stub: generator_pb2_grpc.GeneratorServiceStub, num_to_generate: int
) -> typing.List[deepsmith_pb2.Testcase]:
  app.Log(1, f"Generating batch of {num_to_generate} testcases")
  request = services.BuildDefaultRequest(generator_pb2.GenerateTestcasesRequest)
  response = generator_stub.GenerateTestcases(request)
  services.AssertResponseStatus(response.status)
  num_generated = len(response.testcases)
  if num_generated != num_to_generate:
    app.Warning(
      f"Requested {num_to_generate} testcases, received {num_generated}"
    )
  return response.testcases


def SubmitTestcases(
  datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
  testcases: typing.List[deepsmith_pb2.Testcase],
) -> None:
  request = services.BuildDefaultRequest(datastore_pb2.SubmitTestcasesRequest)
  request.testcases.extend(testcases)
  response = datastore_stub.SubmitTestcases(request)
  services.AssertResponseStatus(response.status)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Unrecognized arguments")
  if FLAGS.generator_batch_size <= 0:
    raise app.UsageError("--generator_batch_size must be positive")
  datastore_config = services.ServiceConfigFromFlag(
    "datastore_config", datastore_pb2.DataStore()
  )
  generator_config = services.ServiceConfigFromFlag(
    "generator_config", generator_pb2.ClgenGenerator()
  )

  datastore_stub = services.GetServiceStub(
    datastore_config, datastore_pb2_grpc.DataStoreServiceStub
  )
  generator_stub = services.GetServiceStub(
    generator_config, generator_pb2_grpc.GeneratorServiceStub
  )

  target_total_testcases = FLAGS.target_total_testcases
  generator_batch_size = FLAGS.generator_batch_size
  capabilities = GetGeneratorCapabilities(generator_stub)

  while True:
    num_testcases = GetNumberOfTestcasesInDataStore(
      datastore_stub, capabilities
    )
    app.Log(1, f"Number of testcases in datastore: %d", num_testcases)
    if 0 <= target_total_testcases <= num_testcases:
      app.Log(
        1,
        "Stopping generation with %d testcases in the DataStore.",
        num_testcases,
      )
      break

    num_to_generate = generator_batch_size
    if target_total_testcases >= 0:
      num_to_generate = min(
        generator_batch_size, target_total_testcases - num_testcases
      )

    testcases = GenerateTestcases(generator_stub, num_to_generate)
    SubmitTestcases(datastore_stub, testcases)


if __name__ == "__main__":
  app.RunWithArgs(main)
