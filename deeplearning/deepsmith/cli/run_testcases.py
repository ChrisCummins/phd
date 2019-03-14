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
import collections
import typing

from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import datastore_pb2_grpc
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_string('datastore_config', None, 'Path to a DataStore message.')
app.DEFINE_string('harness_config', None, 'Path to a Harness config.')
app.DEFINE_integer(
    'target_total_results', -1,
    'The number of results to collect from each Testbed. Results already in the '
    'datastore contribute towards this total. If --target_total_results is '
    'negative, results are collected for all testcases in the DataStore.')
app.DEFINE_integer('harness_batch_size', 100,
                   'The number of results to collect in each batch.')


def GetHarnessCapabilities(harness_stub: harness_pb2_grpc.HarnessServiceStub
                          ) -> harness_pb2.GetHarnessCapabilitiesResponse:
  request = services.BuildDefaultRequest(
      harness_pb2.GetHarnessCapabilitiesRequest)
  response = harness_stub.GetHarnessCapabilities(request)
  services.AssertResponseStatus(response.status)
  return response


def GetNumberOfResultsForTestbed(
    datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
    harness: deepsmith_pb2.Harness, testbed: deepsmith_pb2.Testbed) -> int:
  request = services.BuildDefaultRequest(datastore_pb2.GetResultsRequest)
  request.return_total_matching_count = True
  request.return_results = False
  request.toolchain = testbed.toolchain
  request.harness.CopyFrom(harness)
  request.testbed.CopyFrom(testbed)
  response = datastore_stub.GetTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.total_matching_count


def GetNumberOfTestcases(
    datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
    harness: deepsmith_pb2.Harness, testbed: deepsmith_pb2.Testbed) -> int:
  request = services.BuildDefaultRequest(datastore_pb2.GetTestcasesRequest)
  request.return_total_matching_count = True
  request.return_testcases = False
  request.mark_pending_results = False
  request.include_testcases_with_results = True
  request.include_testcases_with_pending_results = True
  request.toolchain = testbed.toolchain
  request.harness.CopyFrom(harness)
  response = datastore_stub.GetTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.total_matching_count


def GetTestcasesToRun(datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
                      harness: deepsmith_pb2.Harness,
                      testbed: deepsmith_pb2.Testbed, target_total_results: int,
                      batch_size: int) -> typing.List[deepsmith_pb2.Testcase]:
  if target_total_results >= 0:
    total_results = GetNumberOfResultsForTestbed(datastore_stub, harness,
                                                 testbed)
    total_testcases = GetNumberOfTestcases(datastore_stub, harness, testbed)
    batch_size = min(batch_size, total_results - total_testcases)

  request = services.BuildDefaultRequest(datastore_pb2.GetTestcasesRequest)
  request.toolchain = testbed.toolchain
  request.harness.CopyFrom(harness)
  request.testbed.CopyFrom(testbed)
  request.max_num_testcases_to_return = batch_size
  request.mark_results_pending.extend([testbed])
  response = datastore_stub.GetTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.testcases


def RunTestcases(harness_stub: harness_pb2_grpc.HarnessServiceStub,
                 testbed: deepsmith_pb2.Testbed,
                 testcases: typing.List[deepsmith_pb2.Testcase]
                ) -> typing.List[deepsmith_pb2.Result]:
  request = services.BuildDefaultRequest(harness_pb2.RunTestcasesRequest)
  request.testbed.CopyFrom(testbed)
  request.testcases.extend(testcases)
  response = harness_stub.RunTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.results


def SubmitResults(datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
                  results: typing.List[deepsmith_pb2.Result]) -> None:
  request = services.BuildDefaultRequest(datastore_pb2.SubmitResultsRequest)
  request.results.extend(results)
  response = datastore_stub.SubmitTestcases(request)
  services.AssertResponseStatus(response.status)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  if FLAGS.harness_batch_size <= 0:
    raise app.UsageError('--harness_batch_size must be positive')
  datastore_config = services.ServiceConfigFromFlag('datastore_config',
                                                    datastore_pb2.DataStore())
  harness_config = services.ServiceConfigFromFlag('harness_config',
                                                  harness_pb2.CldriveHarness())

  datastore_stub = services.GetServiceStub(
      datastore_config, datastore_pb2_grpc.DataStoreServiceStub)
  harness_stub = services.GetServiceStub(harness_config,
                                         harness_pb2_grpc.HarnessServiceStub)

  target_total_results = FLAGS.target_total_results
  harness_batch_size = FLAGS.harness_batch_size
  capabilities = GetHarnessCapabilities(harness_stub)
  testbeds = collections.deque(capabilities.testbeds)
  if testbeds:
    app.Log(1, '%d testbeds: %s', len(capabilities.testbeds),
            ', '.join(x.name for x in capabilities.testbeds))
    while testbeds:
      testbed = testbeds.popleft()
      testcases = GetTestcasesToRun(datastore_stub, capabilities.harness,
                                    testbed, target_total_results,
                                    harness_batch_size)
      app.Log(1, 'Received %d testcases to execute on %s', len(testcases),
              testbed.name)
      if testcases:
        results = RunTestcases(harness_stub, testbed, testcases)
        SubmitResults(datastore_stub, results)
        # If there are testcases to run, then we add it back to the testbeds
        # queue, as there may be more.
        testbeds.append(testbed)
    app.Log(1, 'done')
  else:
    app.Warning('No testbeds, nothing to do!')


if __name__ == '__main__':
  app.RunWithArgs(main)
