import collections
import typing

from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import datastore_pb2_grpc
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import harness_pb2_grpc
from deeplearning.deepsmith.services import services

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'datastore_config', None,
  'Path to a DataStore message.')
flags.DEFINE_string(
  'harness_config', None,
  'Path to a Harness config.')
flags.DEFINE_integer(
  'target_total_testcases', -1,
  'The number of testcases to runon each Testbed. Results already in the '
  'datastore contribute towards this total. If --target_total_testcases is '
  'negative, all testcases in the datastore are run.')
flags.DEFINE_integer(
  'harness_batch_size', 100,
  'The number of testcases to generate in each batch.')


def GetHarnessCapabilities(
    harness_stub: harness_pb2_grpc.HarnessServiceStub
) -> harness_pb2.GetHarnessCapabilitiesResponse:
  request = services.BuildDefaultRequest(
    harness_pb2.GetHarnessCapabilitiesRequest)
  response = harness_stub.GetHarnessCapabilities(request)
  services.AssertResponseStatus(response.status)
  return response


def GetTestcasesToRun(
    datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
    harness: deepsmith_pb2.Harness,
    testbed: deepsmith_pb2.Testbed,
    max_testcases: int) -> typing.List[deepsmith_pb2.Testcase]:
  request = services.BuildDefaultRequest(datastore_pb2.GetTestcasesRequest)
  request.toolchain = testbed.toolchain
  request.harness.CopyFrom(harness)
  request.testbed.CopyFrom(testbed)
  request.max_num_testcases_to_return = max_testcases
  response = datastore_stub.GetTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.testcases


def RunTestcases(
    harness_stub: harness_pb2_grpc.HarnessServiceStub,
    testbed_: deepsmith_pb2.Testbed,
    testcases: typing.List[deepsmith_pb2.Testcase]
) -> typing.List[deepsmith_pb2.Result]:
  request = services.BuildDefaultRequest(harness_pb2.RunTestcasesRequest)
  request.testbed.CopyFrom(testbed_)
  request.testcases.extend(testcases)
  response = harness_stub.RunTestcases(request)
  services.AssertResponseStatus(response.status)
  return response.results


def SubmitResults(
    datastore_stub: datastore_pb2_grpc.DataStoreServiceStub,
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
  datastore_config = services.ServiceConfigFromFlag(
    'datastore_config', datastore_pb2.DataStore())
  harness_config = services.ServiceConfigFromFlag(
    'harness_config', harness_pb2.CldriveHarness())

  datastore_stub = services.GetServiceStub(
    datastore_config, datastore_pb2_grpc.DataStoreServiceStub)
  harness_stub = services.GetServiceStub(
    harness_config, harness_pb2_grpc.HarnessServiceStub)

  target_total_testcases = FLAGS.target_total_testcases
  harness_batch_size = FLAGS.harness_batch_size
  capabilities = GetHarnessCapabilities(harness_stub)
  testbeds = collections.deque(capabilities.testbeds)

  while testbeds:
    testbed_ = testbeds.popleft()
    testcases = GetTestcasesToRun(
      datastore_stub, capabilities.harness, testbed_, harness_batch_size)
    logging.info(
      'Received %d testcases to execute on %s', len(testcases), testbed_.name)
    if testcases:
      results = RunTestcases(harness_stub, testbed_, testcases)
      SubmitResults(datastore_stub, results)
      # If there are testcases to run, then we add it back to the testbeds
      # queue, as there may be more.
      testbeds.append(testbed_)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
