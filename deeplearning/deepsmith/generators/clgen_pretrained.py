"""A clgen generator for pre-trained models.

By supporting only pre-trained models, this module does not depend on the CLgen
corpus implementation, allowing for a much smaller dependency set, i.e. without
pulling all of the LLVM libraries required by CLgen's corpus preprocessors.
"""
import grpc
import math
import sys
import time
import typing
from absl import app
from absl import flags
from absl import logging
from concurrent import futures

from deeplearning.clgen import sample
from deeplearning.clgen.proto import model_pb2
from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc


FLAGS = flags.FLAGS


def ClgenInstanceToGenerator(
    instance: sample.Instance) -> deepsmith_pb2.Generator:
  """Convert a CLgen instance to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = f'clgen'
  g.opts['model'] = str(instance.model.path)
  g.opts['sampler'] = instance.sampler.hash
  return g


class ClgenGenerator(generator.GeneratorBase,
                     generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator,
               no_init: bool = False):
    """

    Args:
      config: The Generator config.
      no_init: If True, do not initialize the instance and generator values.
    """
    super(ClgenGenerator, self).__init__(config)
    if not no_init:
      self.instance = sample.Instance(self.config.instance)
      self.generator = ClgenInstanceToGenerator(self.instance)

  def GetGeneratorCapabilities(
      self, request: generator_pb2.GetGeneratorCapabilitiesRequest,
      context) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GetGeneratorCapabilitiesRequest)
    response.toolchain = self.config.model.corpus.language
    response.generator = self.generator
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)
    with self.instance.Session():
      num_programs = math.ceil(
          request.num_testcases / len(self.config.testcase_skeleton))
      for i, sample in enumerate(self.instance.model.Sample(
          self.instance.sampler, num_programs)):
        logging.info('Generated sample %d.', i + 1)
        response.testcases.extend(self.SampleToTestcases(sample))

    # Flush any remaining output generated during Sample().
    sys.stdout.flush()
    return response

  def SampleToTestcases(self, sample: model_pb2.Sample) -> typing.List[
    deepsmith_pb2.Testcase]:
    """Convert a CLgen sample to a list of DeepSmith testcase protos."""
    testcases = []
    for skeleton in self.config.testcase_skeleton:
      t = deepsmith_pb2.Testcase()
      t.CopyFrom(skeleton)
      p = t.profiling_events.add()
      p.type = 'generation'
      p.duration_ms = sample.wall_time_ms
      p.event_start_epoch_ms = sample.sample_start_epoch_ms_utc
      t.inputs['src'] = sample.text
      testcases.append(t)
    return testcases


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
      'generator_config', generator_pb2.ClgenGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config.service)
  service = ClgenGenerator(generator_config)
  generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{generator_config.service.port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               generator_config.service.hostname, generator_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
