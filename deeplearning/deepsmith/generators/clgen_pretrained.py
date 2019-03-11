"""A clgen generator for pre-trained models.

By supporting only pre-trained models, this module does not depend on the CLgen
corpus implementation, allowing for a much smaller dependency set, i.e. without
pulling all of the LLVM libraries required by CLgen's corpus preprocessors.
"""
import math
import sys
import typing

from deeplearning.clgen import sample
from deeplearning.clgen.proto import model_pb2
from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8 import app

FLAGS = app.FLAGS


def ClgenInstanceToGenerator(
    instance: sample.Instance) -> deepsmith_pb2.Generator:
  """Convert a CLgen instance to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = 'clgen'
  g.opts['model'] = str(instance.model.path)
  g.opts['sampler'] = instance.sampler.hash
  return g


class ClgenGenerator(generator.GeneratorServiceBase,
                     generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self,
               config: generator_pb2.ClgenGenerator,
               no_init: bool = False):
    """

    Args:
      config: The Generator config.
      no_init: If True, do not initialize the instance and generator values.
    """
    super(ClgenGenerator, self).__init__(config)
    if not no_init:
      self.instance = sample.Instance(self.config.instance)
      self.toolchain = 'opencl'
      self.generator = ClgenInstanceToGenerator(self.instance)
      if not self.config.testcase_skeleton:
        raise ValueError('No testcase skeletons provided')
      for skeleton in self.config.testcase_skeleton:
        skeleton.generator.CopyFrom(self.generator)

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
      for i, sample_ in enumerate(
          self.instance.model.Sample(self.instance.sampler, num_programs)):
        app.Info('Generated sample %d.', i + 1)
        response.testcases.extend(self.SampleToTestcases(sample_))

    # Flush any remaining output generated during Sample().
    sys.stdout.flush()
    return response

  def SampleToTestcases(
      self, sample_: model_pb2.Sample) -> typing.List[deepsmith_pb2.Testcase]:
    """Convert a CLgen sample to a list of DeepSmith testcase protos."""
    testcases = []
    for skeleton in self.config.testcase_skeleton:
      t = deepsmith_pb2.Testcase()
      t.CopyFrom(skeleton)
      p = t.profiling_events.add()
      p.type = 'generation'
      p.duration_ms = sample_.wall_time_ms
      p.event_start_epoch_ms = sample_.sample_start_epoch_ms_utc
      t.inputs['src'] = sample_.text
      testcases.append(t)
    return testcases


if __name__ == '__main__':
  app.RunWithArgs(ClgenGenerator.Main(generator_pb2.ClgenGenerator))
