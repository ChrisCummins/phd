"""A very basic "generator" which always returns the same testcase."""
from absl import app
from absl import flags

from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8 import labdate

FLAGS = flags.FLAGS


class DummyGenerator(generator.GeneratorServiceBase,
                     generator_pb2_grpc.GeneratorServiceServicer):
  """A very basic "generator" which always returns the same testcase."""

  def __init__(self, config: generator_pb2.RandCharGenerator):
    super(DummyGenerator, self).__init__(config)
    self.toolchain = self.config.model.corpus.language
    self.generator = deepsmith_pb2.Generator(name='dummy_generator')

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    """Generate testcases."""
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)

    # Generate random strings.
    for _ in range(request.num_testcases):
      # Instantiate a testcase.
      testcase = response.testcases.add()
      testcase.CopyFrom(self.config.testcase_to_generate)
      testcase.generator.CopyFrom(self.generator)
      start_time = labdate.MillisecondsTimestamp()
      end_time = labdate.MillisecondsTimestamp()
      p = testcase.profiling_events.add()
      p.type = 'generation'
      p.event_start_epoch_ms = start_time
      p.duration_ms = end_time - start_time

    return response


if __name__ == '__main__':
  app.run(DummyGenerator.Main(generator_pb2.DummyGenerator))
