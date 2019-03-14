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
"""A very basic "generator" which always returns the same testcase."""
from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8 import app
from labm8 import labdate

FLAGS = app.FLAGS


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
  app.RunWithArgs(DummyGenerator.Main(generator_pb2.DummyGenerator))
