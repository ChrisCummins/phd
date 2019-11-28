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
"""A very basic "generator" which produces strings of random characters."""
import random
import string

from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from labm8.py import app
from labm8.py import labdate
from labm8.py import pbutil

FLAGS = app.FLAGS


class RandCharGenerator(generator.GeneratorServiceBase,
                        generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.RandCharGenerator):
    super(RandCharGenerator, self).__init__(config)
    self.toolchain = self.config.model.corpus.language
    self.generator = deepsmith_pb2.Generator(
        name='randchar',
        opts={
            'toolchain':
            str(
                pbutil.AssertFieldConstraint(self.config, 'toolchain',
                                             lambda x: len(x))),
            'min_len':
            str(
                pbutil.AssertFieldConstraint(self.config, 'string_min_len',
                                             lambda x: x > 0)),
            'max_len':
            str(
                pbutil.AssertFieldConstraint(
                    self.config, 'string_max_len',
                    lambda x: x > 0 and x >= self.config.string_min_len)),
        })

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    """Generate testcases."""
    del context
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)

    # Generate random strings.
    for _ in range(request.num_testcases):
      # Pick a length for the random string.
      n = random.randint(self.config.string_min_len,
                         self.config.string_max_len + 1)
      # Instantiate a testcase.
      testcase = response.testcases.add()
      testcase.toolchain = self.config.toolchain
      testcase.generator.CopyFrom(self.generator)
      start_time = labdate.MillisecondsTimestamp()
      testcase.inputs['src'] = ''.join(
          random.choice(string.ascii_lowercase) for _ in range(n))
      end_time = labdate.MillisecondsTimestamp()
      p = testcase.profiling_events.add()
      p.type = 'generation'
      p.event_start_epoch_ms = start_time
      p.duration_ms = end_time - start_time

    return response


if __name__ == '__main__':
  app.RunWithArgs(RandCharGenerator.Main(generator_pb2.RandCharGenerator))
