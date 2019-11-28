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
"""A CLSmith program generator."""
import math
import typing

from compilers.clsmith import clsmith
from deeplearning.deepsmith import services
from deeplearning.deepsmith.generators import generator
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.proto import service_pb2
from labm8.py import app
from labm8.py import labdate

FLAGS = app.FLAGS


def ConfigToGenerator(
    config: generator_pb2.ClsmithGenerator) -> deepsmith_pb2.Generator:
  """Convert a config proto to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = 'clsmith'
  g.opts['opts'] = ' '.join(config.opt)
  return g


class ClsmithGenerator(generator.GeneratorServiceBase,
                       generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClsmithGenerator, self).__init__(config)
    self.toolchain = 'opencl'
    self.generator = ConfigToGenerator(self.config)
    if not self.config.testcase_skeleton:
      raise ValueError('No testcase skeletons provided')
    for skeleton in self.config.testcase_skeleton:
      skeleton.generator.CopyFrom(self.generator)

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    num_programs = int(
        math.ceil(request.num_testcases / len(self.config.testcase_skeleton)))
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)
    try:
      for i in range(num_programs):
        response.testcases.extend(
            self.SourceToTestcases(*self.GenerateOneSource()))
        app.Log(1, 'Generated file %d.', i + 1)
    except clsmith.CLSmithError as e:
      response.status.returncode = service_pb2.ServiceStatus.ERROR
      response.status.error_message = str(e)
    return response

  def GenerateOneSource(self) -> typing.Tuple[str, int, int]:
    """Generate and return a single CLSmith program.

    Returns:
      A tuple of the source code as a string, the generation time, and the start
      time.
    """
    start_epoch_ms_utc = labdate.MillisecondsTimestamp()
    src = clsmith.Exec(*list(self.config.opt))
    wall_time_ms = labdate.MillisecondsTimestamp() - start_epoch_ms_utc
    return src, wall_time_ms, start_epoch_ms_utc

  def SourceToTestcases(
      self, src: str, wall_time_ms: int,
      start_epoch_ms_utc: int) -> typing.List[deepsmith_pb2.Testcase]:
    """Make testcases from a CLSmith generated source."""
    testcases = []
    for skeleton in self.config.testcase_skeleton:
      t = deepsmith_pb2.Testcase()
      t.CopyFrom(skeleton)
      p = t.profiling_events.add()
      p.type = 'generation'
      p.duration_ms = wall_time_ms
      p.event_start_epoch_ms = start_epoch_ms_utc
      t.inputs['src'] = src
      testcases.append(t)
    return testcases


if __name__ == '__main__':
  app.RunWithArgs(ClsmithGenerator.Main(generator_pb2.ClsmithGenerator))
