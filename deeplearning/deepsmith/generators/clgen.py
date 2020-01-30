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
from deeplearning.clgen import clgen
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from labm8.py import app

FLAGS = app.FLAGS


def ClgenInstanceToGenerator(
  instance: clgen.Instance,
) -> deepsmith_pb2.Generator:
  """Convert a CLgen instance to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = "clgen"
  g.opts["model"] = instance.model.hash
  g.opts["sampler"] = instance.sampler.hash
  return g


class ClgenGenerator(clgen_pretrained.ClgenGenerator):
  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClgenGenerator, self).__init__(config, no_init=True)
    self.instance = clgen.Instance(self.config.instance)
    self.toolchain = "opencl"
    self.generator = ClgenInstanceToGenerator(self.instance)
    if not self.config.testcase_skeleton:
      raise ValueError("No testcase skeletons provided")
    for skeleton in self.config.testcase_skeleton:
      skeleton.generator.CopyFrom(self.generator)


if __name__ == "__main__":
  app.RunWithArgs(ClgenGenerator.Main(generator_pb2.ClgenGenerator))
