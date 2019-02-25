from absl import app
from absl import flags

from deeplearning.clgen import clgen
from deeplearning.deepsmith.generators import clgen_pretrained
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2

FLAGS = flags.FLAGS


def ClgenInstanceToGenerator(
    instance: clgen.Instance) -> deepsmith_pb2.Generator:
  """Convert a CLgen instance to a DeepSmith generator proto."""
  g = deepsmith_pb2.Generator()
  g.name = 'clgen'
  g.opts['model'] = instance.model.hash
  g.opts['sampler'] = instance.sampler.hash
  return g


class ClgenGenerator(clgen_pretrained.ClgenGenerator):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClgenGenerator, self).__init__(config, no_init=True)
    self.instance = clgen.Instance(self.config.instance)
    self.toolchain = 'opencl'
    self.generator = ClgenInstanceToGenerator(self.instance)
    if not self.config.testcase_skeleton:
      raise ValueError('No testcase skeletons provided')
    for skeleton in self.config.testcase_skeleton:
      skeleton.generator.CopyFrom(self.generator)


if __name__ == '__main__':
  app.run(ClgenGenerator.Main(generator_pb2.ClgenGenerator))
