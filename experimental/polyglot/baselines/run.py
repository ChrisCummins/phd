"""Run a baseline."""
import pathlib

from absl import app
from absl import flags

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('corpus', None, 'Path to corpus config.')
flags.DEFINE_string('model', None, 'Path to model config.')
flags.DEFINE_string('sampler', None, 'Path to sampler config.')
flags.DEFINE_string('working_dir', '/var/phd/clgen/baseline',
                    'Path to CLgen working directory')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  corpus = pbutil.FromFile(pathlib.Path(FLAGS.corpus), corpus_pb2.Corpus())
  model = pbutil.FromFile(pathlib.Path(FLAGS.model), model_pb2.Model())
  model.corpus.CopyFrom(corpus)
  sampler = pbutil.FromFile(pathlib.Path(FLAGS.sampler), sampler_pb2.Sampler())

  config = clgen_pb2.Instance(model=model, sampler=sampler)
  config.working_dir = FLAGS.working_dir
  instance = clgen.Instance(config)

  target_samples = 1000
  num_samples = 0
  sample_dir = instance.model.SamplerCache(instance.sampler)
  if sample_dir.is_dir():
    num_samples = len(list(sample_dir.iterdir()))
  if num_samples < target_samples:
    instance.Sample(min_num_samples=target_samples - num_samples)


if __name__ == '__main__':
  app.run(main)
