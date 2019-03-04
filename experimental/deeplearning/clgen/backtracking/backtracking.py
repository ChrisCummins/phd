"""Sample the experimental backtracking sampler."""
import os
import typing

from absl import app
from absl import flags

from deeplearning.clgen import samplers
from experimental.deeplearning.clgen.backtracking import backtracking_model
from research.cummins_2017_cgo import generative_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('sample_seed', 0, 'Random seed.')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  config = generative_model.CreateInstanceProtoFromFlags()

  os.environ['CLGEN_CACHE'] = config.working_dir
  model = backtracking_model.BacktrackingModel(config.model)
  sampler = samplers.Sampler(config.sampler)

  model.Sample(sampler, FLAGS.clgen_min_sample_count, FLAGS.sample_seed)


if __name__ == '__main__':
  app.run(main)
