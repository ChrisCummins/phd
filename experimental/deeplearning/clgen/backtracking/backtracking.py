"""Sample the experimental backtracking sampler.

Usage:
  bazel run //experimental/deeplearning/clgen/backtracking --
    --clgen_sample_batch_size=1 --clgen_max_sample_length=0
    --clgen_min_sample_count=100
    --clgen_seed_text='kernel void A(global int* a, global float* b, const int c) {'
    --experimental_clgen_backtracking_target_features='100,10,0,0'
    --experimental_clgen_backtracking_feature_distance_epsilon=0.01
    --experimental_clgen_backtracking_max_feature_distance=0.01
    --db='file:///var/phd/db/cc1.mysql?backtracking?charset=utf8'
"""
import os
import typing

from absl import app
from absl import flags

from deeplearning.clgen import samplers
from experimental.deeplearning.clgen.backtracking import backtracking_db
from experimental.deeplearning.clgen.backtracking import backtracking_model
from experimental.deeplearning.clgen.backtracking.logger import \
  BacktrackingDatabaseLogger
from research.cummins_2017_cgo import generative_model

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db', 'sqlite:///tmp/phd/experimental/deplearning/clgen/backtracking/db.db',
    'URL of the database for backtracking.')
flags.DEFINE_integer('sample_seed', 0, 'Random seed.')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  config = generative_model.CreateInstanceProtoFromFlags()

  os.environ['CLGEN_CACHE'] = config.working_dir

  logger = BacktrackingDatabaseLogger(backtracking_db.Database(FLAGS.db))

  model = backtracking_model.BacktrackingModel(config.model, logger=logger)
  sampler = samplers.Sampler(config.sampler)

  model.Sample(sampler, FLAGS.clgen_min_sample_count, FLAGS.sample_seed)


if __name__ == '__main__':
  app.run(main)
