"""Extract OpenCL kernel features and import to database."""
import pathlib
import typing

from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'kernels_dir', None,
    'Path to directory containing kernels to import.')
flags.DEFINE_string(
    'origin', None,
    'Name of the origin of the kernels, e.g. "github".')
flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  kernel_dir = pathlib.Path(FLAGS.kernels_dir)
  if not kernel_dir.is_dir():
    raise app.UsageError('Kernel dir not found')

  if not FLAGS.origin:
    raise app.UsageError('Must set an origin')

  paths_to_import = list(kernel_dir.iterdir())
  if not all(x.is_file() for x in paths_to_import):
    raise app.UsageError('Non-file input found')

  db.ImportStaticFeaturesFromPaths(paths_to_import, FLAGS.origin)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
