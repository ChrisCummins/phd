"""Extract OpenCL kernel features and import to database."""
import multiprocessing
import pathlib
import random
import threading
import typing

import humanize
import progressbar
from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from gpu.portable_mapping_of_data_parallel_programs_to_opencl import \
  feature_extractor as grewe_features


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
flags.DEFINE_integer(
    'num_processes', multiprocessing.cpu_count(),
    'The number of import processes to spawn.')
flags.DEFINE_integer(
    'commit_to_db_every', 100,
    'How many files to process before committing results to database.')


def _ProcessPath(path: pathlib.Path) -> typing.Optional[
  typing.Tuple[str, grewe_features.GreweEtAlFeatures]]:
  try:
    features = grewe_features.ExtractFeaturesFromPath(path)
    with open(path) as f:
      src = f.read()
    return src, features
  except grewe_features.FeatureExtractionError as e:
    print(e)
    return None, None


class AsyncWorker(threading.Thread):
  """Thread which clones github repos."""

  def __init__(self, db: grewe_features_db.Database,
               paths_to_import: typing.List[pathlib.Path]):
    super(AsyncWorker, self).__init__()
    self.db = db
    self.paths_to_import = paths_to_import
    self.max = len(paths_to_import)
    self.i = 0

  def run(self):
    pool = multiprocessing.Pool(FLAGS.num_processes)
    with self.db.Session() as session:
      for self.i, (src, features) in enumerate(pool.imap_unordered(
            _ProcessPath, self.paths_to_import)):
        # None type return if feature extraction failed.
        if not src:
          continue
        session.add(
            grewe_features_db.OpenCLKernelWithRawGreweFeatures.FromSrcOriginAndFeatures(
                src, FLAGS.origin, features))
        if not i % FLAGS.commit_to_db_every:
          session.commit()


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

  random.shuffle(paths_to_import)
  worker = AsyncWorker(db, paths_to_import)
  logging.info('Importing %s files ...',
               humanize.intcomma(worker.max))
  bar = progressbar.ProgressBar(max_value=worker.max, redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.i)
    worker.join(.1)
  bar.update(worker.i)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
