"""Extract OpenCL kernel features and import to database."""
import multiprocessing
import pathlib
import random
import threading
import typing

import humanize
import progressbar
import sqlalchemy as sql
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


def _ProcessPath(path: pathlib.Path) -> typing.Optional[
  typing.Tuple[str, grewe_features.GreweEtAlFeatures]]:
  try:
    features = list(grewe_features.ExtractFeaturesFromPath(path))
  except grewe_features.FeatureExtractionError as e:
    logging.debug("Feature extraction failed with message: %s", e)
    return None, None

  if len(features) != 1:
    logging.debug("Expected 1 feature vector in %s, found %d",
                  path, len(features))
    return None, None

  try:
    with open(path) as f:
      src = f.read()
  except UnicodeEncodeError:
    logging.debug("Failed to encode %s", src)
    return None, None

  return src, features[0]

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
  logging.info('Importing %s files ...',
               humanize.intcomma(len(paths_to_import)))
  bar = progressbar.ProgressBar(max_value=len(paths_to_import), redirect_stderr=True)
  pool = multiprocessing.Pool(FLAGS.num_processes)
  for i, (src, features) in enumerate(pool.imap_unordered(
        _ProcessPath, paths_to_import)):
    bar.update(i)
    # None type return if feature extraction failed.
    if src:
      with db.Session(commit=False) as session:
        obj = grewe_features_db.OpenCLKernelWithRawGreweFeatures.FromSrcOriginAndFeatures(
            src, FLAGS.origin, features)
        # Check if it already exists in the database.
        exists = session.query(grewe_features_db.OpenCLKernelWithRawGreweFeatures)\
            .filter_by(src_sha256=obj.src_sha256).first()
        if not exists:
          session.add(obj)
          try:
            session.commit()
          except (sql.exc.OperationalError, sql.exc.DataError) as e:
            logging.warning('Failed to commit database entry: %s', e)

  logging.info('done')



if __name__ == '__main__':
  app.run(main)
