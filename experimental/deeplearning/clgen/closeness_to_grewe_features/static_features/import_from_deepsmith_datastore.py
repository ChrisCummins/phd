"""Import OpenCL kernels from DeepSmith datastore."""
import pathlib
import tempfile
import typing

import humanize
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import generator
from deeplearning.deepsmith import testcase
from deeplearning.deepsmith import toolchain
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from labm8 import sqlutil

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')
flags.DEFINE_string('datastore', None,
                    'Path of the datastore config to import form')
flags.DEFINE_integer('batch_size', 256,
                     'The number of testcases to process in a batch.')
flags.DEFINE_integer('start_at', 0, 'The initial offset into the results set.')


def CreateTempFileFromTestcase(tempdir: pathlib.Path,
                               tc: testcase.Testcase) -> pathlib.Path:
  """Write testcase to a file in directory."""
  path = tempdir / f'{tc.id}.cl'
  with open(path, 'w') as f:
    f.write(tc.inputs['src'])
  return path


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.datastore:
    raise app.UsageError('--datastore flag is required')

  db = grewe_features_db.Database(FLAGS.db)
  ds = datastore.DataStore.FromFile(pathlib.Path(FLAGS.datastore))

  with ds.Session(commit=False) as session:
    generator_id = session.query(generator.Generator.id).filter(
        generator.Generator.name == 'clgen').first()
    if not generator_id:
      raise app.UsageError('Datastore contains no CLgen generator')

    toolchain_id = session.query(toolchain.Toolchain.id).filter(
        toolchain.Toolchain.string == 'opencl').first()
    if not toolchain_id:
      raise app.UsageError('Datastore contains no opencl toolchain')

    q = session.query(testcase.Testcase) \
      .filter(testcase.Testcase.generator_id == generator_id[0]) \
      .filter(testcase.Testcase.toolchain_id == toolchain_id[0]) \
      .order_by(testcase.Testcase.id)

    batches = sqlutil.OffsetLimitBatchedQuery(
        q,
        batch_size=FLAGS.batch_size,
        start_at=FLAGS.start_at,
        compute_max_rows=True)

    for batch in batches:
      logging.info('Batch %03d containing testcases %s..%s of %s',
                   batch.batch_num, humanize.intcomma(batch.offset),
                   humanize.intcomma(batch.limit),
                   humanize.intcomma(batch.max_rows))
      prefix = 'phd_experimental_deeplearning_clgen_'
      with tempfile.TemporaryDirectory(prefix=prefix) as d:
        d = pathlib.Path(d)
        paths_to_import = [CreateTempFileFromTestcase(d, r) for r in batch.rows]
        db.ImportStaticFeaturesFromPaths(paths_to_import, 'clgen_dsmith')
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
