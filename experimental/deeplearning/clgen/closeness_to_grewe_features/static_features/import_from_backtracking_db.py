"""Import from backtracking database."""
import multiprocessing
import pathlib
import tempfile
import typing

from deeplearning.clgen.preprocessors import preprocessors
from experimental.deeplearning.clgen.backtracking import backtracking_db
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from labm8.py import app
from labm8.py import sqlutil

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')
app.DEFINE_string(
    'backtracking_db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/backtracking/db.db',
    'URL of the backtracking database.')
app.DEFINE_integer('batch_size', 512, 'Number of samples to make per batch.')
app.DEFINE_string('origin', 'clgen_backtracking',
                  'Name of the origin of the kernels, e.g. "clgen".')


def _PrettifySource(src: str) -> str:
  """Format an OpenCL source."""
  return preprocessors.Preprocess(src, [
      'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
      'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
      'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
      'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
      'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
      'deeplearning.clgen.preprocessors.cxx:ClangFormat',
  ])


def ProcessBatch(batch: typing.List[typing.Tuple[str,]],
                 db: grewe_features_db.Database, pool: multiprocessing.Pool):
  app.Log(1, 'Formatting files')
  srcs = pool.imap_unordered(_PrettifySource, [row[0] for row in batch])
  srcs = [s for s in srcs if s]

  with tempfile.TemporaryDirectory(prefix='phd_clgen_import_') as d:
    d = pathlib.Path(d)
    paths = [d / f'{i}.cl' for i in range(1, len(srcs) + 1)]
    for path, src in zip(paths, srcs):
      with open(path, 'w') as f:
        f.write(src)

    app.Log(1, 'Importing files')
    success_count, new_row_count = db.ImportStaticFeaturesFromPaths(
        paths, FLAGS.origin, pool)
    app.Log(1, 'Extracted features from %d of %d kernels, %d new rows',
            success_count, len(batch), new_row_count)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  bdb = backtracking_db.Database(FLAGS.backtracking_db)

  with bdb.Session() as s, multiprocessing.Pool() as pool:
    batches = sqlutil.OffsetLimitBatchedQuery(s.query(
        backtracking_db.BacktrackingStep.src),
                                              batch_size=FLAGS.batch_size,
                                              compute_max_rows=True)
    for batch in batches:
      app.Log(1, 'Batch %d of %d rows', batch.batch_num, batch.max_rows)
      ProcessBatch(batch.rows, db, pool)


if __name__ == '__main__':
  app.RunWithArgs(main)
