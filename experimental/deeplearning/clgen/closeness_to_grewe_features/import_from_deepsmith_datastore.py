"""Import OpenCL kernels from DeepSmith datastore."""
import pathlib
import typing

from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith import db as dsmith_db
from labm8 import pbutil
from labm8 import sqlutil


FLAGS = flags.FLAGS

flags.DEFINE_string('datastore', None,
                    'Path of the datastore config to import form')
flags.DEFINE_integer('batch_size', 64,
                     'The number of testcases to process in a batch.')
flags.DEFINE_integer('start_at', 0,
                     'The initial offset into the results set.')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.datastore:
    raise app.UsageError('--datastore flag is required')

  config = pbutil.FromFile(pathlib.Path(FLAGS.datastore))
  db = dsmith_db.MakeEngine(config)

  with db.Session(commit=False) as session:
    q = session.query(testcase.Testcase) \
      .filter(testcase.Testcase.generator.name == 'clgen') \
      .filter(testcase.Testcase.toolchain.name == 'opencl')

    batches = sqlutil.OffsetLimitBatchedQuery(
        q, batch_size=FLAGS.batch_size, start_at=FLAGS.start_at,
        compute_max_rows=True)

    for batch in batches:
      logging.info('Batch %03d of testcases %s..%s of %s',
                   batch.batch_num, batch.offset, batch.limit, batch.max_rows)
      for testcase in batch.rows:
        print(str(testcase[:10]))
        # TODO(cec): Write a tempfile, then import into db.


if __name__ == '__main__':
  app.run(main)
