"""A command-line interface for importing protos to the datastore."""
import pathlib
from absl import app
from absl import flags
from absl import logging

import deeplearning.deepsmith.result
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db

FLAGS = flags.FLAGS

flags.DEFINE_list('results', [], 'Result proto paths to import')
flags.DEFINE_string('results_dir', None, 'Directory containing result protos')
flags.DEFINE_list('testcases', [], 'Testcase proto paths to import')
flags.DEFINE_string('testcases_dir', None, 'Directory containing testcase protos')


def ImportResultsFromDirectory(session: db.session_t,
                               results_dir: pathlib.Path) -> None:
  """Import Results from a directory of protos.

  Args:
    session: A database session.
    results_dir: Directory containing (only) Result protos.
  """
  batch_size = 1000
  if not results_dir.is_dir():
    logging.fatal('directory %s does not exist', results_dir)
  for i, path in enumerate(results_dir.iterdir()):
    deeplearning.deepsmith.result.Result.FromFile(session, path)
    if not i % batch_size:
      session.commit()
  session.commit()


def ImportTestcasesFromDirectory(session: db.session_t,
                                 testcases_dir: pathlib.Path) -> None:
  """Import Testcases from a directory of protos.

  Args:
    session: A database session.
    testcases_dir: Directory containing (only) Testcase protos.
  """
  batch_size = 1000
  if not testcases_dir.is_dir():
    logging.fatal('directory %s does not exist', testcases_dir)
  for i, path in enumerate(testcases_dir.iterdir()):
    deeplearning.deepsmith.testcase.Testcase.FromFile(session, path)
    if not i % batch_size:
      session.commit()
  session.commit()


def main(argv):
  del argv
  ds = datastore.DataStore.FromFlags()
  with ds.Session(commit=True) as session:
    for path in FLAGS.results:
      deeplearning.deepsmith.result.Result.FromFile(session, pathlib.Path(path))
    session.commit()
    if FLAGS.results_dir:
      ImportResultsFromDirectory(session, pathlib.Path(FLAGS.results_dir))
    for path in FLAGS.testcases:
      deeplearning.deepsmith.testcase.Testcase.FromFile(session, pathlib.Path(path))
    session.commit()
    if FLAGS.testcases_dir:
      ImportTestcasesFromDirectory(session, pathlib.Path(FLAGS.testcases_dir))
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
