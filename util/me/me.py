"""me - Aggregate health and time tracking data."""

import pathlib
import typing
from absl import app
from absl import flags
from absl import logging

from util.me import db
from util.me import importers
from util.me.life_cycle import life_cycle
from util.me.ynab import ynab


FLAGS = flags.FLAGS

flags.DEFINE_string('inbox', None, 'Path to inbox.')
flags.DEFINE_string('db', 'me.db',
                    'Path to database.')


def CreateTasksFromInbox(inbox: pathlib.Path) -> typing.Iterator[
  importers.ImporterTask]:
  yield from life_cycle.CreateTasksFromInbox(inbox)
  yield from ynab.CreateTasksFromInbox(inbox)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')

  db_ = db.Database(pathlib.Path(FLAGS.db))
  logging.info('Using database `%s`', db_.database_path)

  tasks = CreateTasksFromInbox(pathlib.Path(FLAGS.inbox))
  with db_.Session(commit=True) as session:
    db_.AddMeasurementsFromImporterTasks(session, tasks)


if __name__ == '__main__':
  app.run(main)
