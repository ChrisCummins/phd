"""me - Aggregate health and time tracking data."""
import datetime
import multiprocessing
import pathlib
import time
import typing

import sqlalchemy as sql
from absl import app
from absl import flags
from absl import logging
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from datasets.me_db import importers
from datasets.me_db import me_pb2
from datasets.me_db.health_kit import health_kit
from datasets.me_db.life_cycle import life_cycle
from datasets.me_db.ynab import ynab
from labm8 import labdate
from labm8 import sqlutil


FLAGS = flags.FLAGS

flags.DEFINE_string('inbox', None, 'Path to inbox.')
flags.DEFINE_string('db', 'me.db', 'Path to database.')
flags.DEFINE_bool('replace_existing', False,
                  'Remove existing database, if it exists.')

Base = declarative.declarative_base()

# The list of inbox importers. An inbox importer is a function that takes a
# path to a directory (the inbox) and a Queue. The function, when called, must
# place a single SeriesCollection proto on the queue.
INBOX_IMPORTERS: typing.List[importers.InboxImporter] = [
  health_kit.ProcessInboxToQueue,
  life_cycle.ProcessInboxToQueue,
  ynab.ProcessInboxToQueue,
]


class Measurement(Base):
  """The measurements table.

  A row in the measurements table is a concatenation of a me.Measurement proto,
  and the non-measurement fields from a me.Series proto.
  """
  __tablename__ = 'measurements'

  id: int = sql.Column(sql.Integer, primary_key=True)

  series: str = sql.Column(sql.String(512), nullable=False)
  date: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False)
  family: str = sql.Column(sql.String(512), nullable=False)
  group: str = sql.Column(sql.String(512), nullable=False)
  value: int = sql.Column(sql.Integer, nullable=False)
  unit: str = sql.Column(sql.String(512), nullable=False)
  source: str = sql.Column(sql.String(512), nullable=False)

  def __repr__(self):
    return f"{self.family}:{self.series} {self.value} {self.unit} {self.date}"


def MeasurementsFromSeries(series: me_pb2.Series) -> typing.List[Measurement]:
  """Create a list of measurements from a me.Series proto."""
  return [
    Measurement(series=series.name,
                date=labdate.DatetimeFromMillisecondsTimestamp(
                    m.ms_since_unix_epoch),
                family=series.family,
                group=m.group,
                value=m.value,
                unit=series.unit,
                source=m.source)
    for m in series.measurement]


class Database(sqlutil.Database):

  def __init__(self, path: pathlib.Path):
    super(Database, self).__init__(
        f'sqlite:///{path}', Base, create_if_not_exist=True)

  @staticmethod
  def AddSeriesCollection(session: sqlutil.Database.session_t,
                          series_collection: me_pb2.SeriesCollection) -> None:
    """Import the given series_collections to database."""
    for series in series_collection.series:
      logging.info('Importing %d %s:%s measurements',
                   len(series.measurement), series.family, series.name)
      session.add_all(MeasurementsFromSeries(series))

  def ImportMeasurementsFromInboxImporters(
      self, inbox: pathlib.Path, inbox_importers: typing.Iterator[
        importers.InboxImporter] = INBOX_IMPORTERS):
    """Import and commit measurements from inbox directory."""
    start_time = time.time()
    queue = multiprocessing.Queue()

    # Start each importer in a separate process.
    processes = []
    for importer in inbox_importers:
      process = multiprocessing.Process(target=importer, args=(inbox, queue))
      process.start()
      processes.append(process)
    logging.info('Started %d importer processes', len(processes))

    # Get the results of each process as it finishes.
    for _ in range(len(processes)):
      series_collections = queue.get()
      with self.Session(commit=True) as session:
        self.AddSeriesCollection(session, series_collections)

    logging.info('Processed records in %s seconds', time.time() - start_time)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')

  db_path = pathlib.Path(FLAGS.db)
  inbox_path = pathlib.Path(FLAGS.inbox)

  # Validate the flags paths.
  if not inbox_path.is_dir():
    raise app.UsageError(f'Inbox is not a directory: "{inbox_path}"')

  if not db_path.parent.is_dir():
    raise app.UsageError(
        f'Database path parent is not a directory: "{db_path.parent}"')

  # Remove the existing database if requested.
  if FLAGS.replace_existing and db_path.exists():
    db_path.unlink()

  db = Database(db_path)
  logging.info('Using database `%s`', db.url)

  db.ImportMeasurementsFromInboxImporters(inbox_path)


if __name__ == '__main__':
  app.run(main)
