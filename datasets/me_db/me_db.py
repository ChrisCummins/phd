"""me - Aggregate health and time tracking data."""
import datetime
import pathlib
import typing
from concurrent import futures

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


def CreateTasksFromInbox(inbox: pathlib.Path) -> typing.Iterator[
  importers.ImporterTask]:
  """Return ImporterTasks for all importers.

  Args:
    inbox: The inbox path.

  Returns:
    An iterator of ImporterTask objects.
  """
  if not inbox.is_dir():
    raise FileNotFoundError(f"Directory not found: '{inbox}'")
  yield from health_kit.CreateTasksFromInbox(inbox)
  yield from ynab.CreateTasksFromInbox(inbox)
  yield from life_cycle.CreateTasksFromInbox(inbox)


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
    super(Database, self).__init__(path, Base)

  @classmethod
  def AddMeasurementsFromImporterTasks(
      cls, session: sqlutil.Database.session_t,
      importer_tasks: importers.ImporterTasks):
    """Schedule and execute the given importer_tasks, and import to database."""
    with futures.ThreadPoolExecutor() as executor:
      scheduled_tasks = [executor.submit(task) for task in importer_tasks]
      logging.info('Submitted %d tasks', len(scheduled_tasks))

      for future in futures.as_completed(scheduled_tasks):
        executor.submit(cls.AddSeriesCollections(session, future.result()))

  @staticmethod
  def AddSeriesCollections(session: sqlutil.Database.session_t,
                           series_collections: typing.Iterator[
                             me_pb2.SeriesCollection]) -> None:
    """Import the given series_collections to database."""
    for proto in series_collections:
      for series in proto.series:
        logging.info('Importing %d %s:%s measurements',
                     len(series.measurement), series.family, series.name)
        session.add_all(MeasurementsFromSeries(series))

  def ImportMeasurementsFromInbox(self, inbox: pathlib.Path):
    """Import and commit measurements from inbox directory."""
    tasks = CreateTasksFromInbox(inbox)
    with self.Session(commit=True) as session:
      self.AddMeasurementsFromImporterTasks(session, tasks)
      logging.info('Committing records to database')


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
  logging.info('Using database `%s`', db.database_path)

  db.ImportMeasurementsFromInbox(inbox_path)


if __name__ == '__main__':
  app.run(main)
