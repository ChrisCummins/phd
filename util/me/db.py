"""An SQL backend for Series data."""
import datetime
import pathlib
import sqlalchemy as sql
import typing
from absl import flags
from absl import logging
from concurrent import futures
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from lib.labm8 import labdate
from lib.labm8 import sqlutil
from util.me import importers
from util.me import me_pb2


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  """The database metadata table."""
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class Measurement(Base):
  """The measurements table.

  A row in the measurements table is a concatenation of a me.Measurement proto,
  and the non-measurement fields from a me.Series proto.
  """
  __tablename__ = 'measurements'

  id: int = sql.Column(sql.Integer, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False, default=labdate.GetUtcMillisecondsNow)

  series: str = sql.Column(sql.String(512), nullable=False)
  date: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False)
  family: str = sql.Column(sql.String(512), nullable=False)
  group: str = sql.Column(sql.String(512), nullable=False)
  value: int = sql.Column(sql.Integer, nullable=False)
  unit: str = sql.Column(sql.String(512), nullable=False)
  source: str = sql.Column(sql.String(512), nullable=False)


def MeasurementsFromSeries(series: me_pb2.Series) -> typing.List[Measurement]:
  """Create a list of measurements from a me.Series proto."""
  return [
    Measurement(series=series.name,
                date=labdate.DatetimeFromMillisecondsTimestamp(
                    m.ms_since_epoch_utc),
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
        logging.info('Importing %d %s measurements',
                     len(series.measurement), series.name)
        session.add_all(MeasurementsFromSeries(series))
