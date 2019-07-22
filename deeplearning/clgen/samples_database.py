"""A module for databases of CLgen samples."""

import time

import contextlib
import datetime
import sqlalchemy as sql
import typing
from sqlalchemy.ext import declarative

from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import model_pb2
from labm8 import app
from labm8 import crypto
from labm8 import labdate
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Sample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__ = 'samples'
  proto_t = model_pb2.Sample

  id: int = sql.Column(sql.Integer, primary_key=True)
  text: str = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)
  # Checksum of the sample text.
  sha256: str = sql.Column(sql.String(64), nullable=False, index=True)
  num_tokens: int = sql.Column(sql.Integer, nullable=False)
  sample_time_ms: int = sql.Column(sql.Integer, nullable=False)
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  sample_date: datetime.datetime = sql.Column(sql.DateTime, nullable=False)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  def SetProto(self, proto: model_pb2.Sample) -> None:
    proto.text = self.text
    proto.num_tokens = self.num_tokens
    proto.wall_time_ms = self.wall_time_ms
    proto.sample_start_epoch_ms_utc = labdate.MillisecondsTimestamp(
        self.sample_date)

  @classmethod
  def FromProto(cls, proto: model_pb2.Sample) -> typing.Dict[str, typing.Any]:
    return {
        'text':
        proto.text,
        'sha256':
        crypto.sha256_str(proto.text),
        'num_tokens':
        proto.num_tokens,
        'sample_time_ms':
        proto.sample_time_ms,
        'wall_time_ms':
        proto.wall_time_ms,
        'sample_date':
        labdate.DatetimeFromMillisecondsTimestamp(
            proto.sample_start_epoch_ms_utc),
    }


class SamplesDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(SamplesDatabase, self).__init__(url, Base, must_exist=must_exist)

  @contextlib.contextmanager
  def Observer(self) -> sample_observers.SampleObserver:
    """Return an observer that imports samples into database."""
    observer = SamplesDatabaseObserver(self)
    yield observer
    observer.Flush()


class SamplesDatabaseObserver(sample_observers.SampleObserver):
  """A sample observer that imports samples to a database.

  The observer buffers the records that it recieves and commits them to the
  database in batches.
  """

  def __init__(self,
               db: SamplesDatabase,
               commit_seconds_frequency: int = 30,
               commit_sample_frequency: int = 1024):
    self._db = db
    self._last_commit = time.time()
    self._to_commit = []
    self._commit_seconds_frequency = commit_seconds_frequency
    self._commit_sample_frequency = commit_sample_frequency

  def __del__(self):
    self.Flush()

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback."""
    self._to_commit.append(Sample(**Sample.FromProto(sample)))

    # Commit records if required.
    if (len(self._to_commit) > self._commit_sample_frequency or
        (time.time() - self._last_commit) > self._commit_seconds_frequency):
      self.Flush()

    return True

  def Flush(self) -> None:
    """Commit all pending records to database."""
    with self._db.Session(commit=True) as session:
      session.add_all(self._to_commit)
    self._to_commit = []
    self._last_commit = time.time()
