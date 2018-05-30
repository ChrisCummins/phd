"""This file defines a database for pre-preprocessed content files."""
import binascii
import datetime
import multiprocessing
import pathlib
import time

import numpy as np
import progressbar
import sqlalchemy as sql
from absl import flags
from sqlalchemy.ext import declarative

from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import preprocessed
from lib.labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class EncodedContentFile(Base):
  __tablename__ = 'encoded_contentfiles'

  # Checksum of the input preprocessed file.
  sha256: str = sql.Column(sql.Binary(32), primary_key=True)
  data: str = sql.Column(sql.Binary(), nullable=False)
  # The number of milliseconds encoding took.
  encoding_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @property
  def indices_array(self) -> np.ndarray:
    return np.fromstring(self.data, dtype=np.int32)

  @property
  def sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of sha256."""
    return binascii.hexlify(self.sha256).decode('utf-8')

  @classmethod
  def FromPreprocessed(
      cls, preprocessed_db: preprocessed.PreprocessedContentFile,
      atomizer: atomizers.AtomizerBase) -> 'EncodedContentFile':
    start_time = time.time()
    data = atomizer.AtomizeString(preprocessed_db.text)
    encoding_time_ms = (time.time() - start_time) * 1000
    return EncodedContentFile(sha256=preprocessed_db.sha256, data=data,
                              encoding_time_ms=encoding_time_ms,
                              date_added=datetime.datetime.utcnow())


class EncodedContentFiles(sqlutil.Database):
  """A database of pre-processed contentfiles."""

  def __init__(self, path: pathlib.Path, atomizer: atomizers.AtomizerBase):
    super(EncodedContentFiles, self).__init__(path, Base)
    self.atomizer = atomizers

  def Create(self, p: preprocessed.PreprocessedContentFiles,
             atomizer: atomizers.AtomizerBase) -> bool:
    """Create the

    Args:
      p: A PreprocessedContentFiles database.
      atomizer: An AtomizerBase instance.

    Returns:
      True if work was done, else False.
    """
    with self.Session() as session:
      if self.IsDone(session):
        return False
      else:
        self.Import(session, p, atomizer)
        self.SetDone(session)
        session.commit()
        return True

  def IsDone(self, session: sqlutil.Database.session_t):
    if session.query(Meta).filter(Meta.key == 'done').first():
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Database.session_t):
    session.add(Meta(key='done', value='yes'))

  def Import(self, session: sqlutil.Database.session_t,
             preprocessed_db: preprocessed.PreprocessedContentFiles) -> None:
    with preprocessed_db.Session() as p_session:
      query = p_session.query(preprocessed.PreprocessedContentFile).filter(
          ~preprocessed.PreprocessedContentFile.sha256.in_(
              session.query(EncodedContentFile).sha256))
      pool = multiprocessing.Pool()
      bar = progressbar.ProgressBar()
      for encoded_cfg in bar(pool.imap_unordered(
          lambda x: EncodedContentFile.FromPreprocessed(x, self.atomizer),
          query)):
        session.add(encoded_cfg)
