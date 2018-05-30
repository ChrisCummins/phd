"""This file defines a database for pre-preprocessed content files."""
import binascii
import datetime
import pathlib

import sqlalchemy as sql
from absl import flags
from sqlalchemy.ext import declarative

from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import internal_pb2
from lib.labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class PreprocessedContentFile(Base):
  __tablename__ = 'preprocessed_contentfiles'

  # Relative path of the input file within the content files.
  input_relpath: str = sql.Column(sql.String(4096), primary_key=True)
  # Checksum of the input file.
  input_sha256: str = sql.Column(sql.Binary(32), nullable=False)
  # Checksum of the preprocessed file.
  sha256: str = sql.Column(sql.Binary(32), nullable=False, index=True)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(sql.UnicodeText(), nullable=False)
  # True if pre-processing succeeded, else False.
  preprocessing_succeeded: bool = sql.Column(sql.Boolean, nullable=False)
  # The number of milliseconds pre-preprocessing took.
  preprocess_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=datetime.datetime.utcnow)

  @property
  def input_sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of input_sha256."""
    return binascii.hexlify(self.input_sha256).decode('utf-8')

  @property
  def sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of sha256."""
    return binascii.hexlify(self.sha256).decode('utf-8')

  def FromContentFile(
      self,
      input: internal_pb2.CreatePreprocessedContentFile) -> 'PreprocessedContentFile':
    # TODO(cec):
    preprocessors.Preprocess()
    raise NotImplementedError


class PreprocessedContentFiles(sqlutil.Database):
  """A database of pre-processed contentfiles."""

  def __init__(self, path: pathlib.Path):
    super(PreprocessedContentFiles, self).__init__(path, Base)

  def IsDone(self, session: sqlutil.Database.session_t):
    if session.query(Meta).filter(Meta.key == 'done').first():
      return True
    else:
      return False

  def Create(self, contentfiles_root: pathlib.Path) -> None:
    with self.Session() as session:
      if self.IsDone(session):
        return
      else:
        pass  # TODO(cec): Implement!
