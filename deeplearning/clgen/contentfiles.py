"""This file defines a database for corpus contentfiles."""

import binascii
import datetime
import pathlib

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from lib.labm8 import sqlutil


Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class ContentFile(Base):
  __tablename__ = 'contentfiles'

  id: int = sql.Column(sql.Integer, primary_key=True)
  # Relative path of input file with the corpus.
  relpath: str = sql.Column(sql.String(1024), nullable=False, index=True)
  # Checksum of the input file.
  sha256_input: str = sql.Column(sql.Binary(32), nullable=False)
  # Checksum of the preprocessed file.
  sha256: str = sql.Column(sql.Binary(32), nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(sql.UnicodeText(), nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=datetime.datetime.utcnow)

  @property
  def sha256_input_hex(self) -> str:
    """Return the 64 character hexadecimal representation of sha256_input."""
    return binascii.hexlify(self.sha256).decode('utf-8')

  @property
  def sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of sha256."""
    return binascii.hexlify(self.sha256).decode('utf-8')


class ContentFiles(sqlutil.Database):
  """A database of contentfiles."""

  def __init__(self, path: pathlib.Path):
    super(ContentFiles, self).__init__(path, Base)
