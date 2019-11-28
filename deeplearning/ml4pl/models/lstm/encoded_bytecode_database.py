"""Database backend for encoded graphs."""
import codecs
import pickle
import typing

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from labm8.py import app
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class EncodedBytecode(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for an encoded bytecode."""
  # Same as input graph ID.
  bytecode_id: int = sql.Column(sql.Integer, primary_key=True)
  binary_encoded_sequence: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                              nullable=False)
  binary_segment_ids: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                         nullable=False)
  binary_node_mask: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                       nullable=False)

  @property
  def encoded_sequence(self) -> typing.Any:
    return pickle.loads(codecs.decode(self.binary_encoded_sequence, 'zlib'))

  @encoded_sequence.setter
  def encoded_sequence(self, data) -> None:
    self.binary_encoded_sequence = codecs.encode(pickle.dumps(data), 'zlib')

  @property
  def segment_ids(self) -> typing.Any:
    return pickle.loads(codecs.decode(self.binary_segment_ids, 'zlib'))

  @segment_ids.setter
  def segment_ids(self, data) -> None:
    self.binary_segment_ids = codecs.encode(pickle.dumps(data), 'zlib')

  @property
  def node_mask(self) -> typing.Any:
    return pickle.loads(codecs.decode(self.binary_node_mask, 'zlib'))

  @node_mask.setter
  def node_mask(self, data) -> None:
    self.binary_node_mask = codecs.encode(pickle.dumps(data), 'zlib')


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
