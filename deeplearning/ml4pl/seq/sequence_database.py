"""A database backend for storing sequences."""
import codecs
import pickle
from typing import Dict
from typing import Optional

import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from labm8.py import app
from labm8.py import crypto
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class SequenceEncoder(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """An encoder for graphs to sequences."""

  id: int = sql.Column(sql.Integer, primary_key=True)

  vocab_sha1: str = 1
  binary_vocab: bytes = 2

  @property
  def vocab(self) -> Dict[str, int]:
    return pickle.loads(self.binary_vocab)

  @classmethod
  def Create(cls, vocab: Dict[str, int]):
    binary_vocab = pickle.dumps(vocab)
    return cls(vocab_sha1=crypto.sha1(binary_vocab), binary_vocab=binary_vocab)


class Sequence(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for an encoded sequence."""

  ir_id: int = sql.Column(
    sql.Integer, nullable=False, index=True,
  )

  # The sequence encoder.
  encoder_id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey(
      "sequence_encoders.id", onupdate="CASCADE", ondelete="CASCADE"
    ),
    index=True,
  )
  encoder: SequenceEncoder = sql.orm.relationship(
    "SequenceEncoder",
    uselist=False,
    single_parent=True,
    cascade="all, delete-orphan",
  )

  sequence_length: int = sql.Column(sql.Integer, nullable=False)
  vocab_size: int = sql.Column(sql.Integer, nullable=False)

  binary_encoded_sequence: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )
  binary_segment_ids: Optional[bytes] = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )
  binary_node_mask: Optional[bytes] = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @property
  def encoded_sequence(self) -> np.array:
    """Return the encoded sequence, with shape (sequence_length, vocab_size)"""
    return pickle.loads(codecs.decode(self.binary_encoded_sequence, "zlib"))

  @property
  def segment_ids(self) -> np.array:
    return pickle.loads(codecs.decode(self.binary_segment_ids, "zlib"))

  @property
  def node_mask(self) -> np.array:
    return pickle.loads(codecs.decode(self.binary_node_mask, "zlib"))

  @classmethod
  def Create(
    cls,
    ir_id: int,
    encoded_sequence: np.array,
    segment_ids: np.array,
    node_mask: np.array,
  ):
    return cls(
      ir_id=ir_id,
      binary_encoded_sequence=codecs.encode(
        pickle.dumps(encoded_sequence), "zlib"
      ),
      binary_segment_ids=codecs.encode(pickle.dumps(segment_ids), "zlib"),
      binary_node_mask=codecs.encode(pickle.dumps(node_mask), "zlib"),
    )


###############################################################################
# Database.
###############################################################################


class Database(sqlutil.Database):
  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
