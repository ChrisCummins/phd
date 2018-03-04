"""This file defines the testcase generator."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2


class Generator(db.Table):
  id_t = sql.Integer
  __tablename__ = "generators"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  name: str = sql.Column(sql.String(1024), nullable=False)
  version: str = sql.Column(sql.String(1024), nullable=False)

  # Relationships:
  testcases: typing.List["Testcase"] = orm.relationship(
      "Testcase", back_populates="generator")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('name', 'version', name='unique_generator'),
  )

  def SetProto(self, proto: deepsmith_pb2.Generator) -> deepsmith_pb2.Generator:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Generator message.
    """
    proto.name = self.name
    proto.version = self.version
    return proto

  def ToProto(self) -> deepsmith_pb2.Generator:
    """Create protocol buffer representation.
    
    Returns:
      A Generator message.
    """
    proto = deepsmith_pb2.Generator()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t,
               proto: deepsmith_pb2.Generator) -> "Generator":
    return db.GetOrAdd(
        session, cls,
        name=proto.name,
        version=proto.version,
    )
