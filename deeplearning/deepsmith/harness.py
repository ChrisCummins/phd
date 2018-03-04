"""This file implements the testcase harness."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

from deeplearning.deepsmith import db
from deeplearning.deepsmith.protos import deepsmith_pb2


class Harness(db.Table):
  id_t = sql.Integer
  __tablename__ = "harnesses"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=db.now)
  name: str = sql.Column(sql.String(1024), nullable=False)
  version: str = sql.Column(sql.String(1024), nullable=False)

  # Relationships:
  testcases: typing.List["Testcase"] = orm.relationship(
      "Testcase", back_populates="harness")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('name', 'version', name='unique_harness'),
  )

  def SetProto(self, proto: deepsmith_pb2.Harness) -> deepsmith_pb2.Harness:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Harness message.
    """
    proto.name = self.name
    proto.version = self.version
    return proto

  def ToProto(self) -> deepsmith_pb2.Harness:
    """Create protocol buffer representation.

    Returns:
      A Harness message.
    """
    proto = deepsmith_pb2.Harness()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t,
               proto: deepsmith_pb2.Harness) -> "Harness":
    return db.GetOrAdd(
        session, cls,
        name=proto.name,
        version=proto.version,
    )
