"""This file implements profiling events."""
import datetime

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects import mysql

import deeplearning.deepsmith.client
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import labdate
from labm8 import sqlutil


class ProfilingEventType(db.StringTable):
  id_t = db.StringTable.id_t
  __tablename__ = 'proviling_event_types'


class TestcaseProfilingEvent(db.Table):
  id_t = sql.Integer
  __tablename__ = 'testcase_profiling_events'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  testcase_id: int = sql.Column(
      sql.Integer, sql.ForeignKey('testcases.id'), nullable=False)
  client_id: int = sql.Column(
      deeplearning.deepsmith.client.Client.id_t,
      sql.ForeignKey('clients.id'),
      nullable=False)
  type_id: int = sql.Column(
      ProfilingEventType.id_t,
      sql.ForeignKey('proviling_event_types.id'),
      nullable=False)
  duration_ms: int = sql.Column(sql.Integer, nullable=False)
  event_start: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False)

  # Relationships.
  testcase: 'deeplearning.deepsmith.testcase.Testcase' = orm.relationship(
      'Testcase', back_populates='profiling_events')
  client: deeplearning.deepsmith.client.Client = orm.relationship('Client')
  type: ProfilingEventType = orm.relationship('ProfilingEventType')

  # Constraints:
  __table_args__ = (sql.UniqueConstraint(
      'testcase_id',
      'client_id',
      'type_id',
      name='unique_testcase_profiling_event'),)

  def SetProto(self,
               proto: deepsmith_pb2.ProfilingEvent) -> \
      deepsmith_pb2.ProfilingEvent:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A ProfilingEvent message.
    """
    proto.client = self.client.string
    proto.type = self.type.string
    proto.duration_ms = self.duration_ms
    proto.event_start_epoch_ms = labdate.MillisecondsTimestamp(self.event_start)
    return proto

  def ToProto(self) -> deepsmith_pb2.ProfilingEvent:
    """Create protocol buffer representation.

    Returns:
      A ProfilingEvent message.
    """
    proto = deepsmith_pb2.ProfilingEvent()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t, proto: deepsmith_pb2.ProfilingEvent,
               testcase: 'testcase.Testcase') -> 'ProfilingEvent':
    return sqlutil.GetOrAdd(
        session,
        cls,
        testcase=testcase,
        client=deeplearning.deepsmith.client.Client.GetOrAdd(
            session, proto.client),
        type=ProfilingEventType.GetOrAdd(session, proto.type),
        duration_ms=proto.duration_ms,
        event_start=labdate.DatetimeFromMillisecondsTimestamp(
            proto.event_start_epoch_ms))


class ResultProfilingEvent(db.Table):
  id_t = sql.Integer
  __tablename__ = 'result_profiling_events'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  result_id: int = sql.Column(
      sql.Integer, sql.ForeignKey('results.id'), nullable=False)
  client_id: int = sql.Column(
      deeplearning.deepsmith.client.Client.id_t,
      sql.ForeignKey('clients.id'),
      nullable=False)
  type_id: int = sql.Column(
      ProfilingEventType.id_t,
      sql.ForeignKey('proviling_event_types.id'),
      nullable=False)
  duration_ms: int = sql.Column(sql.Integer, nullable=False)
  event_start: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False)

  # Relationships.
  result: 'deeplearning.deepsmith.result.Result' = orm.relationship(
      'Result', back_populates='profiling_events')
  client: deeplearning.deepsmith.client.Client = orm.relationship('Client')
  type: ProfilingEventType = orm.relationship('ProfilingEventType')

  # Constraints:
  __table_args__ = (sql.UniqueConstraint(
      'result_id', 'client_id', 'type_id',
      name='unique_result_profiling_event'),)

  def SetProto(self,
               proto: deepsmith_pb2.ProfilingEvent) -> \
      deepsmith_pb2.ProfilingEvent:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A ProfilingEvent message.
    """
    proto.client = self.client.string
    proto.type = self.type.string
    proto.duration_ms = self.duration_ms
    proto.event_start_epoch_ms = labdate.MillisecondsTimestamp(self.event_start)
    return proto

  def ToProto(self) -> deepsmith_pb2.ProfilingEvent:
    """Create protocol buffer representation.

    Returns:
      A ProfilingEvent message.
    """
    proto = deepsmith_pb2.ProfilingEvent()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t, proto: deepsmith_pb2.ProfilingEvent,
               result: 'result.Result') -> 'ProfilingEvent':
    return sqlutil.GetOrAdd(
        session,
        cls,
        result=result,
        client=deeplearning.deepsmith.client.Client.GetOrAdd(
            session, proto.client),
        type=ProfilingEventType.GetOrAdd(session, proto.type),
        duration_ms=proto.duration_ms,
        event_start=labdate.DatetimeFromMillisecondsTimestamp(
            proto.event_start_epoch_ms))
