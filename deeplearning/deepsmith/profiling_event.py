"""This file implements profiling events."""
import datetime

import sqlalchemy as sql
from sqlalchemy import orm

import deeplearning.deepsmith.client
import deeplearning.deepsmith.result
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import db


class ProfilingEventType(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "proviling_event_types"


class TestcaseProfilingEvent(db.Table):
  id_t = sql.Integer
  __tablename__ = "testcase_profiling_events"

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=db.now)
  testcase_id: int = sql.Column(deeplearning.deepsmith.testcase.Testcase.id_t,
                                sql.ForeignKey("testcases.id"), nullable=False)
  client_id: int = sql.Column(deeplearning.deepsmith.client.Client.id_t,
                              sql.ForeignKey("clients.id"), nullable=False)
  type_id: int = sql.Column(ProfilingEventType.id_t,
                            sql.ForeignKey("proviling_event_types.id"),
                            nullable=False)
  duration_seconds: float = sql.Column(sql.Float, nullable=False)
  date: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  # Relationships.
  testcase: deeplearning.deepsmith.testcase.Testcase = orm.relationship(
      "Testcase", back_populates="profiling_events")
  client: deeplearning.deepsmith.client.Client = orm.relationship("Client")
  type: ProfilingEventType = orm.relationship("ProfilingEventType")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint(
        'testcase_id', 'client_id', 'type_id', name='unique_testcase_timing'),
  )


# class ResultTiming(db.Table):
#   id_t = sql.Integer
#   __tablename__ = "result_timings"
#
#   # Columns:
#   id: int = sql.Column(id_t, primary_key=True)
#   date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False, default=db.now)
#   result_id: int = sql.Column(deeplearning.deepsmith.result.Result.id_t,
#                               sql.ForeignKey("results.id"), nullable=False)
#   name_id: int = sql.Column(ProfilingEventType.id_t,
#                             sql.ForeignKey("proviling_event_names.id"), nullable=False)
#   client_id: int = sql.Column(deeplearning.deepsmith.client.Client.id_t,
#                               sql.ForeignKey("clients.id"), nullable=False)
#   duration_seconds: float = sql.Column(sql.Float, nullable=False)
#   date: datetime.datetime = sql.Column(sql.DateTime, nullable=False)
#
#   # Relationships:
#   result: deeplearning.deepsmith.result.Result = orm.relationship(
#       "Result", back_populates="timings")
#   name: ProfilingEventType = orm.relationship("ProfilingEventType")
#   client: deeplearning.deepsmith.client.Client = orm.relationship("Client")
#
#   # Constraints:
#   __table_args__ = (
#     sql.UniqueConstraint('result_id', 'name_id', name='unique_result_timing'),
#   )
