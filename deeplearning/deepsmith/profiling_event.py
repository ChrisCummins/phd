"""This file implements profiling events."""
from datetime import datetime

from sqlalchemy import Integer, Column, DateTime, ForeignKey, Float, UniqueConstraint
from sqlalchemy.orm import relationship

import deeplearning.deepsmith.client
import deeplearning.deepsmith.result
import deeplearning.deepsmith.testcase

from deeplearning.deepsmith import db


class ProfilingEventName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "proviling_event_names"


class TestcaseTiming(db.Base):
  id_t = Integer
  __tablename__ = "testcase_timings"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  testcase_id: int = Column(deeplearning.deepsmith.testcase.Testcase.id_t,
                            ForeignKey("testcases.id"), nullable=False)
  name_id: int = Column(ProfilingEventName.id_t,
                        ForeignKey("proviling_event_names.id"), nullable=False)
  client_id: int = Column(deeplearning.deepsmith.client.Client.id_t,
                          ForeignKey("clients.id"), nullable=False)
  duration_seconds: float = Column(Float, nullable=False)
  date: datetime = Column(DateTime, nullable=False)

  # Relationships:
  testcase: deeplearning.deepsmith.testcase.Testcase = relationship(
      "Testcase", back_populates="timings")
  name: ProfilingEventName = relationship("ProfilingEventName")
  client: deeplearning.deepsmith.client.Client = relationship("Client")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('testcase_id', 'name_id', 'client_id',
                     name='unique_testcase_timing'),
  )


class ResultTiming(db.Base):
  id_t = Integer
  __tablename__ = "result_timings"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  result_id: int = Column(deeplearning.deepsmith.result.Result.id_t,
                          ForeignKey("results.id"), nullable=False)
  name_id: int = Column(ProfilingEventName.id_t,
                        ForeignKey("proviling_event_names.id"), nullable=False)
  client_id: int = Column(deeplearning.deepsmith.client.Client.id_t,
                          ForeignKey("clients.id"), nullable=False)
  duration_seconds: float = Column(Float, nullable=False)
  date: datetime = Column(DateTime, nullable=False)

  # Relationships:
  result: deeplearning.deepsmith.result.Result = relationship(
      "Result", back_populates="timings")
  name: ProfilingEventName = relationship("ProfilingEventName")
  client: deeplearning.deepsmith.client.Client = relationship("Client")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('result_id', 'name_id', name='unique_result_timing'),
  )
