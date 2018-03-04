"""This file implements the testcase harness."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import db


class Harness(db.Base):
  id_t = sql.Integer
  __tablename__ = "harnesses"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=db.now)
  name: str = sql.Column(sql.String(1024), nullable=False)
  version: str = sql.Column(sql.String(1024), nullable=False)

  # Relationships:
  testcases: typing.List[deeplearning.deepsmith.testcase.Testcase] = orm.relationship(
      "Testcase", back_populates="harness")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('name', 'version', name='unique_harness'),
  )
