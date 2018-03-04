"""This file defines the testcase generator."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

from deeplearning.deepsmith import db


class Generator(db.Base):
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
