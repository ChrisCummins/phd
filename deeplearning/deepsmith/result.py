"""This file defines the result class."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

import deeplearning.deepsmith.testbed
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import db


class Result(db.Table):
  id_t = sql.Integer
  __tablename__ = "results"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  testcase_id: int = sql.Column(
      deeplearning.deepsmith.testcase.Testcase.id_t,
      sql.ForeignKey("testcases.id"), nullable=False)
  testbed_id: int = sql.Column(
      deeplearning.deepsmith.testbed.Testbed.id_t,
      sql.ForeignKey("testbeds.id"), nullable=False)
  returncode: int = sql.Column(sql.SmallInteger, nullable=False)

  # Relationships:
  testcase: deeplearning.deepsmith.testcase.Testcase = orm.relationship(
      "Testcase", back_populates="results")
  testbed: deeplearning.deepsmith.testbed.Testbed = orm.relationship(
      "Testbed", back_populates="results")
  outputs = orm.relationship(
      "ResultOutput", secondary="result_output_associations",
      primaryjoin="ResultOutputAssociation.result_id == Result.id",
      secondaryjoin="ResultOutputAssociation.output_id == ResultOutput.id")
  # timings: typing.List["ResultTiming"] = orm.relationship(
  #     "ResultTiming", back_populates="result")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('testcase_id', 'testbed_id', name='unique_result'),
  )


class PendingResult(db.Table):
  """A pending result is created when a testcase is issued to a testbed.

  It is used to prevent a testcase from being issued to the same testbed
  multiple times. When a testbed requests a testcase, a PendingResult is
  created. Pending results have a deadline by which the result is expected.
  The testcase will not be issued again to a matching testbed until this
  deadline has passed.

  PendingResults are removed in two cases:
    - A Result is received with the same testcase and testbed.
    - The deadline passes (this is to prevent the result being permanently
      lost in case of a testbed which never responds with a result).
  """
  id_t = Result.id_t
  __tablename__ = "pending_results"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  # The date that the result is due by.
  deadline: datetime.datetime = sql.Column(sql.DateTime, nullable=False)
  testcase_id: int = sql.Column(
      deeplearning.deepsmith.testcase.Testcase.id_t,
      sql.ForeignKey("testcases.id"), nullable=False)
  testbed_id: int = sql.Column(
      deeplearning.deepsmith.testbed.Testbed.id_t,
      sql.ForeignKey("testbeds.id"), nullable=False)

  # Relationships:
  testcase: deeplearning.deepsmith.testcase.Testcase = orm.relationship(
      "Testcase", back_populates="pending_results")
  testbed: deeplearning.deepsmith.testbed.Testbed = orm.relationship(
      "Testbed", back_populates="pending_results")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint(
        'testcase_id', 'testbed_id', name='unique_pending_result'),
  )


class ResultOutputName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "result_output_names"

  # Relationships:
  outputs: typing.List["ResultOutput"] = orm.relationship(
      "ResultOutput", back_populates="name")


class ResultOutput(db.Table):
  id_t = sql.Integer
  __tablename__ = "result_outputs"

  # Truncate everything after
  MAX_LEN = 128000

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  name_id: ResultOutputName.id_t = sql.Column(
      ResultOutputName.id_t, sql.ForeignKey("result_output_names.id"),
      nullable=False)
  original_sha1: str = sql.Column(sql.String(40), nullable=False, index=True)
  original_linecount = sql.Column(sql.Integer, nullable=False)
  original_charcount = sql.Column(sql.Integer, nullable=False)
  truncated_output: str = sql.Column(
      sql.UnicodeText(length=MAX_LEN), nullable=False)
  truncated: bool = sql.Column(sql.Boolean, nullable=False)
  truncated_linecount = sql.Column(sql.Integer, nullable=False)
  truncated_charcount = sql.Column(sql.Integer, nullable=False)

  # Relationships:
  name: ResultOutputName = orm.relationship(
      "ResultOutputName", back_populates="outputs")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint(
        "name_id", "original_sha1", name="unique_result_output"),
  )


class ResultOutputAssociation(db.Table):
  __tablename__ = "result_output_associations"

  # Columns:
  result_id: int = sql.Column(Result.id_t, sql.ForeignKey("results.id"),
                              nullable=False)
  output_id: int = sql.Column(ResultOutput.id_t,
                              sql.ForeignKey("result_outputs.id"),
                              nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint(
        'result_id', 'output_id', name='unique_result_output'),
  )

  # Relationships:
  result: deeplearning.deepsmith.testcase.Testcase = orm.relationship("Result")
  output: ResultOutput = orm.relationship("ResultOutput")
