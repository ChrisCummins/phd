"""This file implements testcases."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.language
from deeplearning.deepsmith import db


class Testcase(db.Base):
  id_t = sql.Integer
  __tablename__ = "testcases"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  language_id: int = sql.Column(
      deeplearning.deepsmith.language.Language.id_t,
      sql.ForeignKey("languages.id"), nullable=False)
  generator_id: int = sql.Column(
      deeplearning.deepsmith.generator.Generator.id_t,
      sql.ForeignKey("generators.id"), nullable=False)
  harness_id: int = sql.Column(
      deeplearning.deepsmith.harness.Harness.id_t,
      sql.ForeignKey("harnesses.id"), nullable=False)

  # Relationships:
  language: deeplearning.deepsmith.language.Language = orm.relationship("Language")
  generator: "Generator" = orm.relationship("Generator", back_populates="testcases")
  harness: "Harness" = orm.relationship("Harness", back_populates="testcases")
  inputs = orm.relationship(
      "TestcaseInput", secondary="testcase_input_associations",
      primaryjoin="TestcaseInputAssociation.testcase_id == Testcase.id",
      secondaryjoin="TestcaseInputAssociation.input_id == TestcaseInput.id")
  timings: typing.List["TimingTiming"] = orm.relationship(
      "TestcaseTiming", back_populates="testcase")
  results: typing.List["Result"] = orm.relationship(
      "Result", back_populates="testcase")
  pending_results: typing.List["PendingResult"] = orm.relationship(
      "PendingResult", back_populates="testcase")


class TestcaseInputName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "testcase_input_names"

  # Relationships:
  inputs: typing.List["TestcaseInput"] = orm.relationship(
      "TestcaseInput", back_populates="name")


class TestcaseInput(db.Base):
  id_t = sql.Integer
  __tablename__ = "testcase_inputs"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False, default=db.now)
  name_id: TestcaseInputName.id_t = sql.Column(
      TestcaseInputName.id_t, sql.ForeignKey("testcase_input_names.id"), nullable=False)
  # TODO(cec): Use Binary column type.
  sha1: str = sql.Column(sql.String(40), nullable=False, index=True)
  linecount = sql.Column(sql.Integer, nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  input: str = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Relationships:
  name: TestcaseInputName = orm.relationship(
      "TestcaseInputName", back_populates="inputs")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('name_id', 'sha1', name='unique_testcase_input'),
  )


class TestcaseInputAssociation(db.Base):
  __tablename__ = "testcase_input_associations"

  # Columns:
  testcase_id: int = sql.Column(Testcase.id_t,
                                sql.ForeignKey("testcases.id"), nullable=False)
  input_id: int = sql.Column(TestcaseInput.id_t,
                             sql.ForeignKey("testcase_inputs.id"),
                             nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint(
        'testcase_id', 'input_id', name='unique_testcase_input'),
  )

  # Relationships:
  testcase: Testcase = orm.relationship("Testcase")
  input: TestcaseInput = orm.relationship("TestcaseInput")
