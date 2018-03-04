"""This file implements testbeds."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy import orm

import deeplearning.deepsmith.language
from deeplearning.deepsmith import db


class Testbed(db.Base):
  id_t = sql.Integer
  __tablename__ = "testbeds"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=db.now)
  language_id: int = sql.Column(deeplearning.deepsmith.language.Language.id_t,
                                sql.ForeignKey("languages.id"), nullable=False)
  name: str = sql.Column(sql.String(1024), nullable=False)
  version: str = sql.Column(sql.String(1024), nullable=False)

  # Relationships:
  lang: deeplearning.deepsmith.language.Language = orm.relationship("Language")
  results: typing.List["Result"] = orm.relationship(
      "Result", back_populates="testbed")
  pending_results: typing.List["PendingResult"] = orm.relationship(
      "PendingResult", back_populates="testbed")
  opts = orm.relationship(
      "TestbedOpt", secondary="testbed_opt_associations",
      primaryjoin="TestbedOptAssociation.testbed_id == Testbed.id",
      secondaryjoin="TestbedOptAssociation.opt_id == TestbedOpt.id")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint(
        'language_id', 'name', 'version', name='unique_testbed'),
  )


class TestbedOptName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "testbed_opt_names"

  # Relationships:
  opts: typing.List["TestbedOpt"] = orm.relationship(
      "TestbedOpt", back_populates="name")


class TestbedOpt(db.Base):
  id_t = sql.Integer
  __tablename__ = "testbed_opts"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=db.now)
  name_id: TestbedOptName.id_t = sql.Column(
      TestbedOptName.id_t, sql.ForeignKey("testbed_opt_names.id"), nullable=False)
  # TODO(cec): Use Binary column type.
  sha1: str = sql.Column(sql.String(40), nullable=False, index=True)
  opt: str = sql.Column(sql.UnicodeText(length=4096), nullable=False)

  # Relationships:
  name: TestbedOptName = orm.relationship(
      "TestbedOptName", back_populates="opts")

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint('name_id', 'sha1', name='unique_testbed_opt'),
  )


class TestbedOptAssociation(db.Base):
  __tablename__ = "testbed_opt_associations"

  # Columns:
  testbed_id: int = sql.Column(Testbed.id_t, sql.ForeignKey("testbeds.id"),
                               nullable=False)
  opt_id: int = sql.Column(TestbedOpt.id_t, sql.ForeignKey("testbed_opts.id"),
                           nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('testbed_id', 'opt_id', name='unique_testbed_opt'),
  )

  # Relationships:
  testbed: Testbed = orm.relationship("Testbed")
  opt: TestbedOpt = orm.relationship("TestbedOpt")
