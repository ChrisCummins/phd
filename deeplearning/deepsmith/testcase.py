from datetime import datetime
from typing import List

import sqlalchemy as sql
from sqlalchemy import Integer, Column, DateTime, ForeignKey, String, UnicodeText, UniqueConstraint, \
  PrimaryKeyConstraint
from sqlalchemy.orm import relationship

import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.language

from deeplearning.deepsmith import db


class Testcase(db.Base):
  id_t = Integer
  __tablename__ = "testcases"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  language_id: int = Column(deeplearning.deepsmith.language.Language.id_t,
                            ForeignKey("languages.id"), nullable=False)
  generator_id: int = Column(deeplearning.deepsmith.generator.Generator.id_t,
                             ForeignKey("generators.id"), nullable=False)
  harness_id: int = Column(deeplearning.deepsmith.harness.Harness.id_t,
                           ForeignKey("harnesses.id"), nullable=False)

  # Relationships:
  language: deeplearning.deepsmith.language.Language = relationship("Language")
  generator: "Generator" = relationship("Generator", back_populates="testcases")
  harness: "Harness" = relationship("Harness", back_populates="testcases")
  inputs = relationship(
      "TestcaseInput", secondary="testcase_input_associations",
      primaryjoin="TestcaseInputAssociation.testcase_id == Testcase.id",
      secondaryjoin="TestcaseInputAssociation.input_id == TestcaseInput.id")
  timings: List["TimingTiming"] = relationship("TestcaseTiming",
                                               back_populates="testcase")
  results: List["Result"] = relationship("Result", back_populates="testcase")
  pending_results: List["PendingResult"] = relationship(
      "PendingResult", back_populates="testcase")


class TestcaseInputName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "testcase_input_names"

  # Relationships:
  inputs: List["TestcaseInput"] = relationship("TestcaseInput", back_populates="name")


class TestcaseInput(db.Base):
  id_t = Integer
  __tablename__ = "testcase_inputs"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  name_id: TestcaseInputName.id_t = Column(
      TestcaseInputName.id_t, ForeignKey("testcase_input_names.id"), nullable=False)
  # TODO(cec): Use Binary column type.
  sha1: str = Column(String(40), nullable=False, index=True)
  linecount = sql.Column(sql.Integer, nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  input: str = Column(UnicodeText(length=2 ** 31), nullable=False)

  # Relationships:
  name: TestcaseInputName = relationship("TestcaseInputName", back_populates="inputs")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('name_id', 'sha1', name='unique_testcase_input'),
  )


class TestcaseInputAssociation(db.Base):
  __tablename__ = "testcase_input_associations"

  # Columns:
  testcase_id: int = Column(Testcase.id_t, ForeignKey("testcases.id"), nullable=False)
  input_id: int = Column(TestcaseInput.id_t, ForeignKey("testcase_inputs.id"), nullable=False)
  __table_args__ = (
    PrimaryKeyConstraint('testcase_id', 'input_id', name='unique_testcase_input'),)

  # Relationships:
  testcase: Testcase = relationship("Testcase")
  input: TestcaseInput = relationship("TestcaseInput")
