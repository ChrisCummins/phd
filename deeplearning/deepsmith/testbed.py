from datetime import datetime
from typing import List

from sqlalchemy import Integer, Column, DateTime, ForeignKey, String, UniqueConstraint, UnicodeText, \
  PrimaryKeyConstraint
from sqlalchemy.orm import relationship

import deeplearning.deepsmith.language

from deeplearning.deepsmith import db


class Testbed(db.Base):
  id_t = Integer
  __tablename__ = "testbeds"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  language_id: int = Column(deeplearning.deepsmith.language.Language.id_t,
                            ForeignKey("languages.id"), nullable=False)
  name: str = Column(String(1024), nullable=False)
  version: str = Column(String(1024), nullable=False)

  # Relationships:
  lang: deeplearning.deepsmith.language.Language = relationship("Language")
  results: List["Result"] = relationship("Result", back_populates="testbed")
  pending_results: List["PendingResult"] = relationship(
      "PendingResult", back_populates="testbed")
  opts = relationship(
      "TestbedOpt", secondary="testbed_opt_associations",
      primaryjoin="TestbedOptAssociation.testbed_id == Testbed.id",
      secondaryjoin="TestbedOptAssociation.opt_id == TestbedOpt.id")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('language_id', 'name', 'version', name='unique_testbed'),
  )


class TestbedOptName(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "testbed_opt_names"

  # Relationships:
  opts: List["TestbedOpt"] = relationship("TestbedOpt", back_populates="name")


class TestbedOpt(db.Base):
  id_t = Integer
  __tablename__ = "testbed_opts"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  name_id: TestbedOptName.id_t = Column(
      TestbedOptName.id_t, ForeignKey("testbed_opt_names.id"), nullable=False)
  # TODO(cec): Use Binary column type.
  sha1: str = Column(String(40), nullable=False, index=True)
  opt: str = Column(UnicodeText(length=4096), nullable=False)

  # Relationships:
  name: TestbedOptName = relationship("TestbedOptName", back_populates="opts")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('name_id', 'sha1', name='unique_testbed_opt'),
  )


class TestbedOptAssociation(db.Base):
  __tablename__ = "testbed_opt_associations"

  # Columns:
  testbed_id: int = Column(Testbed.id_t, ForeignKey("testbeds.id"), nullable=False)
  opt_id: int = Column(TestbedOpt.id_t, ForeignKey("testbed_opts.id"), nullable=False)
  __table_args__ = (
    PrimaryKeyConstraint('testbed_id', 'opt_id', name='unique_testbed_opt'),)

  # Relationships:
  testbed: Testbed = relationship("Testbed")
  opt: TestbedOpt = relationship("TestbedOpt")
