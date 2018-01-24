#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Database backend.
"""
import sqlalchemy as sql

from contextlib import contextmanager
from datetime import datetime
from sqlalchemy import DateTime
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import SmallInteger
from sqlalchemy import String
from sqlalchemy import UnicodeText
from sqlalchemy import UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import dsmith


# Type aliases:
session_t = sql.orm.session.Session
query_t = sql.orm.query.Query

# SQLAlchemy:
Base = declarative_base()

# Shorthand:
now = datetime.utcnow


# Tables ######################################################################


class Client(Base):
    id_t = Integer
    __tablename__ = "clients"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    client: str = Column(String(512), nullable=False, unique=True)

    # Relationships:
    events: List['Event'] = relationship("Event", back_populates="client")


class Event(Base):
    id_t = Integer
    __tablename__ = "events"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    event: str = Column(String(128), nullable=False)
    client_id: int = Column(Client.id_t, ForeignKey("clients.id"), nullable=False)

    # Relationships:
    client: List[Client] = relationship("Client", back_populates="events")


class Generator(Base):
    id_t = Integer
    __tablename__ = "generators"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    generator: str = Column(String(128), nullable=False, unique=True)

    # Relationships:
    testcases: List['Testcase'] = relationship("Testcase", back_populates="generator")


class Testcase(Base):
    id_t = Integer
    __tablename__ = "testcases"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    generator_id: int = Column(Generator.id_t, ForeignKey("generators.id"), nullable=False)

    # Relationships:
    generator: "Generator" = relationship("Generator", back_populates="testcases")
    inputs = relationship(
        "TestcaseInput", secondary="testcase_input_associations",
        primaryjoin="TestcaseInputAssociation.testcase_id == Testcase.id",
        secondaryjoin="TestcaseInputAssociation.input_id == TestcaseInput.id")
    opts = relationship(
        "TestcaseOpt", secondary="testcase_opt_associations",
        primaryjoin="TestcaseOptAssociation.testcase_id == Testcase.id",
        secondaryjoin="TestcaseOptAssociation.opt_id == TestcaseOpt.id")
    timings: List["TimingTiming"] = relationship("TestcaseTiming", back_populates="testcase")
    results: List["Result"] = relationship("Result", back_populates="testcase")


class TestcaseInput(Base):
    id_t = Integer
    __tablename__ = "testcase_inputs"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    sha1: str = Column(String(40), nullable=False, unique=True, index=True)
    input: str = Column(UnicodeText(length=2**31), nullable=False)


class TestcaseInputAssociation(Base):
    __tablename__ = "testcase_input_associations"

    # Columns:
    testcase_id: int = Column(Testcase.id_t, ForeignKey("testcases.id"), nullable=False)
    input_id: int = Column(TestcaseInput.id_t, ForeignKey("testcase_inputs.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('testcase_id', 'input_id', name='unique_testcase_input'),)

    # Relationships:
    testcase: Testcase = relationship("Testcase")
    input: TestcaseInput = relationship("TestcaseInput")


class TestcaseOpt(Base):
    id_t = Integer
    __tablename__ = "testcase_opts"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    opt: str = Column(String(1024), nullable=False, unique=True)

    # Relationships:
    # testcases: List["Testcase"] = relationship("Testcase", back_populates="opts")


class TestcaseOptAssociation(Base):
    __tablename__ = "testcase_opt_associations"

    # Columns:
    testcase_id: int = Column(Testcase.id_t, ForeignKey("testcases.id"), nullable=False)
    opt_id: int = Column(TestcaseOpt.id_t, ForeignKey("testcase_opts.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('testcase_id', 'opt_id', name='_uid'),)

    # Relationships:
    testcase: Testcase = relationship("Testcase")
    opt: TestcaseOpt = relationship("TestcaseOpt")


class TestcaseTiming(Base):
    id_t = Integer
    __tablename__ = "testcase_timings"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    testcase_id: int = Column(Testcase.id_t, ForeignKey("testcases.id"), nullable=False)
    event_id: int = Column(Event.id_t, ForeignKey("events.id"), nullable=False)
    duration: float = Column(Float, nullable=False)
    date: datetime = Column(DateTime, nullable=False)

    # Relationships:
    testcase: Testcase = relationship("Testcase", back_populates="timings")
    event: Event = relationship("Event")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('testcase_id', 'event_id', name='unique_testcase_timing'),
    )


class Harness(Base):
    id_t = Integer
    __tablename__ = "harnesses"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    name: str = Column(String(256), nullable=False)
    version: str = Column(String(256), nullable=False)

    # Relationships:
    results: List["Result"] = relationship("Result", back_populates="harness")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('name', 'version', name='unique_harness'),
    )


class Language(Base):
    id_t = Integer
    __tablename__ = "languages"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    name: str = Column(String(256), nullable=False, unique=True)

    # Relationships:
    testbeds: List["Testbed"] = relationship("Testbed", back_populates="lang")


class Testbed(Base):
    id_t = Integer
    __tablename__ = "testbeds"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    lang_id: int = Column(Language.id_t, ForeignKey("languages.id"), nullable=False)
    name: str = Column(String(256), nullable=False)
    version: str = Column(String(256), nullable=False)

    # Relationships:
    lang: Language = relationship("Language", back_populates="testbeds")
    results: List["Result"] = relationship("Result", back_populates="testbed")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('lang_id', 'name', 'version', name='unique_testbed'),
    )


class Result(Base):
    id_t = Integer
    __tablename__ = "results"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    testcase_id: int = Column(Testcase.id_t, ForeignKey("testcases.id"), nullable=False)
    testbed_id: int = Column(Testbed.id_t, ForeignKey("testbeds.id"), nullable=False)
    harness_id: int = Column(Harness.id_t, ForeignKey("harnesses.id"), nullable=False)
    returncode: int = Column(SmallInteger, nullable=False)

    # Relationships:
    testcase: Testcase = relationship("Testcase", back_populates="results")
    testbed: Testbed = relationship("Testbed", back_populates="results")
    harness: Harness = relationship("Harness", back_populates="results")
    outputs = relationship(
        "ResultOutput", secondary="result_output_associations",
        primaryjoin="ResultOutputAssociation.result_id == Result.id",
        secondaryjoin="ResultOutputAssociation.output_id == ResultOutput.id")
    timings: List["ResultTiming"] = relationship("ResultTiming", back_populates="result")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('testcase_id', 'testbed_id', 'harness_id', name='unique_result'),
    )


class ResultOutput(Base):
    id_t = Integer
    __tablename__ = "result_outputs"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    sha1: str = Column(String(40), nullable=False, unique=True, index=True)
    output: str = Column(UnicodeText(length=2**31), nullable=False)


class ResultOutputAssociation(Base):
    __tablename__ = "result_output_associations"

    # Columns:
    result_id: int = Column(Result.id_t, ForeignKey("results.id"), nullable=False)
    output_id: int = Column(ResultOutput.id_t, ForeignKey("result_outputs.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('result_id', 'output_id', name='unique_result_output'),)

    # Relationships:
    result: Testcase = relationship("Result")
    output: ResultOutput = relationship("ResultOutput")


class ResultTiming(Base):
    id_t = Integer
    __tablename__ = "result_timings"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    result_id: int = Column(Result.id_t, ForeignKey("results.id"), nullable=False)
    event_id: int = Column(Event.id_t, ForeignKey("events.id"), nullable=False)
    duration: float = Column(Float, nullable=False)
    date: datetime = Column(DateTime, nullable=False)

    # Relationships:
    result: Result = relationship("Result", back_populates="timings")
    event: Event = relationship("Event")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('result_id', 'event_id', name='unique_result_timing'),
    )
