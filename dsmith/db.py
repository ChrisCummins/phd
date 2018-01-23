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
import sqlalchemy

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
from dsmith import Colors
from dsmith import db_base
from dsmith.db_base import *


# Global state to manage database connections. Must call init() before
# creating sessions.
Base = declarative_base()

now = datetime.datetime.utcnow


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


class TestcaseInput(Base):
    id_t = Integer
    __tablename__ = "testcase_inputs"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    sha1: str = Column(String(40), nullable=False, unique=True, index=True)
    input: str = Column(UnicodeText(length=2**31), nullable=False)

    # Relationships:
    testcases: List['Testcase'] = relationship("Testcase", back_populates="input")


class Testcase(Base):
    id_t = Integer
    __tablename__ = "testcases"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    generator_id: int = Column(Generator.id_t, ForeignKey("generators.id"), nullable=False)
    input_id: int = Column(TestcaseInput.id_t, ForeignKey("testcase_inputs.id"), nullable=False)

    # Relationships:
    generator: "Generator" = relationship("Generator", back_populates="testcases")
    input: "TestcaseInput" = relationship("TestcaseInput", back_populates="testcases")
    opts: "TestcaseOpt" = relationship("TestcaseOpt", back_populates="testcases")
    opts = relationship(
        "TestcaseOpt", secondary="testcase_opt_associations",
        primaryjoin="TestcaseOptAssociation.testcase_id == Testcase.id",
        secondaryjoin="TestcaseOptAssociation.opt_id == TestcaseOpt.id")
    timings: List["TimingTiming"] = relationship("TestcaseTiming", back_populates="testcase")
    results: List["Result"] = relationship("Result", back_populates="testcase")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('generator_id', 'input_id', name='uniq_testcases'),
    )


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
    id_t = Integer
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
    time: float = Column(Float, nullable=False)

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


class Stdout(Base):
    id_t = Integer
    __tablename__ = "stdouts"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    sha1: str = Column(String(40), nullable=False, unique=True, index=True)
    stdout: str = Column(UnicodeText(length=2**31), nullable=False)

    # Relationships:
    results: List["Result"] = relationship("Result", back_populates="stdout")


class Stderr(Base):
    id_t = Integer
    __tablename__ = "stderrs"

    # Columns:
    id: int = Column(id_t, primary_key=True)
    date_added: datetime = Column(DateTime, nullable=False, default=now)
    sha1: str = Column(String(40), nullable=False, unique=True, index=True)
    stderr: str = Column(UnicodeText(length=2**31), nullable=False)

    # Relationships:
    results: List["Result"] = relationship("Result", back_populates="stderr")


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
    stdout_id: int = Column(Stdout.id_t, ForeignKey("stdouts.id"), nullable=False)
    stderr_id: int = Column(Stderr.id_t, ForeignKey("stderrs.id"), nullable=False)

    # Relationships:
    testcase: Testcase = relationship("Testcase", back_populates="results")
    testbed: Testbed = relationship("Testbed", back_populates="results")
    harness: Harness = relationship("Harness", back_populates="results")
    stdout: Stdout = relationship("Stdout", back_populates="results")
    stderr: Stderr = relationship("Stderr", back_populates="results")

    # Constraints:
    __table_args__ = (
        UniqueConstraint('testcase_id', 'testbed_id', 'harness_id', name='unique_result'),
    )
