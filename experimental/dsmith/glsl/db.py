#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
GLSL database backend.
"""
import datetime
import logging
import re
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Union

import progressbar
import sqlalchemy as sql
from experimental.dsmith.db_base import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from experimental import dsmith
from experimental.dsmith import Colors
from experimental.dsmith import db_base
from labm8 import crypto, fs
from labm8 import humanize

# Global state to manage database connections. Must call init() before
# creating sessions.
Base = declarative_base()
engine = None
make_session = None


def init() -> str:
  """
  Initialize database engine.

  Must be called before attempt to create a database connection.

  Returns:
      str: URI of database.

  Raises:
      ValueError: In case of error.
  """
  global engine
  global make_session

  engine, public_uri = db_base.make_engine("glsl")
  Base.metadata.create_all(engine)
  Base.metadata.bind = engine
  make_session = sql.orm.sessionmaker(bind=engine)

  return public_uri


@contextmanager
def Session(commit: bool = False) -> session_t:
  """Provide a transactional scope around a series of operations."""
  session = make_session()
  try:
    yield session
    if commit:
      session.commit()
  except:
    session.rollback()
    raise
  finally:
    session.close()


@contextmanager
def ReuseSession(session: session_t = None, commit: bool = False) -> session_t:
  """
  Same as Session(), except if called with an existing `session` object, it
  will use that rather than creating a new one.
  """
  s = session or make_session()
  try:
    yield s
    if commit:
      s.commit()
  except:
    s.rollback()
    raise
  finally:
    if session is None:
      s.close()


# Programs ####################################################################


class Generators:
  value_t = int
  column_t = sql.SmallInteger

  # Magic numbers
  GITHUB = -1
  DSMITH = 1
  RANDCHAR = 2


class Program(Base):
  id_t = sql.Integer
  __tablename__ = 'programs'

  # Fields
  id = sql.Column(id_t, primary_key=True)
  generator = sql.Column(Generators.column_t, nullable=False)
  sha1 = sql.Column(sql.String(40), nullable=False)
  date = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
  generation_time = sql.Column(sql.Float, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Constraints
  __table_args__ = (sql.UniqueConstraint(
      'generator', 'sha1', name='uniq_program'),)

  # Relationships
  testcases = sql.orm.relationship("Testcase", back_populates="program")

  def __repr__(self):
    return f"program[{self.id}] = {{ generator = {self.generator}, sha1 = {self.sha1} }}"


class ProgramProxy(Proxy):
  """
  A program proxy which does not need to be bound to the lifetime of a
  database session.
  """

  def __init__(self, generator: Generators.column_t, generation_time: float,
               src: str):
    self.generator = generator
    self.sha1 = crypto.sha1_str(src)
    self.date = datetime.datetime.utcnow()
    self.generation_time = generation_time
    self.linecount = len(src.split("\n"))
    self.charcount = len(src)
    self.src = src

  def to_record(self, session: session_t) -> Program:
    return Program(
        generator=self.generator,
        sha1=self.sha1,
        date=self.date,
        generation_time=self.generation_time,
        linecount=self.linecount,
        charcount=self.charcount,
        src=self.src)


# Testcases ###################################################################


class Harnesses(object):
  value_t = int
  column_t = sql.SmallInteger

  # Magic numbers
  GLSLANG_FRAG = 3

  @staticmethod
  def to_str(harness: 'Harnesses.value_t') -> str:
    return {
        Harnesses.GLSLANG_FRAG: "glslang_frag",
    }[harness]


class Testcase(Base):
  id_t = sql.Integer
  __tablename__ = "testcases"

  # Fields
  id = sql.Column(id_t, primary_key=True)
  program_id = sql.Column(
      Program.id_t, sql.ForeignKey("programs.id"), nullable=False)
  harness = sql.Column(Harnesses.column_t, nullable=False)
  timeout = sql.Column(sql.Integer, nullable=False)

  # Constraints
  __table_args__ = (sql.UniqueConstraint(
      "program_id", "harness", "timeout", name="unique_testcase"),)

  # Relationships
  program = sql.orm.relationship("Program", back_populates="testcases")
  results = sql.orm.relationship("Result", back_populates="testcase")
  majority = sql.orm.relationship("Majority", back_populates="testcase")

  def __repr__(self):
    return f"{self.id}"


# Experimental Platforms ######################################################


class Platform(Base):
  id_t = sql.SmallInteger().with_variant(sql.Integer, "sqlite")
  __tablename__ = 'platforms'

  # Fields
  id = sql.Column(id_t, primary_key=True)
  platform = sql.Column(sql.String(255), nullable=False)
  version = sql.Column(sql.String(255), nullable=False)
  host = sql.Column(sql.String(255), nullable=False)

  # Constraints
  __table_args__ = (sql.UniqueConstraint(
      'platform', 'version', 'host', name='unique_platform'),)

  # Relationships
  testbeds = sql.orm.relationship("Testbed", back_populates="platform")

  def __repr__(self) -> str:
    return (f"{self.platform_name} {self.version_name}")

  @property
  def platform_name(self):
    return {}.get(self.platform.strip(), self.platform.strip())

  @property
  def version_name(self):
    return {}.get(self.version.strip(), self.version.strip())

  @property
  def host_name(self):
    return {
        "CentOS Linux 7.1.1503 64bit": "CentOS 7.1 x64",
        "openSUSE  13.1 64bit": "openSUSE 13.1 x64",
        "Ubuntu 16.04 64bit": "Ubuntu 16.04 x64",
    }.get(self.host.strip(), self.host.strip())


class Testbed(Base):
  id_t = sql.SmallInteger().with_variant(sql.Integer, "sqlite")
  __tablename__ = 'testbeds'

  # Fields
  id = sql.Column(id_t, primary_key=True)
  platform_id = sql.Column(
      Platform.id_t, sql.ForeignKey("platforms.id"), nullable=False)
  optimizations = sql.Column(sql.Boolean, nullable=False)

  # Constraints
  __table_args__ = (sql.UniqueConstraint(
      'platform_id', 'optimizations', name='unique_testbed'),)

  # Relationships
  platform = sql.orm.relationship("Platform", back_populates="testbeds")

  def __repr__(self) -> str:
    return f"{Colors.BOLD}{Colors.PURPLE}{self.platform.platform_name}{self.plus_minus}{Colors.END}"

  def _testcase_ids_ran(self, session: session_t, harness,
                        generator) -> query_t:
    """ return IDs of testcases with results """
    return session.query(Result.testcase_id) \
      .join(Testcase) \
      .join(Program) \
      .filter(Result.testbed_id == self.id,
              Testcase.harness == harness.id,
              Program.generator == generator.id)

  def _testcases_to_run(self, session: session_t, harness,
                        generator) -> query_t:
    """ return testcases which do not have results """
    return session.query(Testcase) \
      .join(Program) \
      .filter(Testcase.harness == harness.id,
              Program.generator == generator.id,
              ~Testcase.id.in_(
                  self._testcase_ids_ran(session, harness, generator))) \
      .order_by(Program.date,
                Program.id,
                Testcase.timeout.desc())

  def run_testcases(self, harness, generator, session: session_t = None):
    """ run tests on testbed """
    # Sanity check
    if generator.__name__ not in harness.__generators__:
      raise ValueError(f"incompatible combination {harness}:{generator}")

    class Worker(threading.Thread):
      """ worker thread to run testcases asynchronously """

      def __init__(self, harness, generator, testbed_id):
        self.harness = harness
        self.generator = generator
        self.testbed_id = testbed_id
        self.ndone = 0
        super(Worker, self).__init__()

      def run(self):
        """ main loop"""
        with Session() as s:
          testbed = s.query(Testbed) \
            .filter(Testbed.id == self.testbed_id) \
            .scalar()

          already_done = testbed._testcase_ids_ran(s, self.harness,
                                                   self.generator)
          todo = testbed._testcases_to_run(s, self.harness, self.generator)

          self.ndone = already_done.count()
          ntodo = todo.count()

          buf = []
          try:
            for testcase in todo:
              buf.append(harness.run(s, testcase, testbed))
              self.ndone += 1

              # flush the buffer
              if len(buf) >= dsmith.DB_BUF_SIZE:
                save_proxies(s, buf)
                buf = []
                # update results count with actual
                self.ndone = already_done.count()
          finally:
            save_proxies(s, buf)

    with ReuseSession(session) as s:
      already_done = self._testcase_ids_ran(s, harness, generator)
      todo = self._testcases_to_run(s, harness, generator)

      ndone = already_done.count()
      ntodo = todo.count()

      logging.debug(
          f"run {ntodo} {harness}:{generator} testcases on {self}, {ndone} done"
      )

      # Break early if we have nothing to do
      if not ntodo:
        return

      runtime = s.query(func.sum(Result.runtime)) \
                  .join(Testcase) \
                  .join(Program) \
                  .filter(Result.testbed_id == self.id,
                          Testcase.harness == harness.id,
                          Program.generator == generator.id) \
                  .scalar() or 0

      estimated_time = (runtime / max(ndone, 1)) * ntodo
      eta = humanize.Duration(estimated_time)

      words_ntodo = humanize.Commas(ntodo)
      print(f"Running {Colors.BOLD}{words_ntodo} "
            f"{generator}:{harness} testcases on {self}. "
            f"Estimated runtime is {Colors.BOLD}{eta}{Colors.END}.")

      # asynchronously run testcases, updating progress bar
      bar = progressbar.ProgressBar(
          initial_value=ndone, max_value=ndone + ntodo, redirect_stdout=True)
      worker = Worker(harness, generator, self.id)
      worker.start()
      while worker.is_alive():
        bar.update(worker.ndone)
        worker.join(0.5)

  @property
  def plus_minus(self) -> str:
    return "+" if self.optimizations else "-"

  @staticmethod
  def _get_version(path: Path) -> str:
    """
    Fetch the version string.

    glslandValidator output looks like this:
      $ glslang -v
        Glslang Version: SPIRV99.947 15-Feb-2016
        ESSL Version: OpenGL ES GLSL 3.00 glslang LunarG Khronos.SPIRV99.947 15-Feb-2016
        GLSL Version: 4.20 glslang LunarG Khronos.SPIRV99.947 15-Feb-2016
        SPIR-V Version 0x00010000, Revision 6
        GLSL.std.450 Version 100, Revision 1
        Khronos Tool ID 8
    """
    output = subprocess.check_output([path, '-v'], universal_newlines=True)
    line = output.split("\n")[0]
    return re.sub(r'^Glslang Version: +', '', line)

  @staticmethod
  def from_bin(path: Path = "gslang",
               session: session_t = None) -> List['Testbed']:
    import cldrive

    with ReuseSession(session) as s:
      basename = fs.basename(path)
      version = Testbed._get_version(path)
      platform = get_or_add(
          s,
          Platform,
          platform=basename,
          version=version,
          host=cldrive.host_os())
      s.flush()
      return [
          get_or_add(s, Testbed, platform_id=platform.id, optimizations=True),
      ]

  @staticmethod
  def from_str(string: str, session: session_t = None) -> List['Testbed']:
    """ instantiate testbed(s) from shorthand string, e.g. '1+', '5±', etc. """

    def try_and_match(
        string: str, testbeds: Iterable[Testbed]) -> Union[None, List[Testbed]]:
      for testbed in testbeds:
        if str(testbed.platform.platform) == string[:-1]:
          if string[-1] == "±":
            return [
                get_or_add(
                    s,
                    Testbed,
                    platform_id=testbed.platform.id,
                    optimizations=True),
                get_or_add(
                    s,
                    Testbed,
                    platform_id=testbed.platform.id,
                    optimizations=False)
            ]
          else:
            return [
                get_or_add(
                    s,
                    Testbed,
                    platform_id=testbed.platform.id,
                    optimizations=True if string[-1] == "+" else False)
            ]

    # Strip shell formatting
    string = dsmith.unformat(string)

    # check format
    if string[-1] != "+" and string[-1] != "-" and string[-1] != "±":
      raise ValueError(f"Invalid testbed string '{string}'")

    with ReuseSession(session) as s:
      # Try and match string against an existing record in the database:
      in_db = try_and_match(string, s.query(Testbed))
      if in_db:
        return in_db

      return Testbed.from_bin(string, session=s)


class TestbedProxy(Proxy, dsmith.ReprComparable):
  """
  A testbed proxy which does not need to be bound to the lifetime of a
  database session.
  """

  def __init__(self, testbed: Testbed):
    self.repr = str(testbed)
    self.platform = testbed.platform.platform
    self.version = testbed.platform.version
    self.host = testbed.platform.host_name
    self.optimizations = testbed.optimizations
    self.id = testbed.id

  def to_record(self, session: session_t) -> Testbed:
    record = session.query(Testbed).filter(Testbed.id == self.id).scalar()
    if record:
      return record

    # If there wasn't a record in the database, we need to create a new one:
    testbed = Testbed.from_bin(self.platform, session)
    session.commit()
    logging.debug(f"Added new Testbed {testbed}")
    return testbed

  def run_testcases(self, harness, generator) -> None:
    with Session() as s:
      testbed_ = self.to_record(s)
      testbed_.run_testcases(harness, generator)

  def __repr__(self):
    return self.repr


class Stdout(Base):
  id_t = sql.Integer
  __tablename__ = "stdouts"

  # Fields
  id = sql.Column(id_t, primary_key=True)
  sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
  stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  @staticmethod
  def _escape(string: str) -> str:
    """ filter noise from test harness stdout """
    return '\n'.join(
        line for line in string.split('\n') if line != "ADL Escape failed." and
        line != "WARNING:endless loop detected!" and
        line != "One module without kernel function!")

  @staticmethod
  def from_str(session: session_t, string: str) -> str:
    """
    Instantiate a Stdout object
    """
    # Strip the noise
    string = Stdout._escape(string)

    stdout = get_or_add(
        session, Stdout, sha1=crypto.sha1_str(string), stdout=string)
    return stdout


class Stderr(Base):
  """
  Result stderr output.

  Stderr output may have *UP TO ONE* of the following associated metadata:
      * assertion
      * unreachable
      * stackdump
  """
  id_t = sql.Integer
  __tablename__ = "stderrs"

  # The maximum number of characters to keep before truncating
  max_chars = 64000

  # Fields
  id = sql.Column(id_t, primary_key=True)
  sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
  linecount = sql.Column(sql.Integer, nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  truncated = sql.Column(sql.Boolean, nullable=False)
  stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  def __repr__(self):
    return self.sha1

  @staticmethod
  def _escape(string: str) -> str:
    """ filter noise from test harness stderr """
    return string

  @staticmethod
  def from_str(session: session_t, string: str) -> 'Stderr':
    string = Stderr._escape(string)
    sha1 = crypto.sha1_str(string)

    stderr = get_or_add(
        session,
        Stderr,
        sha1=sha1,
        linecount=len(string.split("\n")),
        charcount=len(string),
        truncated=len(string) > Stderr.max_chars,
        stderr=string[:Stderr.max_chars])
    return stderr


class Outcomes:
  """
  A summary of the result of running a testcase on a testbed.
  """
  type = int
  column_t = sql.SmallInteger

  # Magic numbers:
  TODO = -1
  BF = 1
  BC = 2
  BTO = 3
  PASS = 6

  @staticmethod
  def to_str(outcomes: 'Outcomes.value_t') -> str:
    """ convert to long-form string """
    return {
        Outcomes.TODO: "unknown",
        Outcomes.BF: "build failure",
        Outcomes.BC: "build crash",
        Outcomes.BTO: "build timeout",
        Outcomes.PASS: "pass",
    }[outcomes]


class Result(Base):
  """
  The result of running a testcase on a testbed.
  """
  id_t = sql.Integer
  __tablename__ = "results"

  # Fields
  id = sql.Column(id_t, primary_key=True)
  testbed_id = sql.Column(
      Testbed.id_t, sql.ForeignKey("testbeds.id"), nullable=False, index=True)
  testcase_id = sql.Column(
      Testcase.id_t, sql.ForeignKey("testcases.id"), nullable=False, index=True)
  date = sql.Column(
      sql.DateTime,
      nullable=False,
      index=True,
      default=datetime.datetime.utcnow)
  returncode = sql.Column(sql.SmallInteger, nullable=False)
  outcome = sql.Column(Outcomes.column_t, nullable=False, index=True)
  runtime = sql.Column(sql.Float, nullable=False)
  stdout_id = sql.Column(
      Stdout.id_t, sql.ForeignKey("stdouts.id"), nullable=False)
  stderr_id = sql.Column(
      Stderr.id_t, sql.ForeignKey("stderrs.id"), nullable=False)

  # Constraints
  __table_args__ = (sql.UniqueConstraint(
      'testbed_id', 'testcase_id', name='unique_result_triple'),)

  # Relationships
  meta = sql.orm.relation("ResultMeta", back_populates="result")
  classification = sql.orm.relation("Classification", back_populates="result")
  testbed = sql.orm.relationship("Testbed")
  testcase = sql.orm.relationship("Testcase")
  stdout = sql.orm.relationship("Stdout")
  stderr = sql.orm.relationship("Stderr")

  def __repr__(self):
    return str(self.id)


class ResultProxy(object):
  """
  A result proxy which does not need to be bound to the lifetime of a
  database session.
  """

  def __init__(self, testbed_id: Testbed.id_t, testcase_id: Testcase.id_t,
               returncode: int, outcome: Outcomes.type, runtime: float,
               stdout: str, stderr: str):
    self.testbed_id = testbed_id
    self.testcase_id = testcase_id
    self.returncode = returncode
    self.outcome = outcome
    self.runtime = runtime
    self.stdout = stdout
    self.stderr = stderr
    self.date = datetime.datetime.utcnow()  # default value

  def to_record(self, session: session_t) -> Result:
    # stdout / stderr
    stdout = Stdout.from_str(session, self.stdout)
    stderr = Stderr.from_str(session, self.stderr)
    session.flush()  # required to get IDs

    return Result(
        testbed_id=self.testbed_id,
        testcase_id=self.testcase_id,
        returncode=self.returncode,
        outcome=self.outcome,
        runtime=self.runtime,
        stdout=stdout,
        stderr=stderr,
        date=self.date)


class GlslResult(Result):

  @staticmethod
  def get_outcome(returncode: int, stderr: str, runtime: float,
                  timeout: int) -> Outcomes.type:
    """
    Given a result, determine its outcome.
    See Outcomes for list of possible outcomes.
    """
    if returncode == 0:
      return Outcomes.PASS
    elif returncode == 1:
      return Outcomes.BF
    elif returncode == 2:
      return Outcomes.BF
    elif returncode == -9 and runtime >= timeout:
      return Outcomes.BTO
    elif returncode == -9:
      logging.warn(f"SIGKILL, but only ran for {runtime:.2f}s")
    return Outcomes.BC


class ResultMeta(Base):
  id_t = Result.id_t
  __tablename__ = "results_metas"

  # Fields
  id = sql.Column(id_t, sql.ForeignKey("results.id"), primary_key=True)
  total_time = sql.Column(
      sql.Float, nullable=False)  # time to generate and run test case
  cumtime = sql.Column(
      sql.Float, nullable=False)  # culumative time for this testbed time

  # Relationships
  result = sql.orm.relationship("Result", back_populates="meta")

  def __repr__(self):
    return (f"result: {self.id} total_time: {self.total_time:.3f}s, " +
            f"cumtime: {self.cumtime:.1f}s")


class Classifications:
  value_t = int
  column_t = sql.SmallInteger

  # Magic numbers
  BC = 1
  BTO = 2
  ABF = 3

  @staticmethod
  def to_str(outcomes: 'Classifications.value_t') -> str:
    return {
        Classifications.BC: "build crash",
        Classifications.BTO: "build timeout",
        Classifications.ABF: "anomylous build failure",
    }[outcomes]


class Classification(Base):
  id_t = Result.id_t
  __tablename__ = "classifications"

  # Fields
  id = sql.Column(id_t, sql.ForeignKey("results.id"), primary_key=True)
  classification = sql.Column(Classifications.column_t, nullable=False)

  # Relationships
  result = sql.orm.relationship("Result", back_populates="classification")


class Majority(Base):
  id_t = Testcase.id_t
  __tablename__ = "majorities"

  # Fields
  id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
  num_results = sql.Column(sql.SmallInteger, nullable=False)
  maj_outcome = sql.Column(Outcomes.column_t, nullable=False)
  outcome_majsize = sql.Column(sql.SmallInteger, nullable=False)

  # Relationships
  testcase = sql.orm.relationship("Testcase", back_populates="majority")
