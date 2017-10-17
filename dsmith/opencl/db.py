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
OpenCL database backend.
"""
import cldrive
import clgen
import datetime
import humanize
import logging
import os
import progressbar
import re
import sqlalchemy as sql
import threading

from contextlib import contextmanager
from labm8 import crypto, fs, prof, system
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Dict, Iterable, List, Tuple, Union

import dsmith
from dsmith import Colors
from dsmith.opencl import opencl_pb2 as pb

# Global state to manage database connections. Must call init() before
# creating sessions.
Base = declarative_base()
engine = None
make_session = None


session_t = sql.orm.session.Session
query_t = sql.orm.query.Query


def init() -> str:
    """
    Initialize database engine.

    Must be called before attempt to create a database connection.

    Returns:
        str: URI of database.
    """
    global engine
    global make_session
    username, password = dsmith.MYSQL_CREDENTIALS
    hostname = dsmith.MYSQL_HOSTNAME
    table = dsmith.MYSQL_DATABASE
    port = str(dsmith.MYSQL_PORT)

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    public_uri = f"mysql://{username}@{hostname}:{port}/{table}?charset=utf8".format(**vars())
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{table}?charset=utf8"
    echo = True if os.environ.get("DB_DEBUG", None) else False

    logging.debug(f"connecting to {public_uri}")
    engine = sql.create_engine(uri, encoding="utf-8", echo=echo)

    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    make_session = sql.orm.sessionmaker(bind=engine)

    profile = True if os.environ.get("PROF") else False
    if profile:
        prof.enable()

    return public_uri


@contextmanager
def Session(commit: bool=False) -> session_t:
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
def ReuseSession(session: session_t=None, commit: bool=False) -> session_t:
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


def get_or_add(session: sql.orm.session.Session, model,
               defaults: Dict[str, object]=None, **kwargs) -> object:
    """
    Instantiate a mapped database object. If the object is not in the database,
    add it.

    Note that no change is written to disk until commit() is called on the
    session.
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if not instance:
        params = dict((k, v) for k, v in kwargs.items()
                      if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)

    return instance


# Programs ####################################################################


class Generators:
    value_t = int
    column_t = sql.SmallInteger

    # Magic numbers
    CLSMITH = 0
    DSMITH = 1


class Program(Base):
    id_t = sql.Integer
    __tablename__ = 'programs'

    # Fields
    id = sql.Column(id_t, primary_key=True)
    generator = sql.Column(Generators.column_t, nullable=False)
    sha1 = sql.Column(sql.String(40), nullable=False)
    date = sql.Column(sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
    generation_time = sql.Column(sql.Float, nullable=False)
    linecount = sql.Column(sql.Integer, nullable=False)
    charcount = sql.Column(sql.Integer, nullable=False)
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint('generator', 'sha1', name='uniq_program'),
    )

    # Relationships
    testcases = sql.orm.relationship("Testcase", back_populates="program")

    def __repr__(self):
        return f"program[{self.id}] = {{ generator = {self.generator}, sha1 = {self.sha1} }}"


class ClsmithProgramMeta(Base):
    id_t = Program.id_t
    __tablename__ = "clsmith_program_metas"

    # Fields
    id = sql.Column(id_t, sql.ForeignKey("programs.id"), primary_key=True)
    flags = sql.Column(sql.String(80), nullable=False, default="")

    # Relationships
    program = sql.orm.relationship("Program")


class DsmithProgramMeta(Base):
    id_t = Program.id_t
    __tablename__ = "dsmith_program_metas"

    # Fields
    id = sql.Column(id_t, sql.ForeignKey("programs.id"), primary_key=True)
    contains_floats = sql.Column(sql.Boolean)
    vector_inputs = sql.Column(sql.Boolean)
    compiler_warnings = sql.Column(sql.Boolean)

    # Relationships
    program = sql.orm.relationship("Program")

    def get_contains_floats(self, s: session_t) -> bool:
        if self.contains_floats == None:
            q = s.query(func.count(Program.id))\
                .filter(Program.id == self.id,
                        (Program.src.like("%float%") | Program.src.like("%double%")))\
                .scalar()
            self.contains_floats = True if q else False

        return self.contains_floats

    def get_vector_inputs(self, s: session_t) -> bool:
        if self.vector_inputs == None:
            for arg in cldrive.extract_args(self.program.src):
                if arg.is_vector:
                    self.vector_inputs = True
                    break

        return self.vector_inputs

    def get_compiler_warnings(self, s: session_t) -> bool:
        """ check for red-flag compiler warnings """
        if self.compiler_warnings == None:
            q = s.query(func.count(Stderr.id))\
                .join(Result)\
                .filter(
                    Result.testcase_id == self.id,
                    (Stderr.stderr.like("%incompatible pointer to integer conversion%") |
                     Stderr.stderr.like("%comparison between pointer and integer%") |
                     Stderr.stderr.like("%warning: incompatible%") |
                     Stderr.stderr.like("%warning: division by zero is undefined%") |
                     Stderr.stderr.like("%warning: comparison of distinct pointer types%") |
                     Stderr.stderr.like("%is past the end of the array%") |
                     Stderr.stderr.like("%warning: comparison between pointer and%") |
                     Stderr.stderr.like("%warning: array index%") |
                     Stderr.stderr.like("%warning: implicit conversion from%") |
                     Stderr.stderr.like("%array index -1 is before the beginning of the array%") |
                     Stderr.stderr.like("%incompatible pointer%") |
                     Stderr.stderr.like("%incompatible integer to pointer %"))
                )\
                .scalar()
            self.compiler_warnings = True if q else False

        return self.compiler_warnings


# Parameters ##################################################################


class Threads(Base):
    id_t = sql.SmallInteger
    __tablename__ = "threads"

    # Fields
    id = sql.Column(id_t, primary_key=True)
    gsize_x = sql.Column(sql.Integer, nullable=False)
    gsize_y = sql.Column(sql.Integer, nullable=False)
    gsize_z = sql.Column(sql.Integer, nullable=False)
    lsize_x = sql.Column(sql.Integer, nullable=False)
    lsize_y = sql.Column(sql.Integer, nullable=False)
    lsize_z = sql.Column(sql.Integer, nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint(
            'gsize_x', 'gsize_y', 'gsize_z',
            'lsize_x', 'lsize_y', 'lsize_z',
            name='unique_thread_size'),
    )

    # Relationships
    testcases = sql.orm.relationship("Testcase", back_populates="threads")

    @property
    def gsize(self) -> Tuple[int, int, int]:
        return (self.gsize_x, self.gsize_y, self.gsize_z)

    @property
    def lsize(self) -> Tuple[int, int, int]:
        return (self.lsize_x, self.lsize_y, self.lsize_z)

    def __repr__(self) -> str:
        return " ".join(self.to_flags())

    def to_flags(self) -> List[str]:
        return [
            "-g", f"{self.gsize_x},{self.gsize_y},{self.gsize_z}",
            "-l", f"{self.lsize_x},{self.lsize_y},{self.lsize_z}",
        ]


# Testcases ###################################################################


class Harnesses(object):
    value_t = int
    column_t = sql.SmallInteger

    # Magic numbers
    CLANG = -1
    CL_LAUNCHER = 0
    CLDRIVE = 1

    @staticmethod
    def to_str(harness: 'Harnesses.value_t') -> str:
        return {
            Harnesses.CLDRIVE: "cldrive",
            Harnesses.CL_LAUNCHER: "cl_launcher",
            Harnesses.CLANG: "clang",
        }[harness]

    @staticmethod
    def result_t(harness: "Harnesses.value_t") -> Union["Cl_launcherResult", "CldriveResult"]:
        return {
            Harnesses.CLDRIVE: CldriveResult,
            Harnesses.CL_LAUNCHER: Cl_launcherResult,
            Harnesses.CLANG: ClangResult,
        }[harness]


class Testcase(Base):
    id_t = sql.Integer
    __tablename__ = "testcases"

    # Fields
    id = sql.Column(id_t, primary_key=True)
    program_id = sql.Column(Program.id_t, sql.ForeignKey("programs.id"), nullable=False)
    threads_id = sql.Column(Threads.id_t, sql.ForeignKey("threads.id"), nullable=False)
    harness = sql.Column(Harnesses.column_t, nullable=False)
    input_seed = sql.Column(sql.Integer, nullable=False)
    timeout = sql.Column(sql.Integer, nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint("program_id", "threads_id", "harness",
                             "input_seed", "timeout", name="unique_testcase"),
    )

    # Relationships
    program = sql.orm.relationship("Program", back_populates="testcases")
    threads = sql.orm.relationship("Threads", back_populates="testcases")
    results = sql.orm.relationship("Result", back_populates="testcase")
    majority = sql.orm.relationship("Majority", back_populates="testcase")
    clsmith_meta = sql.orm.relationship("ClsmithTestcaseMeta", back_populates="testcase")
    dsmith_meta = sql.orm.relationship("DsmithTestcaseMeta", back_populates="testcase")

    def __repr__(self):
        return f"testcase {self.id} = {{program: {self.program_id}, threads: {self.threads} }}"

    @property
    def meta_t(self):
        if self.harness == Harnesses.CL_LAUNCHER:
            return ClsmithTestcaseMeta
        elif self.harness == Harnesses.CLDRIVE:
            return DsmithTestcaseMeta
        else:
            raise LookupError(f"unknown harness {self.harness}")

    def meta(self, s: session_t):
        return get_or_add(s, self.meta_t, id=self.id)

    def verify_arc(self, s: session_t):
        """
        Verify that a test case is sensible.
        """
        return self.meta(s).verify_arc(s)

    def verify_awo(self, s: session_t):
        """
        Verify that a test case is sensible.

        On first run, this is time consuming, though results are cached for
        later re-use.
        """
        return self.meta(s).verify_awo(s)

    def retract_classifications(self, s: session_t,
                                classification: "Classifications.value_t") -> None:
        """
        Remove classifications for the testcase.
        """
        q = s.query(Result.id)\
            .join(Classification)\
            .filter(Result.testcase_id == self.id,
                    Classification.classification == classification)
        ids_to_update = [x[0] for x in q.all()]
        n = len(ids_to_update)
        assert n > 0
        ids_str = ",".join(str(x) for x in ids_to_update)
        print("retracting", Classifications.to_str(classification),
              f"classifications on {n} results: {ids_str}")
        s.query(Classification)\
            .filter(Classification.id.in_(ids_to_update))\
            .delete(synchronize_session=False)


class ClsmithTestcaseMeta(Base):
    id_t = Testcase.id_t
    __tablename__ = "clsmith_testcase_metas"

    # Fields
    id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
    oclverified = sql.Column(sql.Boolean)

    # Relationships
    testcase = sql.orm.relationship("Testcase", back_populates="clsmith_meta")

    def get_oclverified(self, s: session_t) -> bool:
        if self.oclverified == None:
            prof.start("clsmith oclgrind")
            import oclgrind

            testcase = s.query(Testcase)\
                .filter(Testcase.id == self.id)\
                .scalar()

            self.oclverified = oclgrind.verify_clsmith_testcase(testcase)
            s.commit()
            prof.stop("clsmith oclgrind")

        return self.oclverified

    def verify_arc(self, s: session_t):
        # TODO: why not gpuverify too?
        if not self.get_oclverified(s):
            return False
        return True

    def verify_awo(self, s: session_t):
        # TODO: why not gpuverify too?
        if not self.get_oclverified(s):
            return False
        return True


class DsmithTestcaseMeta(Base):
    id_t = Testcase.id_t
    __tablename__ = "dsmith_testcase_metas"

    # Fields
    id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
    gpuverified = sql.Column(sql.Boolean)
    oclverified = sql.Column(sql.Boolean)

    # Relationships
    testcase = sql.orm.relationship("Testcase", back_populates="dsmith_meta")

    def get_program_meta(self, s: session_t) -> DsmithProgramMeta:
        program_meta = s.query(DsmithProgramMeta)\
            .filter(DsmithProgramMeta.id == self.testcase.program_id)\
            .scalar()
        if not program_meta:
            program_meta = get_or_add(s, DsmithProgramMeta,
                                         id=self.testcase.program_id)
            s.flush()

        return program_meta

    def get_gpuverified(self, s: session_t) -> bool:
        if self.gpuverified == None:
            prof.start("dsmith gpuverify")
            src = s.query(Program.src)\
                .join(Testcase)\
                .filter(Testcase.id == self.id)\
                .scalar()

            try:
                clgen.gpuverify(src, ["--local_size=64", "--num_groups=128"])
                self.gpuverified = True
            except clgen.GPUVerifyException:
                self.gpuverified = False
            s.commit()
            prof.stop("dsmith gpuverify")

        return self.gpuverified

    def get_oclverified(self, s: session_t) -> bool:
        if self.oclverified == None:
            prof.start("dsmith oclgrind")
            import oclgrind

            testcase = s.query(Testcase)\
                .filter(Testcase.id == self.id)\
                .scalar()

            self.oclverified = oclgrind.verify_dsmith_testcase(testcase)
            s.commit()
            prof.stop("dsmith oclgrind")

        return self.oclverified

    def verify_arc(self, s: session_t):
        program_meta = self.get_program_meta(s)

        if program_meta.get_compiler_warnings(s):
            return False
        if not self.get_oclverified(s):
            return False
        return True

    def verify_awo(self, s: session_t):
        program_meta = self.get_program_meta(s)

        if program_meta.get_contains_floats(s):
            return False
        if program_meta.get_vector_inputs(s):
            return False
        if program_meta.get_compiler_warnings(s):
            return False
        if not self.get_gpuverified(s):
            return False
        if not self.get_oclverified(s):
            return False
        return True


# Experimental Platforms ######################################################


class Platform(Base):
    id_t = sql.SmallInteger
    __tablename__ = 'platforms'

    # Fields
    id = sql.Column(id_t, primary_key=True)
    platform = sql.Column(sql.String(255), nullable=False)  # CL_PLATFORM_NAME
    device = sql.Column(sql.String(255), nullable=False)  # CL_DEVICE_NAME
    driver = sql.Column(sql.String(255), nullable=False)  # CL_DRIVER_VERSION
    opencl = sql.Column(sql.String(8), nullable=False)  # CL_PLATFORM_VERSION
    devtype = sql.Column(sql.String(12), nullable=False)  # CL_DEVICE_TYPE
    host = sql.Column(sql.String(255), nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint('platform', 'device', 'driver', 'opencl',
                             'devtype', 'host', name='unique_platform'),
    )

    # Relationships
    testbeds = sql.orm.relationship("Testbed", back_populates="platform")

    def __repr__(self) -> str:
        if self.device_name.startswith(self.platform_name):
            return (f"{self.device_name} {self.driver_name} on {self.host_name}")
        else:
            return (f"{self.platform_name} {self.device_name} {self.driver_name} on {self.host_name}")

    def platform_id(self):
        """ return OpenCL platform index, or KeyError if platform not found """
        import pyopencl as cl

        for i, platform in enumerate(cl.get_platforms()):
            if platform.get_info(cl.platform_info.NAME) == self.platform:
                return i
        raise KeyError(f"platform {self.platform} not found")

    def device_id(self):
        """ return OpenCL device index, or KeyError if device not found """
        import pyopencl as cl

        platform = cl.get_platforms()[self.platform_id()]
        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
        for i, device in enumerate(ctx.get_info(cl.context_info.DEVICES)):
            if device.get_info(cl.device_info.NAME) == self.device:
                return i

        raise KeyError(f"device {self.device} not found")

    @property
    def num(self):
        if self.platform == "clang":
            return f"{self.platform}-{self.driver}"

        return {
            "ComputeAorta (Intel E5-2620)": 9,
            "GeForce GTX 1080": 1,
            "GeForce GTX 780": 2,
            "Intel E5-2620 v4": 4,
            "Intel E5-2650 v2": 5,
            "Intel HD Haswell GT2": 3,
            "Intel i5-4570": 6,
            "Intel Xeon Phi": 7,
            "Oclgrind Simulator": 10,
            "POCL (Intel E5-2620)": 8,
        }.get(self.device_name, -1)

    @property
    def platform_name(self):
        return {
            "Intel Gen OCL Driver": "Beignet",
            "Intel(R) OpenCL": "Intel OpenCL",
            "Portable Computing Language": "POCL",
        }.get(self.platform.strip(), self.platform.strip())

    @property
    def device_name(self):
        return {
            "Codeplay Software Ltd. - host CPU": "ComputeAorta (Intel E5-2620)",
            "Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz": "Intel i5-4570",
            "Intel(R) HD Graphics Haswell GT2 Desktop": "Intel HD Haswell GT2",
            "Intel(R) Many Integrated Core Acceleration Card": "Intel Xeon Phi",
            "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620 v4",
            "Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz": "Intel E5-2650 v2",
            "pthread-Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "POCL (Intel E5-2620)",
        }.get(self.device.strip(), self.device.strip())

    @property
    def driver_name(self):
        return {
            "Oclgrind 16.10": "16.10",
        }.get(self.driver.strip(), self.driver.strip())

    @property
    def devtype_name(self):
        return {
            "3": "CPU",
            "ACCELERATOR": "Accelerator",
        }.get(self.devtype.strip(), self.devtype.strip())

    @property
    def host_name(self):
        return {
            "CentOS Linux 7.1.1503 64bit": "CentOS 7.1 x64",
            "openSUSE  13.1 64bit": "openSUSE 13.1 x64",
            "Ubuntu 16.04 64bit": "Ubuntu 16.04 x64",
        }.get(self.host.strip(), self.host.strip())

    @staticmethod
    def from_env(env: cldrive.OpenCLEnvironment,
                 session: session_t=None) -> 'Testbed':
        with ReuseSession(session) as s:
            return get_or_add(
                s, Platform,
                platform=env.platform,
                device=env.device,
                driver=env.driver_version,
                opencl=env.opencl_version,
                devtype=env.device_type,
                host=cldrive.host_os())


class Testbed(Base):
    id_t = sql.SmallInteger
    __tablename__ = 'testbeds'

    # Fields
    id = sql.Column(id_t, primary_key=True)
    platform_id = sql.Column(Platform.id_t, sql.ForeignKey("platforms.id"), nullable=False)
    optimizations = sql.Column(sql.Boolean, nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint('platform_id', 'optimizations', name='unique_testbed'),
    )

    # Relationships
    platform = sql.orm.relationship("Platform", back_populates="testbeds")

    def __repr__(self) -> str:
        return f"{Colors.BOLD}{Colors.PURPLE}{self.num}{Colors.END}"

    def _testcase_ids_ran(self, session: session_t, harness, generator) -> query_t:
        """ return IDs of testcases with results """
        return session.query(Result.testcase_id)\
            .join(Testcase)\
            .join(Program)\
            .filter(Result.testbed_id == self.id,
                    Testcase.harness == harness.id,
                    Program.generator == generator.id)

    def _testcases_to_run(self, session: session_t, harness, generator) -> query_t:
        """ return testcases which do not have results """
        return  session.query(Testcase)\
            .join(Program)\
            .filter(Testcase.harness == harness.id,
                    Program.generator == generator.id,
                    ~Testcase.id.in_(self._testcase_ids_ran(session, harness, generator)))

    def run_testcases(self, harness, generator, session: session_t=None):
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
                with Session() as s:
                    testbed = Testbed.from_id(s, self.testbed_id)

                    already_done = testbed._testcase_ids_ran(
                        s, self.harness, self.generator)
                    todo = testbed._testcases_to_run(
                        s, self.harness, self.generator)

                    ndone = already_done.count()
                    ntodo = todo.count()

                    for testcase in todo:
                        for _ in range(100):
                            try:
                                harness.run(s, testcase, testbed)
                                self.ndone = already_done.count()
                                break
                            except IntegrityError:
                                logging.warning("database integrity error")
                        else:
                            raise OSError("100 consecutive database integrity errors")

        with ReuseSession(session) as s:
            already_done = self._testcase_ids_ran(s, harness, generator)
            todo = self._testcases_to_run(s, harness, generator)

            ndone = already_done.count()
            ntodo = todo.count()

            # Break early if we have nothing to do
            if not ntodo:
                return

            runtime = s.query(func.sum(Result.runtime))\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == self.id,
                        Testcase.harness == harness.id,
                        Program.generator == generator.id)\
                .scalar() or 0

            estimated_time = (runtime / max(ndone, 1)) * ntodo
            eta = humanize.naturaldelta(datetime.timedelta(seconds=estimated_time))

            words_ntodo = humanize.intcomma(ntodo)
            print(f"Running {Colors.BOLD}{words_ntodo} "
                  f"{Colors.GREEN}{generator}{Colors.END}"
                  f":{Colors.BOLD}{Colors.YELLOW}{harness}{Colors.END} "
                  "testcases "
                  f"on {Colors.BOLD}{Colors.PURPLE}{self.num}{Colors.END}. "
                  f"Estimated runtime is {Colors.BOLD}{eta}{Colors.END}.")

            # asynchronously run testcases, updating progress bar
            bar = progressbar.ProgressBar(initial_value=ndone,
                                          max_value=ndone + ntodo,
                                          redirect_stdout=True)
            worker = Worker(harness, generator, self.id)
            worker.start()
            while worker.is_alive():
                bar.update(worker.ndone)
                worker.join(0.5)

    @property
    def num(self) -> str:
        p = self.platform.platform_name if self.platform.num == -1 else self.platform.num
        return f"{p}{self.plus_minus}"

    @property
    def plus_minus(self) -> str:
        return "+" if self.optimizations else "-"

    @property
    def ids(self) -> Tuple[int, int]:
        try:
            return self._ids
        except AttributeError:
            self._set_ids()
            return self._ids

    def _set_ids(self):
        import pyopencl as cl
        # match platform ID:
        for j, platform in enumerate(cl.get_platforms()):
            platform_name = platform.get_info(cl.platform_info.NAME)
            if platform_name == self.platform.platform:
                logging.debug(f"trying to match '{self.platform.platform}' to '{platform_name}'")
                # match device ID:
                for i, device in enumerate(platform.get_devices()):
                    logging.debug(f"matched platform '{platform_name}'")

                    device_name = device.get_info(cl.device_info.NAME)
                    device_driver = device.get_info(cl.device_info.DRIVER_VERSION)

                    logging.debug(f"trying to match '{self.platform.device} "
                                  f"{self.platform.driver}' to '{device_name} "
                                  f"{device_driver}'")
                    if (device_name == self.platform.device
                        and device_driver == self.platform.driver):
                        logging.debug(f"matched device '{device_name}'")
                        self._ids = j, i
                        return

        # after iterating over all OpenCL platforms and devices, no match found:
        raise LookupError("unable to determine OpenCL IDs of "
                          f"'{self.platform.platform}' '{self.platform.device}'")

    @staticmethod
    def from_env(env: cldrive.OpenCLEnvironment,
                 session: session_t=None) -> 'Testbed':
        with ReuseSession(session) as s:
            platform = Platform.from_env(env, session=s)
            s.flush()

            return [
                get_or_add(s, Testbed, platform_id=platform.id, optimizations=False),
                get_or_add(s, Testbed, platform_id=platform.id, optimizations=True)
            ]

    @staticmethod
    def from_str(string: str, session: session_t=None) -> List['Testbed']:
        """ instantiate testbed(s) from shorthand string, e.g. '1+', '5±', etc. """
        # strip formatting
        string = re.sub(r'\x1b[^m]*m', '', string.split(" ")[0])

        # check format
        if string[-1] != "+" and string[-1] != "-" and string[-1] != "±":
            raise ValueError(f"Invalid testbed string '{string}'")

        with ReuseSession(session) as s:
            for testbed in s.query(Testbed):
                if str(testbed.platform.num) == string[:-1]:
                    if string[-1] == "±":
                        return [
                            get_or_add(
                                s, Testbed,
                                platform_id=testbed.platform.id,
                                optimizations=True),
                            get_or_add(
                                s, Testbed,
                                platform_id=testbed.platform.id,
                                optimizations=False)
                        ]
                    else:
                        return [
                            get_or_add(
                                s, Testbed,
                                platform_id=testbed.platform.id,
                                optimizations=True if string[-1] == "+" else False)
                            ]
            raise LookupError(f"Testbed '{string}' not found in database")

    @staticmethod
    def from_id(session: session_t, id: int) -> 'Testbed':
        return session.query(Testbed).filter(Testbed.id == id).scalar()


class TestbedProxy(object):
    """
    A testbed proxy which does not need to be bound to the lifetime of a
    database session.
    """
    def __init__(self, testbed: Testbed):
        self.repr = str(testbed)
        self.platform = str(testbed.platform)
        self.id = testbed.id

    def testbed(self, session: session_t):
        """ Return a """
        return session.query(Testbed).filter(Testbed.id == self.id).scalar()

    def run_testcases(self, harness, generator):
        with Session() as s:
            testbed_ = self.testbed(s)
            testbed_.run_testcases(harness, generator)

    def __repr__(self):
        return self.repr

    def __lt__(self, other):
        return str(self) < str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return str(self) != str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __ge__(self, other):
        return str(self) >= str(other)


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
        return '\n'.join(line for line in string.split('\n')
                         if line != "ADL Escape failed."
                         and line != "WARNING:endless loop detected!"
                         and line != "One module without kernel function!")

    @staticmethod
    def from_str(session: session_t, string: str) -> str:
        """
        Instantiate a Stdout object
        """
        # Strip the noise
        string = Stdout._escape(string)

        stdout = get_or_add(
            session, Stdout,
            sha1=crypto.sha1_str(string),
            stdout=string)
        return stdout


class CompilerError(object):
    id_t = sql.Integer

    @sql.ext.declarative.declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + "s"

    id = sql.Column(id_t, primary_key=True)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)


class StackDump(CompilerError, Base):
    stackdump = sql.Column(sql.UnicodeText(length=1024), nullable=False)


class Assertion(CompilerError, Base):
    assertion = sql.Column(sql.UnicodeText(length=1024), nullable=False)


class Unreachable(CompilerError, Base):
    unreachable = sql.Column(sql.UnicodeText(length=1024), nullable=False)


class Stderr(Base):
    id_t = sql.Integer
    __tablename__ = "stderrs"

    # Fields
    id = sql.Column(id_t, primary_key=True)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    assertion_id = sql.Column(Assertion.id_t, sql.ForeignKey("assertions.id"))
    unreachable_id = sql.Column(Unreachable.id_t, sql.ForeignKey("unreachables.id"))
    stackdump_id = sql.Column(StackDump.id_t, sql.ForeignKey("stackdumps.id"))
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    # Relationships
    assertion = sql.orm.relationship("Assertion")
    unreachable = sql.orm.relationship("Unreachable")
    stackdump = sql.orm.relationship("StackDump")

    @staticmethod
    def _escape(string: str) -> str:
        """ filter noise from test harness stderr """
        return '\n'.join(line for line in string.split('\n')
                         if "no version information available" not in line)

    @staticmethod
    def _get_assertion(session: session_t, lines: Iterable[str]) -> Union[None, Assertion]:
        clang_assertion = False
        strip = False

        for line in lines:
            if "assertion" in line.lower():
                if strip:
                    if line.startswith("cldrive-harness"):
                        msg = ":".join(line.split(":")[1:])
                    else:
                        msg = line
                    msg = re.sub(r"^ *:[0-9]+: ", "", msg)
                    if "Assertion `(null)' failed." in msg:
                        msg = "Assertion `(null)' failed."
                    elif "Assertion `' failed." in msg:
                        msg = "Assertion `' failed."
                    elif "Assertion `' failed." in msg:
                        msg = "Assertion `' failed."
                elif clang_assertion:
                    msg = ":".join(line.split(":")[3:])
                else:
                    msg = line

                assertion = get_or_add(
                    session, Assertion,
                    sha1=crypto.sha1_str(msg), assertion=msg)
                return assertion

    @staticmethod
    def _get_unreachable(session: session_t, lines: Iterable[str]) -> Union[None, Unreachable]:
        for line in lines:
            if "unreachable" in line.lower():
                unreachable = get_or_add(
                    session, Unreachable,
                    sha1=crypto.sha1_str(line), unreachable=line)
                return unreachable

    @staticmethod
    def _get_stackdump(session: session_t, lines: Iterable[str]) -> Union[None, StackDump]:
        in_stackdump = False
        stackdump = []
        for line in lines:
            if in_stackdump:
                if line and line[0].isdigit():
                    stackdump.append(line)
                else:
                    stackdump_ = "\n".join(stackdump)
                    stackdump = get_or_add(
                        session, StackDump,
                        sha1=crypto.sha1_str(stackdump_), stackdump=stackdump_)
                    return stackdump
            elif "stack dump:" in line.lower():
                in_stackdump = True

    @staticmethod
    def from_str(session: session_t, string: str) -> 'Stderr':
        # Strip the noise:
        string = Stderr._escape(string)

        # Get metadata:
        lines = string.split('\n')
        assertion = Stderr._get_assertion(session, lines)
        unreachable = Stderr._get_unreachable(session, lines)
        stackdump = Stderr._get_stackdump(session, lines)
        session.flush()

        # Sanity check:
        errs = sum(1 if x else 0 for x in [assertion, unreachable, stackdump])
        if errs > 1:
            raise LookupError(f"Multiple errors types found in: {stderr}\n\n" +
                              f"Assertion: {assertion}\n" +
                              f"Unreachable: {unreachable}\n" +
                              f"Stackdump: {stackdump}")

        stderr = get_or_add(
            session, Stderr,
            sha1=crypto.sha1_str(string),
            assertion=assertion,
            unreachable=unreachable,
            stackdump=stackdump,
            stderr=string)
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
    RC = 4
    TO = 5
    PASS = 6

    @staticmethod
    def to_str(outcomes: 'Outcomes.value_t') -> str:
        """ convert to long-form string """
        return {
            Outcomes.TODO: "unknown",
            Outcomes.BF: "build failure",
            Outcomes.BC: "build crash",
            Outcomes.BTO: "build timeout",
            Outcomes.RC: "runtime crash",
            Outcomes.TO: "timeout",
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
    testbed_id = sql.Column(Testbed.id_t, sql.ForeignKey("testbeds.id"), nullable=False, index=True)
    testcase_id = sql.Column(Testcase.id_t, sql.ForeignKey("testcases.id"), nullable=False, index=True)
    date = sql.Column(sql.DateTime, nullable=False, index=True, default=datetime.datetime.utcnow)
    returncode = sql.Column(sql.SmallInteger, nullable=False)
    outcome = sql.Column(Outcomes.column_t, nullable=False, index=True)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout_id = sql.Column(Stdout.id_t, sql.ForeignKey("stdouts.id"), nullable=False)
    stderr_id = sql.Column(Stderr.id_t, sql.ForeignKey("stderrs.id"), nullable=False)

    # Constraints
    __table_args__ = (
        sql.UniqueConstraint('testbed_id', 'testcase_id', name='unique_result_triple'),
    )

    # Relationships
    meta = sql.orm.relation("ResultMeta", back_populates="result")
    classification = sql.orm.relation("Classification", back_populates="result")
    reduction = sql.orm.relationship("Reduction", back_populates="result")
    testbed = sql.orm.relationship("Testbed")
    testcase = sql.orm.relationship("Testcase")
    stdout = sql.orm.relationship("Stdout")
    stderr = sql.orm.relationship("Stderr")

    def __repr__(self):
        return str(self.id)


class Cl_launcherResult(Result):
    @staticmethod
    def get_outcome(returncode: int, stderr: str, runtime: float, timeout: int) -> Outcomes.type:
        """
        Given a cl_launcher result, determine its outcome.
        See OUTCOMES for list of possible outcomes.
        """
        def crash_or_build_failure():
            return Outcomes.RC if "Compilation terminated successfully..." in stderr else Outcomes.BF
        def crash_or_build_crash():
            return Outcomes.RC if "Compilation terminated successfully..." in stderr else Outcomes.BC
        def timeout_or_build_timeout():
            return Outcomes.TO if "Compilation terminated successfully..." in stderr else Outcomes.BTO

        if returncode == 0:
            return Outcomes.PASS
        # 139 is SIGSEV
        elif returncode == 139 or returncode == -11:
            returncode = 139
            return crash_or_build_crash()
        # SIGTRAP
        elif returncode == -5:
            return crash_or_build_crash()
        # SIGKILL
        elif returncode == -9 and runtime >= timeout:
            return timeout_or_build_timeout()
        elif returncode == -9:
            logging.warn(f"SIGKILL, but only ran for {runtime:.2f}s")
            return crash_or_build_crash()
        # SIGILL
        elif returncode == -4:
            return crash_or_build_crash()
        # SIGABRT
        elif returncode == -6:
            return crash_or_build_crash()
        # SIGFPE
        elif returncode == -8:
            return crash_or_build_crash()
        # SIGBUS
        elif returncode == -7:
            return crash_or_build_crash()
        # cl_launcher error
        elif returncode == 1:
            return crash_or_build_failure()
        else:
            print(result)
            try:
                print(Signals(-returncode).name)
            except ValueError:
                print(returncode)
            raise LookupError(f"failed to determine outcome of Cl_launcherResult {returncode} with stderr: {stderr}")


class CldriveResult(Result):
    @staticmethod
    def get_outcome(returncode: int, stderr: str, runtime: float, timeout: int) -> Outcomes.type:
        """
        Given a cldrive result, determine its outcome.
        See OUTCOMES for list of possible outcomes.
        """
        def crash_or_build_failure():
            return Outcomes.RC if "[cldrive] Kernel: " in stderr else Outcomes.BF
        def crash_or_build_crash():
            return Outcomes.RC if "[cldrive] Kernel: " in stderr else Outcomes.BC
        def timeout_or_build_timeout():
            return Outcomes.TO if "[cldrive] Kernel: " in stderr else Outcomes.BTO

        if returncode == 0:
            return Outcomes.PASS
        # 139 is SIGSEV
        elif returncode == 139 or returncode == -11:
            returncode = 139
            return crash_or_build_crash()
        # SIGTRAP
        elif returncode == -5:
            return crash_or_build_crash()
        # SIGKILL
        elif returncode == -9 and runtime >= 60:
            return timeout_or_build_timeout()
        elif returncode == -9:
            logging.warn(f"SIGKILL, but only ran for {runtime:.2f}s")
            return crash_or_build_crash()
        # SIGILL
        elif returncode == -4:
            return crash_or_build_crash()
        # SIGFPE
        elif returncode == -8:
            return crash_or_build_crash()
        # SIGBUS
        elif returncode == -7:
            return crash_or_build_crash()
        # SIGABRT
        elif returncode == -6:
            return crash_or_build_crash()
        # cldrive error
        elif returncode == 1 and runtime >= 60:
            return timeout_or_build_timeout()
        elif returncode == 1:
            return crash_or_build_failure()
        # file not found (check the stderr on this one):
        elif returncode == 127:
            return crash_or_build_failure()
        else:
            print(result)
            try:
                print(Signals(-returncode).name)
            except ValueError:
                print(returncode)
            raise LookupError(f"failed to determine outcome of cldrive returncode {returncode} with stderr: {stderr}")


class ClangResult(Result):
    @staticmethod
    def get_outcome(returncode: int, stderr: str, runtime: float, timeout: int) -> Outcomes.type:
        """
        Given a clang result, determine its outcome.
        See Outcomes for list of possible outcomes.
        """
        def crash_or_build_failure():
            return Outcomes.RC if "Compilation terminated successfully..." in stderr else Outcomes.BF
        def crash_or_build_crash():
            return Outcomes.RC if "Compilation terminated successfully..." in stderr else Outcomes.BC
        def timeout_or_build_timeout():
            return Outcomes.TO if "Compilation terminated successfully..." in stderr else Outcomes.BTO

        if returncode == 0:
            return Outcomes.PASS
        elif returncode == -9 and runtime >= timeout:
            return Outcomes.BTO
        elif returncode == -9:
            logging.warn(f"SIGKILL, but only ran for {runtime:.2f}s")
        return Outcomes.BC


class ResultMeta(Base):
    id_t = Result.id_t

    __tablename__ = "results_metas"
    id = sql.Column(id_t, sql.ForeignKey("results.id"), primary_key=True)
    total_time = sql.Column(sql.Float, nullable=False)  # time to generate and run test case
    cumtime = sql.Column(sql.Float, nullable=False)  # culumative time for this testbed time

    result = sql.orm.relationship("Result", back_populates="meta")

    def __repr__(self):
        return (f"result: {self.id} total_time: {self.total_time:.3f}s, " +
                f"cumtime: {self.cumtime:.1f}s")


class Classifications:
    value_t = int
    column_t = sql.SmallInteger

    BC = 1
    BTO = 2
    ABF = 3
    ARC = 4
    AWO = 5

    @staticmethod
    def to_str(outcomes: 'Classifications.value_t') -> str:
        return {
            Classifications.BC: "build crash",
            Classifications.BTO: "build timeout",
            Classifications.ABF: "anomylous build failure",
            Classifications.ARC: "anomylous runtime crash",
            Classifications.AWO: "anomylous wrong output",
        }[outcomes]


class Classification(Base):
    id_t = Result.id_t

    __tablename__ = "classifications"
    id = sql.Column(id_t, sql.ForeignKey("results.id"), primary_key=True)
    classification = sql.Column(Classifications.column_t, nullable=False)

    result = sql.orm.relationship("Result", back_populates="classification")


class Majority(Base):
    id_t = Testcase.id_t

    __tablename__ = "majorities"
    id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
    num_results = sql.Column(sql.SmallInteger, nullable=False)
    maj_outcome = sql.Column(Outcomes.column_t, nullable=False)
    outcome_majsize = sql.Column(sql.SmallInteger, nullable=False)
    maj_stdout_id = sql.Column(Stdout.id_t, sql.ForeignKey("stdouts.id"), nullable=False)
    stdout_majsize = sql.Column(sql.SmallInteger, nullable=False)

    testcase = sql.orm.relationship("Testcase", back_populates="majority")
    maj_stdout = sql.orm.relationship("Stdout")


class Reduction(Base):
    id_t = Result.id_t

    __tablename__ = "reductions"
    id = sql.Column(id_t, sql.ForeignKey("results.id"), primary_key=True)
    date = sql.Column(sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    reduced_src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    linecount = sql.Column(sql.Integer, nullable=False)
    log = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    result = sql.orm.relationship("Result", back_populates="reduction")
