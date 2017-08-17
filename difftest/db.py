import datetime
import multiprocessing
import os
import sqlalchemy as sql

from collections import namedtuple
from contextlib import contextmanager
from enum import Enum
from labm8 import crypto, fs, system
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, List, Tuple, Union

import cfg
import deepsmith_pb2 as pb

# Global state to manage database connections. Must call init() before
# creating sessions.
Base = declarative_base()
engine = None
make_session = None


session_t = sql.orm.session.Session


def init(hostname: str, echo: bool=False) -> str:
    """
    Initialize database engine.

    Must be called before attempt to create a database connection.

    Arguments:
        hostname (str): Hostname of machine running MySQL database.

    Returns:
        str: URI of database.
    """
    global engine
    global make_session
    username, password = cfg.get_mysql_creds()
    table = cfg.DATABASE
    port = str(cfg.PORT)

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{table}?charset=utf8"
    echo = True if echo else True if os.environ.get("ECHO") else False
    engine = sql.create_engine(uri, encoding="utf-8", echo=echo)

    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    make_session = sql.orm.sessionmaker(bind=engine)

    return "mysql://{hostname}:{port}/{table}".format(**vars())


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


def get_or_create(session: sql.orm.session.Session, model,
                  defaults: Dict[str, object]=None, **kwargs) -> object:
    """
    Instantiate a mapped database object. If the object is not in the database,
    add it.
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if not instance:
        params = dict((k, v) for k, v in kwargs.items()
                      if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)

    return instance


def get_or_add(session: sql.orm.session.Session, model,
                  defaults: Dict[str, object]=None, **kwargs) -> object:
    """
    Instantiate a mapped database object. If the object is not in the database,
    add it.
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if not instance:
        params = dict((k, v) for k, v in kwargs.items()
                      if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)
        session.flush()

    return instance


# Programs ####################################################################


class Generators:
    value_t = int
    column_t = sql.SmallInteger

    CLSMITH = 0
    DSMITH = 1


class Program(Base):
    id_t = sql.Integer

    __tablename__ = 'programs'
    id = sql.Column(id_t, primary_key=True)
    generator = sql.Column(Generators.column_t, nullable=False)
    sha1 = sql.Column(sql.String(40), nullable=False)
    date = sql.Column(sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
    generation_time = sql.Column(sql.Float, nullable=False)
    linecount = sql.Column(sql.Integer, nullable=False)
    charcount = sql.Column(sql.Integer, nullable=False)
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    __table_args__ = (
        sql.UniqueConstraint('generator', 'sha1', name='uniq_program'),
    )

    testcases = sql.orm.relationship("Testcase", back_populates="program")

    def __repr__(self):
        return f"program[{self.id}] = {{ generator = {self.generator}, sha1 = {self.sha1} }}"


class ClsmithProgramMeta(Base):
    id_t = Program.id_t

    __tablename__ = "clsmith_program_metas"
    id = sql.Column(id_t, sql.ForeignKey("programs.id"), primary_key=True)
    flags = sql.Column(sql.String(80), nullable=False, default="")

    program = sql.orm.relationship("Program")


class DsmithProgramMeta(Base):
    id_t = Program.id_t

    __tablename__ = "dsmith_program_metas"
    id = sql.Column(id_t, sql.ForeignKey("programs.id"), primary_key=True)
    contains_floats = sql.Column(sql.Boolean)
    compiler_warnings = sql.Column(sql.Boolean)

    program = sql.orm.relationship("Program")


# Parameters ##################################################################


class Threads(Base):
    id_t = sql.SmallInteger

    __tablename__ = "threads"
    id = sql.Column(id_t, primary_key=True)
    gsize_x = sql.Column(sql.Integer, nullable=False)
    gsize_y = sql.Column(sql.Integer, nullable=False)
    gsize_z = sql.Column(sql.Integer, nullable=False)
    lsize_x = sql.Column(sql.Integer, nullable=False)
    lsize_y = sql.Column(sql.Integer, nullable=False)
    lsize_z = sql.Column(sql.Integer, nullable=False)

    __table_args__ = (sql.UniqueConstraint(
        'gsize_x', 'gsize_y', 'gsize_z',
        'lsize_x', 'lsize_y', 'lsize_z',
        name='unique_thread_size'),)

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

    COMPILE_ONLY = -1
    CLSMITH = 0
    DSMITH = 1

    @staticmethod
    def to_str(harness: 'Harnesses.value_t') -> str:
        return {
            Harnesses.COMPILE_ONLY: "compile only",
            Harnesses.CLSMITH: "CLsmith",
            Harnesses.DSMITH: "DeepSmith",
        }[harness]

    @staticmethod
    def result_t(harness: "Harnesses.value_t") -> Union[ClsmithResult, DsmithResult]
        return {
            Harnesses.CLSMITH: ClsmithResult,
            Harnesses.DSMITH: DsmithResult,
        }[harness]


class Testcase(Base):
    id_t = sql.Integer

    __tablename__ = "testcases"
    id = sql.Column(id_t, primary_key=True)
    program_id = sql.Column(Program.id_t, sql.ForeignKey("programs.id"), nullable=False)
    threads_id = sql.Column(Threads.id_t, sql.ForeignKey("threads.id"), nullable=False)
    harness = sql.Column(Harnesses.column_t, nullable=False)
    input_seed = sql.Column(sql.Integer, nullable=False)
    timeout = sql.Column(sql.Integer, nullable=False)

    __table_args__ = (
        sql.UniqueConstraint("program_id", "threads_id", "harness",
                             "input_seed", "timeout", name="unique_testcase"),
    )

    program = sql.orm.relationship("Program", back_populates="testcases")
    threads = sql.orm.relationship("Threads", back_populates="testcases")
    results = sql.orm.relationship("Result", back_populates="testcase")
    majority = sql.orm.relationship("Majority", back_populates="testcase")
    clsmith_meta = sql.orm.relationship("ClsmithTestcaseMeta", back_populates="testcase")
    dsmith_meta = sql.orm.relationship("DsmithTestcaseMeta", back_populates="testcase")

    def __repr__(self):
        return f"testcase {self.id} = {{program: {self.program_id}, threads: {self.threads} }}"


class ClsmithTestcaseMeta(Base):
    id_t = Testcase.id_t

    __tablename__ = "clsmith_testcase_metas"
    id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
    oclverified = sql.Column(sql.Boolean)

    testcase = sql.orm.relationship("Testcase", back_populates="clsmith_meta")

    def get_oclverified():
        raise NotImplementedError


class DsmithTestcaseMeta(Base):
    id_t = Testcase.id_t

    __tablename__ = "dsmith_testcase_metas"
    id = sql.Column(id_t, sql.ForeignKey("testcases.id"), primary_key=True)
    gpuverified = sql.Column(sql.Boolean)
    oclverified = sql.Column(sql.Boolean)

    testcase = sql.orm.relationship("Testcase", back_populates="dsmith_meta")

    def get_oclverified():
        raise NotImplementedError

    def get_gpuverified():
        raise NotImplementedError

    def get_oclverified():
        raise NotImplementedError

    def get_contains_floats():
        raise NotImplementedError

    def get_compiler_warnings():
        raise NotImplementedError


# Experimental Platforms ######################################################


class Platform(Base):
    id_t = sql.SmallInteger

    __tablename__ = 'platforms'
    id = sql.Column(id_t, primary_key=True)
    platform = sql.Column(sql.String(255), nullable=False)  # CL_PLATFORM_NAME
    device = sql.Column(sql.String(255), nullable=False)  # CL_DEVICE_NAME
    driver = sql.Column(sql.String(255), nullable=False)  # CL_DRIVER_VERSION
    opencl = sql.Column(sql.String(8), nullable=False)  # CL_PLATFORM_VERSION
    devtype = sql.Column(sql.String(12), nullable=False)  # CL_DEVICE_TYPE
    host = sql.Column(sql.String(255), nullable=False)

    __table_args__ = (
        sql.UniqueConstraint('platform', 'device', 'driver', 'opencl',
                             'devtype', 'host', name='unique_platform'),
    )

    testbeds = sql.orm.relationship("Testbed", back_populates="platform")

    def __repr__(self) -> str:
        return (f"Platform: {self.platform}, Device: {self.device}, " +
                f"Driver: {self.driver}, Host: {self.host}")

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
    def num(self):
        return {
            "ComputeAorta (Intel E5-2620)": 9,
            "GeForce GTX 1080": 1,
            "GeForce GTX 780": 2,
            "Intel E5-2620 v4": 4,
            "Intel E5-2650 v2": 5,
            "Intel HD Haswell GT2": 3,
            "Intel i5-4570": 6,
            "Intel Xeon Phi": 7,
            "Olcgrind Simulator": 10,
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
            "CentOS Linux 7.1.1503 64bit": "CentOS 7.1 64bit"
        }.get(self.host.strip(), self.host.strip())


class Testbed(Base):
    id_t = sql.SmallInteger

    __tablename__ = 'testbeds'
    id = sql.Column(id_t, primary_key=True)
    platform_id = sql.Column(Platform.id_t, sql.ForeignKey("platforms.id"), nullable=False)
    optimizations = sql.Column(sql.Boolean, nullable=False)

    __table_args__ = (
        sql.UniqueConstraint('platform_id', 'optimizations', name='unique_testbed'),
    )

    platform = sql.orm.relationship("Platform", back_populates="testbeds")

    def __repr__(self) -> str:
        return self.num

    @property
    def num(self) -> str:
        p = self.platform.platform_name if self.platform.num == -1 else self.platform.num
        return f"{p}{self.plus_minus}"

    @property
    def plus_minus(self) -> str:
        return "+" if self.optimizations else "-"



class Stdout(Base):
    id_t = sql.Integer

    __tablename__ = "stdouts"
    id = sql.Column(id_t, primary_key=True)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    @staticmethod
    def escape(string):
        """ filter noise from test harness stdout """
        return '\n'.join(line for line in stdout.split('\n')
                         if line != "ADL Escape failed."
                         and line != "WARNING:endless loop detected!"
                         and line != "One module without kernel function!")


class CompilerError(object):
    id_t = sql.Integer

    @sql.ext.declarative.declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + "s"

    id = sql.Column(id_t, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    stackdump = sql.Column(sql.UnicodeText(length=1024), nullable=False)


class StackDump(CompilerError, Base):
    pass


class Assertion(CompilerError, Base):
    pass


class Unreachable(CompilerError, Base):
    pass


class Stderr(Base):
    id_t = sql.Integer

    __tablename__ = "stderrs"
    id = sql.Column(id_t, primary_key=True)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    assertion_id = sql.Column(Assertion.id_t, sql.ForeignKey("assertions.id"))
    unreachable_id = sql.Column(Unreachable.id_t, sql.ForeignKey("unreachables.id"))
    stackdump_id = sql.Column(StackDump.id_t, sql.ForeignKey("stackdumps.id"))
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    assertion = sql.orm.relationship("Assertion")
    unreachable = sql.orm.relationship("Unreachable")
    stackdump = sql.orm.relationship("StackDump")

    @staticmethod
    def escape(string):
        """ filter noise from test harness stderr """
        return '\n'.join(line for line in stderr.split('\n')
                         if "no version information available" not in line)


class Outcomes:
    type = int
    column_t = sql.SmallInteger

    TODO = -1
    BF = 1
    BC = 2
    BTO = 3
    C = 4
    TO = 5
    PASS = 6

    @staticmethod
    def to_str(outcomes: 'Outcomes.value_t') -> str:
        return {
            Outcomes.TODO: "unknown",
            Outcomes.BF: "build failure",
            Outcomes.BC: "build crash",
            Outcomes.BTO: "build timeout",
            Outcomes.C: "runtime crash",
            Outcomes.TO: "timeout",
            Outcomes.PASS: "pass",
        }[outcomes]


class Result(Base):
    id_t = sql.Integer

    __tablename__ = "results"
    id = sql.Column(id_t, primary_key=True)
    testbed_id = sql.Column(Testbed.id_t, sql.ForeignKey("testbeds.id"), nullable=False, index=True)
    testcase_id = sql.Column(Testcase.id_t, sql.ForeignKey("testcases.id"), nullable=False, index=True)
    date = sql.Column(sql.DateTime, nullable=False, index=True, default=datetime.datetime.utcnow)
    returncode = sql.Column(sql.SmallInteger, nullable=False)
    outcome = sql.Column(Outcomes.column_t, nullable=False, index=True)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout_id = sql.Column(Stdout.id_t, sql.ForeignKey("stdouts.id"), nullable=False)
    stderr_id = sql.Column(Stderr.id_t, sql.ForeignKey("stderrs.id"), nullable=False)

    __table_args__ = (
        sql.UniqueConstraint('testbed_id', 'testcase_id', name='unique_result_triple'),
    )

    meta = sql.orm.relation("ResultMeta", back_populates="result")
    classification = sql.orm.relation("Classification", back_populates="result")
    reduction = sql.orm.relationship("Reduction", back_populates="result")
    testbed = sql.orm.relationship("Testbed")
    testcase = sql.orm.relationship("Testcase")
    stdout = sql.orm.relationship("Stdout")
    stderr = sql.orm.relationship("Stderr")

    # FIXME:
    # def __repr__(self):
    #     return (f"result[{self.id}] = {{ {self.testbed.platform.device}, " +
    #             f"testcase: {self.testcase.id}, returncode: {self.returncode}, runtime: {self.runtime:.2}s }}")


class ClsmithResult(Result):
    @staticmethod
    def get_outcome(returncode: int, stderr: str, runtime: float, timeout: int) -> int:
        """
        Given a cl_launcher result, determine and set it's outcome.

        See OUTCOMES for list of possible outcomes.
        """
        def crash_or_build_failure():
            return Outcomes.C if "Compilation terminated successfully..." in stderr else Outcomes.BF
        def crash_or_build_crash():
            return Outcomes.C if "Compilation terminated successfully..." in stderr else Outcomes.BC
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
            print(f"SIGKILL, but only ran for {runtime:.2f}s")
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
            raise LookupError(f"failed to determine outcome of ClsmithResult {returncode} with stderr: {stderr}")


class DsmithResult(Result):
    @staticmethod
    def get_outcome(returncode: int, stderr: str, runtime: float, timeout: int) -> int:
        def crash_or_build_failure():
            return Outcomes.C if "[cldrive] Kernel: " in stderr else Outcomes.BF
        def crash_or_build_crash():
            return Outcomes.C if "[cldrive] Kernel: " in stderr else Outcomes.BC
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
            print(f"SIGKILL, but only ran for {runtime:.2f}s")
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
    type = int
    column_t = sql.SmallInteger

    BC = 1
    BTO = 2
    ABF = 3
    ARC = 4
    AWO = 5
    PASS = 6


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


# Utility #####################################################################


def get_testbed(session: session_t, platform: str, device: str) -> Testbed:
    """
    Get the testbed for the specified hardware.

    Arguments:
        platform (str): Name of the OpenCL platform.
        device (str): Name of the OpenCL device.

    Returns:
        Testbed: If no testbed already exists, create one.
    """
    import pyopencl as cl
    import cldrive

    env = cldrive.make_env(platform=platform, device=device)

    return get_or_add(
        session, Testbed,
        platform=platform,
        device=device,
        driver=env.driver_version,
        host=cldrive.host_os(),
        opencl=env.opencl_version,
        devtype=env.device_type)
