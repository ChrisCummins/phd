import clgen
import datetime
import enum
import sqlalchemy as sql
import os

from collections import namedtuple
from configparser import ConfigParser
from contextlib import contextmanager
from labm8 import system
from labm8 import fs
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
from typing import Dict, List, Tuple

Base = declarative_base()

# must call init() first
engine = None
make_session = None


OUTCOMES = {
    1: "bf",
    2: "bc",
    3: "bto",
    4: "c",
    5: "to",
    6: "pass",
}

OUTCOMES_TO_INT = {
    "bf": 1,
    "bc": 2,
    "bto": 3,
    "c": 4,
    "to": 5,
    "pass": 6,
}

CLASSIFICATIONS = {
    1: "w",
    2: "bf",
    3: "c",
    4: "to",
    5: "pass",
}

CLASSIFICATIONS_TO_INT = {
    "w": 1,
    "bf": 2,
    "c": 3,
    "to": 4,
    "pass": 5,
}


def get_mysql_creds() -> Tuple[str, str]:
    """ read default MySQL credentials in ~/.my.cnf """
    config = ConfigParser()
    config.read(fs.path("~/.my.cnf"))
    return config['mysql']['user'], config['mysql']['password']


def init(hostname: str) -> str:
    """ must be called before using anything """
    global engine
    global make_session
    username, password = get_mysql_creds()
    table = "project_b"
    port = "3306"

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{table}?charset=utf8"
    echo = True if os.environ.get("ECHO") else False
    engine = sql.create_engine(uri, encoding="utf-8", echo=echo)

    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    make_session = sql.orm.sessionmaker(bind=engine)

    return "mysql://{hostname}:{port}/{table}".format(**vars())


# session type
session_t = sql.orm.session.Session


@contextmanager
def Session(commit: bool=True) -> session_t:
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
        params = dict((k, v) for k, v in kwargs.items() if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)

    return instance


# Programs & Harnesses ########################################################


class CLSmithProgram(Base):
    """ programs """
    __tablename__ = 'CLSmithPrograms'
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True, index=True)

    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    # additional flags passed to CLSmith
    flags = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    # time taken to produce program (in seconds).
    runtime = sql.Column(sql.Float, nullable=False)

    # production output
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    linecount = sql.Column(sql.Integer, nullable=False)

    # relation back to results:
    testcases = sql.orm.relationship("CLSmithTestCase", back_populates="program")

    def __repr__(self):
        return self.id


class CLgenProgram(Base):
    """ programs """
    __tablename__ = 'CLgenPrograms'
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True, index=True)

    date_added = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    # time taken to produce program (in seconds).
    runtime = sql.Column(sql.Float, nullable=False)

    # production output
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    linecount = sql.Column(sql.Integer, nullable=False)

    # stats
    cl_launchable = sql.Column(sql.Boolean)
    gpuverified = sql.Column(sql.Boolean)
    throws_warnings = sql.Column(sql.Boolean)

    # relations:
    testcases = sql.orm.relationship("CLgenTestCase", back_populates="program")

    def __repr__(self) -> str:
        return self.id


# Parameters ##################################################################


class cl_launcherParams(Base):
    """ params used by cl_launcher to run kernel """
    __tablename__ = "cl_launcherParams"
    id = sql.Column(sql.Integer, primary_key=True)
    optimizations = sql.Column(sql.Boolean, nullable=False)
    gsize_x = sql.Column(sql.Integer, nullable=False)
    gsize_y = sql.Column(sql.Integer, nullable=False)
    gsize_z = sql.Column(sql.Integer, nullable=False)
    lsize_x = sql.Column(sql.Integer, nullable=False)
    lsize_y = sql.Column(sql.Integer, nullable=False)
    lsize_z = sql.Column(sql.Integer, nullable=False)
    # unique combination of values:
    __table_args__ = (sql.UniqueConstraint(
        'optimizations', 'gsize_x', 'gsize_y', 'gsize_z',
        'lsize_x', 'lsize_y', 'lsize_z', name='_uid'),)

    def to_flags(self) -> List[str]:
        flags = [
            "-g", "{self.gsize_x},{self.gsize_y},{self.gsize_z}".format(**vars()),
            "-l", "{self.lsize_x},{self.lsize_y},{self.lsize_z}".format(**vars())
        ]
        if not self.optimizations:
            flags.append("---disable_opts")
        return flags

    @property
    def optimizations_on_off(self) -> str:
        return "on" if self.optimizations else "off"

    @property
    def gsize(self) -> Tuple[int, int, int]:
        return (self.gsize_x, self.gsize_y, self.gsize_z)

    @property
    def lsize(self) -> Tuple[int, int, int]:
        return (self.lsize_x, self.lsize_y, self.lsize_z)

    def __repr__(self) -> str:
        return " ".join(self.to_flags())


class cldriveParams(Base):
    """ params used by cldrive to run kernel """
    __tablename__ = "cldriveParams"
    id = sql.Column(sql.Integer, primary_key=True)
    size = sql.Column(sql.Integer, nullable=False)
    generator = sql.Column(sql.String(12), nullable=False)
    scalar_val = sql.Column(sql.Integer)
    gsize_x = sql.Column(sql.Integer, nullable=False)
    gsize_y = sql.Column(sql.Integer, nullable=False)
    gsize_z = sql.Column(sql.Integer, nullable=False)
    lsize_x = sql.Column(sql.Integer, nullable=False)
    lsize_y = sql.Column(sql.Integer, nullable=False)
    lsize_z = sql.Column(sql.Integer, nullable=False)
    optimizations = sql.Column(sql.Boolean, nullable=False)
    # unique combination of values:
    __table_args__ = (sql.UniqueConstraint(
        'size', 'generator', 'scalar_val', 'gsize_x', 'gsize_y', 'gsize_z',
        'lsize_x', 'lsize_y', 'lsize_z', 'optimizations', name='_uid'),)

    def to_flags(self):
        flags = [
            "-s", f"{self.size}",
            "-i", f"{self.generator}",
            "-g", f"{self.gsize_x},{self.gsize_y},{self.gsize_z}",
            "-l", f"{self.lsize_x},{self.lsize_y},{self.lsize_z}"
        ]
        if self.scalar_val is not None:
            flags.append(f"--scalar-val={self.scalar_val}")
        if not self.optimizations:
            flags.append("--no-opts")
        return flags

    @property
    def optimizations_on_off(self):
        return "on" if self.optimizations else "off"

    @property
    def gsize(self):
        return (self.gsize_x, self.gsize_y, self.gsize_z)

    @property
    def lsize(self):
        return (self.lsize_x, self.lsize_y, self.lsize_z)

    def __repr__(self):
        return " ".join(self.to_flags())


# Testcases ###################################################################


class CLSmithTestCase(Base):
    __tablename__ = "CLSmithTestCases"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithPrograms.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cl_launcherParams.id"),
                           nullable=False)

    oclverified = sql.Column(sql.Boolean)

    __table_args__ = (
        sql.UniqueConstraint("program_id", "params_id", name="_uid"),)

    program = sql.orm.relationship("CLSmithProgram", back_populates="testcases")
    params = sql.orm.relationship("cl_launcherParams")
    results = sql.orm.relationship("CLSmithResult", back_populates="testcase")

    def __repr__(self):
        return f"testcase {self.id} = {{program: {self.program_id}, params: {self.params_id} }}"


class CLgenTestCase(Base):
    __tablename__ = "CLgenTestCases"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenPrograms.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cldriveParams.id"),
                           nullable=False)

    gpuverified = sql.Column(sql.Boolean)
    oclverified = sql.Column(sql.Boolean)
    contains_floats = sql.Column(sql.Boolean)
    compiler_warnings = sql.Column(sql.Boolean)

    __table_args__ = (
        sql.UniqueConstraint("program_id", "params_id", name="_uid"),)

    program = sql.orm.relationship("CLgenProgram", back_populates="testcases")
    params = sql.orm.relationship("cldriveParams")
    harness = sql.orm.relationship("CLgenHarness", back_populates="testcase")
    results = sql.orm.relationship("CLgenResult", back_populates="testcase")


class CLgenHarness(Base):
    """ cldrive-generated test harnesses for CLgen programs """
    __tablename__ = 'CLgenHarnesses'
    id = sql.Column(sql.Integer, sql.ForeignKey("CLgenTestCases.id"),
                    primary_key=True)

    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    # cldrive version which generated harness
    cldrive_version = sql.Column(sql.String(12))
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    compile_only = sql.Column(sql.Boolean)

    # time taken to create harness
    generation_time = sql.Column(sql.Float, nullable=False)
    compile_time = sql.Column(sql.Float, nullable=False)

    # relations:
    testcase = sql.orm.relationship("CLgenTestCase", back_populates="harness")

    def __repr__(self) -> str:
        return self.id


# Testbeds ####################################################################


class Testbed(Base):
    """ devices """
    __tablename__ = 'Testbeds'
    id = sql.Column(sql.Integer, primary_key=True)
    platform = sql.Column(sql.String(255), nullable=False)  # CL_PLATFORM_NAME
    device = sql.Column(sql.String(255), nullable=False)  # CL_DEVICE_NAME
    driver = sql.Column(sql.String(255), nullable=False)  # CL_DRIVER_VERSION
    host = sql.Column(sql.String(255), nullable=False)
    opencl = sql.Column(sql.String(8), nullable=False)  # CL_PLATFORM_VERSION
    devtype = sql.Column(sql.String(12), nullable=False)  # CL_DEVICE_TYPE

    __table_args__ = (
        sql.UniqueConstraint('platform', 'device', 'driver', 'host',
                             'opencl', 'devtype', name='_uid'),)

    clsmith_results = sql.orm.relationship("CLSmithResult", back_populates="testbed")
    clgen_results = sql.orm.relationship("CLgenResult", back_populates="testbed")
    bug_reports = sql.orm.relationship("BugReport", back_populates="testbed")

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


# CLSmith Results #############################################################


class CLSmithResult(Base):
    __tablename__ = "CLSmithResults"
    id = sql.Column(sql.Integer, primary_key=True)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False, index=True)
    testcase_id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithTestCases.id"),
                             nullable=False, index=True)

    # stats
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow,
                      nullable=False, index=True)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    # output
    stdout_id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithStdouts.id"))
    stderr_id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithStderrs.id"))

    outcome = sql.Column(sql.Integer, index=True, nullable=False)

    # unique
    __table_args__ = (
        sql.UniqueConstraint('testbed_id', 'testcase_id', name='_uid'),)

    # relations:
    meta = sql.orm.relation("CLSmithMeta", back_populates="result")
    classification = sql.orm.relation("CLSmithClassification", back_populates="result")
    testbed = sql.orm.relationship("Testbed", back_populates="clsmith_results")
    testcase = sql.orm.relationship("CLSmithTestCase", back_populates="results")
    stdout = sql.orm.relationship("CLSmithStdout")
    stderr = sql.orm.relationship("CLSmithStderr")

    def __repr__(self):
        return (f"result: {self.id} testbed: {self.testbed.device}, " +
                f"program: {self.program_id}, params: {self.params}, " +
                f"status: {self.status}, runtime: {self.runtime:.2f}s")


class CLSmithStdout(Base):
    __tablename__ = "CLSmithStdouts"
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


class CLSmithStderr(Base):
    __tablename__ = "CLSmithStderrs"
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


class CLSmithMeta(Base):
    __tablename__ = "CLSmithMetas"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithResults.id"),
                    primary_key=True)
    total_time = sql.Column(sql.Float, nullable=False)  # time to generate and run test case
    cumtime = sql.Column(sql.Float, nullable=False)  # culumative time for this device and optimization level

    # relations:
    result = sql.orm.relationship("CLSmithResult", back_populates="meta")

    def __repr__(self):
        return (f"result: {self.id} total_time: {self.total_time:.3f}s, " +
                f"cumtime: {self.cumtime:.1f}s")


class CLSmithClassification(Base):
    __tablename__ = "CLSmithClassifications"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithResults.id"),
                    primary_key=True)
    classification = sql.Column(sql.Integer, nullable=False)

    result = sql.orm.relationship("CLSmithResult", back_populates="classification")

    @property
    def label(self):
        return INT_TO_CLASSIFICATIONS[self.classification]


# CLgen Results ###############################################################


class CLgenResult(Base):
    __tablename__ = "CLgenResults"
    id = sql.Column(sql.Integer, primary_key=True)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    testcase_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenTestCases.id"),
                             nullable=False)

    # stats
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow,
                      nullable=False, index=True)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    # output
    stdout_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenStdouts.id"))
    stderr_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenStderrs.id"))

    outcome = sql.Column(sql.Integer, index=True, nullable=False)

    __table_args__ = (
        sql.UniqueConstraint('testbed_id', 'testcase_id', name='_uid'),
        sql.Index('testcase_outcome', 'testcase_id', 'outcome')
    )

    # relations:
    meta = sql.orm.relationship("CLgenMeta", back_populates="result")
    classification = sql.orm.relation("CLgenClassification", back_populates="result")
    testcase = sql.orm.relationship("CLgenTestCase", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="clgen_results")
    stdout = sql.orm.relationship("CLgenStdout")
    stderr = sql.orm.relationship("CLgenStderr")

    def __repr__(self) -> str:
        return (f"program: {self.program_id}, testcase: {self.testcase_id}, " +
                f"status: {self.status}, runtime: {self.runtime:.2f}s")


class CLgenStdout(Base):
    __tablename__ = "CLgenStdouts"
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


class CLgenStderr(Base):
    __tablename__ = "CLgenStderrs"
    id = sql.Column(sql.Integer, primary_key=True)
    hash = sql.Column(sql.String(40), nullable=False, unique=True)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


class CLgenMeta(Base):
    __tablename__ = "CLgenMetas"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLgenResults.id"),
                    primary_key=True)
    total_time = sql.Column(sql.Float, nullable=False)  # time to generate and run test case
    cumtime = sql.Column(sql.Float, nullable=False)  # culumative time for this device and optimization level

    # relations:
    result = sql.orm.relationship("CLgenResult", back_populates="meta")

    def __repr__(self):
        return ("result: {self.id} "
                "total_time: {self.total_time:.3f}s, "
                "cumtime: {self.cumtime:.1f}s, "
                .format(**vars()))


class CLgenClassification(Base):
    __tablename__ = "CLgenClassifications"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLgenResults.id"),
                    primary_key=True)
    classification = sql.Column(sql.Integer, nullable=False)

    result = sql.orm.relationship("CLgenResult", back_populates="classification")

    @property
    def label(self):
        return INT_TO_CLASSIFICATIONS[self.classification]


# Reductions ##################################################################


class CLSmithReduction(Base):
    __tablename__ = "CLSmithReductions"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithResults.id"), primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    log = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    result = sql.orm.relationship("CLSmithResult")


class CLgenReduction(Base):
    __tablename__ = "CLgenReductions"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLgenResults.id"), primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    log = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    result = sql.orm.relationship("CLgenResult")


# Compile-only tests ##########################################################


class coParams(Base):
    """ params used by compile-only """
    __tablename__ = "coParams"
    id = sql.Column(sql.Integer, primary_key=True)
    optimizations = sql.Column(sql.Boolean, nullable=False)
    build_kernel = sql.Column(sql.Boolean, nullable=False)

    # unique combination of values:
    __table_args__ = (sql.UniqueConstraint(
        'optimizations', 'build_kernel', name='_uid'),)
    # relation back to results:
    clgen_results = sql.orm.relationship("coCLgenResult", back_populates="params")

    def to_flags(self) -> List[str]:
        flags = ['--emit-c', '--compile-only']
        if self.build_kernel:
            flags.append("--with-kernel")
        if not self.optimizations:
            flags.append("--no-opts")
        return flags

    @property
    def optimizations_on_off(self) -> str:
        return "on" if self.optimizations else "off"

    def __repr__(self) -> str:
        return " ".join(self.to_flags())


class coCLgenResult(Base):
    """ CLgen programs ran using --compile-only """
    __tablename__ = "coCLgenResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLgenPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("coParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))
    classification = sql.Column(sql.String(16))
    submitted = sql.Column(sql.Boolean)
    dupe = sql.Column(sql.Integer, sql.ForeignKey("coCLgenResults.id"))

    program = sql.orm.relationship("CLgenProgram")
    testbed = sql.orm.relationship("Testbed")
    params = sql.orm.relationship("coParams")

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


# GitHub tests ################################################################


class GitHubProgram(Base):
    """ programs """
    __tablename__ = 'GitHubPrograms'
    id = sql.Column(sql.String(255), primary_key=True)
    date_added = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    status = sql.Column(sql.Integer)

    def __repr__(self) -> str:
        return self.id


# cl_launcher tests ###########################################################


class cl_launcherCLgenResult(Base):
    """ CLgen programs ran using cl_launcher """
    __tablename__ = "cl_launcherCLgenResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLgenPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cl_launcherParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    flags = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))
    classification = sql.Column(sql.String(16))
    submitted = sql.Column(sql.Boolean)
    dupe = sql.Column(sql.Boolean)

    program = sql.orm.relationship("CLgenProgram")
    testbed = sql.orm.relationship("Testbed")
    params = sql.orm.relationship("cl_launcherParams")

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


# Miscellaneous ###############################################################


class BugReport(Base):
    __tablename__ = "BugReports"
    id = sql.Column(sql.Integer, primary_key=True)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    classification = sql.Column(sql.String(12))

    testcase_url = sql.Column(sql.String(255))
    reported_url = sql.Column(sql.String(255))
    notes = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow,
                      nullable=False)
    rejected = sql.Column(sql.Boolean)
    fixed = sql.Column(sql.Boolean)

    testbed = sql.orm.relationship("Testbed", back_populates="bug_reports")

    __table_args__ = (
        sql.UniqueConstraint('testbed_id', 'testcase_url', name='_uid'),)

    def __repr__(self) -> str:
        return ("report: {self.id}, "
                "testbed: {self.testbed_id}, "
                "url: {self.url}, "
                "rejected: {self.rejected}, "
                "fixed: {self.fixed}"
                .format(**vars()))


class CLgenProgramTranslation(Base):
    __tablename__ = 'tmp_clgenprogram_translate'
    old_id = sql.Column(sql.String(40), primary_key=True)
    new_id = sql.Column(sql.Integer, nullable=False, unique=True, index=True)


class CLSmithProgramTranslation(Base):
    __tablename__ = 'tmp_clsmithprogram_translate'
    old_id = sql.Column(sql.String(40), primary_key=True)
    new_id = sql.Column(sql.Integer, nullable=False, unique=True, index=True)


# Utility #####################################################################


def get_testbed(session: session_t, platform: str, device: str) -> Testbed:
    """
    Get the testbed for the specified hardware.
    """
    import pyopencl as cl
    import cldrive

    env = cldrive.make_env(platform=platform, device=device)

    return get_or_create(session, Testbed,
                         platform=platform,
                         device=device,
                         driver=env.driver_version,
                         host=cldrive.host_os(),
                         opencl=env.opencl_version,
                         devtype=env.device_type)


# Tablesets ###################################################################
Tableset = namedtuple('Tableset', [
        'name',
        'results',
        'testcases',
        'programs',
        'harnesses',
        'params',
        'reductions',
        'meta',
        'classifications',
        'stdouts',
        'stderrs',
    ])

CLSMITH_TABLES = Tableset(name="CLSmith",
    results=CLSmithResult, testcases=CLSmithTestCase,
    programs=CLSmithProgram, harnesses=None,
    params=cl_launcherParams, reductions=CLSmithReduction,
    meta=CLSmithMeta, classifications=CLSmithClassification,
    stdouts=CLSmithStdout, stderrs=CLSmithStderr)
CLGEN_TABLES = Tableset(name="CLgen",
    results=CLgenResult, testcases=CLgenTestCase,
    programs=CLgenProgram, harnesses=CLgenHarness,
    params=cldriveParams, reductions=CLgenReduction,
    meta=CLgenMeta, classifications=CLgenClassification,
    stdouts=CLgenStdout, stderrs=CLgenStderr)


class InsufficientDataError(ValueError):
    """ raised if not enough results """
    pass


def results_in_order(session, tables: Tableset, testbed_id: int,
                     no_opt: bool, *return_values, reverse=False):
    if not len(return_values):
        return_values = (tables.results,)

    optimizations = not no_opt
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    q = session.query(*return_values)\
        .join(tables.meta)\
        .join(tables.testcases)\
        .outerjoin(tables.classifications)\
        .filter(tables.results.testbed_id == testbed_id,
                tables.testcases.params_id.in_(param_ids))

    if reverse:
        q = q.order_by(tables.meta.cumtime.desc())
    else:
        q = q.order_by(tables.meta.cumtime)

    return q


def results_in_timelimit(session, tables: Tableset, testbed_id: int,
                         no_opt: bool, time_limit: int,
                         *return_values, filter=None):
    """
    Raises:
        InsufficientDataError: If run out of results before time_limit is
            reached.
    """
    q = results_in_order(session, tables, testbed_id, no_opt, *return_values, tables.meta.cumtime)

    if filter is not None:
        q = q.filter(filter)

    vals = [0]
    for vals in q:
        if vals[-1] > time_limit:
            break
        yield vals[:-1]
    else:
        # Didn't reach time limit
        import util
        total_hours = vals[-1] / 3600
        testbed = session.query(Testbed).filter(Testbed.id == testbed_id).first()
        devname = util.device_str(testbed.device)
        raise InsufficientDataError(f"insufficient {tables.results.__tablename__} for {devname} {no_opt} ({total_hours:.1f} hs)")
