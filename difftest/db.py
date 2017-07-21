import datetime
import sqlalchemy as sql

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
make_session = None


def get_mysql_creds() -> Tuple[str, str]:
    """ read default MySQL credentials in ~/.my.cnf """
    config = ConfigParser()
    config.read(fs.path("~/.my.cnf"))
    return config['mysql']['user'], config['mysql']['password']


def init(hostname: str) -> str:
    """ must be called before using anything """
    global make_session
    username, password = get_mysql_creds()
    table = "project_b"
    port = "3306"

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{table}?charset=utf8"
    engine = sql.create_engine(uri, encoding="utf-8")

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


# Database Schema


class CLSmithProgram(Base):
    """ programs """
    __tablename__ = 'CLSmithPrograms'
    id = sql.Column(sql.String(40), primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    # additional flags passed to CLSmith
    flags = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    # time taken to produce program (in seconds).
    runtime = sql.Column(sql.Float, nullable=False)

    # production output
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    # relation back to results:
    cl_launcher_results = sql.orm.relationship("CLSmithResult", back_populates="program")
    cldrive_results = sql.orm.relationship("cldriveCLSmithResult", back_populates="program")

    def __repr__(self):
        return self.id


class CLgenProgram(Base):
    """ programs """
    __tablename__ = 'CLgenPrograms'
    id = sql.Column(sql.String(40), primary_key=True)
    date_added = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    clgen_version = sql.Column(sql.String(12))
    model = sql.Column(sql.String(40))
    sampler = sql.Column(sql.String(40))

    # time taken to produce program (in seconds).
    runtime = sql.Column(sql.Float)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    status = sql.Column(sql.Integer)
    gpuverified = sql.Column(sql.Boolean)
    cl_launchable = sql.Column(sql.Boolean)
    handchecked = sql.Column(sql.Boolean)

    # relations:
    results = sql.orm.relationship("CLgenResult", back_populates="program")
    cl_launcher_results = sql.orm.relationship("cl_launcherCLgenResult",
                                               back_populates="program")
    harnesses = sql.orm.relationship("CLgenHarness", back_populates="program")

    def __repr__(self) -> str:
        return self.id


class CLgenReduction(Base):
    __tablename__ = "CLgenReductions"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLgenResults.id"), primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    log = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    result = sql.orm.relationship("CLgenResult")


class CLgenHarness(Base):
    """ cldrive-generated test harnesses for CLgen programs """
    __tablename__ = 'CLgenHarnesses'
    id = sql.Column(sql.Integer, primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLgenPrograms.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cldriveParams.id"),
                           nullable=False)

    # cldrive version which generated harness
    cldrive_version = sql.Column(sql.String(12))
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    compile_only = sql.Column(sql.Boolean)

    # time taken to create harness
    generation_time = sql.Column(sql.Float, nullable=False)
    compile_time = sql.Column(sql.Float, nullable=False)

    # relations:
    program = sql.orm.relationship("CLgenProgram", back_populates="harnesses")
    params = sql.orm.relationship("cldriveParams", back_populates="harnesses")

    def __repr__(self) -> str:
        return self.id


class GitHubProgram(Base):
    """ programs """
    __tablename__ = 'GitHubPrograms'
    id = sql.Column(sql.String(255), primary_key=True)
    date_added = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    status = sql.Column(sql.Integer)

    results = sql.orm.relationship("GitHubResult", back_populates="program")

    def __repr__(self) -> str:
        return self.id


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
    cl_launcher_clgen_results = sql.orm.relationship("cl_launcherCLgenResult", back_populates="testbed")
    cldrive_clsmith_results = sql.orm.relationship("cldriveCLSmithResult", back_populates="testbed")
    github_results = sql.orm.relationship("GitHubResult", back_populates="testbed")

    def __repr__(self) -> str:
        return ("Platform: {self.platform}, "
                "Device: {self.device}, "
                "Driver: {self.driver}, "
                "Host: {self.host}".format(**vars()))

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
    # relation back to results:
    results = sql.orm.relationship("CLSmithResult", back_populates="params")
    clgen_results = sql.orm.relationship("cl_launcherCLgenResult", back_populates="params")

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


def cl_launcher_params_groups(session):
    """
    Return list of Param IDs for distinct params, grouped by optimizations on/off.
    """
    id_groups = []
    for gx, gy, gz, lx, ly, lz in session.query(
            cl_launcherParams.gsize_x, cl_launcherParams.gsize_y, cl_launcherParams.gsize_z,
            cl_launcherParams.lsize_x, cl_launcherParams.lsize_y, cl_launcherParams.lsize_z
        ).group_by(
            cl_launcherParams.gsize_x, cl_launcherParams.gsize_y, cl_launcherParams.gsize_z,
            cl_launcherParams.lsize_x, cl_launcherParams.lsize_y, cl_launcherParams.lsize_z
        ):
        id_groups.append([x[0] for x in
            session.query(cl_launcherParams.id).filter(
                cl_launcherParams.gsize_x == gx,
                cl_launcherParams.gsize_y == gy,
                cl_launcherParams.gsize_z == gz,
                cl_launcherParams.lsize_x == lx,
                cl_launcherParams.lsize_y == ly,
                cl_launcherParams.lsize_z == lz)])
    return id_groups


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

    # relations:
    clgen_results = sql.orm.relationship("CLgenResult", back_populates="params")
    github_results = sql.orm.relationship("GitHubResult", back_populates="params")
    harnesses = sql.orm.relationship("CLgenHarness", back_populates="params")

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


class CLSmithResult(Base):
    __tablename__ = "CLSmithResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLSmithPrograms.id"),
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

    program = sql.orm.relationship("CLSmithProgram", back_populates="cl_launcher_results")
    testbed = sql.orm.relationship("Testbed", back_populates="clsmith_results")
    params = sql.orm.relationship("cl_launcherParams", back_populates="results")
    reduction = sql.orm.relation("CLSmithReduction", back_populates="result")

    def __repr__(self):
        return ("result: {self.id} "
                "testbed: {self.testbed.device}, "
                "program: {self.program_id}, "
                "params: {self.params}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


class CLSmithReduction(Base):
    __tablename__ = "CLSmithReductions"
    id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithResults.id"), primary_key=True)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    log = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    result = sql.orm.relationship("CLSmithResult")


class cldriveCLSmithResult(Base):
    __tablename__ = "cldriveCLSmithResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLSmithPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cldriveParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    cli = sql.Column(sql.String(255), nullable=False)
    # cldrive_version = sql.Column(sql.String(12))
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.LargeBinary(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))
    classification = sql.Column(sql.String(16))

    program = sql.orm.relationship("CLSmithProgram", back_populates="cldrive_results")
    testbed = sql.orm.relationship("Testbed", back_populates="cldrive_clsmith_results")
    params = sql.orm.relationship("cldriveParams")

    def __repr__(self) -> str:
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


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

    program = sql.orm.relationship("CLgenProgram", back_populates="cl_launcher_results")
    testbed = sql.orm.relationship("Testbed", back_populates="cl_launcher_clgen_results")
    params = sql.orm.relationship("cl_launcherParams", back_populates="clgen_results")

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


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


class CLgenResult(Base):
    __tablename__ = "CLgenResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLgenPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cldriveParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))
    classification = sql.Column(sql.String(16))
    submitted = sql.Column(sql.Boolean)
    dupe = sql.Column(sql.Integer, sql.ForeignKey("CLgenResults.id"))

    # relations:
    program = sql.orm.relationship("CLgenProgram", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="clgen_results")
    params = sql.orm.relationship("cldriveParams", back_populates="clgen_results")

    def __repr__(self) -> str:
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


class GitHubResult(Base):
    __tablename__ = "GitHubResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(255), sql.ForeignKey("GitHubPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("cldriveParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    cli = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.LargeBinary(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))
    classification = sql.Column(sql.String(16))

    program = sql.orm.relationship("GitHubProgram", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="github_results")
    params = sql.orm.relationship("cldriveParams", back_populates="github_results")

    def __repr__(self) -> str:
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


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
Tableset = namedtuple('Tableset', ['results', 'programs', 'params', 'reductions'])

CLSMITH_TABLES = Tableset(results=CLSmithResult, programs=CLSmithProgram,
                          params=cl_launcherParams, reductions=CLSmithReduction)
CLGEN_TABLES = Tableset(results=CLgenResult, programs=CLgenProgram,
                        params=cldriveParams, reductions=CLgenReduction)


class InsufficientDataError(ValueError):
    """ raised if not enough results """
    pass


def results_in_timelimit(session, tables: Tableset, testbed_id: int,
                         no_opt: bool, time_limit: int,
                         *return_values, filter=None):
    """
    Raises:
        InsufficientDataError: If run out of results before time_limit is
            reached.
    """
    if not len(return_values):
        return_values = (tables.results,)

    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == no_opt)

    clgen_generation_time = .9  # FIXME
    generation_time = sql.sql.func.ifnull(tables.programs.runtime, clgen_generation_time)
    runtime = tables.results.runtime
    reduction_time = sql.sql.func.ifnull(tables.reductions.runtime, 0)
    result_time = generation_time + runtime + reduction_time

    q = session.query(
            *return_values, result_time)\
        .outerjoin(tables.programs)\
        .outerjoin(tables.reductions)\
        .filter(tables.results.testbed_id == testbed_id,
                tables.results.params_id.in_(param_ids),
                tables.results.outcome != None)\
        .order_by(tables.results.date)

    if filter is not None:
        q = q.filter(filter)

    total_time = 0  # elapsed time
    for vals in q:
        if total_time + vals[-1] > time_limit:
            break
        total_time += vals[-1]
        yield vals[:-1], vals[-1], total_time
    else:
        # Didn't break
        import util
        total_hours = total_time / 3600
        testbed = session.query(Testbed).filter(Testbed.id == testbed_id).first()
        devname = util.device_str(testbed.device)
        raise InsufficientDataError(f"insufficient {tables.results.__tablename__} for {devname} {no_opt} ({total_hours:.1f} hs)")
