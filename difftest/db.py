import datetime
import sqlalchemy as sql

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
    table = "clsmith"
    port = "3306"

    uri = "mysql://{username}:{password}@{hostname}:{port}/{table}".format(**vars())
    engine = sql.create_engine(uri)

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
    cldrive_results = sql.orm.relation("cldriveCLSmithResult", back_populates="program")

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

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    status = sql.Column(sql.Integer)

    # relation back to results:
    results = sql.orm.relationship("CLgenResult", back_populates="program")

    def __repr__(self) -> str:
        return self.id


class GitHubProgram(Base):
    """ programs """
    __tablename__ = 'GitHubPrograms'
    id = sql.Column(sql.String(40), primary_key=True)
    date_added = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)

    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    status = sql.Column(sql.Integer)

    def __repr__(self) -> str:
        return self.id


class Testbed(Base):
    """ devices """
    __tablename__ = 'Testbeds'
    id = sql.Column(sql.Integer, primary_key=True)
    platform = sql.Column(sql.String(255), nullable=False)  # CL_DEVICE_NAME
    device = sql.Column(sql.String(255), nullable=False)  # CL_PLATFORM_NAME
    driver = sql.Column(sql.String(255), nullable=False)  # CL_DRIVER_VERSION
    host = sql.Column(sql.String(255), nullable=False)
    # unique combination of values:
    __table_args__ = (
        sql.UniqueConstraint('platform', 'device', 'driver', name='_uid'),)
    # relation back to results:
    clsmith_results = sql.orm.relationship("CLSmithResult", back_populates="testbed")
    clgen_results = sql.orm.relationship("CLgenResult", back_populates="testbed")
    cldrive_clsmith_results = sql.orm.relationship("cldriveCLSmithResult", back_populates="testbed")
    github_results = sql.orm.relationship("GitHubResult", back_populates="testbed")

    def __repr__(self) -> str:
        return ("Platform: {self.platform}, "
                "Device: {self.device}, "
                "Driver: {self.driver}, "
                "Host: {self.host}".format(**vars()))


class CLSmithParams(Base):
    """ params used by cl_launcher to run kernel """
    __tablename__ = "CLSmithParams"
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


class CLgenParams(Base):
    """ params used by cldrive to run kernel """
    __tablename__ = "CLgenParams"
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
    # relation back to results:
    clgen_results = sql.orm.relationship("CLgenResult", back_populates="params")

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
    params_id = sql.Column(sql.Integer, sql.ForeignKey("CLSmithParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    flags = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))

    program = sql.orm.relationship("CLSmithProgram", back_populates="cl_launcher_results")
    testbed = sql.orm.relationship("Testbed", back_populates="clsmith_results")
    params = sql.orm.relationship("CLSmithParams", back_populates="results")

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


class cldriveCLSmithResult(Base):
    __tablename__ = "cldriveCLSmithResults"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("CLSmithPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    cli = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.LargeBinary(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))

    program = sql.orm.relationship("CLSmithProgram", back_populates="cldrive_results")
    testbed = sql.orm.relationship("Testbed", back_populates="cldrive_clsmith_results")
    params = sql.orm.relationship("CLgenParams")

    def __repr__(self) -> str:
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
    params_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    cli = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.LargeBinary(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))

    program = sql.orm.relationship("CLgenProgram", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="clgen_results")
    params = sql.orm.relationship("CLgenParams", back_populates="clgen_results")

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
    program_id = sql.Column(sql.String(40), sql.ForeignKey("GitHubPrograms.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("CLgenParams.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    cli = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.LargeBinary(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    outcome = sql.Column(sql.String(255))

    program = sql.orm.relationship("GitHubProgram", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="github_results")
    params = sql.orm.relationship("CLgenParams", back_populates="github_results")

    def __repr__(self) -> str:
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))
