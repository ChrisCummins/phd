import datetime
import sqlalchemy as sql

from configparser import ConfigParser
from contextlib import contextmanager
from labm8 import system
from labm8 import fs
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError

import clinfo

Base = declarative_base()

# must call init() first
make_session = None


def get_mysql_creds():
    """ read default MySQL credentials in ~/.my.cnf """
    config = ConfigParser()
    config.read(fs.path("~/.my.cnf"))
    return config['mysql']['user'], config['mysql']['password']


def init(hostname: str):
    """ must be called before using anything """
    global make_session
    username, password = get_mysql_creds()

    engine = sql.create_engine(
        "mysql://{username}:{password}@{hostname}:3306/clsmith".format(**vars()))
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    make_session = sql.orm.sessionmaker(bind=engine)


@contextmanager
def Session():
    """Provide a transactional scope around a series of operations."""
    session = make_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# Database Schema


class Program(Base):
    """ CLSmith programs """
    __tablename__ = 'Programs'
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
    results = sql.orm.relationship("Result", backref="program")

    def __repr__(self):
        return self.id


class Testbed(Base):
    """ devices """
    __tablename__ = 'Testbeds'
    id = sql.Column(sql.Integer, primary_key=True)
    platform = sql.Column(sql.String(255), nullable=False)  # CL_DEVICE_NAME
    device = sql.Column(sql.String(255), nullable=False)  # CL_PLATFORM_NAME
    driver = sql.Column(sql.String(255), nullable=False)  # CL_DRIVER_VERSION
    # unique combination of values:
    __table_args__ = (
        sql.UniqueConstraint('platform', 'device', 'driver', name='_uid'),)
    # relation back to results:
    results = sql.orm.relationship("Result", backref="testbed")

    def __repr__(self):
        return ("Platform: {self.platform}, "
                "Device: {self.device}, "
                "Driver: {self.driver}".format(**vars()))


class Params(Base):
    """ params """
    __tablename__ = "Params"
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
    results = sql.orm.relationship("Result", backref="params")

    def to_flags(self):
        flags = [
            "-g", "{self.gsize_x},{self.gsize_y},{self.gsize_z}".format(**vars()),
            "-l", "{self.lsize_x},{self.lsize_y},{self.lsize_z}".format(**vars())
        ]
        if not self.optimizations:
            flags.append("---disable_opts")
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
        return ("optimizations: {self.optimizations_on_off}, "
                "global: {self.gsize}, "
                "local: {self.lsize}"
                .format(**vars()))


class Result(Base):
    __tablename__ = "Results"
    id = sql.Column(sql.Integer, primary_key=True)
    program_id = sql.Column(sql.String(40), sql.ForeignKey("Programs.id"),
                            nullable=False)
    testbed_id = sql.Column(sql.Integer, sql.ForeignKey("Testbeds.id"),
                            nullable=False)
    params_id = sql.Column(sql.Integer, sql.ForeignKey("Params.id"),
                           nullable=False)
    date = sql.Column(sql.DateTime, default=datetime.datetime.utcnow)
    flags = sql.Column(sql.String(255), nullable=False)
    status = sql.Column(sql.Integer, nullable=False)
    runtime = sql.Column(sql.Float, nullable=False)
    stdout = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))


# Helper functions


def get_testbed(platform_name: str,
                device_name: str,
                driver_version: str) -> Testbed:
    return Testbed(platform=platform_name,
                   device=device_name,
                   driver=driver_version)


def register_testbed(platform_id: int, device_id: int) -> int:
    """
    Returns:
        int: Testbed ID.
    """
    platform_name = clinfo.get_platform_name(platform_id)
    device_name = clinfo.get_device_name(platform_id, device_id)
    driver_version = clinfo.get_driver_version(platform_id, device_id)

    try:
        with Session() as session:
            d = get_testbed(platform_name, device_name, driver_version)
            session.add(d)
    except IntegrityError:
        with Session() as session:
            d = get_testbed(platform_name, device_name, driver_version)
            assert(session.query(Testbed).filter(
                Testbed.platform == d.platform,
                Testbed.device == d.device,
                Testbed.driver == d.driver).count() == 1)

    with Session() as session:
        d = get_testbed(platform_name, device_name, driver_version)
        testbed_id = session.query(Testbed).filter(
            Testbed.platform == d.platform,
            Testbed.device == d.device,
            Testbed.driver == d.driver).one().id

    return testbed_id


def get_num_progs_to_run(testbed_id, params):
    with Session() as session:
        subquery = session.query(Result.program_id).filter(
            Result.testbed_id == testbed_id, Result.params_id == params.id)
        ran = session.query(Program.id).filter(Program.id.in_(subquery)).count()
        subquery = session.query(Result.program_id).filter(
            Result.testbed_id == testbed_id)
        total = session.query(Program.id).count()
        return ran, total
