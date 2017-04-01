import datetime
import sqlalchemy as sql

from configparser import ConfigParser
from contextlib import contextmanager
from labm8 import system
from labm8 import fs
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError


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


def get_or_create(session, model, defaults=None, **kwargs):
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
    results = sql.orm.relationship("Result", back_populates="program")

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
    results = sql.orm.relationship("Result", back_populates="testbed")

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
    results = sql.orm.relationship("Result", back_populates="params")

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
        return ("Optimizations: {self.optimizations_on_off}, "
                "Global size: {self.gsize}, "
                "Local size: {self.lsize}"
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

    program = sql.orm.relationship("Program", back_populates="results")
    testbed = sql.orm.relationship("Testbed", back_populates="results")
    params = sql.orm.relationship("Params", back_populates="results")

    def __repr__(self):
        return ("program: {self.program_id}, "
                "testbed: {self.testbed_id}, "
                "params: {self.params_id}, "
                "status: {self.status}, "
                "runtime: {self.runtime:.2f}s"
                .format(**vars()))
