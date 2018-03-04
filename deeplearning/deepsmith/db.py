"""
Database backend.
"""
import collections
import datetime
import os
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging
from sqlalchemy.ext.declarative import declarative_base

FLAGS = flags.FLAGS

# TODO(cec): Document these flags and set appropriate defaults.
flags.DEFINE_string("db_engine", None, "")
flags.DEFINE_string("db_hostname", None, "")
flags.DEFINE_integer("db_port", None, "")
flags.DEFINE_string("db_username", None, "")
flags.DEFINE_string("db_password", None, "")
flags.DEFINE_string("db_dir", None, "")

__version__ = "1.0.0.dev1"

_major = int(__version__.split(".")[0])
_minor = (int(__version__.split('.')[1])
          if len(__version__.split('.')) > 1 else 0)
_micro = (int(__version__.split('.')[2])
          if len(__version__.split('.')) > 2 else 0)
_releaselevel = (__version__.split('.')[3]
                 if len(__version__.split('.')) > 3 else 'final')

version_info_t = collections.namedtuple(
    'version_info_t', ['major', 'minor', 'micro', 'releaselevel'])
version_info = version_info_t(_major, _minor, _micro, _releaselevel)

# Type aliases:
session_t = sql.orm.session.Session
query_t = sql.orm.query.Query

# SQLAlchemy:
Base = declarative_base()

# Shorthand:
now = datetime.datetime.utcnow


class ListOfNames(Base):
  id_t = sql.Integer
  __abstract__ = True

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=now)
  name: str = sql.Column(sql.String(1024), nullable=False, unique=True)


def MakeEngine(**kwargs) -> sql.engine.Engine:
  """
  Raises:
      ValueError: If DB_ENGINE config value is invalid.
  """
  prefix = kwargs.get("prefix", "")
  engine = kwargs.get("engine", FLAGS.db_engine)

  name = f"{prefix}dsmith_{version_info.major}{version_info.minor}"

  if engine == "mysql":
    flags_credentials = (FLAGS.db_username, FLAGS.db_password)
    username, password = kwargs.get("credentials", flags_credentials)
    hostname = kwargs.get("hostname", FLAGS.db_hostname)
    port = str(kwargs.get("port", FLAGS.db_port))

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    public_uri = f"mysql://{username}@{hostname}:{port}/{name}?charset=utf8".format(**vars())
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{name}?charset=utf8"
  elif engine == "sqlite":
    db_dir = kwargs.get("db_dir", FLAGS.db_dir)
    if not db_dir:
      raise ValueError(f"no database directory specified")
    os.makedirs(db_dir, exist_ok=True)
    path = os.path.join(db_dir, f"{name}.db")
    uri = f"sqlite:///{path}"
    public_uri = uri
  else:
    raise ValueError(f"unsupported database engine {engine}")

  # Determine whether to enable logging of SQL statements:
  echo = True if os.environ.get("DB_DEBUG", None) else False

  logging.debug("connecting to database %s", public_uri)
  return sql.create_engine(uri, encoding="utf-8", echo=echo), public_uri


def GetOrAdd(session: sql.orm.session.Session, model,
             defaults: typing.Dict[str, object] = None, **kwargs) -> object:
  """
  Instantiate a mapped database object. If the object is not in the database,
  add it.

  Note that no change is written to disk until commit() is called on the
  session.
  """
  instance = session.query(model).filter_by(**kwargs).first()
  if not instance:
    params = {k: v for k, v in kwargs.items()
              if not isinstance(v, sql.sql.expression.ClauseElement)}
    params.update(defaults or {})
    instance = model(**params)
    session.add(instance)

    # logging
    logging.debug("new %s record", model.__name__)

  return instance

