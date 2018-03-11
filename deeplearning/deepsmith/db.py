"""
Database backend.
"""
import datetime
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging
from sqlalchemy.ext.declarative import declarative_base

from deeplearning.deepsmith.proto import datastore_pb2

FLAGS = flags.FLAGS

flags.DEFINE_bool('sql_echo', None, 'Print all executed SQL statements')

# The database session type.
session_t = sql.orm.session.Session

# The database query type.
query_t = sql.orm.query.Query

# A type alias for annotating methods which take or return protocol buffers.
ProtocolBuffer = typing.Any

# The SQLAlchemy base table.
Base = declarative_base()

# A shorthand declaration for the current time.
now = datetime.datetime.utcnow


class InvalidInputError(ValueError):
  pass


class StringTooLongError(ValueError):
  def __init__(self, column_name: str, string: str, max_len: int):
    self.column_name = column_name
    self.string = string
    self.max_len = max_len

  def __repr__(self):
    n = len(self.max_len)
    s = string[:20]
    return (f"String '{s}...' too long for '{self.column_name}'. " +
            f"Max length: {self.max_len}, actual length: {n}. ")


class Table(Base):
  """A database-backed object.

  This extends the standard SQLAlchemy 'Base' object by adding features
  specific to Deepsmith: methods for serializing to and from protobufs, and
  an index type for use when declaring foreign keys.
  """
  __abstract__ = True
  id_t = None

  @classmethod
  def GetOrAdd(cls, session: session_t, proto: ProtocolBuffer) -> 'Table':
    """Instantiate an object from a protocol buffer message.

    This is the preferred method for creating database-backed instances.
    If the created instance does not already exist in the database, it is
    added.

    Args:
      session: A database session.
      proto: A protocol buffer.

    Returns:
      An instance.

    Raises:
      InvalidInputError: In case one or more values contained in the protocol
        buffer cannot be stored in the database schema.
    """
    raise NotImplementedError(type(cls).__name__ + ".GetOrAdd() not implemented")

  def ToProto(self) -> ProtocolBuffer:
    """Create protocol buffer representation.

    Returns:
      A protocol buffer.
    """
    raise NotImplementedError(type(self).__name__ + ".ToProto() not implemented")

  def SetProto(self, proto: ProtocolBuffer) -> ProtocolBuffer:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer.

    Returns:
      The same protocol buffer that is passed as argument.
    """
    raise NotImplementedError(type(self).__name__ + ".SetProto() not implemented")

  def __repr__(self):
    try:
      return str(self.ToProto())
    except NotImplementedError:
      typename = type(self).__name__
      return f"TODO: Define {typename}.ToProto() method"


class ListOfNames(Table):
  """A list of names table.
  """
  __abstract__ = True
  id_t = sql.Integer
  name_len = 4096

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=now)
  name: str = sql.Column(sql.String(name_len), nullable=False, unique=True)

  @classmethod
  def GetOrAdd(cls, session: session_t, name: str) -> 'ListOfNames':
    """Instantiate a ListOfNames entry from a name.

    This is the preferred method for creating database-backed instances.
    If the created instance does not already exist in the database, it is
    added.

    Args:
      session: A database session.
      name: The name.

    Returns:
      A ListOfNames instance.
    """
    if len(name) > cls.name_len:
      raise StringTooLongError(cls, name, cls.name_len)

    return GetOrAdd(session, cls, name=name)

  def __repr__(self):
    return self.name[:50] or ""


def MakeEngine(config: datastore_pb2.DataStore) -> sql.engine.Engine:
  """
  Raises:
      ValueError: If DB_ENGINE config value is invalid.
  """
  if config.HasField('sqlite'):
    url, public_url = config.sqlite.url, config.sqlite.url
  elif config.HasField('mysql'):
    username, password = config.mysql.username, config.mysql.password
    hostname = config.mysql.password
    port = config.mysql.password

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    public_url = f'mysql://{username}@{hostname}:{port}/{name}?charset=utf8'
    url = f'mysql+mysqldb://{username}:{password}@{hostname}:{port}/{name}?charset=utf8'
  else:
    raise ValueError(f'unsupported database engine {engine}')

  logging.info('creating database engine %s', public_url)
  return sql.create_engine(url, encoding='utf-8', echo=FLAGS.sql_echo)


def GetOrAdd(session: sql.orm.session.Session, model,
             defaults: typing.Dict[str, object] = None, **kwargs):
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
