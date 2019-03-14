# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""Database backend."""
import datetime
import pathlib

import sqlalchemy as sql
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.declarative import declarative_base

from deeplearning.deepsmith.proto import datastore_pb2
from labm8 import app
from labm8 import labdate
from labm8 import pbutil
from labm8.sqlutil import GetOrAdd

FLAGS = app.FLAGS

app.DEFINE_boolean('sql_echo', None, 'Print all executed SQL statements')

# The database session type.
session_t = sql.orm.session.Session

# The database query type.
query_t = sql.orm.query.Query

# The SQLAlchemy base table.
Base = declarative_base()


class InvalidDatabaseConfig(ValueError):
  """Raise if the datastore config contains invalid values."""
  pass


class DatabaseDoesNotExist(EnvironmentError):
  """Raised if the database does not exist."""

  def __init__(self):
    super(DatabaseDoesNotExist, self).__init__(
        'Database does not exist. Either create it yourself, or set field '
        'create_database_if_not_exist in DataStore proto to create it '
        'automatically.')


class InvalidInputError(ValueError):
  pass


class StringTooLongError(ValueError):

  def __init__(self, column_name: str, string: str, max_len: int):
    self.column_name = column_name
    self.string = string
    self.max_len = max_len

  def __repr__(self):
    n = len(self.max_len)
    s = self.string[:20]
    return (f'String "{s}..." too long for "{self.column_name}". ' + f'Max '
            f'length: '
            f'{self.max_len}, actual length: {n}. ')


class Table(Base):
  """A database-backed object.

  This extends the standard SQLAlchemy 'Base' object by adding features
  specific to Deepsmith: methods for serializing to and from protobufs, and
  an index type for use when declaring foreign keys.
  """
  __abstract__ = True
  id_t = None

  @classmethod
  def GetOrAdd(cls, session: session_t,
               proto: pbutil.ProtocolBuffer) -> 'Table':
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
    typename = type(cls).__name__
    raise NotImplementedError(f'{typename}.GetOrAdd() not implemented')

  def ToProto(self) -> pbutil.ProtocolBuffer:
    """Create protocol buffer representation.

    Returns:
      A protocol buffer.
    """
    typename = type(self).__name__
    raise NotImplementedError(f'{typename}.ToProto() not implemented')

  def SetProto(self, proto: pbutil.ProtocolBuffer) -> pbutil.ProtocolBuffer:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer.

    Returns:
      The same protocol buffer that is passed as argument.
    """
    typename = type(self).__name__
    raise NotImplementedError(f'{typename}.SetProto() not implemented')

  @classmethod
  def ProtoFromFile(cls, path: pathlib.Path) -> pbutil.ProtocolBuffer:
    """Instantiate a protocol buffer representation from file.

    Args:
      path: Path to the proto file.

    Returns:
      Protocol buffer message instance.
    """
    typename = type(cls).__name__
    raise NotImplementedError(f'{typename}.ProtoFromFile() not implemented')

  @classmethod
  def FromFile(cls, session: session_t, path: pathlib.Path) -> 'Table':
    """Instantiate an object from a serialized protocol buffer on file.

    Args:
      session: A database session.
      path: Path to the proto file.

    Returns:
      An instance.
    """
    raise NotImplementedError(
        type(cls).__name__ + '.FromFile() not implemented')

  def __repr__(self):
    try:
      return str(self.ToProto())
    except NotImplementedError:
      typename = type(self).__name__
      return f'TODO: Define {typename}.ToProto() method'


class StringTable(Table):
  """A table of unique strings.

  A string table maps a unique string to a unique integer. In most cases, it is
  better to use a string table than to store strings directly in columns. The
  advantage of a string table is that it saves space for duplicate strings, and
  reduces table sizes by having tables contain only integer indexes. This makes
  grouping rows by string values faster, as well as reducing the cost of
  modifying a string.

  The downside of a string table is that it requires one extra table lookup to
  resolve the string itself.

  Note that the maximum length of strings is hardcoded to StringTable.maxlen.
  You should only use the StringTable.GetOrAdd() method to insert new strings,
  as this method performs the bounds checking and will raise a
  StringTooLongError if required. Instantiating a StringTable directly with a
  string which is too long will cause some SQL-based error which is harder to
  catch and potentially backend-specific.
  """
  __abstract__ = True
  id_t = sql.Integer

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  # MySQL maximum key length is 3072 bytes, with 3 bytes per character.
  string: str = sql.Column(
      sql.String(4096).with_variant(sql.String(3072 // 3), 'mysql'),
      nullable=False,
      unique=True)

  # The maximum number of characters in the string column.
  maxlen = string.type.length

  @classmethod
  def GetOrAdd(cls, session: session_t, string: str) -> 'StringTable':
    """Instantiate a StringTable entry from a string.

    This is the preferred method for creating database-backed instances.
    If the created instance does not already exist in the database, it is
    added.

    Args:
      session: A database session.
      string: The string.

    Returns:
      A StringTable instance.

    Raises:
      StringTooLongError: If the string is too long.
    """
    if len(string) > cls.maxlen:
      raise StringTooLongError(cls, string, cls.maxlen)

    return GetOrAdd(session, cls, string=string)

  def TruncatedString(self, n=80):
    """Return the truncated first 'n' characters of the string.

    Args:
      n: The maximum length of the string to return.

    Returns:
      A truncated string.
    """
    if self.string and len(self.string) > n:
      return self.string[:n - 3] + '...'
    elif self.string:
      return self.string
    else:
      return ''

  def __repr__(self):
    return self.TruncatedString(n=52)


def MakeEngine(config: datastore_pb2.DataStore) -> sql.engine.Engine:
  """Instantiate a database engine.

  Raises:
    InvalidDatabaseConfig: If the config contains illegal or missing values.
    DatabaseDoesNotExist: If the database does not exist and
      config.create_database_if_not_exist not set.
    NotImplementedError: If the datastore backend is not supported.
  """
  # Force database creation on testonly databases:
  if config.testonly:
    config.create_database_if_not_exist = True

  if config.HasField('sqlite'):
    if config.sqlite.inmemory:
      url = 'sqlite://'
    else:
      path = pathlib.Path(
          pbutil.RaiseIfNotSet(config.sqlite, 'path',
                               InvalidDatabaseConfig)).absolute()
      if not config.create_database_if_not_exist and not path.is_file():
        raise DatabaseDoesNotExist()
      path.parent.mkdir(parents=True, exist_ok=True)
      abspath = path.absolute()
      url = f'sqlite:///{abspath}'
    public_url = url
  elif config.HasField('mysql'):
    username = pbutil.RaiseIfNotSet(config.mysql, 'username',
                                    InvalidDatabaseConfig)
    password = pbutil.RaiseIfNotSet(config.mysql, 'password',
                                    InvalidDatabaseConfig)
    hostname = pbutil.RaiseIfNotSet(config.mysql, 'hostname',
                                    InvalidDatabaseConfig)
    port = pbutil.RaiseIfNotSet(config.mysql, 'port', InvalidDatabaseConfig)
    database = pbutil.RaiseIfNotSet(config.mysql, 'database',
                                    InvalidDatabaseConfig)
    url_base = f'mysql://{username}:{password}@{hostname}:{port}'
    if '`' in database:
      raise InvalidDatabaseConfig('MySQL database cannot have backtick in name')
    engine = sql.create_engine(url_base)
    query = engine.execute(
        sql.text('SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE '
                 'SCHEMA_NAME = :database'),
        database=database)
    if not query.first():
      if config.create_database_if_not_exist:
        # We can't use sql.text() escaping here becuase it uses singlequotes
        # for escaping. MySQL only accepts backticks for quoting database
        # names.
        engine.execute(f'CREATE DATABASE `{database}`')
      else:
        raise DatabaseDoesNotExist()
    engine.dispose()
    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    public_url = f'mysql://{username}@{hostname}:{port}/{database}?charset=utf8'
    url = f'{url_base}/{database}?charset=utf8'
  elif config.HasField('postgresql'):
    username = pbutil.RaiseIfNotSet(config.postgresql, 'username',
                                    InvalidDatabaseConfig)
    password = pbutil.RaiseIfNotSet(config.postgresql, 'password',
                                    InvalidDatabaseConfig)
    hostname = pbutil.RaiseIfNotSet(config.postgresql, 'hostname',
                                    InvalidDatabaseConfig)
    port = pbutil.RaiseIfNotSet(config.postgresql, 'port',
                                InvalidDatabaseConfig)
    database = pbutil.RaiseIfNotSet(config.postgresql, 'database',
                                    InvalidDatabaseConfig)
    if "'" in database:
      raise InvalidDatabaseConfig(
          'PostgreSQL database name cannot contain single quotes')
    url_base = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}'

    engine = sql.create_engine(f'{url_base}/postgres')
    conn = engine.connect()
    query = conn.execute(
        sql.text('SELECT 1 FROM pg_database WHERE datname = :database'),
        database=database)
    if not query.first():
      if config.create_database_if_not_exist:
        # PostgreSQL does not let you create databases within a transaction, so
        # manually complete the transaction before creating the database.
        conn.execute(sql.text('COMMIT'))
        # PostgreSQL does not allow single quoting of database names.
        conn.execute(f'CREATE DATABASE {database}')
      else:
        raise DatabaseDoesNotExist()
    conn.close()
    engine.dispose()
    public_url = f'postgresql://{username}@{hostname}:{port}/{database}'
    url = f'{url_base}/{database}'
  else:
    raise NotImplementedError(f'unsupported database engine')

  app.Log(1, "Database engine: '%s'", public_url)
  return sql.create_engine(url, encoding='utf-8', echo=FLAGS.sql_echo)


def DestroyTestonlyEngine(config: datastore_pb2.DataStore):
  """Permamently erase all data in a testonly datastore engine.

  Args:
    config: The datastore config.

  Raises:
    OSError: If the datastore is not configured as testonly.
    NotImplementedError: If the datastore backend is not supported.
  """
  if not config.testonly:
    raise OSError('Cannot destroy non-testonly dataset')

  if config.HasField('sqlite'):
    if not config.sqlite.inmemory:
      pbutil.RaiseIfNotSet(config.sqlite, 'path', InvalidDatabaseConfig)
      pathlib.Path(config.sqlite.path).unlink()
  elif config.HasField('mysql'):
    username = pbutil.RaiseIfNotSet(config.mysql, 'username',
                                    InvalidDatabaseConfig)
    password = pbutil.RaiseIfNotSet(config.mysql, 'password',
                                    InvalidDatabaseConfig)
    hostname = pbutil.RaiseIfNotSet(config.mysql, 'hostname',
                                    InvalidDatabaseConfig)
    port = pbutil.RaiseIfNotSet(config.mysql, 'port', InvalidDatabaseConfig)
    database = pbutil.RaiseIfNotSet(config.mysql, 'database',
                                    InvalidDatabaseConfig)
    url_base = f'mysql://{username}:{password}@{hostname}:{port}'

    engine = sql.create_engine(url_base)
    engine.execute(f'DROP DATABASE {database}')
  elif config.HasField('postgresql'):
    username = pbutil.RaiseIfNotSet(config.postgresql, 'username',
                                    InvalidDatabaseConfig)
    password = pbutil.RaiseIfNotSet(config.postgresql, 'password',
                                    InvalidDatabaseConfig)
    hostname = pbutil.RaiseIfNotSet(config.postgresql, 'hostname',
                                    InvalidDatabaseConfig)
    port = pbutil.RaiseIfNotSet(config.postgresql, 'port',
                                InvalidDatabaseConfig)
    database = pbutil.RaiseIfNotSet(config.postgresql, 'database',
                                    InvalidDatabaseConfig)
    url_base = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}'

    engine = sql.create_engine(f'{url_base}/postgres')
    conn = engine.connect()
    # PostgreSQL does not let you delete databases within a transaction, so
    # manually complete the transaction before creating the database.
    conn.execute('COMMIT')
    conn.execute(f'DROP DATABASE {database}')
    conn.close()
  else:
    raise NotImplementedError(f'unsupported database engine {engine}')
