"""Utility code for working with sqlalchemy."""
import contextlib
import pathlib
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging
from sqlalchemy import orm
from sqlalchemy.ext import declarative


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'sqlutil_echo', False,
    'If True, the Engine will log all statements as well as a repr() of their '
    'parameter lists to the engines logger, which defaults to sys.stdout.')


class DatabaseNotFound(FileNotFoundError):
  """An error that is raised if the requested database cannot be found."""

  def __init__(self, url: str):
    self._url = url

  @property
  def url(self):
    return self._url

  def __repr__(self) -> str:
    return f"Database not found: '{self.url}'"

  def __str__(self) -> str:
    return repr(self)


def GetOrAdd(session: sql.orm.session.Session, model,
             defaults: typing.Dict[str, object] = None, **kwargs):
  """Instantiate a mapped database object.

  If the object is not in the database,
  add it. Note that no change is written to disk until commit() is called on the
  session.

  Args:
    session: The database session.
    model: The database table class.
    defaults: Default values for mapped objects.
    kwargs: The values for the table row.

  Returns:
    An instance of the model class, with the values specified.
  """
  instance = session.query(model).filter_by(**kwargs).first()
  if not instance:
    params = {k: v for k, v in kwargs.items() if
              not isinstance(v, sql.sql.expression.ClauseElement)}
    params.update(defaults or {})
    instance = model(**params)
    session.add(instance)
    if logging.level_debug():
      logging.debug('New record: %s(%s)', model.__name__,
                    ', '.join([f'{k}={v}' for k, v in params.items()]))
  return instance


def Get(session: sql.orm.session.Session, model,
        defaults: typing.Dict[str, object] = None, **kwargs):
  """Determine if a database object exists.

  Args:
    session: The database session.
    model: The database table class.
    defaults: Default values for mapped objects.
    kwargs: The values for the table row.

  Returns:
    An instance of the model class with the values specified, or None if the
    object is not in the database.
  """
  del defaults
  return session.query(model).filter_by(**kwargs).first()


def CreateEngine(url: str,
                 create_if_not_exist: bool = True) -> sql.engine.Engine:
  """Create an sqlalchemy database engine.

  This is a convenience wrapper for creating an sqlalchemy engine, that also
  creates the database if required, and checks that the database exists. This
  means that it is less flexible than SqlAlchemy's create_engine() - only three
  combination of dialects and drivers are supported: sqlite, mysql, and
  postgresql.

  See https://docs.sqlalchemy.org/en/latest/core/engines.html for details.

  Examples:
    Create in-memory SQLite database:
    >>> engine = CreateEngine('sqlite://')

    Connect to an SQLite database at relative.db:
    >>> engine = CreateEngine('sqlite:///relative.db')

    Connect to an SQLite database at /absolute/path/to/db:
    >>> engine = CreateEngine('sqlite:////absolute/path/to/db')

    Connect to MySQL database:
    >>> engine = CreateEngine(
        'mysql://bob:password@localhost:1234/database?charset=utf8')

    Connect to PostgreSQL database:
    >>> engine.CreateEngine(
      'postgresql://bob:password@localhost:1234/database')

  Args:
    url: The URL of the database to connect to.

  Returns:
    An SQLalchemy Engine instance.

  Raises:
    DatabaseNotFound: If the database does not exist and create_if_not_exist not
      set.
    ValueError: If the datastore backend is not supported.
  """
  if url.startswith('mysql://'):
    # Support for MySQL dialect.

    # We create a throwaway engine that we use to check if the requested
    # database exists.
    engine = sql.create_engine(url)
    database = url.split('/')[-1].split('?')[0]
    query = engine.execute(sql.text('SELECT SCHEMA_NAME FROM '
                                    'INFORMATION_SCHEMA.SCHEMATA WHERE '
                                    'SCHEMA_NAME = :database'),
                           database=database)
    if not query.first():
      if create_if_not_exist:
        # We can't use sql.text() escaping here because it uses single quotes
        # for escaping. MySQL only accepts backticks for quoting database
        # names.
        engine.execute(f'CREATE DATABASE `{database}`')
      else:
        raise DatabaseNotFound(url)
    engine.dispose()
  elif url.startswith('sqlite://'):
    # Support for SQLite dialect.

    # This project (phd) deliberately disallows relative paths due to Bazel
    # sandboxing.
    if url != 'sqlite://' and not url.startswith('sqlite:////'):
      raise ValueError("Relative path to SQLite database is not allowed")

    if url == 'sqlite://':
      if not create_if_not_exist:
        raise ValueError(
            'create_if_exist=False not valid for in-memory SQLite database')
    else:
      path = pathlib.Path(url[len('sqlite:///'):])
      if create_if_not_exist:
        # Make the parent directory for SQLite database if creating a new
        # database.
        path.parent.mkdir(parents=True, exist_ok=True)
      else:
        if not path.is_file():
          raise DatabaseNotFound(url)
  elif url.startswith('postgresql://'):
    # Support for PostgreSQL dialect.

    database = url.split('/')[-1]
    engine = sql.create_engine(f'{url_base}/postgres')
    conn = engine.connect()
    query = conn.execute(
        sql.text('SELECT 1 FROM pg_database WHERE datname = :database'),
        database=database)
    if not query.first():
      if create_if_not_exist:
        # PostgreSQL does not let you create databases within a transaction, so
        # manually complete the transaction before creating the database.
        conn.execute(sql.text('COMMIT'))
        # PostgreSQL does not allow single quoting of database names.
        conn.execute(f'CREATE DATABASE {database}')
      else:
        raise DatabaseNotFound(url)
    conn.close()
    engine.dispose()
  else:
    raise ValueError(f"Unsupported database URL='{url}'")

  # Create the engine.
  engine = sql.create_engine(url, encoding='utf-8', echo=FLAGS.sqlutil_echo)

  # Create and immediately close a connection. This is because SQLAlchemy engine
  # is lazily instantiated, so for connections such as SQLite, this line
  # actually creates the file.
  engine.connect().close()

  return engine


class Session(orm.session.Session):
  """A subclass of the default SQLAlchemy Session with added functionality.

  An instance of this class is returned by Database.Session().
  """

  def GetOrAdd(self, model, defaults: typing.Dict[str, object] = None,
               **kwargs):
    """Instantiate a mapped database object.

    If the object is not in the database, add it. Note that no change is written
    to disk until commit() is called on the session.

    Args:
      model: The database table class.
      defaults: Default values for mapped objects.
      kwargs: The values for the table row.

    Returns:
      An instance of the model class, with the values specified.
    """
    return GetOrAdd(self, model, defaults, **kwargs)


class Database(object):
  """A base class for implementing databases."""

  def __init__(self, url: str, declarative_base,
               create_if_not_exist: bool = True):
    """Instantiate a database object.

    Example:
      >>> db = Database('sqlite:////tmp/foo.db',
                        sqlalchemy.ext.declarative.declarative_base())

    Args:
      url: The URL of the database to connect to.
      declarative_base: The SQLAlchemy declarative base instance.
      create_if_not_exist: If True, create database if it doesn't exist.

    Raises:
      DatabaseNotFound: If the database does not exist and create_if_not_exist
        not set.
      ValueError: If the datastore backend is not supported.
    """
    self._url = url
    self.engine = CreateEngine(url, create_if_not_exist=create_if_not_exist)
    declarative_base.metadata.create_all(self.engine)
    declarative_base.metadata.bind = self.engine

    # Bind the Engine to a session maker, which instantiates our own Session
    # class, which is a subclass of the default SQLAlchemy Session with added
    # functionality.
    self.make_session = orm.sessionmaker(bind=self.engine, class_=Session)

  @contextlib.contextmanager
  def Session(self, commit: bool = False) -> Session:
    """Provide a transactional scope around a session.

    Args:
      commit: If true, commit session at the end of scope.

    Returns:
      A database session.
    """
    session = self.make_session()
    try:
      yield session
      if commit:
        session.commit()
    except:
      session.rollback()
      raise
    finally:
      session.close()

  def __repr__(self) -> str:
    return f'Database[{self.database_uri}]'
