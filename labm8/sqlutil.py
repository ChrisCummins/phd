"""Utility code for working with sqlalchemy."""
import contextlib
import pathlib
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging
from sqlalchemy import orm


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'sqlutil_echo', False,
    'If True, the Engine will log all statements as well as a repr() of their '
    'parameter lists to the engines logger, which defaults to sys.stdout.')


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


class Database(object):
  """A base class for implementing database objects."""

  session_t = orm.session.Session

  def __init__(self, path: pathlib.Path, declarative_base):
    self.database_path = path.absolute()
    if not self.database_path.is_file():
      logging.info("Creating sqlite database: '%s'.", self.database_path)
    self.database_uri = f'sqlite:///{self.database_path}'
    self.engine = sql.create_engine(
        self.database_uri, encoding='utf-8', echo=FLAGS.sqlutil_echo)
    declarative_base.metadata.create_all(self.engine)
    declarative_base.metadata.bind = self.engine
    self.make_session = orm.sessionmaker(bind=self.engine)

  @contextlib.contextmanager
  def Session(self, commit: bool = False) -> session_t:
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
