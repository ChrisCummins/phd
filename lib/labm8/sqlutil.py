"""Utility code for working with sqlalchemy."""
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging


FLAGS = flags.FLAGS


def GetOrAdd(session: sql.orm.session.Session, model,
             defaults: typing.Dict[str, object] = None, **kwargs):
  """Instantiate a mapped database object.

  If the object is not in the database,
  add it. Note that no change is written to disk until commit() is called on the
  session.
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
