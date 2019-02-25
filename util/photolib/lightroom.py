"""Functions for working with Lightroom."""
import datetime
import os
import typing

import sqlalchemy as sql
from absl import flags
from absl import logging
from libxmp import utils as xmputils
from sqlalchemy import Binary
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import orm
from sqlalchemy.ext import declarative

from util.photolib import common

FLAGS = flags.FLAGS

Base = declarative.declarative_base()  # pylint: disable=invalid-name

# Global state for the keywords cache database is initialized by
# a call to InitializeKeywordsCache().
ENGINE = None
MAKE_SESSION = None
SESSION = None


class Keywords(Base):
  """A set of image keywords."""
  __tablename__ = "keywords"

  id: int = Column(Integer, primary_key=True)
  keywords: int = Column(String(4096), nullable=False, unique=True)


class KeywordCacheEntry(Base):
  """A keyword cache entry, mapping a file to a set of keywords.

  Each KeywordCacheEntry requires 56 (40 + 8 + 8 + ?) bytes.
  """
  __tablename__ = "files"

  relpath_md5: str = Column(Binary(16), primary_key=True)
  mtime: int = Column(Integer, nullable=False)
  keywords_id: int = Column(Integer, ForeignKey("keywords.id"), nullable=False)
  date_added: datetime.datetime = Column(
      DateTime, nullable=False, default=datetime.datetime.utcnow)

  keywords: Keywords = orm.relationship("Keywords")


def InitializeKeywordsCache(workspace_abspath: str) -> None:
  """
  Initialize the keywords cache database.

  Args:
    workspace_abspath: The absolute path to the workspace root.
  """
  global ENGINE
  global MAKE_SESSION
  global SESSION

  if ENGINE:
    raise ValueError("InitializeKeywordsCache() already called.")

  cache_dir = os.path.join(workspace_abspath, ".cache")
  os.makedirs(cache_dir, exist_ok=True)
  path = os.path.join(cache_dir, "keywords.db")
  uri = f"sqlite:///{path}"
  logging.debug("Keywords cache %s", uri)

  ENGINE = sql.create_engine(uri, encoding="utf-8")

  Base.metadata.create_all(ENGINE)
  Base.metadata.bind = ENGINE
  MAKE_SESSION = orm.sessionmaker(bind=ENGINE)
  SESSION = MAKE_SESSION()


def GetOrAdd(session,
             model,
             defaults: typing.Dict[str, typing.Any] = None,
             **kwargs) -> object:
  """
  Instantiate a mapped database object. If the object is not in the database,
  add it.

  Note that no change is written to disk until commit() is called on the
  session.
  """
  instance = session.query(model).filter_by(**kwargs).first()
  if not instance:
    params = dict((k, v)
                  for k, v in kwargs.items()
                  if not isinstance(v, sql.sql.expression.ClauseElement))
    params.update(defaults or {})
    instance = model(**params)
    session.add(instance)

  return instance


def _AddKeywordsToCache(relpath_md5: str, mtime: float,
                        keywords: typing.Set[str]) -> None:
  """
  Add a new database entry for the given values.

  Args:
    relpath_md5: The md5sum of the workspace relpath to the file.
    mtime: Seconds since the epoch that the file was last modified.
    keywords: The set of keywords to record.
  """
  keywords_ = GetOrAdd(SESSION, Keywords, keywords=",".join(keywords))
  entry = KeywordCacheEntry(
      relpath_md5=relpath_md5,
      mtime=int(mtime),
      keywords=keywords_,
  )
  SESSION.add(entry)
  SESSION.commit()


def _ReadKeywordsFromFile(abspath: str) -> typing.Set[str]:
  """
  Read the lightroom keywords for a file.

  Args:
    abspath: Path to the file.

  Returns:
    A set of lightroom keywords. An empty set is returned on failure.
  """
  try:
    xmp = xmputils.file_to_dict(abspath)
    lrtags = xmp['http://ns.adobe.com/lightroom/1.0/']
    keywords = set([e[1] for e in lrtags if e[1]])
    return keywords
  except KeyError:
    logging.error(abspath)
    return set()


def GetLightroomKeywords(abspath: str, relpath: str) -> typing.Set[str]:
  """Fetch the lightroom keywords for the given file.

  Nested keywords are separated using the '|' symbol.

  Args:
    abspath: Absolute path of the file.
    relpath: Workspace-relative path to the file.

  Returns:
    A set of lightroom keywords. An empty set is returned on failure.
  """
  relpath_md5 = common.Md5String(relpath).digest()
  mtime = int(os.path.getmtime(abspath))
  entry = SESSION \
    .query(KeywordCacheEntry) \
    .filter(KeywordCacheEntry.relpath_md5 == relpath_md5) \
    .first()

  if entry and entry.mtime == mtime:
    # logging.debug("keywords cache hit %s", relpath)
    keywords = set(entry.keywords.keywords.split(","))
  elif entry and entry.mtime != mtime and not abspath.endswith('.mov'):
    SESSION.delete(entry)
    keywords = _ReadKeywordsFromFile(abspath)
    _AddKeywordsToCache(relpath_md5, mtime, keywords)
    logging.debug("refreshed keywords cache %s", relpath)
  else:
    keywords = _ReadKeywordsFromFile(abspath)
    _AddKeywordsToCache(relpath_md5, mtime, keywords)
    logging.debug("cached keywords %s", relpath)

  return keywords
