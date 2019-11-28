"""This file defines a cache for linting results."""
import datetime
import hashlib
import os
import pathlib
import typing

import sqlalchemy as sql
from sqlalchemy import Binary
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

import build_info
from labm8.py import app
from labm8.py import shell
from labm8.py import sqlutil
from util.photolib import common
from util.photolib import workspace

FLAGS = app.FLAGS

Base = declarative.declarative_base()  # pylint: disable=invalid-name


class Meta(Base):
  __tablename__ = "meta"

  key: str = Column(String(1024), primary_key=True)
  value: str = Column(String(1024))


class Directory(Base):
  """Directory cache entry."""
  __tablename__ = "directories"

  relpath_md5: str = Column(Binary(16), primary_key=True)
  checksum: bytes = Column(sql.Binary(16).with_variant(mysql.BINARY(16),
                                                       'mysql'),
                           nullable=False)
  date_added: datetime.datetime = Column(DateTime,
                                         nullable=False,
                                         default=datetime.datetime.utcnow)

  def __repr__(self):
    return (f'{self.relpath}:  '
            f'{shell.ShellEscapeCodes.YELLOW}{self.message}'
            f'{shell.ShellEscapeCodes.END}  [{self.category}]')


class CachedError(Base):
  """Cached linter error."""
  __tablename__ = "errors"

  id: int = Column(Integer, primary_key=True)
  dir: str = Column(Binary(16),
                    ForeignKey("directories.relpath_md5"),
                    nullable=False)
  relpath: str = Column(String(1024), nullable=False)
  category: str = Column(String(512), nullable=False)
  message: str = Column(String(512), nullable=False)
  fix_it: str = Column(String(512), nullable=False)

  directory: Directory = orm.relationship("Directory")

  __table_args__ = (UniqueConstraint('dir',
                                     'relpath',
                                     'category',
                                     'message',
                                     'fix_it',
                                     name='unique_error'),)

  def __repr__(self):
    return (f'{self.relpath}:  '
            f'{shell.ShellEscapeCodes.YELLOW}{self.message}'
            f'{shell.ShellEscapeCodes.END}  [{self.category}]')


class CacheLookupResult(object):
  """Contains results of a cache lookup."""

  def __init__(self, exists: bool, checksum: bytes, relpath: str,
               relpath_md5: str, errors: typing.List[CachedError]):
    self.exists = exists
    self.checksum = checksum
    self.relpath = relpath
    self.relpath_md5 = relpath_md5
    self.errors = errors


class LinterCache(sqlutil.Database):
  """A database consisting of linter cache entries."""

  def __init__(self, workspace_: workspace.Workspace, must_exist: bool = False):
    cache_dir = workspace_.workspace_root / ".photolib"
    cache_dir.mkdir(exist_ok=True)
    url = f"sqlite:///{cache_dir}/errors.db"

    super(LinterCache, self).__init__(url, Base, must_exist)
    self.session = self.MakeSession()
    self.RefreshLintersVersion()

  def Empty(self, commit: bool = True):
    self.session.query(Directory).delete()
    self.session.query(CachedError).delete()
    if commit:
      self.session.commit()

  def RefreshLintersVersion(self):
    """Check that """
    meta_key = "version"

    cached_version = self.session.query(Meta) \
      .filter(Meta.key == meta_key) \
      .first()
    cached_version_str = (cached_version.value if cached_version else "")

    actual_version = Meta(key=meta_key, value=build_info.Version())

    if cached_version_str != actual_version.value:
      app.Log(1, "Version has changed, emptying cache ...")
      self.Empty(commit=False)
      if cached_version:
        self.session.delete(cached_version)
      self.session.add(actual_version)
      self.session.commit()

  def AddLinterErrors(self, entry: CacheLookupResult,
                      errors: typing.List[str]) -> None:
    """Record linter errors in the cache."""
    # Create a directory cache entry.
    directory = Directory(relpath_md5=entry.relpath_md5,
                          checksum=entry.checksum)

    self.session.add(directory)

    # Create entries for the errors.
    errors_ = [
        CachedError(dir=directory.relpath_md5,
                    relpath=e.relpath,
                    category=e.category,
                    message=e.message,
                    fix_it=e.fix_it or "") for e in errors
    ]
    if errors_:
      self.session.bulk_save_objects(errors_)
    self.session.commit()
    app.Log(2, "cached directory %s", entry.relpath)

  def GetLinterErrors(self, abspath: str, relpath: str) -> CacheLookupResult:
    """Looks up the given directory and returns cached results (if any)."""
    relpath_md5 = common.Md5String(relpath).digest()

    # Get the time of the most-recently modified file in the directory.
    checksum = GetDirectoryChecksum(abspath).digest()

    ret = CacheLookupResult(exists=False,
                            checksum=checksum,
                            relpath=relpath,
                            relpath_md5=relpath_md5,
                            errors=[])

    directory = self.session \
      .query(Directory) \
      .filter(Directory.relpath_md5 == ret.relpath_md5) \
      .first()

    if directory and directory.checksum == ret.checksum:
      ret.exists = True
      ret.errors = self.session \
        .query(CachedError) \
        .filter(CachedError.dir == ret.relpath_md5)
    elif directory:
      app.Log(2, "Removing stale directory cache: `%s`", relpath)

      # Delete all existing cache entries.
      self.session.delete(directory)
      self.session \
        .query(CachedError) \
        .filter(CachedError.dir == ret.relpath_md5) \
        .delete()

    return ret


def GetDirectoryMTime(abspath) -> int:
  """Get the timestamp of the most recently modified file/dir in directory.

  Params:
    abspath: The absolute path to the directory.

  Returns:
    The seconds since epoch of the last modification. If the directory is
    empty, zero is returned.
  """
  paths = [os.path.join(abspath, filename) for filename in os.listdir(abspath)]
  if paths:
    return int(max(os.path.getmtime(path) for path in paths))
  else:
    return 0


def GetDirectoryChecksum(abspath: pathlib.Path) -> str:
  """Compute a checksum to determine the contents and status of the directory.

  The checksum is computed

  Params:
    abspath:
  :return:
  """
  hash = hashlib.md5()
  paths = [os.path.join(abspath, filename) for filename in os.listdir(abspath)]
  if paths:
    directory_mtime = int(max(os.path.getmtime(path) for path in paths))
    hash.update(str(directory_mtime).encode('utf-8'))
    for path in paths:
      if os.path.isfile(path):
        hash.update(str(path).encode('utf-8'))
  return hash
