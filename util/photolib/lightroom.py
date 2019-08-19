"""Functions for working with Lightroom."""
import datetime
import os
import typing

import sqlalchemy as sql
from libxmp import utils as xmputils
from sqlalchemy import Binary
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import orm
from sqlalchemy.ext import declarative

from labm8 import app
from util.photolib import common


FLAGS = app.FLAGS

Base = declarative.declarative_base()  # pylint: disable=invalid-name

# Global state for the keywords cache database is initialized by
# a call to InitializeKeywordsCache().
ENGINE = None
MAKE_SESSION = None
SESSION = None


class XmpCacheEntry(Base):
  """A keyword cache entry, mapping a file to a set of keywords.

  Each XmpCacheEntry requires 64 (40 + 8 + 8 + ?) bytes.
  """
  __tablename__ = "files"

  relpath_md5: str = Column(Binary(16), primary_key=True)
  mtime: int = Column(Integer, nullable=False)
  keywords_id: int = Column(Integer, ForeignKey("keywords.id"), nullable=False)
  iso: int = Column(Integer, nullable=False)
  camera: str = Column(String(1024), nullable=False)
  lens: str = Column(String(1024), nullable=False)
  shutter_speed: str = Column(String(128), nullable=False)
  aperture: str = Column(String(128), nullable=False)
  focal_length_35mm: str = Column(String(128), nullable=False)
  flash_fired: bool = Column(Boolean, nullable=False)
  date_added: datetime.datetime = Column(DateTime,
                                         nullable=False,
                                         default=datetime.datetime.utcnow)

  keywords: 'Keywords' = orm.relationship("Keywords")


class Keywords(Base):
  """A set of image keywords."""
  __tablename__ = "keywords"

  id: int = Column(Integer, primary_key=True)
  keywords: str = Column(String(4096), nullable=False, unique=True)

  def AsList(self) -> typing.List[str]:
    return self.keywords.split(',')

  def AsSet(self) -> typing.Set[str]:
    return set(self.AsList())


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
  app.Log(2, "Keywords cache %s", uri)

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


def _GetFromXmpDict(
    xmp: typing.Dict[str, typing.Tuple[str, str, typing.Dict[str, str]]],
    xmp_type: str,
    tag_name: str,
    default=''):
  keywords = xmp.get(xmp_type, {})
  for tag_, val, _ in keywords:
    if tag_ == tag_name:
      return val
  return default


def _CreateXmpCacheEntry(abspath: str, relpath_md5: str, mtime: float) -> None:
  """
  Add a new database entry for the given values.

  Args:
    relpath_md5: The md5sum of the workspace relpath to the file.
    mtime: Seconds since the epoch that the file was last modified.
    keywords: The set of keywords to record.
  """
  try:
    xmp = xmputils.file_to_dict(abspath)
    lightroom_tags = xmp['http://ns.adobe.com/lightroom/1.0/']

    keywords = set([e[1] for e in lightroom_tags if e[1]])

    iso = int(
        _GetFromXmpDict(xmp, 'http://cipa.jp/exif/1.0/',
                        'exifEX:PhotographicSensitivity', 0))

    camera_make = _GetFromXmpDict(xmp, 'http://ns.adobe.com/tiff/1.0/',
                                  'tiff:Make')
    camera_model = _GetFromXmpDict(xmp, 'http://ns.adobe.com/tiff/1.0/',
                                   'tiff:Model')
    if camera_make and camera_model:
      camera = f'{camera_make} {camera_model}'
    else:
      camera = ''

    shutter_speed = _GetFromXmpDict(xmp, 'http://ns.adobe.com/exif/1.0/',
                                    'exif:ExposureTime')
    aperture = _GetFromXmpDict(xmp, 'http://ns.adobe.com/exif/1.0/',
                               'exif:FNumber')
    focal_length_35mm = _GetFromXmpDict(xmp, 'http://ns.adobe.com/exif/1.0/',
                                        'exif:FocalLengthIn35mmFilm')
    flash_fired = (True if _GetFromXmpDict(xmp, 'http://ns.adobe.com/exif/1.0/',
                                           'exif:Flash/exif:Fired') == 'True'
                   else False)

    lens_make = _GetFromXmpDict(xmp, 'http://cipa.jp/exif/1.0/',
                                'exifEX:LensMake')
    lens_model = _GetFromXmpDict(xmp, 'http://cipa.jp/exif/1.0/',
                                 'exifEX:LensModel')
    if lens_make and lens_model:
      lens = f'{lens_make} {lens_model}'
    else:
      lens = ''

  except KeyError:
    app.Log(2, 'Failed to read keywords of file: `%s`', abspath)
    keywords = []
    iso = 0
    camera = ''
    lens = ''
    shutter_speed = ''
    aperture = ''
    focal_length_35mm = ''
    flash_fired = False

  keywords_ = GetOrAdd(SESSION, Keywords, keywords=",".join(keywords))
  entry = XmpCacheEntry(
      relpath_md5=relpath_md5,
      mtime=int(mtime),
      keywords=keywords_,
      iso=iso,
      camera=camera,
      lens=lens,
      shutter_speed=shutter_speed,
      aperture=aperture,
      focal_length_35mm=focal_length_35mm,
      flash_fired=flash_fired,
  )
  SESSION.add(entry)
  SESSION.commit()

  return entry


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
    .query(XmpCacheEntry) \
    .filter(XmpCacheEntry.relpath_md5 == relpath_md5) \
    .first()

  if entry and entry.mtime == mtime:
    keywords = set(entry.keywords.keywords.split(","))
  elif entry and entry.mtime != mtime and not abspath.endswith('.mov'):
    SESSION.delete(entry)
    entry = _CreateXmpCacheEntry(abspath, relpath_md5, mtime)
    keywords = entry.keywords.AsSet()
    app.Log(2, "Refreshed keywords cache `%s`", relpath)
  else:
    entry = _CreateXmpCacheEntry(abspath, relpath_md5, mtime)
    keywords = entry.keywords.AsSet()
    app.Log(2, "Cached keywords `%s`", relpath)

  return keywords
