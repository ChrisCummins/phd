"""Module for extracting and caching XMP data from image files."""
import datetime
import os
import typing

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
from labm8 import sqlutil
from util.photolib import common

FLAGS = app.FLAGS

Base = declarative.declarative_base()  # pylint: disable=invalid-name


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


class XmpCache(sqlutil.Database):
  """Database of keywords"""

  def __init__(self, workspace_root_path: str, must_exist: bool = False):
    cache_dir = os.path.join(workspace_root_path, ".photolib")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "xmp.db")
    url = f"sqlite:///{path}"
    app.Log(2, "Errors cache %s", url)

    super(XmpCache, self).__init__(url, Base, must_exist)
    self.session = self.MakeSession()

  def _CreateXmpCacheEntry(self, abspath: str, relpath_md5: str,
                           mtime: float) -> None:
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
      flash_fired = (True if _GetFromXmpDict(
          xmp, 'http://ns.adobe.com/exif/1.0/',
          'exif:Flash/exif:Fired') == 'True' else False)

      lens_make = _GetFromXmpDict(xmp, 'http://cipa.jp/exif/1.0/',
                                  'exifEX:LensMake')
      lens_model = _GetFromXmpDict(xmp, 'http://cipa.jp/exif/1.0/',
                                   'exifEX:LensModel')
      if lens_make and lens_model:
        lens = f'{lens_make} {lens_model}'
      elif lens_make:
        lens = lens_make
      elif lens_model:
        lens = lens_model
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

    keywords_ = sqlutil.GetOrAdd(self.session,
                                 Keywords,
                                 keywords=",".join(keywords))
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
    self.session.add(entry)
    self.session.commit()

    return entry

  def GetLightroomKeywords(self, abspath: str, relpath: str) -> typing.Set[str]:
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
    entry = self.session \
      .query(XmpCacheEntry) \
      .filter(XmpCacheEntry.relpath_md5 == relpath_md5) \
      .first()

    if entry and entry.mtime == mtime:
      keywords = set(entry.keywords.keywords.split(","))
    elif entry and entry.mtime != mtime and not abspath.endswith('.mov'):
      self.session.delete(entry)
      entry = self._CreateXmpCacheEntry(abspath, relpath_md5, mtime)
      keywords = entry.keywords.AsSet()
      app.Log(2, "Refreshed keywords cache `%s`", relpath)
    else:
      entry = self._CreateXmpCacheEntry(abspath, relpath_md5, mtime)
      keywords = entry.keywords.AsSet()
      app.Log(2, "Cached keywords `%s`", relpath)

    return keywords


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
