"""Module for extracting and caching XMP data from image files."""
import os

import datetime
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

import build_info
from labm8 import app
from labm8 import sqlutil
from util.photolib import common
from util.photolib import workspace

FLAGS = app.FLAGS

Base = declarative.declarative_base()  # pylint: disable=invalid-name


class XmpCacheEntry(Base):
  """A keyword cache entry, mapping a file to a set of keywords.

  Each XmpCacheEntry requires 64 (40 + 8 + 8 + ?) bytes.
  """
  __tablename__ = "files"

  relpath_md5: str = Column(Binary(16), primary_key=True)
  relpath: str = Column(String(1024), nullable=False)
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

  def ToDict(self) -> typing.Dict[str, str]:
    return {
        "relpath": self.relpath,
        "camera": self.camera,
        "lens": self.lens,
        "iso": self.iso,
        "shutter_speed": self.shutter_speed,
        "aperture": self.aperture,
        "focal_length_35mm": self.focal_length_35mm,
        "flash_fired": self.flash_fired,
        "keywords": ",".join(self.keywords.AsList())
    }


class Keywords(Base):
  """A set of image keywords."""
  __tablename__ = "keywords"

  id: int = Column(Integer, primary_key=True)
  keywords: str = Column(String(4096), nullable=False, unique=True)

  def AsList(self) -> typing.List[str]:
    return self.keywords.split(',')

  def AsSet(self) -> typing.Set[str]:
    return set(self.AsList())


class Meta(Base):
  __tablename__ = "meta"

  key: str = Column(String(1024), primary_key=True)
  value: str = Column(String(1024))


class XmpCache(sqlutil.Database):
  """Database of keywords"""

  def __init__(self, workspace_: workspace.Workspace, must_exist: bool = False):
    cache_dir = workspace_.workspace_root / ".photolib"
    cache_dir.mkdir(exist_ok=True)
    url = f"sqlite:///{cache_dir}/xmp.db"

    super(XmpCache, self).__init__(url, Base, must_exist)
    self.session = self.MakeSession()
    self.RefreshVersion()

  def RefreshVersion(self):
    """Refresh version."""
    meta_key = "version"

    cached_version = self.session.query(Meta) \
      .filter(Meta.key == meta_key) \
      .first()
    cached_version_str = (cached_version.value if cached_version else "")

    actual_version = Meta(key=meta_key, value=build_info.Version())

    if cached_version_str != actual_version.value:
      app.Log(1, "Version has changed, emptying cache ...")
      self.session.query(XmpCacheEntry).delete()
      self.session.query(Keywords).delete()
      if cached_version:
        self.session.delete(cached_version)
      self.session.add(actual_version)
      self.session.commit()

  def _CreateXmpCacheEntry(self, abspath: str, relpath: str, relpath_md5: str,
                           mtime: float) -> XmpCacheEntry:
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
        relpath=relpath,
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

  def GetOrCreateXmpCacheEntry(self, abspath: str,
                               relpath: str) -> XmpCacheEntry:
    relpath_md5 = common.Md5String(relpath).digest()
    mtime = int(os.path.getmtime(abspath))
    entry = self.session \
      .query(XmpCacheEntry) \
      .filter(XmpCacheEntry.relpath_md5 == relpath_md5) \
      .first()

    if entry and entry.mtime == mtime:
      return entry
    elif entry and entry.mtime != mtime and not abspath.endswith('.mov'):
      self.session.delete(entry)
      entry = self._CreateXmpCacheEntry(abspath, relpath, relpath_md5, mtime)
      app.Log(2, "Refreshed cached XMP metadata `%s`", relpath)
    else:
      entry = self._CreateXmpCacheEntry(abspath, relpath, relpath_md5, mtime)
      app.Log(2, "Cached XMP metadata `%s`", relpath)

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
    entry = self.GetOrCreateXmpCacheEntry(abspath, relpath)
    return entry.keywords.AsSet()


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
