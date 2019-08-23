"""Representation of a file in the photo library."""
import os

import pathlib
import typing

from labm8 import app
from labm8 import decorators
from util.photolib import common
from util.photolib import xmp_cache

FLAGS = app.FLAGS


class Contentfile(object):

  def __init__(self, abspath: str, relpath: str, filename: str,
               xmp_cache_: xmp_cache.XmpCache):
    self.path = pathlib.Path(abspath)
    self.abspath = abspath
    self.relpath = relpath
    self.filename = filename
    self.xmp_cache = xmp_cache_

  @decorators.memoized_property
  def filename_noext(self):
    """Get the file name without an extension."""
    return os.path.splitext(self.filename)[0]

  @decorators.memoized_property
  def extension(self):
    """Get the file name extension."""
    return os.path.splitext(self.filename)[1]

  @decorators.memoized_property
  def keywords(self) -> typing.Set[str]:
    return self.xmp_cache.GetLightroomKeywords(self.abspath, self.relpath)

  @property
  def is_composite_file(self) -> bool:
    return (self.filename_noext.endswith('-HDR') or
            self.filename_noext.endswith('-Pano') or
            self.filename_noext.endswith('-Edit'))

  @decorators.memoized_property
  def composite_file_types(self) -> typing.Optional[typing.List[str]]:
    if not self.is_composite_file:
      return None
    components = self.filename.split('-')
    return [c for c in components if c in {'HDR', 'Pano', 'Edit'}]

  @decorators.memoized_property
  def composite_file_base(self) -> typing.Optional['Contentfile']:
    """Guess the base file for a composite."""
    if not self.is_composite_file:
      return None

    # Get the length of the shared prefix for all other file names in the
    # directory.
    names_and_prefixes = [
        (path.name, _GetLengthOfCommonPrefix(path.name, self.filename))
        for path in self.path.parent.iterdir()
        if path.name != self.filename and path.suffix in
        common.KNOWN_IMG_FILE_EXTENSIONS and len(path.name) < len(self.filename)
    ]
    if not names_and_prefixes:
      return None

    # Select the file which has the longest shared file name prefix.
    closest_match = list(sorted(names_and_prefixes, key=lambda x: x[1]))[-1][0]
    return Contentfile(str(self.path.absolute().parent / closest_match),
                       self.relpath[:-len(self.filename)] + closest_match,
                       closest_match, self.xmp_cache)


def _GetLengthOfCommonPrefix(a: str, b: str) -> int:
  n = min(len(a), len(b))
  for i in range(n):
    if a[i] != b[i]:
      return i + 1
  return n


def get_yyyy(
    string: str) -> typing.Tuple[typing.Optional[int], typing.Optional[str]]:
  """Parse string or return error."""
  if len(string) != 4:
    return None, f"'{string}' should be a four digit year"

  try:
    n = int(string)
  except ValueError:
    return None, f"'{string}' should be a four digit year"

  if n < 1900 or n > 2100:
    return None, f"year '{string}' out of range"

  return n, None


def get_mm(
    string: str) -> typing.Tuple[typing.Optional[int], typing.Optional[str]]:
  """Parse string or return error."""
  if len(string) != 2:
    return None, f"'{string}' should be a two digit month"

  try:
    n = int(string)
  except ValueError:
    return None, f"'{string}' should be a two digit month"

  if n < 1 or n > 12:
    return None, f"month '{string}' out of range"

  return n, None


def get_dd(
    string: str) -> typing.Tuple[typing.Optional[int], typing.Optional[str]]:
  """Parse string or return error."""
  if len(string) != 2:
    return None, f"'{string}' should be a two digit day"

  try:
    n = int(string)
  except ValueError:
    return None, f"'{string}' should be a two digit day"

  if n < 1 or n > 31:
    return None, f"day '{string}' out of range"

  return n, None


def get_yyyy_mm(
    string: str
) -> typing.Tuple[typing.Tuple[typing.Optional[int], typing.
                               Optional[int]], typing.Optional[str]]:
  """Parse string or return error."""
  string_components = string.split("-")
  if len(string_components) != 2:
    return (None, None), f"'{string}' should be YYYY-MM"

  year, err = get_yyyy(string_components[0])
  if err:
    return (None, None), err

  month, err = get_mm(string_components[1])
  if err:
    return (None, None), err

  return (year, month), None


def get_yyyy_mm_dd(
    string: str
) -> typing.Tuple[typing.Tuple[typing.Optional[int], typing.
                               Optional[int], typing.Optional[int]], typing.
                  Optional[str]]:
  """Parse string or return error."""
  string_components = string.split("-")
  if len(string_components) != 3:
    return (None, None, None), f"'{string}' should be YYYY-MM-DD"

  year, err = get_yyyy(string_components[0])
  if err:
    return (None, None, None), err

  month, err = get_mm(string_components[1])
  if err:
    return (None, None, None), err

  day, err = get_dd(string_components[2])
  if err:
    return (None, None, None), err

  return (year, month, day), None
