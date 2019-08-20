"""Representation of a file in the photo library."""
import os
import pathlib
import typing

from labm8 import app
from labm8 import decorators
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
