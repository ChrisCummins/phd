"""Implementations of custom flag types for absl.flags."""
import pathlib
from absl import flags as absl_flags
from typing import Callable


class PathParser(absl_flags.ArgumentParser):
  """Parser of path values."""

  def __init__(self,
               must_exist: bool = True,
               exist_ok: bool = True,
               is_dir: bool = False):
    """Create a path values parser.

    Args:
      must_exist: If true, the path mst exist.
      exist_ok: If not true, the path must not exist. Not effect if true.
        Implied true if must_exist.
      is_dir: If true, the path must be a directory.
    """
    self.must_exist = must_exist
    self.exist_ok = exist_ok
    self.is_dir = is_dir

  def parse(self, argument) -> pathlib.Path:
    """See base class."""
    val = self.convert(argument)
    if self.must_exist:
      if not val.exists():
        raise ValueError('not found')
      if self.is_dir and not val.is_dir():
        raise ValueError('not a directory')
      elif not self.is_dir and not val.is_file():
        raise ValueError('not a file')
    elif not self.exist_ok and val.exists():
      raise ValueError('already exists')
    return val

  def convert(self, argument: str) -> pathlib.Path:
    """Returns the value of this argument."""
    if not argument:
      raise TypeError('Path flag must be set')
    return pathlib.Path(argument)


class DatabaseParser(absl_flags.ArgumentParser):
  """Parser of path values."""

  def __init__(self, database_class, must_exist: bool = True):
    """Create a path values parser.

    Args:
      must_exist: If true, the database must exist. Else, it is created.
    """
    # TODO(cec): Raise TypeError if database_class is not a subclass of
    # 'sqlutil.Database'.
    self.database_class = database_class
    self.must_exist = must_exist

  def parse(self, argument) -> 'sqlutil.Database':
    """See base class."""
    return self.convert(argument)

  def convert(self, argument: str) -> Callable[[], 'sqlutil.Database']:
    """Returns the value of this argument."""
    if not argument:
      raise TypeError('Path flag must be set')
    return lambda: self.database_class(url=argument, must_exist=self.must_exist)
