#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Deep learning program generator

Attributes
----------
__version__ : str
    PEP 440 compliant version string.
version_info : namedtuple['major', 'minor', 'micro', 'releaselevel'])
    Version tuple.
"""
import os
import platform
from collections import namedtuple
from contextlib import contextmanager

import psutil
from deeplearning.clgen._config import *
from pkg_resources import require, resource_filename, resource_string

from deeplearning.clgen.clgen import CLgenError, InternalError
from deeplearning.clgen.clgen.cache import cachepath
from deeplearning.clgen.clgen.errors import CLgenError, File404, InternalError, UserError
from lib.labm8 import fs, system


__author__ = "Chris Cummins"
__copyright__ = "Copyright 2016, 2017, 2018 Chris Cummins"
__license__ = "GPL v3"
__version__ = require("clgen")[0].version
__maintainer__ = __author__
__email__ = "chrisc.101@gmail.com"
__status__ = "Prototype"

# version_info tuple
_major = int(__version__.split(".")[0])
_minor = int(__version__.split('.')[1]) if len(__version__.split('.')) > 1 else 0
_micro = int(__version__.split('.')[2]) if len(__version__.split('.')) > 2 else 0
_releaselevel = __version__.split('.')[3] if len(__version__.split('.')) > 3 else 'final'

version_info_t = namedtuple('version_info_t', ['major', 'minor', 'micro', 'releaselevel'])
version_info = version_info_t(_major, _minor, _micro, _releaselevel)


def get_default_author() -> str:
  """
  Get a default author name.

  If CLGEN_AUTHOR environment variable is set, use that. Else, author
  is $USER@$HOSTNAME.

  Returns
  -------
  str
      Author name.
  """
  return os.environ.get("CLGEN_AUTHOR",
                        "{user}@{host}".format(user=system.USERNAME, host=system.HOSTNAME))


def must_exist(*path_components: str, **kwargs) -> str:
  """
  Require that a file exists.

  Parameters
  ----------
  *path_components : str
      Components of the path.
  **kwargs
      Key "Error" specifies the exception type to throw.

  Returns
  -------
  str
      Path.
  """
  assert (len(path_components))

  path = os.path.expanduser(os.path.join(*path_components))
  if not os.path.exists(path):
    Error = kwargs.get("Error", File404)
    e = Error("path '{}' does not exist".format(path))
    e.path = path
    raise e
  return path


_must_exist = must_exist  # prevent variable scope shadowing


def package_path(*path) -> str:
  """
  Path to package file.

  Parameters
  ----------
  *path : str
      Path components.

  Returns
  -------
  str
      Path.
  """
  path = os.path.expanduser(os.path.join(*path))
  abspath = resource_filename(__name__, path)
  return must_exist(abspath)


def _shorthash(hash: str, cachedir: str, min_len: int = 7) -> str:
  """
  Truncate the hash to a shorter length, while maintaining uniqueness.

  This returns the shortest hash required to uniquely identify all elements
  in the cache.

  Parameters
  ----------
  hash : str
      Hash to truncate.
  cachedir : str
      Path to cache.
  min_len : int, optional
      Minimum length of hash to try.

  Returns
  -------
  str
      Truncated hash.
  """
  for shorthash_len in range(min_len, len(hash)):
    entries = [x[:shorthash_len] for x in fs.ls(cachedir)]
    if len(entries) == len(set(entries)):
      break

  return hash[:shorthash_len]


def data_path(*path) -> str:
  """
  Path to package file.

  Parameters
  ----------
  *path : str
      Path components.

  Returns
  -------
  str
      Path.
  """
  return package_path("data", *path)


def package_data(*path) -> bytes:
  """
  Read package data file.

  Parameters
  ----------
  path : str
      The relative path to the data file, e.g. 'share/foo.txt'.

  Returns
  -------
  bytes
      File contents.

  Raises
  ------
  InternalError
      In case of IO error.
  """
  # throw exception if file doesn't exist
  package_path(*path)

  try:
    return resource_string(__name__, fs.path(*path))
  except Exception:
    raise InternalError("failed to read package data '{}'".format(path))


def package_str(*path) -> str:
  """
  Read package data file as a string.

  Parameters
  ----------
  path : str
      The relative path to the text file, e.g. 'share/foo.txt'.

  Returns
  -------
  str
      File contents.

  Raises
  ------
  InternalError
      In case of IO error.
  """
  try:
    return package_data(*path).decode('utf-8')
  except UnicodeDecodeError:
    raise InternalError("failed to decode package data '{}'".format(path))


def sql_script(name: str) -> str:
  """
  Read SQL script to string.

  Parameters
  ----------
  name : str
      The name of the SQL script (without file extension).

  Returns
  -------
  str
      SQL script.
  """
  path = fs.path('data', 'sql', str(name) + ".sql")
  return package_str(path)


@contextmanager
def terminating(thing):
  """
  Context manager to terminate object at end of scope.
  """
  try:
    yield thing
  finally:
    thing.terminate()


def platform_info(printfn=print) -> None:
  """
  Log platform information.

  Parameters
  ----------
  printfn : fn, optional
      Function to call to print output to. Default `print()`.
  """
  if USE_CUDA:
    features_str = "(with CUDA)"
  else:
    features_str = ""

  printfn("CLgen:      mater")
  printfn("Platform:  ", platform.system())
  printfn("Memory:    ", round(psutil.virtual_memory().total / (1024 ** 2)), "MB")


# package level imports
from deeplearning.clgen._langs import *
from deeplearning.clgen._fetch import *
from deeplearning.clgen._explore import *
from deeplearning.clgen._atomizer import *
from deeplearning.clgen._corpus import *
from deeplearning.clgen._model import *
from deeplearning.clgen._preprocess import *
from deeplearning.clgen._sampler import *
