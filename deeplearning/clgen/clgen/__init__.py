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
"""CLgen is a Deep learning program generator."""
import platform
from contextlib import contextmanager

import psutil
from deeplearning.clgen._config import *

from deeplearning.clgen.clgen import CLgenError, InternalError
from deeplearning.clgen.clgen.cache import cachepath
from deeplearning.clgen.clgen.errors import (
  CLgenError, File404, InternalError, UserError,
)
from deeplearning.clgen.clgen.package_util import must_exist
from lib.labm8 import fs


__author__ = "Chris Cummins"
__copyright__ = "Copyright 2016, 2017, 2018 Chris Cummins"
__license__ = "GPL v3"
__maintainer__ = __author__
__email__ = "chrisc.101@gmail.com"
__status__ = "Prototype"

_must_exist = must_exist  # prevent variable scope shadowing


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
    features_str = " (with CUDA)"
  else:
    features_str = ""

  printfn(f"CLgen{features_str}")
  printfn("Platform:  ", platform.system())
  printfn("Memory:    ", round(psutil.virtual_memory().total / (1024 ** 2)),
          "MB")


# package level imports
from deeplearning.clgen._langs import *
from deeplearning.clgen._fetch import *
from deeplearning.clgen._explore import *
from deeplearning.clgen._atomizer import *
from deeplearning.clgen._corpus import *
from deeplearning.clgen._model import *
from deeplearning.clgen._preprocess import *
from deeplearning.clgen._sampler import *
