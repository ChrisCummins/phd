# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file contains the logic for managing CLgen filesystem caches."""
import os
import pathlib

from labm8.py import app
from labm8.py import cache
from labm8.py import fs

FLAGS = app.FLAGS


def cachepath(*relative_path_components: str) -> pathlib.Path:
  """Return path to file system cache.

  Args:
    *relative_path_components: Relative path of cache.

  Returns:
    Absolute path of file system cache.
  """
  cache_root = pathlib.Path(os.environ.get("CLGEN_CACHE", "~/.cache/clgen/"))
  cache_root.expanduser().mkdir(parents=True, exist_ok=True)
  return pathlib.Path(fs.path(cache_root, *relative_path_components))


def mkcache(*relative_path_components: str) -> cache.FSCache:
  """Instantiate a file system cache.

  If the cache does not exist, one is created.

  Args:
    *relative_path_components: Relative path of cache.

  Returns:
    A filesystem cache instance.
  """
  return cache.FSCache(
    cachepath(*relative_path_components), escape_key=cache.escape_path
  )
