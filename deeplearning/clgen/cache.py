"""This file contains the logic for managing CLgen filesystem caches."""
import os
import pathlib

from absl import flags

from phd.lib.labm8 import cache
from phd.lib.labm8 import fs


FLAGS = flags.FLAGS


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
  return cache.FSCache(cachepath(*relative_path_components),
                       escape_key=cache.escape_path)
