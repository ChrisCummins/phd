import os
import pathlib
import typing

from lib.labm8 import cache
from lib.labm8 import fs


def cachepath(*relative_path_components: typing.List[str]) -> pathlib.Path:
  """
  Return path to file system cache.

  Parameters
  ----------
  *relative_path_components
      Relative path of cache.

  Returns
  -------
  str
      Absolute path of file system cache.
  """
  cache_root = pathlib.Path(os.environ.get("CLGEN_CACHE", "~/.cache/clgen/"))
  cache_root.expanduser().mkdir(parents=True, exist_ok=True)
  return pathlib.Path(fs.path(cache_root, *relative_path_components))


def mkcache(*relative_path_components: typing.List[str]) -> cache.FSCache:
  """
  Instantiae a file system cache.

  If the cache does not exist, one is created.

  Parameters
  ----------
  lang
      Programming language.
  *relative_path_components
      Relative path of cache.

  Returns
  -------
  labm8.FSCache
      Filesystem cache.
  """
  return cache.FSCache(cachepath(*relative_path_components),
                       escape_key=cache.escape_path)


def ShortHash(hash: str, cachedir: str, min_len: int = 7) -> str:
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
