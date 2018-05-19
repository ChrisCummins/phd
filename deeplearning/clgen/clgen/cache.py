import os

from lib.labm8 import cache, fs


def cachepath(*relative_path_components: list) -> str:
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
  cache_root = os.environ.get("CLGEN_CACHE",
                              f"~/.cache/clgen/{version_info.major}.{version_info.minor}.x")

  fs.mkdir(cache_root)
  return fs.path(cache_root, *relative_path_components)


def mkcache(*relative_path_components: list) -> cache.FSCache:
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
  return cache.FSCache(cachepath(*relative_path_components), escape_key=cache.escape_path)
