"""Tarball util.
"""

import tarfile

from labm8 import fs


def unpack_archive(*components, **kwargs) -> str:
  """
  Unpack a compressed archive.

  Arguments:
      *components (str[]): Absolute path.
      **kwargs (dict, optional): Set "compression" to compression type.
          Default: bz2. Set "dir" to destination directory. Defaults to the
          directory of the archive.

  Returns:
      str: Path to directory.
  """
  path = fs.abspath(*components)
  compression = kwargs.get("compression", "bz2")
  dir = kwargs.get("dir", fs.dirname(path))

  fs.cd(dir)
  tar = tarfile.open(path, "r:" + compression)
  tar.extractall()
  tar.close()
  fs.cdpop()

  return dir
