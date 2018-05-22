"""The fast hash cache.

Checksum directories and cache results. If a directory has not been modified,
subsequent hashes are cache hits. Hashes are recomputed lazily, when a
directory (or any of its subdirectories) have been modified.
"""
import datetime
import os
import pathlib
import sqlite3
import time

import checksumdir

from lib.labm8 import fs


# A list of all the supported hash functions.
HASH_FUNCTIONS = ['md5', 'sha1', 'sha256', 'sha512']


class DirHashCacheError(sqlite3.OperationalError):
  """Exception thrown in case of an error with the cache database."""
  pass


class DirHashCache(object):
  """A persistent database for directory checksums."""

  def __init__(self, database_path: pathlib.Path, hash_function: str = 'sha1'):
    """
    Instantiate a directory checksum cache.

    Arguments:
      database_path: Path to persistent cache store.
      hash_function: The name of the hash algorithm to use. See HASH_FUNCTIONS
        for a list of acceptable values.

    Raises:
      ValueError: If hash_function is not one of HASH_FUNCTIONS.
      DirHashCacheError: If the requested database_path could not be connected
        to.
    """
    self.path = fs.path(database_path)
    self.hash = hash_function

    if hash_function not in HASH_FUNCTIONS:
      raise ValueError(f'Unsupported hash_function "{hash_function}". '
                       'Available hash functions: ' + ', '.join(HASH_FUNCTIONS))

    try:
      db = sqlite3.connect(self.path)
    except sqlite3.OperationalError:
      raise DirHashCacheError(f'could not connect to database {self.path}')
    c = db.cursor()
    c.execute("""\
CREATE TABLE IF NOT EXISTS dirhashcache (
        path TEXT NOT NULL,
        date DATETIME NOT NULL,
        hash TEXT NOT NULL,
        PRIMARY KEY(path)
)""")
    db.commit()
    db.close()

  def clear(self):
    """
    Remove all cache entries.
    """
    db = sqlite3.connect(self.path)
    c = db.cursor()
    c.execute("DELETE FROM dirhashcache")
    db.commit()
    db.close()

  def dirhash(self, path: pathlib.Path, **dirhash_opts) -> str:
    """Compute the hash of a directory.

    Arguments:
      path: Directory.
      **dirhash_opts: Additional options to checksumdir.dirhash().

    Returns:
      The checksum of the directory.

    Raises:
      ValueError: If the path does not exist, or is not a directory.
    """
    path = fs.path(path)

    if not fs.isdir(path):
      raise ValueError(f'Path "{path}" is not a directory')

    if fs.directory_is_empty(path):
      last_modified = datetime.datetime.now()
    else:
      last_modified = time.ctime(max(
        max(os.path.getmtime(os.path.join(root, file)) for file in files) for
        root, _, files in os.walk(path)))

    db = sqlite3.connect(self.path)
    c = db.cursor()
    c.execute("SELECT date, hash FROM dirhashcache WHERE path=?", (path,))
    cached = c.fetchone()

    if cached:
      cached_date, cached_hash = cached
      if cached_date == last_modified:
        # cache hit
        dirhash = cached_hash
      else:
        # out of date cache
        dirhash = checksumdir.dirhash(path, self.hash, **dirhash_opts)
        c.execute("UPDATE dirhashcache SET date=?, hash=? WHERE path=?",
                  (last_modified, dirhash, path))
        db.commit()
    else:
      # new entry
      dirhash = checksumdir.dirhash(path, self.hash, **dirhash_opts)
      c.execute("INSERT INTO dirhashcache VALUES (?,?,?)",
                (path, last_modified, dirhash))
      db.commit()

    db.close()
    return dirhash
