"""Unit tests for //lib/labm8/dirhashcache.py."""
import pathlib
import sys
import tempfile

import pytest
from absl import app
from absl import flags

from lib.labm8 import dirhashcache


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def database_path() -> str:
  with tempfile.TemporaryDirectory(prefix='labm8_dirhashcache_') as d:
    yield d + '/dirhashcache.db'


def test_DirHashCache_invalid_hash_function(database_path):
  """Test that an unrecognized hash function raises an error."""
  with pytest.raises(ValueError):
    dirhashcache.DirHashCache(database_path,
                              hash_function='not a valid hash function')


def test_dirhash_empty_directory(database_path):
  """Test the hash of an empty directory."""
  c = dirhashcache.DirHashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    assert c.dirhash(d) == 'd41d8cd98f00b204e9800998ecf8427e'


def test_dirhash_non_existent_path(database_path):
  """Test that a non-existent path raises an error."""
  c = dirhashcache.DirHashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(ValueError):
      c.dirhash(pathlib.Path(d) / 'a')


def test_dirhash_file_path(database_path):
  """Test that a path to a file raises an error."""
  c = dirhashcache.DirHashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    with pytest.raises(ValueError):
      c.dirhash((pathlib.Path(d) / 'a'))


def test_dirhash_modified_directory(database_path):
  """Test that modifying a directory changes the hash."""
  c = dirhashcache.DirHashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    hash_1 = c.dirhash(d)
    (pathlib.Path(d) / 'a').touch()
    hash_2 = c.dirhash(d)
    assert hash_1 != hash_2


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
