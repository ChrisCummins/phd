"""Unit tests for //lib/labm8/hashcache.py."""
import pathlib
import tempfile

import pytest
import sys
from absl import app
from absl import flags

from lib.labm8 import hashcache


FLAGS = flags.FLAGS

HASH_FUNCTIONS = ['md5', 'sha1', 'sha256']


@pytest.fixture(scope='function')
def database_path() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='labm8_hashcache_') as d:
    yield pathlib.Path(d) / 'hashcache.db'


def test_HashCache_unrecognized_hash_fn(database_path):
  """Test that a non-existent path raises an error."""
  with pytest.raises(ValueError) as e_info:
    hashcache.HashCache(database_path, 'null')
  assert "Hash function not recognized: 'null'" == str(e_info.value)


def test_HashCache_GetHash_non_existent_path(database_path):
  """Test that a non-existent path raises an error."""
  c = hashcache.HashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(FileNotFoundError) as e_info:
      c.GetHash(pathlib.Path(d) / 'a')
    assert f"File not found: '{d}/a'" == str(e_info.value)


def test_HashCache_GetHash_empty_file(database_path):
  """Test the hash of an empty file."""
  c = hashcache.HashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    assert 'd41d8cd98f00b204e9800998ecf8427e' == c.GetHash(
        (pathlib.Path(d) / 'a'))


def test_HashCache_GetHash_empty_directory(database_path):
  """Test the hash of an empty directory."""
  c = hashcache.HashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    assert 'd41d8cd98f00b204e9800998ecf8427e' == c.GetHash(pathlib.Path(d))


def test_HashCache_GetHash_modified_directory(database_path):
  """Test that modifying a directory changes the hash."""
  c = hashcache.HashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    hash_1 = c.GetHash(pathlib.Path(d))
    (pathlib.Path(d) / 'a').touch()
    hash_2 = c.GetHash(pathlib.Path(d))
    assert hash_1 != hash_2


def test_HashCache_GetHash_modified_file(database_path):
  """Test that modifying a file changes the hash."""
  c = hashcache.HashCache(database_path, 'md5')
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    hash_1 = c.GetHash(pathlib.Path(d) / 'a')
    (pathlib.Path(d) / 'a').touch()
    hash_2 = c.GetHash(pathlib.Path(d))
    assert hash_1 != hash_2


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
