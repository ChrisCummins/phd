"""Unit tests for //lib/labm8:latex."""
import pathlib
import tempfile

import pytest
import sys
from absl import app

from lib.labm8 import fs
from lib.labm8 import lockfile


def test_LockFile_file_exists():
  """Test that lockfile is created on acquire."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    assert not lock.path.is_file()
    lock.acquire()
    assert lock.path.is_file()


def test_LockFile_islocked():
  """Test that lockfile.islocked returns True after acquired."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    assert not lock.islocked
    lock.acquire()
    assert lock.islocked


def test_LockFile_owned_by_self():
  """Test that lockfile.owned_by_self returns True after acquired."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    assert not lock.owned_by_self
    lock.acquire()
    assert lock.owned_by_self


def test_LockFile_release_deletes_file():
  """Test that lockfile is removed after lockfile.release()."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    lock.acquire()
    lock.release()
    assert not lock.path.is_file()


def test_LockFile_replace_stale():
  """Test that lockfile is replaced if stale."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
    lock.acquire(pid=MAX_PROCESSES + 1)
    assert lock.islocked
    assert not lock.owned_by_self
    with pytest.raises(lockfile.UnableToAcquireLockError):
      lock.acquire()
    lock.acquire(replace_stale=True)
    assert lock.islocked
    assert lock.owned_by_self
    lock.release()
    assert not lock.path.is_file()


def test_LockFile_replace_stale():
  """Test that lockfile is replaced if forced."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'LOCK'
    lock = lockfile.LockFile(path)
    MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
    lock.acquire(pid=MAX_PROCESSES + 1)
    assert lock.islocked
    assert not lock.owned_by_self
    with pytest.raises(lockfile.UnableToAcquireLockError):
      lock.acquire()
    lock.acquire(force=True)
    assert lock.islocked
    assert lock.owned_by_self
    lock.release()
    assert not fs.exists(lock.path)


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
