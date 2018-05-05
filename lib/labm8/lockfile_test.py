"""Unit tests for //lib/labm8:latex."""
import sys
import time

import pytest
from absl import app

from lib.labm8 import fs
from lib.labm8 import lockfile


def test_path():
  lock = lockfile.LockFile("/tmp/labm8.lock")
  fs.rm(lock.path)
  assert not lock.islocked
  lock.acquire()
  assert fs.exists(lock.path)
  assert lock.islocked
  assert lock.owned_by_self
  lock.acquire()
  assert lock.islocked
  assert lock.owned_by_self
  lock.release()
  assert not fs.exists(lock.path)


def test_replace_stale():
  lock = lockfile.LockFile("/tmp/labm8.stale.lock")
  fs.rm(lock.path)
  assert not lock.islocked
  MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
  lockfile.LockFile.write(lock.path, MAX_PROCESSES + 1, time.time())
  assert lock.islocked
  assert not lock.owned_by_self
  with pytest.raises(lockfile.UnableToAcquireLockError):
    lock.acquire()
  lock.acquire(replace_stale=True)
  assert lock.islocked
  assert lock.owned_by_self
  lock.release()
  assert not fs.exists(lock.path)


def test_force():
  lock = lockfile.LockFile("/tmp/labm8.force.lock")
  fs.rm(lock.path)
  assert not lock.islocked
  lockfile.LockFile.write(lock.path, 0, time.time())
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
