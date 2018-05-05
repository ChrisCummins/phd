# Copyright (C) 2015-2018 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
import time

from lib.labm8 import fs
from lib.labm8 import lockfile
from lib.labm8.tests.testutil import TestCase


class TestLockFile(TestCase):

  def test_path(self):
    lock = lockfile.LockFile("/tmp/labm8.lock")
    fs.rm(lock.path)

    self.assertFalse(lock.islocked)

    lock.acquire()
    self.assertTrue(fs.exists(lock.path))
    self.assertTrue(lock.islocked)
    self.assertTrue(lock.owned_by_self)

    lock.acquire()
    self.assertTrue(lock.islocked)
    self.assertTrue(lock.owned_by_self)

    lock.release()
    self.assertFalse(fs.exists(lock.path))

  def test_replace_stale(self):
    lock = lockfile.LockFile("/tmp/labm8.stale.lock")
    fs.rm(lock.path)

    self.assertFalse(lock.islocked)

    MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
    lockfile.LockFile.write(lock.path, MAX_PROCESSES + 1, time.time())

    self.assertTrue(lock.islocked)
    self.assertFalse(lock.owned_by_self)

    with self.assertRaises(lockfile.UnableToAcquireLockError):
      lock.acquire()

    lock.acquire(replace_stale=True)
    self.assertTrue(lock.islocked)
    self.assertTrue(lock.owned_by_self)

    lock.release()
    self.assertFalse(fs.exists(lock.path))

  def test_force(self):
    lock = lockfile.LockFile("/tmp/labm8.force.lock")
    fs.rm(lock.path)

    self.assertFalse(lock.islocked)

    lockfile.LockFile.write(lock.path, 0, time.time())

    self.assertTrue(lock.islocked)
    self.assertFalse(lock.owned_by_self)

    with self.assertRaises(lockfile.UnableToAcquireLockError):
      lock.acquire()

    lock.acquire(force=True)
    self.assertTrue(lock.islocked)
    self.assertTrue(lock.owned_by_self)

    lock.release()
    self.assertFalse(fs.exists(lock.path))
