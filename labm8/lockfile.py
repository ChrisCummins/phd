# Copyright (C) 2017 Chris Cummins.
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
#
"""
Lock file mechanism.
"""
from __future__ import print_function
# Use absolute paths for imports so as to prevent a conflict with the
# system "time" module.
from __future__ import absolute_import

import datetime
import os
import time

from labm8 import fs


class Error(Exception):
    pass


class UnableToAcquireLockError(Error):
    """ thrown if cannot acquire lock """
    def __init__(self, lock):
        self.lock = lock

    def __str__(self):
        path = self.lock.path
        pid = self.lock.pid
        date = self.lock.date
        return """\
Unable to acquire file lock owned by a different process.
Lock acquired by process {pid} on {date}.
Lock path: {path}""".format(**vars())


class UnableToReleaseLockError(Error):
    """ thrown if cannot release lock """
    def __init__(self, lock):
        self.lock = lock

    def __str__(self):
        path = self.lock.path
        pid = self.lock.pid
        date = self.lock.date
        return """\
Unable to release file lock owned by a different process.
Lock acquired by process {pid} on {date}.
Lock path: {path}""".format(**vars())


class LockFile:
    """
    A lock file.

    Attributes:
        path (str): Path of lock file.
    """
    def __init__(self, path):
        """
        Create a new directory lock.
        Arguments:
            path (str): Path to lock file.
        """
        self.path = fs.path(path)

    @property
    def pid(self):
        """
        The process ID of the lock. Value is None if lock is not claimed.
        """
        if fs.exists(self.path):
            with open(self.path) as infile:
                data = infile.read()
                components = data.split()
                pid = int(components[0])
                return pid
        else:
            return None

    @property
    def date(self):
        """
        The date that the lock was acquired. Value is None if lock is unclaimed.
        """
        if fs.exists(self.path):
            with open(self.path) as lockfile:
                data = lockfile.read()
                components = data.split()
                date = datetime.date.fromtimestamp(float(components[1]))
                return date
        else:
            return None

    @property
    def islocked(self):
        """
        Whether the directory is locked.
        """
        return fs.exists(self.path)

    @property
    def owned_by_self(self):
        """
        Whether the directory is locked by the current process.
        """
        return self.pid == os.getpid()

    def acquire(self, force=False):
        """
        Acquire the lock.

        A lock can be claimed if any of these conditions are true:
            1. The lock is unheld by anyone.
            2. The lock is held but the 'force' argument is set.
            3. The lock is held by the current process.

        Arguments:
            force (boolean, optional): If true, ignore any existing
              lock. If false, fail if lock already claimed.

        Returns:
            LockFile: self.

        Raises:
            UnableToAcquireLockError: If the lock is already claimed
              (not raised if force option is used).
        """
        if not self.islocked or force or self.pid == os.getpid():
            with open(self.path, "w") as lockfile:
                print(os.getpid(), time.time(), file=lockfile)
            return self
        else:
            raise UnableToAcquireLockError(self)

    def release(self, force=False):
        """
        Release lock.

        To release a lock, we must already own the lock.

        Arguments:
            force (bool, optional): If true, ignore any existing lock owner.

        Raises:
            UnableToReleaseLockError: If the lock is claimed by another
              process (not raised if force option is used).
        """
        # There's no lock, so do nothing.
        if not self.islocked:
            return

        if self.owned_by_self or force:
            os.remove(self.path)
        else:
            raise UnableToReleaseLockError(self)

    def __repr__(self):
        return self.path

    def __enter__(self):
        return self.acquire()

    def __exit__(self, type, value, tb):
        self.release()
