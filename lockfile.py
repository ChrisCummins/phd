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
from labm8 import system


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
        pid, _ = LockFile.read(self.path)
        return pid

    @property
    def date(self):
        """
        The date that the lock was acquired. Value is None if lock is unclaimed.
        """
        _, date = LockFile.read(self.path)
        return date

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

    def acquire(self, replace_stale=False, force=False):
        """
        Acquire the lock.

        A lock can be claimed if any of these conditions are true:
            1. The lock is unheld by anyone.
            2. The lock is held but the 'force' argument is set.
            3. The lock is held by the current process.

        Arguments:
            replace_stale (bool, optional) If true, lock can be aquired from
                stale processes. A stale process is one which currently owns
                the parent lock, but no process with that PID is alive.
            force (bool, optional): If true, ignore any existing
              lock. If false, fail if lock already claimed.

        Returns:
            LockFile: self.

        Raises:
            UnableToAcquireLockError: If the lock is already claimed
              (not raised if force option is used).
        """

        def _create_lock():
            LockFile.write(self.path, os.getpid(), time.time())

        if self.islocked:
            lock_owner_pid = self.pid

            if self.owned_by_self:
                pass  # don't replace existing lock
            elif force:
                _create_lock()
            elif replace_stale and not system.isprocess(lock_owner_pid):
                _create_lock()
            else:
                raise UnableToAcquireLockError(self)
        else:  # new lock
            _create_lock()

        return self

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

    @staticmethod
    def read(path):
        """
        Read the contents of a LockFile.

        Arguments:
            path (str): Path to lockfile.

        Returns:
            Tuple(int, datetime): The integer PID of the lock owner, and the
                date the lock was required. If the lock is not claimed, both
                values are None.
        """
        if fs.exists(path):
            with open(path) as infile:
                components = infile.read().split()
                pid = int(components[0])
                date = datetime.date.fromtimestamp(float(components[1]))
            return pid, date
        else:
            return None, None

    @staticmethod
    def write(path, pid, timestamp):
        """
        Write the contents of a LockFile.

        Arguments:
            path (str): Path to lockfile.
            pid (int): The integer process ID.
            timestamp (datetime): The time the lock was aquired.
        """
        with open(path, "w") as lockfile:
            print(pid, timestamp, file=lockfile)
