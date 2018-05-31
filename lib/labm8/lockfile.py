"""Lock file mechanism."""
# Use absolute paths for imports so as to prevent a conflict with the
# system "time" module.
from __future__ import absolute_import
from __future__ import print_function

import datetime
import os
import pathlib
import sys
import time
import typing

from lib.labm8 import labdate
from lib.labm8 import pbutil
from lib.labm8 import system
from lib.labm8.proto import lockfile_pb2


class Error(Exception):
  pass


class UnableToAcquireLockError(Error):
  """ thrown if cannot acquire lock """

  def __init__(self, lock):
    self.lock = lock

  def __str__(self):
    return f"""\
Unable to acquire file lock owned by a different process.
Lock acquired by process {self.lock.pid} on {self.lock.date}.
Lock path: {self.lock.path}"""


class UnableToReleaseLockError(Error):
  """ thrown if cannot release lock """

  def __init__(self, lock: 'LockFile'):
    self.lock = lock

  def __str__(self):
    return f"""\
Unable to release file lock owned by a different process.
Lock acquired by process {self.lock.pid} on {self.lock.date}.
Lock path: {self.lock.path}"""


class LockFile:
  """A lock file.

  Attributes:
    path: Path of lock file.
  """

  def __init__(self, path: typing.Union[str, pathlib.Path]):
    """Create a new directory lock.

    Args:
      path: Path to lock file.
    """
    self.path = pathlib.Path(path).expanduser().absolute()

  @property
  def pid(self) -> typing.Optional[datetime.datetime]:
    """The process ID of the lock. Value is None if lock is not claimed."""
    lockfile = self.read(self.path)
    if lockfile.HasField('owner_process_id'):
      return lockfile.owner_process_id
    else:
      return None

  @property
  def date(self) -> typing.Optional[datetime.datetime]:
    """The date that the lock was acquired. Value is None if lock is unclaimed.
    """
    lockfile = self.read(self.path)
    if lockfile.date_acquired_utc_epoch_ms:
      return labdate.DatetimeFromMillisecondsTimestamp(
          lockfile.date_acquired_utc_epoch_ms)
    else:
      return None

  @property
  def islocked(self) -> bool:
    """Whether the directory is locked."""
    return self.path.is_file()

  @property
  def owned_by_self(self):
    """
    Whether the directory is locked by the current process.
    """
    return self.pid == os.getpid()

  def acquire(self, replace_stale: bool = False, force: bool = False,
              pid: int = None, block: bool = False):
    """Acquire the lock.

    A lock can be claimed if any of these conditions are true:
      1. The lock is unheld by anyone.
      2. The lock is held but the 'force' argument is set.
      3. The lock is held by the current process.

    Args:
      replace_stale: If true, lock can be aquired from stale processes. A stale
        process is one which currently owns the parent lock, but no process with
        that PID is alive.
      force: If true, ignore any existing lock. If false, fail if lock already
        claimed.
      pid: If provided, force the process ID of the lock to this value.
        Otherwise the ID of the current process is used.
      block: If True, block indefinitely until the lock is available. Use with
        care!

    Returns:
      Self.

    Raises:
      UnableToAcquireLockError: If the lock is already claimed
        (not raised if force option is used).
    """

    def _create_lock():
      lockfile = lockfile_pb2.LockFile(
          owner_process_id=os.getpid() if pid is None else pid,
          owner_process_argv=' '.join(sys.argv),
          date_acquired_utc_epoch_ms=labdate.MillisecondsTimestamp(
              labdate.GetUtcMillisecondsNow()))
      pbutil.ToFile(lockfile, self.path, assume_filename='LOCK.pbtxt')

    while block:
      if self.islocked:
        lock_owner_pid = self.pid
        if self.owned_by_self:
          pass  # don't replace existing lock
          return self
        elif force:
          _create_lock()
          return self
        elif replace_stale and not system.isprocess(lock_owner_pid):
          _create_lock()
          return self
        elif block:
          raise UnableToAcquireLockError(self)
        # Block and try again later.
        time.sleep(5.0)
      else:  # new lock
        _create_lock()
        return self

  def release(self, force=False):
    """Release lock.

    To release a lock, we must already own the lock.

    Args:
      force: If true, ignore any existing lock owner.

    Raises:
      UnableToReleaseLockError: If the lock is claimed by another process (not
        raised if force option is used).
    """
    # There's no lock, so do nothing.
    if not self.islocked:
      return

    if self.owned_by_self or force:
      os.remove(self.path)
    else:
      raise UnableToReleaseLockError(self)

  def __repr__(self):
    return str(self.path)

  def __enter__(self):
    return self.acquire()

  def __exit__(self, type, value, tb):
    self.release()

  @staticmethod
  def read(path: typing.Union[str, pathlib.Path]) -> lockfile_pb2.LockFile:
    """Read the contents of a LockFile.

    Args:
      path: Path to lockfile.

    Returns:
      A LockFile proto.
    """
    path = pathlib.Path(path)
    if path.is_file():
      return pbutil.FromFile(path, lockfile_pb2.LockFile(),
                             assume_filename='LOCK.pbtxt')
    else:
      return lockfile_pb2.LockFile()
