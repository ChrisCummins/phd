"""Unit tests for //labm8:latex."""
import pathlib
import tempfile

import pytest
from absl import flags

from labm8 import fs
from labm8 import lockfile
from labm8 import pbutil
from labm8 import test
from labm8.proto import lockfile_pb2


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def dummy_lockfile_proto() -> lockfile_pb2.LockFile:
  """A simple lockfile proto."""
  return lockfile_pb2.LockFile(
      owner_process_id=100,
      owner_process_argv='./foo --bar',
      date_acquired_utc_epoch_ms=1529403585000,
      owner_hostname='foo',
      owner_user='bar',
  )


@pytest.fixture(scope='function')
def dummy_lockfile_path(
    dummy_lockfile_proto: lockfile_pb2.LockFile) -> pathlib.Path:
  """Yield a path to a lockfile proto."""
  with tempfile.TemporaryDirectory() as d:
    pbutil.ToFile(dummy_lockfile_proto, pathlib.Path(d) / 'LOCK.pbtxt')
    yield pathlib.Path(d) / 'LOCK.pbtxt'


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


def test_LockFile_acquire_fail(dummy_lockfile_path):
  """Test that acquiring a lock owned by a different host fails."""
  lock = lockfile.LockFile(dummy_lockfile_path)
  assert lock.islocked
  with pytest.raises(lockfile.UnableToAcquireLockError) as e_ctx:
    lock.acquire()
  assert str(e_ctx.value) == f"""\
Unable to acquire file lock owned by a different process.
Lock acquired by process 100 on bar@foo at 2018-06-19 10:19:45.

If you believe that this is an error and that no other
process holds the lock, you may remove the lock file:

   {dummy_lockfile_path}"""


def test_LockFile_release_fail(dummy_lockfile_path):
  """Test that releasing a lock owned by a different host fails."""
  lock = lockfile.LockFile(dummy_lockfile_path)
  assert lock.islocked
  with pytest.raises(lockfile.UnableToReleaseLockError) as e_ctx:
    lock.release()
  assert str(e_ctx.value) == f"""\
Unable to release file lock owned by a different process.
Lock acquired by process 100 on bar@foo at 2018-06-19 10:19:45.

If you believe that this is an error and that no other
process holds the lock, you may remove the lock file:

   {dummy_lockfile_path}"""


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


def test_LockFile_force_replace_stale():
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


@pytest.mark.parametrize('granularity', ('line', 'function', 'module'))
def test_AutoLockFile_path(granularity: str):
  """Test the path of an automatic lock file."""
  # FRAGILE TEST: Moving this line of code requires updating the path tested!
  lock = lockfile.AutoLockFile(granularity=granularity)

  # This is a fragile test: if the position of the line above changes, this test
  # will fail. To fix it, change the number at the end of the path below to the
  # line number of the lockfile declaration:
  if granularity == 'line':
    path = ('/tmp/phd/labm8/autolockfiles/'
            'lockfile_test_test_AutoLockFile_path_151.pbtxt')
  elif granularity == 'function':
    path = ('/tmp/phd/labm8/autolockfiles/'
            'lockfile_test_test_AutoLockFile_path.pbtxt')
  elif granularity == 'module':
    path = '/tmp/phd/labm8/autolockfiles/lockfile_test.pbtxt'

  assert lock.path == pathlib.Path(path)


def test_AutoLockFile_unknown_granularity():
  """Test that unknown granularity raises an error."""
  with pytest.raises(TypeError):
    lockfile.AutoLockFile(granularity='unknown')


@pytest.mark.parametrize('granularity', ('line', 'function', 'module'))
def test_AutoLockFile_acquire(granularity: str):
  """Test that an auto lockfile can be acquired."""
  lock = lockfile.AutoLockFile(granularity=granularity)
  assert not lock.islocked
  with lock:
    assert lock.islocked
  assert not lock.islocked


def test_AutoLockFile_acquire_fail(dummy_lockfile_proto: lockfile_pb2.LockFile):
  """Test that acquiring a lock owned by a different host fails."""
  lock = lockfile.AutoLockFile()

  # Mock that a lock has been "acquired" by another process.
  pbutil.ToFile(dummy_lockfile_proto, lock.path)

  with pytest.raises(lockfile.UnableToAcquireLockError):
    lock.acquire()


if __name__ == '__main__':
  test.Main()
