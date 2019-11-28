"""Unit tests for //system/machines:mirrored_directory.py."""
import pathlib
import subprocess
import tempfile
import time
import typing

import pytest

from labm8.py import app
from labm8.py import fs
from labm8.py import labtypes
from labm8.py import test
from system.machines import mirrored_directory
from system.machines.proto import machine_spec_pb2

FLAGS = app.FLAGS


@pytest.fixture(scope='module')
def test_host() -> machine_spec_pb2.Host:
  return machine_spec_pb2.Host(host='localhost', port=22)


def _MakeMirroredDirectory():
  with tempfile.TemporaryDirectory(
      prefix='phd_mirrored_directory_remote_path_') as d1:
    with tempfile.TemporaryDirectory(
        prefix='phd_mirrored_directory_local_path_') as d2:
      yield machine_spec_pb2.MirroredDirectory(name='test_mirrored_directory',
                                               remote_path=d1,
                                               local_path=d2)


@pytest.fixture(scope='function')
def test_mirrored_directory() -> machine_spec_pb2.MirroredDirectory:
  yield from _MakeMirroredDirectory()


@pytest.fixture(scope='function')
def test_mirrored_directory2() -> machine_spec_pb2.MirroredDirectory:
  yield from _MakeMirroredDirectory()


class LocalMirroredDirectory(mirrored_directory.MirroredDirectory):
  """Mirrored directory which overrides rsync to support local remote paths."""

  def _Ssh(self, cmd: str) -> str:
    """Perform command locally."""
    return subprocess.check_output(cmd, shell=True, universal_newlines=True)

  @staticmethod
  def Rsync(src: str, dst: str, host_port: int, excludes: typing.List[str],
            dry_run: bool, verbose: bool, delete: bool, progress: bool):
    """Rsync for local transfers without ssh."""
    del host_port
    src = str(src).replace('localhost:', '')
    dst = str(dst).replace('localhost:', '')
    cmd = ['rsync', '-ah', str(src), str(dst)] + labtypes.flatten(
        [['--exclude', p] for p in excludes])
    if dry_run:
      cmd.append('--dry-run')
    if verbose:
      cmd.append('--verbose')
    if delete:
      cmd.append('--delete')
    if progress:
      cmd.append('--progress')
    app.Log(1, ' '.join(cmd))
    p = subprocess.Popen(cmd)
    p.communicate()
    if p.returncode:
      raise subprocess.SubprocessError(
          f'rsync failed with returncode {p.returncode}')


# rsync_exclude


def test_PushLocalToRemote(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test pushing a file to remote."""
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  with open(fs.path(m.local_path, 'a'), 'w') as f:
    f.write('Hello, world!')

  m.PushFromLocalToRemote()

  assert pathlib.Path(m.remote_path).is_dir()
  assert (pathlib.Path(m.remote_path) / 'a').is_file()
  with open(fs.path(m.remote_path, 'a')) as f:
    assert f.read() == 'Hello, world!'


def test_PullFromRemoteToLocal(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test pulling a file from remote."""
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  with open(fs.path(m.remote_path, 'a'), 'w') as f:
    f.write('Hello, world!')

  m.PullFromRemoteToLocal()

  assert pathlib.Path(m.local_path).is_dir()
  assert (pathlib.Path(m.local_path) / 'a').is_file()
  with open(fs.path(m.local_path, 'a')) as f:
    assert f.read() == 'Hello, world!'


def test_PushLocalToRemote_exclude(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test pushing with an excluded file."""
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  with open(fs.path(m.local_path, 'a'), 'w') as f:
    f.write('Hello, world!')

  # Exclude /a from transfer.
  m.spec.rsync_exclude.extend(['/a'])
  m.PushFromLocalToRemote()

  assert not (pathlib.Path(m.remote_path) / 'a').is_file()


def test_PushLocalToRemote_pull_only_push_error(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that pull_only disables push."""
  test_mirrored_directory.pull_only = True
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)

  with pytest.raises(mirrored_directory.InvalidOperation):
    m.PushFromLocalToRemote()


def test_PushLocalToRemote_push_only_pull_error(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that push_only disables pull."""
  test_mirrored_directory.push_only = True
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)

  with pytest.raises(mirrored_directory.InvalidOperation):
    m.PullFromRemoteToLocal()


# Tests of timestamp files.


def test_PushLocalToRemote_timestamp_files_created(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that timestamp files are created."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  m.PushFromLocalToRemote()
  assert (pathlib.Path(m.local_path) / 'TIME.txt').is_file()
  assert (pathlib.Path(m.remote_path) / 'TIME.txt').is_file()


def test_PullFromRemoteToLocal_timestamp_files_created(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that timestamp files are created."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  m.PullFromRemoteToLocal()
  assert (pathlib.Path(m.local_path) / 'TIME.txt').is_file()
  assert (pathlib.Path(m.remote_path) / 'TIME.txt').is_file()


def test_PullFromRemoteToLocal_timestamp_files_no_created_in_dry_run(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that timestamp files are created."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)
  m.PullFromRemoteToLocal(dry_run=True)
  assert not (pathlib.Path(m.local_path) / 'TIME.txt').is_file()
  assert not (pathlib.Path(m.remote_path) / 'TIME.txt').is_file()


def test_PushLocalToRemote_timestamp_in_past_cannot_be_pushed(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that local cannot be pushed when behind remote."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)

  fs.Write(pathlib.Path(m.local_path) / 'TIME.txt', '50'.encode('utf-8'))
  fs.Write(pathlib.Path(m.remote_path) / 'TIME.txt', '100'.encode('utf-8'))
  with pytest.raises(mirrored_directory.InvalidOperation):
    m.PushFromLocalToRemote()


def test_PullFromRemoteToLocal_timestamp_in_past_cannot_be_pulled(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory):
  """Test that local cannot be pulled when ahead of remote."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  m = LocalMirroredDirectory(test_host, test_mirrored_directory)

  fs.Write(pathlib.Path(m.local_path) / 'TIME.txt', '100'.encode('utf-8'))
  fs.Write(pathlib.Path(m.remote_path) / 'TIME.txt', '50'.encode('utf-8'))
  with pytest.raises(mirrored_directory.InvalidOperation):
    m.PullFromRemoteToLocal()


def test_push_race(
    test_host: machine_spec_pb2.Host,
    test_mirrored_directory: machine_spec_pb2.MirroredDirectory,
    test_mirrored_directory2: machine_spec_pb2.MirroredDirectory):
  """Test a common push race scenario."""
  test_mirrored_directory.timestamp_relpath = 'TIME.txt'
  test_mirrored_directory2.timestamp_relpath = 'TIME.txt'
  test_mirrored_directory2.remote_path = (test_mirrored_directory.remote_path)
  m1 = LocalMirroredDirectory(test_host, test_mirrored_directory)
  m2 = LocalMirroredDirectory(test_host, test_mirrored_directory2)

  m1.PushFromLocalToRemote()
  with pytest.raises(mirrored_directory.InvalidOperation):
    m2.PushFromLocalToRemote()
  assert m1.local_timestamp == m2.remote_timestamp
  assert m1.remote_timestamp == m2.remote_timestamp

  m1.PushFromLocalToRemote()
  time.sleep(.1)  # Make sure that timestamp increases.
  m2.PullFromRemoteToLocal()
  with pytest.raises(mirrored_directory.InvalidOperation):
    m1.PushFromLocalToRemote()
  m2.PushFromLocalToRemote()


if __name__ == '__main__':
  test.Main()
