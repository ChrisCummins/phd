"""Unit tests for //system/machines:mirrored_directory.py."""
import pathlib
import subprocess
import tempfile
import typing

import pytest
from absl import flags
from absl import logging

from labm8 import fs
from labm8 import labtypes
from labm8 import test
from system.machines import mirrored_directory
from system.machines.proto import machine_spec_pb2

FLAGS = flags.FLAGS


@pytest.fixture(scope='module')
def test_host() -> machine_spec_pb2.Host:
  return machine_spec_pb2.Host(host='localhost', port=22)


@pytest.fixture(scope='function')
def test_mirrored_directory() -> machine_spec_pb2.MirroredDirectory:
  with tempfile.TemporaryDirectory(
      prefix='phd_mirrored_directory_remote_path_') as d1:
    with tempfile.TemporaryDirectory(
        prefix='phd_mirrored_directory_local_path_') as d2:
      yield machine_spec_pb2.MirroredDirectory(
          name='test_mirrored_directory', remote_path=d1, local_path=d2)


class LocalMirroredDirectory(mirrored_directory.MirroredDirectory):
  """Mirrored directory which overrides rsync to support local remote paths."""

  @staticmethod
  def Rsync(src: str, dst: str, host_port: int, excludes: typing.List[str],
            dry_run: bool, verbose: bool, delete: bool, progress: bool):
    """Private helper method to invoke rsync with appropriate arguments."""
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
    logging.info(' '.join(cmd))
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


if __name__ == '__main__':
  test.Main()
