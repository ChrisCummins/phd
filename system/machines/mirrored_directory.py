"""Class for a mirrored directory.

This library depends on 'rsync' and 'ssh' being on the system path.
"""
import datetime
import os
import pathlib
import subprocess
import time
import typing

from labm8 import app
from labm8 import fs
from labm8 import labtypes
from system.machines.proto import machine_spec_pb2

FLAGS = app.FLAGS


class InvalidOperation(ValueError):
  """Error raised when trying to perform an invalid operation."""
  pass


class MirroredDirectory(object):
  """Provides interface for a mirrored directory.

  See //system/machines/proto:machine_spec.proto for the spec.
  """

  def __init__(self, host: machine_spec_pb2.Host,
               spec: machine_spec_pb2.MirroredDirectory):
    self.host = host
    self.spec = spec
    self.rsync_remote_path = f'{self.host.host}:{self.remote_path}'

  @property
  def name(self) -> str:
    return self.spec.name

  @property
  def local_path(self) -> str:
    """Return the local path."""
    if self.spec.local_path[-1] == '/':
      return self.spec.local_path
    else:
      return self.spec.local_path + '/'

  @property
  def remote_path(self) -> str:
    """Return the remote path."""
    if self.spec.remote_path[-1] == '/':
      return self.spec.remote_path
    else:
      return self.spec.remote_path + '/'

  @property
  def timestamp_relpath(self) -> str:
    """Get the relpath of a timestamp file."""
    return self.spec.timestamp_relpath or None

  @property
  def local_timestamp(self) -> datetime.datetime:
    """Return the local timestamp."""
    if not self.timestamp_relpath:
      return datetime.datetime.fromtimestamp(0)
    timestamp_path = os.path.join(self.local_path, self.timestamp_relpath)
    if os.path.isfile(timestamp_path):
      return datetime.datetime.fromtimestamp(int(fs.Read(timestamp_path)) / 1e6)
    else:
      return datetime.datetime.fromtimestamp(0)

  @local_timestamp.setter
  def local_timestamp(self, timestamp: int) -> None:
    """Set the local timestamp."""
    timestamp_path = os.path.join(self.local_path, self.timestamp_relpath)
    fs.AtomicWrite(timestamp_path, str(timestamp).encode('utf-8'))

  @property
  def remote_timestamp(self) -> datetime.datetime:
    """Return the remote timestamp."""
    if not self.timestamp_relpath:
      return datetime.datetime.fromtimestamp(0)
    try:
      return datetime.datetime.fromtimestamp(
          int(
              self._Ssh(
                  f'cat "{os.path.join(self.remote_path, self.timestamp_relpath)}"'
              )) / 1e6)
    except subprocess.CalledProcessError:
      return datetime.datetime.fromtimestamp(0)

  @remote_timestamp.setter
  def remote_timestamp(self, timestamp: int) -> None:
    """Set the remote timestamp."""
    self._Ssh(f'echo {timestamp} > '
              f'"{os.path.join(self.remote_path, self.timestamp_relpath)}"')

  @property
  def skip_if_not_present(self):
    return self.spec.skip_if_not_present

  def RemoteExists(self) -> bool:
    """Test if remote directory exists."""
    try:
      self._Ssh(f'test -d "{self.remote_path}"')
      return True
    except subprocess.SubprocessError:
      return False

  def LocalExists(self) -> bool:
    """Test if local path exists."""
    return pathlib.Path(self.local_path).is_dir()

  def PushFromLocalToRemote(self,
                            dry_run: bool = False,
                            verbose: bool = False,
                            delete: bool = True,
                            progress: bool = False,
                            force: bool = False) -> None:
    """Push from local to remote paths."""
    if self.spec.pull_only:
      raise InvalidOperation("Mirrored directory has been marked 'pull_only'")
    if self.skip_if_not_present and not self.LocalExists():
      app.Log(1, 'Skipping local path that does not exist: `%s`',
              self.local_path)
      return
    if self.timestamp_relpath:
      if not force and self.local_timestamp < self.remote_timestamp:
        raise InvalidOperation(
            "Refusing to push to local directory with out-of-date timestamp")
      if not dry_run:
        new_timestamp = int(time.time() * 1e6)
        self.local_timestamp = new_timestamp
        self.remote_timestamp = new_timestamp
    self.Rsync(self.local_path, self.rsync_remote_path, self.host.port,
               self.spec.rsync_exclude, dry_run, verbose, delete, progress)

  def PullFromRemoteToLocal(self,
                            dry_run: bool = False,
                            verbose: bool = False,
                            delete: bool = True,
                            progress: bool = False,
                            force: bool = False) -> None:
    """Pull from remote to local paths."""
    if self.spec.push_only:
      raise InvalidOperation("Mirrored directory has been marked 'push_only'")
    if self.skip_if_not_present and not self.LocalExists():
      app.Log(1, 'Skipping local path that does not exist: `%s`',
              self.local_path)
      return
    if self.timestamp_relpath:
      if not force and self.local_timestamp > self.remote_timestamp:
        raise InvalidOperation(
            "Refusing to pull from remote directory with out-of-date timestamp")
      if not dry_run:
        new_timestamp = int(time.time() * 1e6)
        self.local_timestamp = new_timestamp
        self.remote_timestamp = new_timestamp
    self.Rsync(self.rsync_remote_path, self.local_path, self.host.port,
               self.spec.rsync_exclude, dry_run, verbose, delete, progress)

  def __repr__(self) -> str:
    return (f"MirroredDirectory(name={self.name}, "
            f"local_path='{self.local_path}', "
            f"remote_path='{self.rsync_remote_path}')")

  def _Ssh(self, cmd: str) -> str:
    """Run command on remote machine and return its output."""
    return subprocess.check_output(
        ['ssh', '-p',
         str(self.host.port), self.host.host, '-t', cmd],
        universal_newlines=True)

  # TODO(cec): Put this in it's own module, with a class RsyncOptions to
  # replace the enormous argument list.
  @staticmethod
  def Rsync(src: str, dst: str, host_port: int, excludes: typing.List[str],
            dry_run: bool, verbose: bool, delete: bool, progress: bool):
    """Private helper method to invoke rsync with appropriate arguments."""
    cmd = [
        'rsync',
        '-ah',
        str(src),
        str(dst),
        '-e',
        f'ssh -p {host_port}',
    ] + labtypes.flatten([['--exclude', p] for p in excludes])
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
