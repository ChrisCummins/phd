"""Class for a mirrored directory.

This library depends on 'rsync' and 'ssh' being on the system path.
"""
import pathlib
import subprocess
import typing

from absl import flags
from absl import logging

from labm8 import labtypes
from system.machines.proto import machine_spec_pb2

FLAGS = flags.FLAGS


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

  def RemoteExists(self) -> bool:
    """Test if remote directory exists."""
    try:
      subprocess.check_output([
          'ssh', '-p',
          str(self.host.port), self.host.host, '-t',
          f'test -d "{self.remote_path}"'
      ])
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
                            progress: bool = False) -> None:
    """Push from local to remote paths."""
    if self.spec.pull_only:
      raise InvalidOperation("Mirrored directory has been marked 'pull_only'")
    self.Rsync(self.local_path, self.rsync_remote_path, self.host.port,
               self.spec.rsync_exclude, dry_run, verbose, delete, progress)

  def PullFromRemoteToLocal(self,
                            dry_run: bool = False,
                            verbose: bool = False,
                            delete: bool = True,
                            progress: bool = False) -> None:
    """Pull from remote to local paths."""
    if self.spec.push_only:
      raise InvalidOperation("Mirrored directory has been marked 'push_only'")
    self.Rsync(self.rsync_remote_path, self.local_path, self.host.port,
               self.spec.rsync_exclude, dry_run, verbose, delete, progress)

  def __repr__(self) -> str:
    return (f"MirroredDirectory(name={self.name}, "
            f"local_path='{self.local_path}', "
            f"remote_path='{self.rsync_remote_path}')")

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
    logging.info(' '.join(cmd))
    p = subprocess.Popen(cmd)
    p.communicate()
    if p.returncode:
      raise subprocess.SubprocessError(
          f'rsync failed with returncode {p.returncode}')
