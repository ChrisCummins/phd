"""This file implements the Machine class.

When executed as a script, it provides basic functionality to push and pull
mirrored directories.
"""
import pathlib
import subprocess
import typing

from absl import app
from absl import flags
from absl import logging

from labm8 import pbutil
from system.machines.mirrored_directory import MirroredDirectory
from system.machines.proto import machine_spec_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('machine', None, 'Path to MachineSpec proto.')
flags.DEFINE_list('push', [], 'Mirrored directories to push.')
flags.DEFINE_list('pull', [], 'Mirrored directories to push.')
flags.DEFINE_bool('dry_run', True, 'Whether to run ops without making changes.')
flags.DEFINE_bool('delete', False, 'Whether to delete files during push/pull'
                  'mirroring.')
flags.DEFINE_bool('progress', False, 'Show progress during file transfers.')


def RespondsToPing(host: str) -> typing.Optional[str]:
  """Return host if it responds to ping.

  Args:
    host: The host address to test.

  Returns:
    The host if it responds to ping, else None.
  """
  try:
    subprocess.check_output(['ping', '-c1', '-W1', host])
    return host
  except subprocess.CalledProcessError:
    return None


def ResolveHost(
    hosts: typing.List[machine_spec_pb2.Host]) -> machine_spec_pb2.Host:
  """Resolve the host from a list.

  Iterate through the list, seeing if any of the hosts respond to ping. If none
  of them do, the last item of the list is returned.

  Args:
    hosts: The list of hosts to ping.

  Returns:
    A Host message instance.
  """
  for host in hosts[:-1]:
    if RespondsToPing(host.host):
      logging.info('Resolved host %s', host.host)
      return host
    else:
      logging.debug('Failed to resolve host %s', host.host)
  return hosts[-1]


class Machine(object):
  """Provides the functionality for interacting with machines."""

  def __init__(self, proto: machine_spec_pb2.MachineSpec):
    self._proto = proto
    self._host = None
    self._mirrored_directories = None

  @property
  def name(self) -> str:
    """Return the name of the machine."""
    return self._proto.name

  @property
  def host(self) -> machine_spec_pb2.Host:
    """Return the host for the machine."""
    if self._host is None:
      self._host = ResolveHost(self._proto.host)
    return self._host

  @property
  def mirrored_directories(self) -> typing.List[MirroredDirectory]:
    """Return a list of mirrored directories."""
    return [
        MirroredDirectory(self.host, m) for m in self._proto.mirrored_directory
    ]

  def MirroredDirectory(self, name: str) -> MirroredDirectory:
    """Lookup a mirrored directory by name."""
    m = {t.spec.name: t for t in self.mirrored_directories}.get(name)
    if not m:
      raise LookupError(f"Cannot find mirrored directory '{name}'")
    return m

  @staticmethod
  def FromProto(proto: machine_spec_pb2.MachineSpec) -> 'Machine':
    """Instantiate machine from proto."""
    return Machine(proto)

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> 'Machine':
    """Instantiate machine from proto file path."""
    return cls.FromProto(pbutil.FromFile(path, machine_spec_pb2.MachineSpec()))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unknown arguments')

  machine_proto_path = pathlib.Path(FLAGS.machine)
  if not machine_proto_path.is_file():
    raise app.UsageError(f"Cannot find --machine proto '{machine_proto_path}'")
  machine = Machine.FromFile(machine_proto_path)

  for mirrored_dir_name in FLAGS.pull:
    mirrored_dir = machine.MirroredDirectory(mirrored_dir_name)
    mirrored_dir.PullFromRemoteToLocal(
        dry_run=FLAGS.dry_run,
        verbose=True,
        delete=FLAGS.delete,
        progress=FLAGS.progress)

  for mirrored_dir_name in FLAGS.push:
    mirrored_dir = machine.MirroredDirectory(mirrored_dir_name)
    mirrored_dir.PushFromLocalToRemote(
        dry_run=FLAGS.dry_run,
        verbose=True,
        delete=FLAGS.delete,
        progress=FLAGS.progress)


if __name__ == '__main__':
  app.run(main)
