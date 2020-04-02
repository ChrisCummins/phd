"""This file implements the Machine class.

When executed as a script, it provides basic functionality to push and pull
mirrored directories.
"""
import pathlib
import subprocess
import typing

from labm8.py import app
from labm8.py import pbutil
from system.machines import mirrored_directory
from system.machines.mirrored_directory import MirroredDirectory
from system.machines.proto import machine_spec_pb2

FLAGS = app.FLAGS

app.DEFINE_input_path("machine", None, "Path to MachineSpec proto.")
app.DEFINE_list("push", [], "Mirrored directories to push.")
app.DEFINE_list("pull", [], "Mirrored directories to push.")
app.DEFINE_boolean(
  "dry_run", True, "Whether to run ops without making changes."
)
app.DEFINE_boolean(
  "delete", False, "Whether to delete files during push/pull" "mirroring."
)
app.DEFINE_boolean("progress", False, "Show progress during file transfers.")
app.DEFINE_boolean("force", False, "Show progress during file transfers.")


def RespondsToPing(host: str) -> typing.Optional[str]:
  """Return host if it responds to ping.

  Args:
    host: The host address to test.

  Returns:
    The host if it responds to ping, else None.
  """
  ping = pathlib.Path("/sbin/ping")
  if not ping.is_file():
    ping = "ping"
  try:
    subprocess.check_call(
      [str(ping), "-c1", "-W1", host],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    return host
  except subprocess.CalledProcessError:
    return None


def ResolveHost(
  hosts: typing.List[machine_spec_pb2.Host],
) -> machine_spec_pb2.Host:
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
      app.Log(1, "Resolved host %s", host.host)
      return host
    else:
      app.Log(2, "Failed to resolve host %s", host.host)
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
  def FromProto(proto: machine_spec_pb2.MachineSpec) -> "Machine":
    """Instantiate machine from proto."""
    return Machine(proto)

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> "Machine":
    """Instantiate machine from proto file path."""
    return cls.FromProto(pbutil.FromFile(path, machine_spec_pb2.MachineSpec()))


def main():
  """Main entry point."""
  if not FLAGS.machine:
    raise app.UsageError("--machine flag required")
  machine_proto_path = FLAGS.machine
  if not machine_proto_path.is_file():
    raise app.UsageError(f"Cannot find --machine proto '{machine_proto_path}'")
  machine = Machine.FromFile(machine_proto_path)

  try:
    for mirrored_dir_name in FLAGS.pull:
      mirrored_dir = machine.MirroredDirectory(mirrored_dir_name)
      mirrored_dir.PullFromRemoteToLocal(
        dry_run=FLAGS.dry_run,
        verbose=True,
        delete=FLAGS.delete,
        progress=FLAGS.progress,
        force=FLAGS.force,
      )

    for mirrored_dir_name in FLAGS.push:
      mirrored_dir = machine.MirroredDirectory(mirrored_dir_name)
      mirrored_dir.PushFromLocalToRemote(
        dry_run=FLAGS.dry_run,
        verbose=True,
        delete=FLAGS.delete,
        progress=FLAGS.progress,
        force=FLAGS.force,
      )
  except (subprocess.SubprocessError, mirrored_directory.InvalidOperation) as e:
    app.FatalWithoutStackTrace(e)


if __name__ == "__main__":
  app.Run(main)
