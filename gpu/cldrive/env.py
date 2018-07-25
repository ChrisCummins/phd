import platform
import subprocess
import sys
import typing
from typing import Iterator

from gpu.clinfo.proto import clinfo_pb2
from gpu.oclgrind import oclgrind
from lib.labm8 import bazelutil
from lib.labm8 import pbutil


CLINFO = bazelutil.DataPath('phd/gpu/clinfo/clinfo')


class OpenCLEnvironment(object):

  def __init__(self, device: clinfo_pb2.OpenClDevice):
    self.name = device.name
    self.platform_name = device.platform_name
    self.device_name = device.device_name
    self.driver_version = device.driver_version
    self.opencl_version = device.opencl_version
    self.device_type = device.device_type
    self.platform_id = device.platform_id
    self.device_id = device.device_id
    self.opencl_opt = device.opencl_opt

  def ids(self) -> typing.Tuple[int, int]:
    """Return platform and device ID numbers.

    The ID numbers can be used to index into the list of platforms and
    devices. Note that the stability of these IDs is *not* guaranteed
    by OpenCL, and may depend on ICD load order or any number of other
    factors.
    """
    return self.platform_id, self.device_id

  def Exec(self, argv: typing.List[str],
           env: typing.Dict[str, str] = None) -> subprocess.Popen:
    """Execute a command in an environment for the OpenCL device.

    This creates a Popen process, executes it, and sets the stdout and stderr
    attributes to the process output.

    This method can be used to wrap OpenCL devices which require a specific
    environment to execute, such as by setting a LD_PRELOAD, or running a
    command inside of another (in the case of oclgrind).

    Args:
      argv: A list of arguments to execute.
      env: An optional environment to use.

    Returns:
      A Popen instance, with string stdout and stderr attributes set.
    """
    # logging.debug('$ %s', ' '.join(argv))
    process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True,
                               env=env)
    stdout, stderr = process.communicate()
    process.stdout, process.stderr = stdout, stderr
    return process


class OclgrindOpenCLEnvironment(OpenCLEnvironment):
  """A mock OpenCLEnvironment for oclgrind."""

  def __init__(self):
    super(OclgrindOpenCLEnvironment, self).__init__(
        oclgrind.CLINFO_DESCRIPTION)

  def Exec(self, argv: typing.List[str],
           env: typing.Dict[str, str] = None) -> subprocess.Popen:
    """Execute a command in the device environment."""
    return oclgrind.Exec(
        ['--max-errors', '1', '--uninitialized', '--data-races',
         '--uniform-writes', '--uniform-writes'] + argv, env=env)


def host_os() -> str:
  """
  Get the type and version of the host operating system.

  Returns
  -------
  str
      Formatted <system> <release> <arch>, where <system> is the
      operating system type, <release> is the release version, and
      <arch> is 32 or 64 bit.
  """
  if sys.platform == "linux" or sys.platform == "linux2":
    dist = platform.linux_distribution()
    system, release = dist[0], dist[1]
  else:
    system, release = platform.system(), platform.release()

  arch = platform.architecture()[0]

  return f"{system} {release} {arch}"


def all_envs() -> Iterator[OpenCLEnvironment]:
  """
  Iterate over all available OpenCL environments on a system.

  Parameters
  ----------
  devtype : str, optional
      OpenCL device type to filter by, one of: {all,cpu,gpu}.

  Returns
  -------
  Iterator[OpenCLEnvironment]
      An iterator over all available OpenCL environments.
  """
  stdout = subprocess.check_output([str(CLINFO)])
  devices = pbutil.FromString(stdout, clinfo_pb2.OpenClDevices())
  for device in devices.device:
    yield OpenCLEnvironment(device)


def has_cpu() -> bool:
  """
  Determine if there is a CPU OpenCL device available.

  Returns
  -------
  bool
      True if device available, else False.
  """
  return any(env.device_type == 'CPU' for env in all_envs())


def has_gpu() -> bool:
  """
  Determine if there is a CPU OpenCL device available.

  Returns
  -------
  bool
      True if device available, else False.
  """
  return any(env.device_type == 'GPU' for env in all_envs())


def GetOpenClEnvironments() -> typing.List[OpenCLEnvironment]:
  """Get a list of available OpenCL testbeds.

  This includes the local oclgrind device.

  Returns:
    A list of OpenCLEnvironment instances.
  """
  return sorted(list(all_envs()) + [OclgrindOpenCLEnvironment()],
                key=lambda x: x.name)


def GetTestbedNames() -> typing.List[str]:
  """Get a list of available OpenCL testbed names."""
  return [env.name for env in GetOpenClEnvironments()]


def PrintOpenClEnvironments() -> None:
  """List the names and details of available OpenCL testbeds."""
  for i, env in enumerate(GetOpenClEnvironments()):
    if i:
      print()
    print(env.name)
    print(f'    Platform:     {env.platform_name}')
    print(f'    Device:       {env.device_name}')
    print(f'    Driver:       {env.driver_version}')
    print(f'    Device Type:  {env.device_type}')
    print(f'    OpenCL:       {env.opencl_version}')
