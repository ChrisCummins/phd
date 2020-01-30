# Copyright (c) 2016-2020 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
import platform
import subprocess
import sys
import typing
from typing import Iterator

from gpu.clinfo.proto import clinfo_pb2
from gpu.oclgrind import oclgrind
from labm8.py import bazelutil
from labm8.py import pbutil

CLINFO = bazelutil.DataPath("phd/gpu/clinfo/clinfo")


class OpenCLEnvironment(object):
  def __init__(self, device: clinfo_pb2.OpenClDevice):
    self._proto = device

  @property
  def name(self) -> str:
    return self._proto.name

  @property
  def platform_name(self) -> str:
    return self._proto.platform_name

  @property
  def device_name(self) -> str:
    return self._proto.device_name

  @property
  def driver_version(self) -> str:
    return self._proto.driver_version

  @property
  def opencl_version(self) -> str:
    return self._proto.opencl_version

  @property
  def device_type(self) -> str:
    return self._proto.device_type

  @property
  def platform_id(self) -> int:
    return self._proto.platform_id

  @property
  def device_id(self) -> int:
    return self._proto.device_id

  @property
  def opencl_opt(self) -> str:
    return self._proto.opencl_opt

  @opencl_opt.setter
  def opencl_opt(self, opt: bool):
    self._proto.opencl_opt = opt

  @property
  def proto(self) -> clinfo_pb2.OpenClDevice:
    return self._proto

  def ids(self) -> typing.Tuple[int, int]:
    """Return platform and device ID numbers.

    The ID numbers can be used to index into the list of platforms and
    devices. Note that the stability of these IDs is *not* guaranteed
    by OpenCL, and may depend on ICD load order or any number of other
    factors.
    """
    return self.platform_id, self.device_id

  def Exec(
    self,
    argv: typing.List[str],
    stdin: typing.Optional[str] = None,
    env: typing.Dict[str, str] = None,
  ) -> subprocess.Popen:
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
    # app.Log(2, '$ %s', ' '.join(argv))
    process = subprocess.Popen(
      argv,
      stdout=subprocess.PIPE,
      stdin=subprocess.PIPE if stdin else None,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      env=env,
    )
    if stdin:
      stdout, stderr = process.communicate(stdin)
    else:
      stdout, stderr = process.communicate()
    process.stdout, process.stderr = stdout, stderr
    return process

  @classmethod
  def FromName(cls, env_name: str) -> "OpenCLEnvironment":
    """Look up OpenCL environment from name.

    Args:
      env_name: The name of the environment.

    Returns:
      An OpenCLEnvironment instance.

    Raises:
      LookupError: If the name is not found.
    """
    all_envs = {env.name: env for env in GetOpenClEnvironments()}
    if env_name in all_envs:
      return all_envs[env_name]
    else:
      available = "\n".join(f"    {n}" for n in sorted(all_envs.keys()))
      raise LookupError(
        f"Requested OpenCL environment not available: '{env_name}'.\n"
        f"Available OpenCL devices:\n{available}"
      )


class OclgrindOpenCLEnvironment(OpenCLEnvironment):
  """A mock OpenCLEnvironment for oclgrind."""

  def __init__(self):
    super(OclgrindOpenCLEnvironment, self).__init__(oclgrind.CLINFO_DESCRIPTION)

  def Exec(
    self,
    argv: typing.List[str],
    stdin: typing.Optional[str] = None,
    env: typing.Dict[str, str] = None,
  ) -> subprocess.Popen:
    """Execute a command in the device environment."""
    return oclgrind.Exec(
      [
        "--max-errors",
        "1",
        "--uninitialized",
        "--data-races",
        "--uniform-writes",
        "--uniform-writes",
      ]
      + argv,
      stdin=stdin,
      env=env,
    )


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
  return any(env.device_type == "CPU" for env in all_envs())


def has_gpu() -> bool:
  """
  Determine if there is a CPU OpenCL device available.

  Returns
  -------
  bool
      True if device available, else False.
  """
  return any(env.device_type == "GPU" for env in all_envs())


def GetOpenClEnvironments() -> typing.List[OpenCLEnvironment]:
  """Get a list of available OpenCL testbeds.

  This includes the local oclgrind device.

  Returns:
    A list of OpenCLEnvironment instances.
  """
  return sorted(
    list(all_envs()) + [OclgrindOpenCLEnvironment()], key=lambda x: x.name
  )


def GetTestbedNames() -> typing.List[str]:
  """Get a list of available OpenCL testbed names."""
  return [env.name for env in GetOpenClEnvironments()]


def PrintOpenClEnvironments() -> None:
  """List the names and details of available OpenCL testbeds."""
  for i, env in enumerate(GetOpenClEnvironments()):
    if i:
      print()
    print(env.name)
    print(f"    Platform:     {env.platform_name}")
    print(f"    Device:       {env.device_name}")
    print(f"    Driver:       {env.driver_version}")
    print(f"    Device Type:  {env.device_type}")
    print(f"    OpenCL:       {env.opencl_version}")
