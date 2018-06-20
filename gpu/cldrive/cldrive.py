"""Main entry point for CLdrive script."""
import typing

from absl import app
from absl import flags

from gpu import cldrive
from gpu.oclgrind import oclgrind


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'ls_env', False,
    'If set, list the names and details of available OpenCL environments, and'
    'exit.')


def GetOpenClEnvironments() -> typing.List[cldrive.OpenCLEnvironment]:
  """Get a list of available OpenCL testbeds.

  This includes the local oclgrind device.

  Returns:
    A list of OpenCLEnvironment instances.
  """
  return sorted(list(cldrive.all_envs()) + [oclgrind.OpenCLEnvironment()],
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if FLAGS.ls_env:
    PrintOpenClEnvironments()


if __name__ == '__main__':
  app.run(main)
