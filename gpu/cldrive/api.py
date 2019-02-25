"""Public API for cldrive."""

from absl import flags

from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = flags.FLAGS

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')


def Drive(instance: cldrive_pb2.CldriveInstance) -> cldrive_pb2.CldriveInstance:
  if instance.device.name == _env.OclgrindOpenCLEnvironment().name:
    command = [str(oclgrind.OCLGRIND_PATH), str(_NATIVE_DRIVER)]
  else:
    command = [str(_NATIVE_DRIVER)]

  pbutil.RunProcessMessageInPlace(command, instance)
  return instance
